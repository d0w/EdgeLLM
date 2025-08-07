// internal/p2p/host.go
package p2p

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/libp2p/go-libp2p"
	dht "github.com/libp2p/go-libp2p-kad-dht"
	pubsub "github.com/libp2p/go-libp2p-pubsub"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/p2p/discovery/mdns"
	"github.com/libp2p/go-libp2p/p2p/discovery/routing"
)

var (
	discoveredNodes = make(map[string]NodeData)
	nodesMutex      sync.RWMutex
	WorkerId        = "worker"
)

// NodeData represents a discovered node in the network
type NodeData struct {
	ID            string    `json:"id"`
	TunnelAddress string    `json:"tunnel_address"`
	Address       string    `json:"address"`
	Port          int       `json:"port"`
	LastSeen      time.Time `json:"last_seen"`
	Status        string    `json:"status"`
	Capabilities  []string  `json:"capabilities"`
	PeerID        string    `json:"peer_id"`
}

// IsOnline checks if the node is considered online
func (n NodeData) IsOnline() bool {
	return time.Since(n.LastSeen) < 5*time.Minute && n.Status == "online"
}

// NetworkId creates a network identifier for a given network and service type
func NetworkId(networkId, serviceType string) string {
	return fmt.Sprintf("%s_%s", networkId, serviceType)
}

// P2PNode represents a libp2p node with discovery capabilities
type P2PNode struct {
	Host         host.Host
	DHT          *dht.IpfsDHT
	PubSub       *pubsub.PubSub
	Discovery    *routing.RoutingDiscovery
	MDNSService  mdns.Service
	ctx          context.Context
	cancel       context.CancelFunc
	capabilities []string
	servicePort  int
}

// NewP2PNode creates a new P2P node with discovery capabilities
func NewP2PNode(ctx context.Context, capabilities []string, servicePort int) (*P2PNode, error) {
	// Create a new libp2p host
	h, err := libp2p.New(
		libp2p.ListenAddrStrings("/ip4/0.0.0.0/tcp/0"),
		libp2p.Ping(false),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create libp2p host: %w", err)
	}

	// Create DHT for peer discovery
	kadDHT, err := dht.New(ctx, h)
	if err != nil {
		return nil, fmt.Errorf("failed to create DHT: %w", err)
	}

	// Bootstrap the DHT
	if err = kadDHT.Bootstrap(ctx); err != nil {
		return nil, fmt.Errorf("failed to bootstrap DHT: %w", err)
	}

	// Create pubsub
	ps, err := pubsub.NewGossipSub(ctx, h)
	if err != nil {
		return nil, fmt.Errorf("failed to create pubsub: %w", err)
	}

	// Create routing discovery
	routingDiscovery := routing.NewRoutingDiscovery(kadDHT)

	nodeCtx, cancel := context.WithCancel(ctx)

	node := &P2PNode{
		Host:         h,
		DHT:          kadDHT,
		PubSub:       ps,
		Discovery:    routingDiscovery,
		ctx:          nodeCtx,
		cancel:       cancel,
		capabilities: capabilities,
		servicePort:  servicePort,
	}

	// Setup MDNS discovery
	if err := node.setupMDNS(); err != nil {
		log.Printf("Failed to setup MDNS: %v", err)
	}

	return node, nil
}

// setupMDNS sets up mDNS discovery for local network
func (p *P2PNode) setupMDNS() error {
	mdnsService := mdns.NewMdnsService(p.Host, "vllm-cluster", p)
	return mdnsService.Start()
}

// HandlePeerFound implements the mdns notifier interface
func (p *P2PNode) HandlePeerFound(pi peer.AddrInfo) {
	log.Printf("Found peer via mDNS: %s", pi.ID)

	// Connect to the peer
	if err := p.Host.Connect(p.ctx, pi); err != nil {
		log.Printf("Failed to connect to peer %s: %v", pi.ID, err)
		return
	}

	// Add to discovered nodes
	p.addDiscoveredPeer(pi)
}

// ServiceDiscoverer starts the service discovery for the P2P network
func (p *P2PNode) ServiceDiscoverer(ctx context.Context, serviceId string, callback func(string, NodeData), advertise bool) error {
	if advertise {
		// Advertise our service
		nodeData := NodeData{
			ID:            p.GetNodeID(),
			TunnelAddress: p.getTunnelAddress(),
			Address:       "localhost",
			Port:          p.servicePort,
			LastSeen:      time.Now(),
			Status:        "online",
			Capabilities:  p.capabilities,
			PeerID:        p.Host.ID().String(),
		}

		if err := p.advertiseService(ctx, serviceId, nodeData); err != nil {
			log.Printf("Failed to advertise service: %v", err)
		} else {
			log.Printf("Advertising service %s with data: %+v", serviceId, nodeData)
		}
	}

	// Start discovery loop
	go p.discoveryLoop(ctx, serviceId, callback)

	return nil
}

// advertiseService advertises our service on the DHT
func (p *P2PNode) advertiseService(ctx context.Context, serviceId string, nodeData NodeData) error {
	// Announce ourselves on the DHT
	_, err := p.Discovery.Advertise(ctx, serviceId)
	if err != nil {
		return fmt.Errorf("failed to advertise on DHT: %w", err)
	}

	// Also publish our node data via pubsub
	topic, err := p.PubSub.Join(serviceId)
	if err != nil {
		return fmt.Errorf("failed to join pubsub topic: %w", err)
	}

	nodeDataBytes, err := json.Marshal(nodeData)
	if err != nil {
		return fmt.Errorf("failed to marshal node data: %w", err)
	}

	if err := topic.Publish(ctx, nodeDataBytes); err != nil {
		return fmt.Errorf("failed to publish node data: %w", err)
	}

	return nil
}

// discoveryLoop runs the discovery process
func (p *P2PNode) discoveryLoop(ctx context.Context, serviceId string, callback func(string, NodeData)) {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	// Subscribe to the pubsub topic for real-time updates
	topic, err := p.PubSub.Join(serviceId)
	if err != nil {
		log.Printf("Failed to join pubsub topic: %v", err)
		return
	}

	sub, err := topic.Subscribe()
	if err != nil {
		log.Printf("Failed to subscribe to topic: %v", err)
		return
	}
	defer sub.Cancel()

	// Handle pubsub messages in a separate goroutine
	go p.handlePubSubMessages(ctx, sub, serviceId, callback)

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			// Discover peers via DHT
			peerChan, err := p.Discovery.FindPeers(ctx, serviceId)
			if err != nil {
				log.Printf("Failed to find peers: %v", err)
				continue
			}

			// Process discovered peers
			go p.processPeers(ctx, peerChan, serviceId, callback)
		}
	}
}

// handlePubSubMessages processes incoming pubsub messages
func (p *P2PNode) handlePubSubMessages(ctx context.Context, sub *pubsub.Subscription, serviceId string, callback func(string, NodeData)) {
	for {
		msg, err := sub.Next(ctx)
		if err != nil {
			if ctx.Err() != nil {
				return
			}
			log.Printf("Failed to get next pubsub message: %v", err)
			continue
		}

		// Skip our own messages
		if msg.ReceivedFrom == p.Host.ID() {
			continue
		}

		var nodeData NodeData
		if err := json.Unmarshal(msg.Data, &nodeData); err != nil {
			log.Printf("Failed to unmarshal node data from pubsub: %v", err)
			continue
		}

		nodeData.LastSeen = time.Now()
		p.storeDiscoveredNode(nodeData)

		if callback != nil {
			callback(serviceId, nodeData)
		}

		log.Printf("Discovered node via pubsub: %s at %s", nodeData.ID, nodeData.TunnelAddress)
	}
}

// processPeers processes discovered peers from DHT
func (p *P2PNode) processPeers(ctx context.Context, peerChan <-chan peer.AddrInfo, serviceId string, callback func(string, NodeData)) {
	for {
		select {
		case <-ctx.Done():
			return
		case pi, ok := <-peerChan:
			if !ok {
				return
			}

			// Connect to peer
			if err := p.Host.Connect(ctx, pi); err != nil {
				log.Printf("Failed to connect to peer %s: %v", pi.ID, err)
				continue
			}

			p.addDiscoveredPeer(pi)
		}
	}
}

// addDiscoveredPeer adds a peer to our discovered nodes
func (p *P2PNode) addDiscoveredPeer(pi peer.AddrInfo) {
	nodeData := NodeData{
		ID:            pi.ID.String(),
		TunnelAddress: p.getAddressFromPeerInfo(pi),
		Address:       "localhost",
		Port:          50051,
		LastSeen:      time.Now(),
		Status:        "online",
		Capabilities:  []string{"vllm", "inference"},
		PeerID:        pi.ID.String(),
	}

	p.storeDiscoveredNode(nodeData)
}

// storeDiscoveredNode stores a node in the global map
func (p *P2PNode) storeDiscoveredNode(nodeData NodeData) {
	nodesMutex.Lock()
	discoveredNodes[nodeData.ID] = nodeData
	nodesMutex.Unlock()
}

// GetAvailableNodes returns all discovered nodes for a given network
func GetAvailableNodes(networkId string) map[string]NodeData {
	nodesMutex.RLock()
	defer nodesMutex.RUnlock()

	result := make(map[string]NodeData)
	for id, node := range discoveredNodes {
		result[id] = node
	}

	return result
}

// GetNodeID returns the node ID
func (p *P2PNode) GetNodeID() string {
	return p.Host.ID().String()
}

// getTunnelAddress gets the tunnel address for this node
func (p *P2PNode) getTunnelAddress() string {
	addrs := p.Host.Addrs()
	if len(addrs) > 0 {
		return addrs[0].String() + "/p2p/" + p.Host.ID().String()
	}
	return ""
}

// getAddressFromPeerInfo extracts address from peer info
func (p *P2PNode) getAddressFromPeerInfo(pi peer.AddrInfo) string {
	if len(pi.Addrs) > 0 {
		return pi.Addrs[0].String() + "/p2p/" + pi.ID.String()
	}
	return ""
}

// StartDiscovery starts the discovery service
// func (p *P2PNode) StartDiscovery() error {
// 	log.Printf("Started P2P discovery for node: %s", p.Host.ID())
// 	return nil
// }

// Close closes the P2P node and cleans up resources
// func (p *P2PNode) Close() error {
// 	p.cancel()
//
// 	if p.MDNSService != nil {
// 		if err := p.MDNSService.Close(); err != nil {
// 			log.Printf("Failed to close MDNS service: %v", err)
// 		}
// 	}
//
// 	if err := p.DHT.Close(); err != nil {
// 		log.Printf("Failed to close DHT: %v", err)
// 	}
//
// 	if err := p.Host.Close(); err != nil {
// 		log.Printf("Failed to close host: %v", err)
// 	}
//
// 	return nil
// }

// func newNodeOptions(token string) ([]node.Option, error) {
// 	defaultInterval := 10 * time.Second
//
// 	noDHT := os.Getenv("P2P_DISABLE_DHT") == "true"
// 	noLimits := os.Getenv("ENABLE_LIMITS") == "true"
//
// 	var listenMaddrs []string
// 	var bootstrapPeers []string
//
// 	laddrs := os.Getenv("P2P_LISTEN_MADDRS")
// 	if laddrs != "" {
// 		listenMaddrs = strings.Split(laddrs, ",")
// 	}
//
// 	bootmaddr := os.Getenv("BOOTSTRAP_PEERS_MADDRS")
// 	if bootmaddr != "" {
// 		bootstrapPeers = strings.Split(bootmaddr, ",")
// 	}
//
// 	dhtAnnounceMaddrs := stringsToMultiAddr(strings.Split(os.Getenv("P2P_DHT_ANNOUNCE_MADDRS"), ","))
// 	// libp2ploglevel := os.Getenv("P2P_LIB_LOGLEVEL")
// 	// if libp2ploglevel == "" {
// 	// 	libp2ploglevel = "fatal"
// 	// }
// 	c := config.Config{
// 		ListenMaddrs:      listenMaddrs,
// 		DHTAnnounceMaddrs: dhtAnnounceMaddrs,
// 		Limit: config.ResourceLimit{
// 			Enable:   noLimits,
// 			MaxConns: 100,
// 		},
// 		NetworkToken:   token,
// 		LowProfile:     false,
// 		LogLevel:       "fatal",
// 		Libp2pLogLevel: "fatal",
// 		Ledger: config.Ledger{
// 			SyncInterval:     defaultInterval,
// 			AnnounceInterval: defaultInterval,
// 		},
// 		NAT: config.NAT{
// 			Service:           true,
// 			Map:               true,
// 			RateLimit:         true,
// 			RateLimitGlobal:   100,
// 			RateLimitPeer:     100,
// 			RateLimitInterval: defaultInterval,
// 		},
// 		Discovery: config.Discovery{
// 			DHT:            !noDHT,
// 			MDNS:           true,
// 			Interval:       10 * time.Second,
// 			BootstrapPeers: bootstrapPeers,
// 		},
// 		Connection: config.Connection{
// 			HolePunch:      true,
// 			AutoRelay:      true,
// 			MaxConnections: 1000,
// 		},
// 	}
//
// 	networkLogger := logger.New(ipfs.LevelFatal)
// 	nodeOpts, _, err := c.ToOpts(networkLogger)
// 	if err != nil {
// 		return nil, fmt.Errorf("creating options: %w", err)
// 	}
//
// 	nodeOpts = append(nodeOpts, services.Alive(30*time.Second, 900*time.Second, 15*time.Minute)...)
//
// 	return nodeOpts, nil
// }
//
// // node.Host.ID()
// func GetNodeID(n *node.Node) string {
// 	if n == nil || n.Host == nil {
// 		return ""
// 	}
// 	// Generate a random number between 0 and 99
// 	randNumber := rand.Intn(1000)
// 	// return random number
// 	return string(randNumber)
// }
//
// func stringsToMultiAddr(peers []string) []multiaddr.Multiaddr {
// 	var maddrs []multiaddr.Multiaddr
// 	for _, p := range peers {
// 		if p == "" {
// 			continue
// 		}
// 		addr, err := multiaddr.NewMultiaddr(p)
// 		if err != nil {
// 			log.Printf("failed to parse multiaddr %s: %v", p, err)
// 			continue
// 		}
// 		maddrs = append(maddrs, addr)
// 	}
// 	return maddrs
// }
