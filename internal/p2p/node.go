// internal/p2p/host.go
package p2p

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	ipfs "github.com/ipfs/go-log"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/mudler/edgevpn/pkg/config"
	"github.com/mudler/edgevpn/pkg/logger"
	"github.com/mudler/edgevpn/pkg/node"
	"github.com/mudler/edgevpn/pkg/services"
	"github.com/multiformats/go-multiaddr"
	// "github.com/libp2p/go-libp2p/core/protocol"
)

type P2PNode struct {
	Node host.Host
	ctx  context.Context
}

func NewNode(ctx context.Context, token string) (*node.Node, error) {
	nodeOpts, err := newNodeOptions(token)
	if err != nil {
		return nil, err
	}

	n, err := node.New(nodeOpts...)
	if err != nil {
		return nil, fmt.Errorf("failed to create new node: %w", err)
	}

	return n, nil
}

func newNodeOptions(token string) ([]node.Option, error) {
	defaultInterval := 10 * time.Second

	noDHT := os.Getenv("P2P_DISABLE_DHT") == "true"
	noLimits := os.Getenv("ENABLE_LIMITS") == "true"

	var listenMaddrs []string
	var bootstrapPeers []string

	laddrs := os.Getenv("P2P_LISTEN_MADDRS")
	if laddrs != "" {
		listenMaddrs = strings.Split(laddrs, ",")
	}

	bootmaddr := os.Getenv("BOOTSTRAP_PEERS_MADDRS")
	if bootmaddr != "" {
		bootstrapPeers = strings.Split(bootmaddr, ",")
	}

	dhtAnnounceMaddrs := stringsToMultiAddr(strings.Split(os.Getenv("P2P_DHT_ANNOUNCE_MADDRS"), ","))
	// libp2ploglevel := os.Getenv("P2P_LIB_LOGLEVEL")
	// if libp2ploglevel == "" {
	// 	libp2ploglevel = "fatal"
	// }
	c := config.Config{
		ListenMaddrs:      listenMaddrs,
		DHTAnnounceMaddrs: dhtAnnounceMaddrs,
		Limit: config.ResourceLimit{
			Enable:   noLimits,
			MaxConns: 100,
		},
		NetworkToken:   token,
		LowProfile:     false,
		LogLevel:       "fatal",
		Libp2pLogLevel: "fatal",
		Ledger: config.Ledger{
			SyncInterval:     defaultInterval,
			AnnounceInterval: defaultInterval,
		},
		NAT: config.NAT{
			Service:           true,
			Map:               true,
			RateLimit:         true,
			RateLimitGlobal:   100,
			RateLimitPeer:     100,
			RateLimitInterval: defaultInterval,
		},
		Discovery: config.Discovery{
			DHT:            !noDHT,
			MDNS:           true,
			Interval:       10 * time.Second,
			BootstrapPeers: bootstrapPeers,
		},
		Connection: config.Connection{
			HolePunch:      true,
			AutoRelay:      true,
			MaxConnections: 1000,
		},
	}

	networkLogger := logger.New(ipfs.LevelFatal)
	nodeOpts, _, err := c.ToOpts(networkLogger)
	if err != nil {
		return nil, fmt.Errorf("creating options: %w", err)
	}

	nodeOpts = append(nodeOpts, services.Alive(30*time.Second, 900*time.Second, 15*time.Minute)...)

	return nodeOpts, nil
}

func stringsToMultiAddr(peers []string) []multiaddr.Multiaddr {
	var maddrs []multiaddr.Multiaddr
	for _, p := range peers {
		if p == "" {
			continue
		}
		addr, err := multiaddr.NewMultiaddr(p)
		if err != nil {
			log.Printf("failed to parse multiaddr %s: %v", p, err)
			continue
		}
		maddrs = append(maddrs, addr)
	}
	return maddrs
}
