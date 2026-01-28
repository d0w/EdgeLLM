package p2p

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/ipfs/go-log"
	"github.com/mudler/edgevpn/pkg/config"
	"github.com/mudler/edgevpn/pkg/logger"
	edgevpn "github.com/mudler/edgevpn/pkg/node"
	"github.com/mudler/edgevpn/pkg/services"
	"github.com/mudler/edgevpn/pkg/vpn"
	"github.com/multiformats/go-multiaddr"
)

// VPNOptions contains configuration for starting a VPN
type VPNOptions struct {
	Address       string
	DNSAddress    string
	DNSForwarder  bool
	DNSCacheSize  int
	DNSForwardSrv []string
}

// StartVPN starts a VPN node with the given token and options
// This follows the edgevpn Main() pattern
func StartVPN(ctx context.Context, token string, opts VPNOptions) error {
	if token == "" {
		return fmt.Errorf("token is required")
	}

	fmt.Printf("Starting VPN with token: %s, address: %s\n", token, opts.Address)

	// Build the base config
	c := buildConfig(token, opts)

	logger := logger.New(log.LevelError)

	// Convert config to node options
	o, _, err := c.ToOpts(logger)
	if err != nil {
		return fmt.Errorf("failed to create options: %w", err)
	}

	// Build VPN options
	vpnOpts := []vpn.Option{}

	// Add alive service (required for VPN)
	o = append(o,
		services.Alive(
			30*time.Second,  // healthcheck interval
			900*time.Second, // scrub interval
			15*time.Minute,  // max interval
		)...,
	)

	// Add DNS if enabled
	if opts.DNSForwarder {
		o = append(o,
			services.DNS(
				logger,
				opts.DNSAddress,
				opts.DNSForwarder,
				opts.DNSForwardSrv,
				opts.DNSCacheSize,
			)...,
		)
	}

	// Register VPN service
	registeredOpts, err := vpn.Register(vpnOpts...)
	if err != nil {
		return fmt.Errorf("failed to register VPN: %w", err)
	}

	// Create the edgevpn instance
	e, err := edgevpn.New(append(o, registeredOpts...)...)
	if err != nil {
		return fmt.Errorf("failed to create edgevpn: %w", err)
	}

	log.Printf("VPN node created successfully")

	// Start the node
	return e.Start(ctx)
}

// buildConfig creates an edgevpn config from token and options
func buildConfig(token string, opts VPNOptions) config.Config {
	defaultInterval := 10 * time.Second

	// Check environment variables
	noDHT := os.Getenv("P2P_DISABLE_DHT") == "true"
	enableLimits := os.Getenv("ENABLE_LIMITS") == "true"

	var listenMaddrs []string
	var bootstrapPeers []string

	// Get listen addresses from environment
	if laddrs := os.Getenv("P2P_LISTEN_MADDRS"); laddrs != "" {
		listenMaddrs = strings.Split(laddrs, ",")
	}

	// Get bootstrap peers from environment
	if bootmaddr := os.Getenv("BOOTSTRAP_PEERS_MADDRS"); bootmaddr != "" {
		bootstrapPeers = strings.Split(bootmaddr, ",")
	}

	// Parse DHT announce addresses
	dhtAnnounceMaddrs := stringsToMultiAddr(
		strings.Split(os.Getenv("P2P_DHT_ANNOUNCE_MADDRS"), ","),
	)

	return config.Config{
		// Network configuration
		NetworkToken:      token,
		Interface:         getEnvOrDefault("VPN_INTERFACE", "edgevpn0"),
		Address:           opts.Address,
		ListenMaddrs:      listenMaddrs,
		DHTAnnounceMaddrs: dhtAnnounceMaddrs,

		// Limits
		Limit: config.ResourceLimit{
			Enable:   enableLimits,
			MaxConns: getEnvIntOrDefault("MAX_CONNECTIONS", 100),
		},

		// Logging
		LowProfile:     false,
		LogLevel:       getLogLevel(),
		Libp2pLogLevel: getLibp2pLogLevel(),

		// Ledger
		Ledger: config.Ledger{
			SyncInterval:     defaultInterval,
			AnnounceInterval: defaultInterval,
		},

		// NAT traversal
		NAT: config.NAT{
			Service:           true,
			Map:               true,
			RateLimit:         true,
			RateLimitGlobal:   100,
			RateLimitPeer:     100,
			RateLimitInterval: defaultInterval,
		},

		// Discovery
		Discovery: config.Discovery{
			DHT:            !noDHT,
			MDNS:           true,
			Interval:       defaultInterval,
			BootstrapPeers: bootstrapPeers,
		},

		// Connection settings
		Connection: config.Connection{
			HolePunch:      true,
			AutoRelay:      true,
			MaxConnections: getEnvIntOrDefault("MAX_CONNECTIONS", 1000),
		},
	}
}

// Helper functions

// stringsToMultiAddr converts string addresses to multiaddr.Multiaddr
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

// getLogLevel returns the log level from environment or default
func getLogLevel() string {
	return getEnvOrDefault("LOG_LEVEL", "info")
}

// getLibp2pLogLevel returns the libp2p log level from environment or default
func getLibp2pLogLevel() string {
	return getEnvOrDefault("LIBP2P_LOG_LEVEL", "fatal")
}

// getEnvOrDefault returns environment variable value or default
func getEnvOrDefault(key, defaultVal string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return defaultVal
}

// getEnvIntOrDefault returns environment variable as int or default
func getEnvIntOrDefault(key string, defaultVal int) int {
	if val := os.Getenv(key); val != "" {
		var intVal int
		if _, err := fmt.Sscanf(val, "%d", &intVal); err == nil {
			return intVal
		}
	}
	return defaultVal
}

// StartVPNWithToken is a convenience function to start VPN with just a token and default options
func StartVPNWithToken(ctx context.Context, token string) error {
	defaultOpts := VPNOptions{
		Address:       "10.1.0.0/24",
		DNSForwarder:  true,
		DNSCacheSize:  200,
		DNSForwardSrv: []string{"8.8.8.8:53", "1.1.1.1:53"},
	}
	return StartVPN(ctx, token, defaultOpts)
}
