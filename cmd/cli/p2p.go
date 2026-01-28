package cli

import (
	"bufio"
	"context"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/d0w/EdgeLLM/internal/p2p"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/multiformats/go-multiaddr"
	"github.com/spf13/cobra"
)

// quiet             bool
// workerModel       string
// workerPort        string
// workerP2PToken    string
// workerNetworkID   string
// workerHFCachePath string
var p2pPort int

var nodeCommand = &cobra.Command{
	Use:   "node",
	Short: "Test p2p",
	// Args:  cobra.ExactArgs(1),
	RunE: nodeHandler,
}

func init() {
	rootCmd.AddCommand(nodeCommand)
	rootCmd.Flags().IntVar(&p2pPort, "port", 8000, "P2P port to listen on")
}

func nodeDebugCLI(ctx context.Context, node *p2p.P2PNode) {
	reader := bufio.NewReader(os.Stdin)
	fmt.Println("P2P Node Debug CLI")
	fmt.Println("listen, close, ping <ping addr>")

OuterLoop:
	for {
		fmt.Print("> ")
		line, err := reader.ReadString('\n')
		if err != nil {
			fmt.Println("Error reading input:", err)
			continue
		}
		line = strings.TrimSpace(line)
		args := strings.Split(line, " ")

		switch args[0] {
		case "listen":
			break OuterLoop
		case "close":
			{
				if node == nil {
					fmt.Println("Node is not initialized")
					continue
				}
				_ = node.Close()
				node = nil
			}
		case "ping":
			{
				if node == nil {
					fmt.Println("Node is not initialized")
					continue
				}
				if len(args) < 2 {
					fmt.Println("Usage: ping <peerID>")
					continue
				}
				addr, err := multiaddr.NewMultiaddr(args[1])
				if err != nil {
					fmt.Println("Invalid peer addr:", err)
					continue
				}

				peer, err := peer.AddrInfoFromP2pAddr(addr)
				if err != nil {
					fmt.Println("Failed to get peer info:", err)
					continue
				}

				if err := node.Host.Connect(ctx, *peer); err != nil {
					fmt.Println("Failed to connect to peer:", err)
					continue
				}
				fmt.Println("sending 5 ping messages to", addr)
				ch := node.PingService.Ping(ctx, peer.ID)
				for i := 0; i < 5; i++ {
					res := <-ch
					fmt.Println("pinged", addr, "in", res.RTT)
				}

			}
		case "vpn":
			{
				if len(args) < 2 {
					fmt.Println("Usage: vpn <token>")
					continue
				}

				token := args[1]
				fmt.Printf("Starting VPN with token: %s\n", token)

				_, err := p2p.StartVPN(ctx, token, p2p.VPNOptions{
					Address:       "10.1.0.1/24",
					DNSAddress:    "",
					DNSForwarder:  true,
					DNSCacheSize:  200,
					DNSForwardSrv: []string{"8.8.8.8:53", "1.1.1.1:53"},
				})
				if err != nil {
					fmt.Printf("Failed to start VPN: %v\n", err)
					continue
				}
				fmt.Println("VPN started successfully!")
			}
		}
	}
}

func nodeHandler(cmd *cobra.Command, args []string) error {
	ctx := cmd.Context()

	capabilities := []string{"test", "test2"}

	node, err := p2p.NewP2PNode(ctx, capabilities, p2pPort)
	if err != nil {
		return fmt.Errorf("failed to create P2P node: %w", err)
	}

	slog.Info(fmt.Sprintf("Node Id: %s", node.Host.ID()))

	nodeDebugCLI(ctx, node)

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
	<-sigCh

	slog.Info("Shutting down P2P node...")

	return nil
}
