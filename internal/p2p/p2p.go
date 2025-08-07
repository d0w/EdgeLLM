package p2p

//
// import (
// 	"context"
// 	"fmt"
// 	"log"
// 	"os"
// 	"strings"
//
// 	// "github.com/mudler/edgevpn/pkg/node"
// )
//
// func StartP2P(ctx context.Context, address, token, networkId string) error {
// 	var n *node.Node
//
// 	if token == "" {
// 		log.Print("No token provided, using generated token")
// 	}
//
// 	if n == nil {
// 		node, err := NewNode(ctx, token)
// 		if err != nil {
// 			return err
// 		}
// 		err = node.Start(ctx)
// 		if err != nil {
// 			return fmt.Errorf("failed to start node: %w", err)
// 		}
// 		n = node
//
// 		// Discovery
// 		log.Printf("Starting P2P node with address: %s, token: %s, networkId: %s", address, token, networkId)
// 		if err := ServiceDiscoverer(ctx, n, token, NetworkId(networkId, WorkerId), func(serviceID string, node NodeData) {
// 			var tunnelAddresses []string
// 			for _, v := range GetAvailableNodes(NetworkId(networkId, WorkerId)) {
// 				if v.IsOnline() {
// 					tunnelAddresses = append(tunnelAddresses, v.TunnelAddress)
// 				} else {
// 					log.Printf("Node %s is offline", v.ID)
// 				}
// 			}
// 			tunnelEnvVar := strings.Join(tunnelAddresses, ",")
//
// 			os.Setenv("GRPC_SERVERS", tunnelEnvVar)
// 			log.Printf("setting LLAMACPP_GRPC_SERVERS to %s", tunnelEnvVar)
// 		}, true); err != nil {
// 			return err
// 		}
//
// 	}
//
// 	return nil
// }
