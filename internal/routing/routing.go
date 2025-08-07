package routing

//
// import (
// 	"container/heap"
// 	"context"
// 	"log"
//
// 	"github.com/d0w/EdgeLLM/internal/p2p"
// 	"github.com/d0w/EdgeLLM/pkg/pq"
// 	"github.com/mudler/edgevpn/pkg/node"
// )
//
// type Route struct {
// 	Nodes    []*node.Node
// 	Distance int // number of hops
// }
//
// type RoutingRequest struct {
// 	RequiredLayers []string // layers that need to be collected
// 	MaxHops        int      // maximum hops allowed
// 	Context        context.Context
// }
//
// /*
// * We need to know:
// *   1. What nodes are available
// *   2. What nodes have what capabilities (model, layers, etc.)
// *   3. The fastest path to run inference through the network
//  */
//
// // State represents the current state in our pathfinding
// type routingState struct {
// 	currentNode     *node.Node
// 	visitedNodes    []*node.Node
// 	collectedLayers map[string]bool
// 	distance        int
// }
//
// // hasAllLayers checks if all required layers have been collected
// func (s *routingState) hasAllLayers(requiredLayers []string) bool {
// 	for _, layer := range requiredLayers {
// 		if !s.collectedLayers[layer] {
// 			return false
// 		}
// 	}
// 	return true
// }
//
// type DHTInterface interface {
// 	// TODO: Capability needs to be a defined type
// 	FindNodesByCapability(ctx context.Context, capability string) ([]*node.Node, error)
// 	GetNodeInfo(ctx context.Context, nodeID string) (*node.Node, error)
// 	GetConnectedNodes(ctx context.Context) ([]*node.Node, error)
// }
//
// // getNodeLayers extracts layers from a node (assuming node has a Layers field)
// func getNodeLayers(n *node.Node) []string {
// 	// This is a placeholder - you'll need to implement based on your node structure
// 	// For now, assuming nodes have some way to expose their layers
// 	// You might need to modify this based on the actual node.Node implementation
// 	return []string{} // Replace with actual layer extraction
// }
//
// func calculateEdgeCost(from, to *node.Node, collectedLayers map[string]bool, requiredLayers []string) float32 {
// 	// Placeholder for edge cost calculation logic
// 	// This could be based on distance, latency, or other factors
// 	return 1
// }
//
// // dijkstra's algorithm implementation to find the shortest path
// func findShortestPathDHT(source *node.Node, dht DHTInterface, request RoutingRequest) *Route {
// 	if len(request.RequiredLayers) == 0 {
// 		return &Route{Nodes: []*node.Node{source}, Distance: 0}
// 	}
//
// 	distances := make(map[string]float32)
// 	previous := make(map[string]*routingState)
// 	visited := make(map[string]bool)
//
// 	queue := &pq.PriorityQueue[routingState, float32]{}
//
// 	nodeCache := make(map[string]*node.Node)
// 	sourceID := p2p.GetNodeID(source)
// 	nodeCache[sourceID] = source
//
// 	initialLayers := make(map[string]bool)
// 	for _, layer := range getNodeLayers(source) {
// 		initialLayers[layer] = true
// 	}
//
// 	initial := &routingState{
// 		currentNode:     source,
// 		visitedNodes:    []*node.Node{source},
// 		collectedLayers: initialLayers,
// 		distance:        0,
// 	}
//
// 	initialKey := generateStateKey(source, initialLayers)
// 	distances[initialKey] = 0
//
// 	heap.Push(queue, &pq.Item[*routingState, float32]{
// 		Value: initial,
// 		Cost:  0,
// 	})
//
// 	for queue.Len() > 0 {
// 		current, ok := heap.Pop(queue).(*pq.Item[*routingState, float32])
// 		if !ok {
// 			log.Fatal("Failed to pop item from priority queue")
// 		}
//
// 		currentKey := generateStateKey(current.Value.currentNode, current.Value.collectedLayers)
//
// 		if visited[currentKey] {
// 			continue
// 		}
//
// 		visited[currentKey] = true
//
// 		if current.Value.hasAllLayers(request.RequiredLayers) {
// 			return &Route{
// 				Nodes:    current.Value.visitedNodes,
// 				Distance: current.Value.distance,
// 			}
// 		}
//
// 		if request.MaxHops > 0 && current.Value.distance >= request.MaxHops {
// 			continue
// 		}
//
// 		missingLayers := []string{}
// 		for _, layer := range request.RequiredLayers {
// 			if !current.Value.collectedLayers[layer] {
// 				missingLayers = append(missingLayers, layer)
// 			}
// 		}
//
// 		candidateNodes := []*node.Node{}
// 		for _, layer := range missingLayers {
// 			nodes, err := dht.FindNodesByCapability(request.Context, layer)
// 			if err != nil {
// 				continue
// 			}
// 			candidateNodes = append(candidateNodes, nodes...)
// 		}
//
// 		connectedNodes, err := dht.GetConnectedNodes(request.Context)
// 		if err == nil {
// 			candidateNodes = append(candidateNodes, connectedNodes...)
// 		}
//
// 		for _, nextNode := range candidateNodes {
// 			nextNodeID := p2p.GetNodeID(nextNode)
//
// 			alreadyVisited := false
// 			for _, visitedNode := range current.Value.visitedNodes {
// 				if p2p.GetNodeID(visitedNode) == nextNodeID {
// 					alreadyVisited = true
// 					break
// 				}
// 			}
//
// 			if alreadyVisited {
// 				continue
// 			}
//
// 			nodeCache[nextNodeID] = nextNode
//
// 			newLayers := make(map[string]bool)
// 			for k, v := range current.Value.collectedLayers {
// 				newLayers[k] = v
// 			}
//
// 			for _, layer := range getNodeLayers(nextNode) {
// 				newLayers[layer] = true
// 			}
//
// 			newState := &routingState{
// 				currentNode:     nextNode,
// 				visitedNodes:    append(current.Value.visitedNodes, nextNode),
// 				collectedLayers: newLayers,
// 				distance:        current.Value.distance + 1,
// 			}
//
// 			edgeCost := calculateEdgeCost(current.Value.currentNode, nextNode, newLayers, request.RequiredLayers)
// 			newCost := current.Cost + edgeCost
// 			newKey := generateStateKey(nextNode, newLayers)
//
// 			if existingCost, found := distances[newKey]; !found || newCost < existingCost {
// 				distances[newKey] = newCost
// 				previous[newKey] = current.Value
//
// 				heap.Push(queue, &pq.Item[*routingState, float32]{
// 					Value: newState,
// 					Cost:  newCost,
// 				})
//
// 			}
//
// 		}
// 	}
//
// 	// no path found
// 	return nil
// }
//
// func generateStateKey(node *node.Node, layers map[string]bool) string {
// 	key := p2p.GetNodeID(node)
// 	for layer, collected := range layers {
// 		if collected {
// 			key += ":" + layer
// 		}
// 	}
// 	return key
// }
//
// func NewRoute(source *node.Node, destination *node.Node, nodes []node.Node) *Route {
// 	// Simple direct route for backward compatibility
// 	return &Route{
// 		Nodes:    []*node.Node{source, destination},
// 		Distance: 1,
// 	}
// }
//
// // NewRouteWithLayers finds the shortest path that collects all required layers
// func NewRouteWithLayers(source *node.Node, dht DHTInterface, request RoutingRequest) *Route {
// 	return findShortestPathDHT(source, dht, request)
// }
