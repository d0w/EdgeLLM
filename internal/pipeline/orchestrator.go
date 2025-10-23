package pipeline

// import (
// 	"context"
// 	"fmt"
// 	"log"
// 	"sync"
// 	"time"
//
// 	pb "github.com/d0w/EdgeLLM/backend/go/proto"
// 	"github.com/d0w/EdgeLLM/internal/p2p"
// 	"github.com/d0w/EdgeLLM/internal/routing"
// 	"github.com/mudler/edgevpn/pkg/node"
// 	"google.golang.org/grpc"
// 	"google.golang.org/grpc/credentials/insecure"
// )
//
// type PipelineOrchestrator struct {
// 	localNode        *node.Node
// 	dht              routing.DHTInterface
// 	modelName        string
// 	totalLayers      int32
// 	nodeCapabilities map[string]*pb.CapabilityResponse
// 	nodeConnections  map[string]*NodeConnection
// 	mu               sync.RWMutex
// 	logger           *log.Logger
// }
//
// type InferenceRoute struct {
// 	Nodes      []*NodeConnection
// 	LayerSpans []LayerSpan
// 	RequestID  string
// }
//
// type NodeConnection struct {
// 	Node       *node.Node
// 	GRPCClient pb.BackendClient
// 	Conn       *grpc.ClientConn
// 	Address    string
// 	NodeID     string
// 	NodeData   *p2p.NodeData
// }
//
// type LayerSpan struct {
// 	StartLayer int32
// 	EndLayer   int32
// 	NodeID     string
// }
//
// func NewPipelineOrchestrator(localNode *node.Node, dht routing.DHTInterface, modelName string) *PipelineOrchestrator {
// 	return &PipelineOrchestrator{
// 		localNode:        localNode,
// 		dht:              dht,
// 		modelName:        modelName,
// 		nodeCapabilities: make(map[string]*pb.CapabilityResponse),
// 		nodeConnections:  make(map[string]*NodeConnection),
// 		logger:           log.New(log.Writer(), "[PipelineOrchestrator] ", log.LstdFlags),
// 	}
// }
//
// // DistributedInference performs inference across multiple nodes using pipeline parallelism
// func (po *PipelineOrchestrator) DistributedInference(ctx context.Context, prompt string, temperature float32, topP float32, maxTokens int32) (*pb.GenerateTextResponse, error) {
// 	po.logger.Printf("Starting distributed inference for prompt: %.50s...", prompt)
//
// 	// Discover available nodes and their capabilities
// 	route, err := po.discoverAndPlanRoute(ctx)
// 	if err != nil {
// 		return nil, fmt.Errorf("failed to plan inference route: %w", err)
// 	}
//
// 	po.logger.Printf("Planned inference route with %d nodes across %d layer spans", len(route.Nodes), len(route.LayerSpans))
//
// 	// 2. Execute the pipeline
// 	response, err := po.executePipeline(ctx, route, prompt, temperature, topP, maxTokens)
// 	if err != nil {
// 		return nil, fmt.Errorf("pipeline execution failed: %w", err)
// 	}
//
// 	po.logger.Printf("Distributed inference completed successfully")
// 	return response, nil
// }
//
// func (po *PipelineOrchestrator) discoverAndPlanRoute(ctx context.Context) (*InferenceRoute, error) {
// 	po.logger.Println("Discovering available nodes...")
//
// 	// Get all available nodes
// 	allNodes := p2p.GetAvailableNodes("")
// 	var availableNodes []*p2p.NodeData
//
// 	for _, nodeData := range allNodes {
// 		if nodeData.IsOnline() && nodeData.ModelName == po.modelName {
// 			availableNodes = append(availableNodes, nodeData)
// 		}
// 	}
//
// 	if len(availableNodes) == 0 {
// 		return nil, fmt.Errorf("no nodes available with model %s", po.modelName)
// 	}
//
// 	po.logger.Printf("Found %d available nodes with model %s", len(availableNodes), po.modelName)
//
// 	// Establish connections to nodes
// 	var nodeConnections []*NodeConnection
// 	for _, nodeData := range availableNodes {
// 		nodeConn, err := po.connectToNodeData(ctx, nodeData)
// 		if err != nil {
// 			po.logger.Printf("Failed to connect to node %s: %v", nodeData.ID, err)
// 			continue
// 		}
//
// 		// Verify capabilities
// 		caps, err := nodeConn.GRPCClient.GetNodeCapabilities(ctx, &pb.CapabilityRequest{})
// 		if err != nil {
// 			po.logger.Printf("Failed to get capabilities from node %s: %v", nodeData.ID, err)
// 			nodeConn.Conn.Close()
// 			continue
// 		}
//
// 		if caps.ModelName == po.modelName {
// 			nodeConnections = append(nodeConnections, nodeConn)
// 			po.mu.Lock()
// 			po.nodeCapabilities[caps.NodeId] = caps
// 			po.nodeConnections[caps.NodeId] = nodeConn
// 			po.mu.Unlock()
// 		} else {
// 			nodeConn.Conn.Close()
// 		}
// 	}
//
// 	if len(nodeConnections) == 0 {
// 		return nil, fmt.Errorf("no nodes could be connected for model %s", po.modelName)
// 	}
//
// 	// Plan the layer distribution
// 	layerSpans, err := po.planOptimalLayerDistribution(nodeConnections)
// 	if err != nil {
// 		return nil, fmt.Errorf("failed to plan layer distribution: %w", err)
// 	}
//
// 	return &InferenceRoute{
// 		Nodes:      nodeConnections,
// 		LayerSpans: layerSpans,
// 		RequestID:  generateRequestID(),
// 	}, nil
// }
//
// func (po *PipelineOrchestrator) planOptimalLayerDistribution(nodes []*NodeConnection) ([]LayerSpan, error) {
// 	po.logger.Printf("Planning optimal layer distribution across %d nodes", len(nodes))
//
// 	po.mu.RLock()
// 	defer po.mu.RUnlock()
//
// 	// Build a coverage map
// 	layerCoverage := make(map[int32][]*NodeConnection)
// 	var maxLayer int32 = 0
//
// 	for _, nodeConn := range nodes {
// 		caps, exists := po.nodeCapabilities[nodeConn.NodeID]
// 		if !exists {
// 			continue
// 		}
//
// 		for _, layer := range caps.AvailableLayers {
// 			layerCoverage[layer] = append(layerCoverage[layer], nodeConn)
// 			if layer > maxLayer {
// 				maxLayer = layer
// 			}
// 		}
// 	}
//
// 	// Create consecutive layer spans
// 	var spans []LayerSpan
// 	var currentSpan *LayerSpan
// 	nodeUsageCount := make(map[string]int)
//
// 	for layer := int32(0); layer <= maxLayer; layer++ {
// 		availableNodes := layerCoverage[layer]
// 		if len(availableNodes) == 0 {
// 			// Gap in coverage - end current span if exists
// 			if currentSpan != nil {
// 				spans = append(spans, *currentSpan)
// 				currentSpan = nil
// 			}
// 			continue
// 		}
//
// 		// Choose the best node for this layer (least used)
// 		var bestNode *NodeConnection
// 		minUsage := int(^uint(0) >> 1) // Max int
// 		for _, node := range availableNodes {
// 			if usage := nodeUsageCount[node.NodeID]; usage < minUsage {
// 				minUsage = usage
// 				bestNode = node
// 			}
// 		}
//
// 		if bestNode == nil {
// 			continue
// 		}
//
// 		// Check if we can extend the current span
// 		if currentSpan != nil && currentSpan.NodeID == bestNode.NodeID && currentSpan.EndLayer == layer {
// 			// Extend current span
// 			currentSpan.EndLayer = layer + 1
// 		} else {
// 			// End current span if exists
// 			if currentSpan != nil {
// 				spans = append(spans, *currentSpan)
// 			}
// 			// Start new span
// 			currentSpan = &LayerSpan{
// 				StartLayer: layer,
// 				EndLayer:   layer + 1,
// 				NodeID:     bestNode.NodeID,
// 			}
// 		}
//
// 		nodeUsageCount[bestNode.NodeID]++
// 	}
//
// 	// Add the final span
// 	if currentSpan != nil {
// 		spans = append(spans, *currentSpan)
// 	}
//
// 	po.logger.Printf("Created %d layer spans covering layers 0-%d", len(spans), maxLayer)
//
// 	// Verify we have complete coverage
// 	if len(spans) == 0 {
// 		return nil, fmt.Errorf("no valid layer spans could be created")
// 	}
//
// 	return spans, nil
// }
//
// func (po *PipelineOrchestrator) executePipeline(ctx context.Context, route *InferenceRoute, prompt string, temperature, topP float32, maxTokens int32) (*pb.GenerateTextResponse, error) {
// 	po.logger.Printf("Executing pipeline with %d spans", len(route.LayerSpans))
//
// 	var currentHiddenStates []byte
// 	var finalResponse *pb.GenerateTextResponse
//
// 	for i, span := range route.LayerSpans {
// 		po.logger.Printf("Processing span %d: layers %d-%d on node %s", i, span.StartLayer, span.EndLayer-1, span.NodeID)
//
// 		nodeConn := po.getNodeConnection(span.NodeID)
// 		if nodeConn == nil {
// 			return nil, fmt.Errorf("node connection not found for node %s", span.NodeID)
// 		}
//
// 		if i == 0 {
// 			// First span: process the prompt
// 			req := &pb.GenerateTextRequest{
// 				Prompt:       prompt,
// 				Temperature:  temperature,
// 				TopP:         topP,
// 				MaxTokens:    maxTokens,
// 				PipelineMode: true,
// 			}
//
// 			response, err := nodeConn.GRPCClient.GenerateText(ctx, req)
// 			if err != nil {
// 				return nil, fmt.Errorf("failed to process first span: %w", err)
// 			}
//
// 			// In a real implementation, this would extract hidden states
// 			currentHiddenStates = []byte(response.Text) // Simplified
// 			finalResponse = response
// 		} else {
// 			// Subsequent spans: process hidden states
// 			layerReq := &pb.LayerProcessRequest{
// 				HiddenStates: currentHiddenStates,
// 				LayerStart:   span.StartLayer,
// 				LayerEnd:     span.EndLayer,
// 				RequestId:    route.RequestID,
// 				Metadata: map[string]string{
// 					"model":       po.modelName,
// 					"span_index":  fmt.Sprintf("%d", i),
// 					"total_spans": fmt.Sprintf("%d", len(route.LayerSpans)),
// 				},
// 			}
//
// 			layerResp, err := nodeConn.GRPCClient.ProcessLayer(ctx, layerReq)
// 			if err != nil {
// 				return nil, fmt.Errorf("failed to process layers %d-%d on node %s: %w",
// 					span.StartLayer, span.EndLayer-1, span.NodeID, err)
// 			}
//
// 			if !layerResp.Success {
// 				return nil, fmt.Errorf("layer processing failed on node %s: %s",
// 					span.NodeID, layerResp.ErrorMessage)
// 			}
//
// 			currentHiddenStates = layerResp.HiddenStates
// 		}
// 	}
//
// 	// Final processing - convert hidden states to text
// 	if len(route.LayerSpans) > 1 {
// 		// If we used multiple spans, we need to decode the final hidden states
// 		finalResponse = &pb.GenerateTextResponse{
// 			Text:    string(currentHiddenStates), // Simplified - would decode properly
// 			IsFinal: true,
// 		}
// 	}
//
// 	return finalResponse, nil
// }
//
// func (po *PipelineOrchestrator) connectToNodeData(ctx context.Context, nodeData *p2p.NodeData) (*NodeConnection, error) {
// 	address := nodeData.GRPCAddress
// 	po.logger.Printf("Connecting to node %s at %s", nodeData.ID, address)
//
// 	conn, err := grpc.NewClient(address,
// 		grpc.WithTransportCredentials(insecure.NewCredentials()),
// 		grpc.WithTimeout(10*time.Second))
// 	if err != nil {
// 		return nil, fmt.Errorf("failed to connect to %s: %w", address, err)
// 	}
//
// 	client := pb.NewBackendClient(conn)
//
// 	// Test the connection
// 	_, err = client.Health(ctx, &pb.HealthRequest{})
// 	if err != nil {
// 		conn.Close()
// 		return nil, fmt.Errorf("health check failed for %s: %w", address, err)
// 	}
//
// 	return &NodeConnection{
// 		Node:       nil, // We don't have the actual node object
// 		GRPCClient: client,
// 		Conn:       conn,
// 		Address:    address,
// 		NodeID:     nodeData.ID,
// 		NodeData:   nodeData,
// 	}, nil
// }
//
// func (po *PipelineOrchestrator) getNodeConnection(nodeID string) *NodeConnection {
// 	po.mu.RLock()
// 	defer po.mu.RUnlock()
// 	return po.nodeConnections[nodeID]
// }
//
// func (po *PipelineOrchestrator) Close() {
// 	po.mu.Lock()
// 	defer po.mu.Unlock()
//
// 	po.logger.Println("Closing pipeline orchestrator connections")
// 	for _, conn := range po.nodeConnections {
// 		if conn.Conn != nil {
// 			conn.Conn.Close()
// 		}
// 	}
// 	po.nodeConnections = make(map[string]*NodeConnection)
// }
//
// // Helper functions
// func generateRequestID() string {
// 	return fmt.Sprintf("req_%d", time.Now().UnixNano())
// }
//
