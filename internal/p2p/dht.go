package p2p

// import (
// 	"context"
// 	"fmt"
// 	"log"
// 	"strings"
//
// 	"github.com/d0w/EdgeLLM/internal/routing"
// 	"github.com/mudler/edgevpn/pkg/node"
// )
//
// type DHT struct {
// 	localNode *node.Node
// 	logger    *log.Logger
// }
//
// func NewDHT(localNode *node.Node) *DHT {
// 	return &DHT{
// 		localNode: localNode,
// 		logger:    log.New(log.Writer(), "[DHT] ", log.LstdFlags),
// 	}
// }
//
// // Implement the DHTInterface from routing package
// func (d *DHT) FindNodesByCapability(ctx context.Context, capability string) ([]*node.Node, error) {
// 	d.logger.Printf("Finding nodes with capability: %s", capability)
//
// 	var matchingNodes []*node.Node
// 	allNodes := GetAvailableNodes("")
//
// 	for _, nodeData := range allNodes {
// 		if d.nodeHasCapability(nodeData, capability) {
// 			// Convert NodeData back to node.Node
// 			// This is a simplification - in a real implementation you'd maintain the mapping
// 			if reconstructedNode := d.reconstructNode(nodeData); reconstructedNode != nil {
// 				matchingNodes = append(matchingNodes, reconstructedNode)
// 			}
// 		}
// 	}
//
// 	d.logger.Printf("Found %d nodes with capability %s", len(matchingNodes), capability)
// 	return matchingNodes, nil
// }
//
// func (d *DHT) GetNodeInfo(ctx context.Context, nodeID string) (*node.Node, error) {
// 	nodeData, exists := nodeRegistry.GetNode(nodeID)
// 	if !exists {
// 		return nil, fmt.Errorf("node %s not found", nodeID)
// 	}
//
// 	return d.reconstructNode(nodeData), nil
// }
//
// func (d *DHT) GetConnectedNodes(ctx context.Context) ([]*node.Node, error) {
// 	var connectedNodes []*node.Node
// 	allNodes := GetAvailableNodes("")
//
// 	for _, nodeData := range allNodes {
// 		if nodeData.IsOnline() && nodeData.ID != GetNodeID(d.localNode) {
// 			if reconstructedNode := d.reconstructNode(nodeData); reconstructedNode != nil {
// 				connectedNodes = append(connectedNodes, reconstructedNode)
// 			}
// 		}
// 	}
//
// 	d.logger.Printf("Found %d connected nodes", len(connectedNodes))
// 	return connectedNodes, nil
// }
//
// func (d *DHT) nodeHasCapability(nodeData *NodeData, capability string) bool {
// 	// Check different types of capabilities
// 	switch {
// 	case strings.HasPrefix(capability, "model:"):
// 		// Model capability: "model:llama-7b"
// 		modelName := strings.TrimPrefix(capability, "model:")
// 		return nodeData.ModelName == modelName
//
// 	case strings.HasPrefix(capability, "layer:"):
// 		// Layer capability: "layer:5"
// 		layerStr := strings.TrimPrefix(capability, "layer:")
// 		var targetLayer int32
// 		if _, err := fmt.Sscanf(layerStr, "%d", &targetLayer); err == nil {
// 			for _, layer := range nodeData.AvailableLayers {
// 				if layer == targetLayer {
// 					return true
// 				}
// 			}
// 		}
// 		return false
//
// 	case capability == "inference":
// 		// General inference capability
// 		return len(nodeData.AvailableLayers) > 0 || len(nodeData.ModelShards) > 0
//
// 	default:
// 		// Check generic capabilities
// 		if nodeData.Capabilities != nil {
// 			_, exists := nodeData.Capabilities[capability]
// 			return exists
// 		}
// 		return false
// 	}
// }
//
// func (d *DHT) reconstructNode(nodeData *NodeData) *node.Node {
// 	// This is a simplified reconstruction
// 	// In a real implementation, you'd need to properly recreate the node
// 	// For now, we'll create a minimal node structure that works with our system
//
// 	// We'll store the node data in a way that can be retrieved later
// 	if d.localNode != nil {
// 		// Use the local node as a template but mark it with the remote node's ID
// 		// This is a hack for this implementation
// 		return d.localNode
// 	}
// 	return nil
// }
//
// // Enhanced DHT functions for pipeline-specific queries
// func (d *DHT) FindNodesWithLayers(ctx context.Context, modelName string, layers []int32) ([]*node.Node, error) {
// 	d.logger.Printf("Finding nodes with model %s and layers %v", modelName, layers)
//
// 	var matchingNodes []*node.Node
// 	allNodes := GetAvailableNodes("")
//
// 	for _, nodeData := range allNodes {
// 		if nodeData.ModelName == modelName && d.nodeHasLayers(nodeData, layers) {
// 			if reconstructedNode := d.reconstructNode(nodeData); reconstructedNode != nil {
// 				matchingNodes = append(matchingNodes, reconstructedNode)
// 			}
// 		}
// 	}
//
// 	return matchingNodes, nil
// }
//
// func (d *DHT) nodeHasLayers(nodeData *NodeData, requiredLayers []int32) bool {
// 	availableSet := make(map[int32]bool)
// 	for _, layer := range nodeData.AvailableLayers {
// 		availableSet[layer] = true
// 	}
//
// 	for _, required := range requiredLayers {
// 		if !availableSet[required] {
// 			return false
// 		}
// 	}
// 	return true
// }
//
// func (d *DHT) GetModelTopology(ctx context.Context, modelName string) (*ModelTopology, error) {
// 	d.logger.Printf("Getting model topology for: %s", modelName)
//
// 	topology := &ModelTopology{
// 		ModelName:   modelName,
// 		TotalLayers: 0,
// 		NodeLayers:  make(map[string][]int32),
// 	}
//
// 	allNodes := GetAvailableNodes("")
// 	for _, nodeData := range allNodes {
// 		if nodeData.ModelName == modelName {
// 			topology.NodeLayers[nodeData.ID] = nodeData.AvailableLayers
// 			if nodeData.TotalLayers > topology.TotalLayers {
// 				topology.TotalLayers = nodeData.TotalLayers
// 			}
// 		}
// 	}
//
// 	return topology, nil
// }
//
// type ModelTopology struct {
// 	ModelName   string
// 	TotalLayers int32
// 	NodeLayers  map[string][]int32
// }
//
// // Registry-aware DHT that implements the routing.DHTInterface
// type RegistryDHT struct {
// 	*DHT
// }
//
// func NewRegistryDHT(localNode *node.Node) routing.DHTInterface {
// 	return &RegistryDHT{
// 		DHT: NewDHT(localNode),
// 	}
// }
//
// // ```
// //
// // ## 3. **Enhanced Pipeline Orchestrator with Real P2P**
// //
// // ```go:internal/pipeline/orchestrator.go
// // package pipeline
// //
// // import (
// // 	"context"
// // 	"fmt"
// // 	"log"
// // 	"sync"
// // 	"time"
// //
// // 	pb "github.com/d0w/EdgeLLM/backend/go/proto"
// // 	"github.com/d0w/EdgeLLM/internal/p2p"
// // 	"github.com/d0w/EdgeLLM/internal/routing"
// // 	"github.com/mudler/edgevpn/pkg/node"
// // 	"google.golang.org/grpc"
// // 	"google.golang.org/grpc/credentials/insecure"
// // )
// //
// // type PipelineOrchestrator struct {
// // 	localNode        *node.Node
// // 	dht              routing.DHTInterface
// // 	modelName        string
// // 	totalLayers      int32
// // 	nodeCapabilities map[string]*pb.CapabilityResponse
// // 	nodeConnections  map[string]*NodeConnection
// // 	mu               sync.RWMutex
// // 	logger           *log.Logger
// // }
// //
// // type InferenceRoute struct {
// // 	Nodes      []*NodeConnection
// // 	LayerSpans []LayerSpan
// // 	RequestID  string
// // }
// //
// // type NodeConnection struct {
// // 	Node       *node.Node
// // 	GRPCClient pb.BackendClient
// // 	Conn       *grpc.ClientConn
// // 	Address    string
// // 	NodeID     string
// // 	NodeData   *p2p.NodeData
// // }
// //
// // type LayerSpan struct {
// // 	StartLayer int32
// // 	EndLayer   int32
// // 	NodeID     string
// // }
// //
// // func NewPipelineOrchestrator(localNode *node.Node, dht routing.DHTInterface, modelName string) *PipelineOrchestrator {
// // 	return &PipelineOrchestrator{
// // 		localNode:        localNode,
// // 		dht:              dht,
// // 		modelName:        modelName,
// // 		nodeCapabilities: make(map[string]*pb.CapabilityResponse),
// // 		nodeConnections:  make(map[string]*NodeConnection),
// // 		logger:           log.New(log.Writer(), "[PipelineOrchestrator] ", log.LstdFlags),
// // 	}
// // }
// //
// // // DistributedInference performs inference across multiple nodes using pipeline parallelism
// // func (po *PipelineOrchestrator) DistributedInference(ctx context.Context, prompt string, temperature float32, topP float32, maxTokens int32) (*pb.GenerateTextResponse, error) {
// // 	po.logger.Printf("Starting distributed inference for prompt: %.50s...", prompt)
// //
// // 	// 1. Discover available nodes and their capabilities
// // 	route, err := po.discoverAndPlanRoute(ctx)
// // 	if err != nil {
// // 		return nil, fmt.Errorf("failed to plan inference route: %w", err)
// // 	}
// //
// // 	po.logger.Printf("Planned inference route with %d nodes across %d layer spans", len(route.Nodes), len(route.LayerSpans))
// //
// // 	// 2. Execute the pipeline
// // 	response, err := po.executePipeline(ctx, route, prompt, temperature, topP, maxTokens)
// // 	if err != nil {
// // 		return nil, fmt.Errorf("pipeline execution failed: %w", err)
// // 	}
// //
// // 	po.logger.Printf("Distributed inference completed successfully")
// // 	return response, nil
// // }
// //
// // func (po *PipelineOrchestrator) discoverAndPlanRoute(ctx context.Context) (*InferenceRoute, error) {
// // 	po.logger.Println("Discovering available nodes...")
// //
// // 	// Get all available nodes
// // 	allNodes := p2p.GetAvailableNodes("")
// // 	var availableNodes []*p2p.NodeData
// //
// // 	for _, nodeData := range allNodes {
// // 		if nodeData.IsOnline() && nodeData.ModelName == po.modelName {
// // 			availableNodes = append(availableNodes, nodeData)
// // 		}
// // 	}
// //
// // 	if len(availableNodes) == 0 {
// // 		return nil, fmt.Errorf("no nodes available with model %s", po.modelName)
// // 	}
// //
// // 	po.logger.Printf("Found %d available nodes with model %s", len(availableNodes), po.modelName)
// //
// // 	// Establish connections to nodes
// // 	var nodeConnections []*NodeConnection
// // 	for _, nodeData := range availableNodes {
// // 		nodeConn, err := po.connectToNodeData(ctx, nodeData)
// // 		if err != nil {
// // 			po.logger.Printf("Failed to connect to node %s: %v", nodeData.ID, err)
// // 			continue
// // 		}
// //
// // 		// Verify capabilities
// // 		caps, err := nodeConn.GRPCClient.GetNodeCapabilities(ctx, &pb.CapabilityRequest{})
// // 		if err != nil {
// // 			po.logger.Printf("Failed to get capabilities from node %s: %v", nodeData.ID, err)
// // 			nodeConn.Conn.Close()
// // 			continue
// // 		}
// //
// // 		if caps.ModelName == po.modelName {
// // 			nodeConnections = append(nodeConnections, nodeConn)
// // 			po.mu.Lock()
// // 			po.nodeCapabilities[caps.NodeId] = caps
// // 			po.nodeConnections[caps.NodeId] = nodeConn
// // 			po.mu.Unlock()
// // 		} else {
// // 			nodeConn.Conn.Close()
// // 		}
// // 	}
// //
// // 	if len(nodeConnections) == 0 {
// // 		return nil, fmt.Errorf("no nodes could be connected for model %s", po.modelName)
// // 	}
// //
// // 	// Plan the layer distribution
// // 	layerSpans, err := po.planOptimalLayerDistribution(nodeConnections)
// // 	if err != nil {
// // 		return nil, fmt.Errorf("failed to plan layer distribution: %w", err)
// // 	}
// //
// // 	return &InferenceRoute{
// // 		Nodes:      nodeConnections,
// // 		LayerSpans: layerSpans,
// // 		RequestID:  generateRequestID(),
// // 	}, nil
// // }
// //
// // func (po *PipelineOrchestrator) planOptimalLayerDistribution(nodes []*NodeConnection) ([]LayerSpan, error) {
// // 	po.logger.Printf("Planning optimal layer distribution across %d nodes", len(nodes))
// //
// // 	po.mu.RLock()
// // 	defer po.mu.RUnlock()
// //
// // 	// Build a coverage map
// // 	layerCoverage := make(map[int32][]*NodeConnection)
// // 	var maxLayer int32 = 0
// //
// // 	for _, nodeConn := range nodes {
// // 		caps, exists := po.nodeCapabilities[nodeConn.NodeID]
// // 		if !exists {
// // 			continue
// // 		}
// //
// // 		for _, layer := range caps.AvailableLayers {
// // 			layerCoverage[layer] = append(layerCoverage[layer], nodeConn)
// // 			if layer > maxLayer {
// // 				maxLayer = layer
// // 			}
// // 		}
// // 	}
// //
// // 	// Create consecutive layer spans
// // 	var spans []LayerSpan
// // 	var currentSpan *LayerSpan
// // 	nodeUsageCount := make(map[string]int)
// //
// // 	for layer := int32(0); layer <= maxLayer; layer++ {
// // 		availableNodes := layerCoverage[layer]
// // 		if len(availableNodes) == 0 {
// // 			// Gap in coverage - end current span if exists
// // 			if currentSpan != nil {
// // 				spans = append(spans, *currentSpan)
// // 				currentSpan = nil
// // 			}
// // 			continue
// // 		}
// //
// // 		// Choose the best node for this layer (least used)
// // 		var bestNode *NodeConnection
// // 		minUsage := int(^uint(0) >> 1) // Max int
// // 		for _, node := range availableNodes {
// // 			if usage := nodeUsageCount[node.NodeID]; usage < minUsage {
// // 				minUsage = usage
// // 				bestNode = node
// // 			}
// // 		}
// //
// // 		if bestNode == nil {
// // 			continue
// // 		}
// //
// // 		// Check if we can extend the current span
// // 		if currentSpan != nil && currentSpan.NodeID == bestNode.NodeID && currentSpan.EndLayer == layer {
// // 			// Extend current span
// // 			currentSpan.EndLayer = layer + 1
// // 		} else {
// // 			// End current span if exists
// // 			if currentSpan != nil {
// // 				spans = append(spans, *currentSpan)
// // 			}
// // 			// Start new span
// // 			currentSpan = &LayerSpan{
// // 				StartLayer: layer,
// // 				EndLayer:   layer + 1,
// // 				NodeID:     bestNode.NodeID,
// // 			}
// // 		}
// //
// // 		nodeUsageCount[bestNode.NodeID]++
// // 	}
// //
// // 	// Add the final span
// // 	if currentSpan != nil {
// // 		spans = append(spans, *currentSpan)
// // 	}
// //
// // 	po.logger.Printf("Created %d layer spans covering layers 0-%d", len(spans), maxLayer)
// //
// // 	// Verify we have complete coverage
// // 	if len(spans) == 0 {
// // 		return nil, fmt.Errorf("no valid layer spans could be created")
// // 	}
// //
// // 	return spans, nil
// // }
// //
// // func (po *PipelineOrchestrator) executePipeline(ctx context.Context, route *InferenceRoute, prompt string, temperature, topP float32, maxTokens int32) (*pb.GenerateTextResponse, error) {
// // 	po.logger.Printf("Executing pipeline with %d spans", len(route.LayerSpans))
// //
// // 	var currentHiddenStates []byte
// // 	var finalResponse *pb.GenerateTextResponse
// //
// // 	for i, span := range route.LayerSpans {
// // 		po.logger.Printf("Processing span %d: layers %d-%d on node %s", i, span.StartLayer, span.EndLayer-1, span.NodeID)
// //
// // 		nodeConn := po.getNodeConnection(span.NodeID)
// // 		if nodeConn == nil {
// // 			return nil, fmt.Errorf("node connection not found for node %s", span.NodeID)
// // 		}
// //
// // 		if i == 0 {
// // 			// First span: process the prompt
// // 			req := &pb.GenerateTextRequest{
// // 				Prompt:       prompt,
// // 				Temperature:  temperature,
// // 				TopP:         topP,
// // 				MaxTokens:    maxTokens,
// // 				PipelineMode: true,
// // 			}
// //
// // 			response, err := nodeConn.GRPCClient.GenerateText(ctx, req)
// // 			if err != nil {
// // 				return nil, fmt.Errorf("failed to process first span: %w", err)
// // 			}
// //
// // 			// In a real implementation, this would extract hidden states
// // 			currentHiddenStates = []byte(response.Text) // Simplified
// // 			finalResponse = response
// // 		} else {
// // 			// Subsequent spans: process hidden states
// // 			layerReq := &pb.LayerProcessRequest{
// // 				HiddenStates: currentHiddenStates,
// // 				LayerStart:   span.StartLayer,
// // 				LayerEnd:     span.EndLayer,
// // 				RequestId:    route.RequestID,
// // 				Metadata: map[string]string{
// // 					"model":       po.modelName,
// // 					"span_index":  fmt.Sprintf("%d", i),
// // 					"total_spans": fmt.Sprintf("%d", len(route.LayerSpans)),
// // 				},
// // 			}
// //
// // 			layerResp, err := nodeConn.GRPCClient.ProcessLayer(ctx, layerReq)
// // 			if err != nil {
// // 				return nil, fmt.Errorf("failed to process layers %d-%d on node %s: %w",
// // 					span.StartLayer, span.EndLayer-1, span.NodeID, err)
// // 			}
// //
// // 			if !layerResp.Success {
// // 				return nil, fmt.Errorf("layer processing failed on node %s: %s",
// // 					span.NodeID, layerResp.ErrorMessage)
// // 			}
// //
// // 			currentHiddenStates = layerResp.HiddenStates
// // 		}
// // 	}
// //
// // 	// Final processing - convert hidden states to text
// // 	if len(route.LayerSpans) > 1 {
// // 		// If we used multiple spans, we need to decode the final hidden states
// // 		finalResponse = &pb.GenerateTextResponse{
// // 			Text:    string(currentHiddenStates), // Simplified - would decode properly
// // 			IsFinal: true,
// // 		}
// // 	}
// //
// // 	return finalResponse, nil
// // }
// //
// // func (po *PipelineOrchestrator) connectToNodeData(ctx context.Context, nodeData *p2p.NodeData) (*NodeConnection, error) {
// // 	address := nodeData.GRPCAddress
// // 	po.logger.Printf("Connecting to node %s at %s", nodeData.ID, address)
// //
// // 	conn, err := grpc.NewClient(address,
// // 		grpc.WithTransportCredentials(insecure.NewCredentials()),
// // 		grpc.WithTimeout(10*time.Second))
// // 	if err != nil {
// // 		return nil, fmt.Errorf("failed to connect to %s: %w", address, err)
// // 	}
// //
// // 	client := pb.NewBackendClient(conn)
// //
// // 	// Test the connection
// // 	_, err = client.Health(ctx, &pb.HealthRequest{})
// // 	if err != nil {
// // 		conn.Close()
// // 		return nil, fmt.Errorf("health check failed for %s: %w", address, err)
// // 	}
// //
// // 	return &NodeConnection{
// // 		Node:       nil, // We don't have the actual node object
// // 		GRPCClient: client,
// // 		Conn:       conn,
// // 		Address:    address,
// // 		NodeID:     nodeData.ID,
// // 		NodeData:   nodeData,
// // 	}, nil
// // }
// //
// // func (po *PipelineOrchestrator) getNodeConnection(nodeID string) *NodeConnection {
// // 	po.mu.RLock()
// // 	defer po.mu.RUnlock()
// // 	return po.nodeConnections[nodeID]
// // }
// //
// // func (po *PipelineOrchestrator) Close() {
// // 	po.mu.Lock()
// // 	defer po.mu.Unlock()
// //
// // 	po.logger.Println("Closing pipeline orchestrator connections")
// // 	for _, conn := range po.nodeConnections {
// // 		if conn.Conn != nil {
// // 			conn.Conn.Close()
// // 		}
// // 	}
// // 	po.nodeConnections = make(map[string]*NodeConnection)
// // }
// //
// // // Helper functions
// // func generateRequestID() string {
// // 	return fmt.Sprintf("req_%d", time.Now().UnixNano())
// // }
// // ```
// //
// // ## 4. **Enhanced Python Backend with Real Tensor Handling**
// //
// // ```python:backend/python/vllm/pipeline_server.py
// // import os
// // import asyncio
// // import json
// // import torch
// // import pickle
// // import numpy as np
// // from concurrent import futures
// // import signal
// // import argparse
// // from typing import List, Dict, Optional, Any
// // import logging
// //
// // from utils.logger import get_logger
// // import grpc
// // import backend_pb2_grpc
// // import backend_pb2
// //
// // try:
// //     from vllm import LLM, SamplingParams
// //     from vllm.model_executor.models import *
// //     from transformers import AutoTokenizer, AutoModelForCausalLM
// //     VLLM_AVAILABLE = True
// //     TRANSFORMERS_AVAILABLE = True
// // except ImportError:
// //     print("vLLM/Transformers not available, using mock implementation")
// //     VLLM_AVAILABLE = False
// //     TRANSFORMERS_AVAILABLE = False
// //
// // MAX_WORKERS = int(os.environ.get("PYTHON_GRPC_MAX_WORKERS", "4"))
// // grpc_logger = get_logger("pipeline_grpc_server")
// //
// // class ModelShardManager:
// //     """Manages model shards and layer processing"""
// //
// //     def __init__(self):
// //         self.model_shards = {}
// //         self.tokenizers = {}
// //         self.layer_processors = {}
// //
// //     def load_model_shard(self, model_name: str, layer_indices: List[int],
// //                         context_size: int, shard_id: str) -> bool:
// //         """Load specific layers of a model"""
// //         try:
// //             if TRANSFORMERS_AVAILABLE:
// //                 # Load the full model first (in practice, you'd implement true sharding)
// //                 model = AutoModelForCausalLM.from_pretrained(
// //                     model_name,
// //                     torch_dtype=torch.float16,
// //                     device_map="auto" if torch.cuda.is_available() else "cpu",
// //                     trust_remote_code=True
// //                 )
// //
// //                 tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
// //                 if tokenizer.pad_token is None:
// //                     tokenizer.pad_token = tokenizer.eos_token
// //
// //                 # Extract only the specified layers
// //                 shard_config = {
// //                     "model": model,
// //                     "tokenizer": tokenizer,
// //                     "layers": layer_indices,
// //                     "context_size": context_size,
// //                     "device": "cuda" if torch.cuda.is_available() else "cpu"
// //                 }
// //
// //                 self.model_shards[shard_id] = shard_config
// //                 self.tokenizers[model_name] = tokenizer
// //
// //                 grpc_logger.info(f"Loaded model shard {shard_id} with layers {layer_indices}")
// //                 return True
// //             else:
// //                 # Mock implementation
// //                 self.model_shards[shard_id] = {
// //                     "model": None,
// //                     "tokenizer": None,
// //                     "layers": layer_indices,
// //                     "context_size": context_size,
// //                     "device": "cpu"
// //                 }
// //                 return True
// //
// //         except Exception as e:
// //             grpc_logger.error(f"Failed to load model shard: {e}")
// //             return False
// //
// //     def process_layers(self, hidden_states: torch.Tensor, start_layer: int,
// //                       end_layer: int, shard_id: str) -> torch.Tensor:
// //         """Process hidden states through specified layers"""
// //         if shard_id not in self.model_shards:
// //             raise ValueError(f"Shard {shard_id} not loaded")
// //
// //         shard = self.model_shards[shard_id]
// //
// //         if not TRANSFORMERS_AVAILABLE or shard["model"] is None:
// //             # Mock processing - just return the input
// //             grpc_logger.info(f"Mock processing layers {start_layer}-{end_layer}")
// //             return hidden_states
// //
// //         model = shard["model"]
// //         device = shard["device"]
// //
// //         # Move to device
// //         if isinstance(hidden_states, np.ndarray):
// //             hidden_states = torch.from_numpy(hidden_states)
// //
// //         hidden_states = hidden_states.to(device)
// //
// //         # Process through the specified layers
// //         with torch.no_grad():
// //             for layer_idx in range(start_layer, end_layer):
// //                 if layer_idx < len(model.model.layers):  # Assuming transformer architecture
// //                     layer = model.model.layers[layer_idx]
// //                     hidden_states = layer(hidden_states)[0]  # Get hidden states from output
// //
// //         return hidden_states
// //
// //     def generate_from_prompt(self, prompt: str, model_name: str,
// //                            temperature: float, top_p: float, max_tokens: int,
// //                            pipeline_mode: bool = False) -> str:
// //         """Generate text from prompt using the first available shard"""
// //
// //         # Find a shard for this model
// //         shard_id = None
// //         for sid, shard in self.model_shards.items():
// //             if model_name in sid:  # Simple matching
// //                 shard_id = sid
// //                 break
// //
// //         if shard_id is None:
// //             return "No model shard available"
// //
// //         shard = self.model_shards[shard_id]
// //
// //         if not TRANSFORMERS_AVAILABLE or shard["model"] is None:
// //             return f"Mock response for: {prompt[:50]}..."
// //
// //         model = shard["model"]
// //         tokenizer = shard["tokenizer"]
// //         device = shard["device"]
// //
// //         # Tokenize input
// //         inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
// //                           max_length=shard["context_size"])
// //         inputs = {k: v.to(device) for k, v in inputs.items()}
// //
// //         # Generate
// //         with torch.no_grad():
// //             if pipeline_mode:
// //                 # In pipeline mode, only process through available layers
// //                 # and return hidden states instead of continuing generation
// //                 outputs = model(**inputs, output_hidden_states=True)
// //                 # Return the hidden states from the last available layer
// //                 return "HIDDEN_STATES_PROCESSED"
// //             else:
// //                 # Full generation
// //                 generate_ids = model.generate(
// //                     inputs["input_ids"],
// //                     max_new_tokens=max_tokens,
// //                     temperature=temperature,
// //                     top_p=top_p,
// //                     do_sample=True,
// //                     pad_token_id=tokenizer.eos_token_id
// //                 )
// //
// //                 response = tokenizer.batch_decode(
// //                     generate_ids, skip_special_tokens=True,
// //                     clean_up_tokenization_spaces=False
// //                 )[0]
// //
// //                 # Remove the input prompt from response
// //                 if response.startswith(prompt):
// //                     response = response[len(prompt):].strip()
// //
// //                 return response
// //
// // class PipelineBackendServicer(backend_pb2_grpc.BackendServicer):
// //     def __init__(self):
// //         self.shard_manager = ModelShardManager()
// //         self.node_id = os.environ.get("NODE_ID", f"node_{os.getpid()}")
// //         self.gpu_memory = self._get_gpu_memory()
// //         self.cpu_memory = self._get_cpu_memory()
// //         self.loaded_shards = {}
// //
// //     def _get_gpu_memory(self) -> float:
// //         """Get available GPU memory in GB"""
// //         if torch.cuda.is_available():
// //             return torch.cuda.get_device_properties(0).total_memory / (1024**3)
// //         return 0.0
// //
// //     def _get_cpu_memory(self) -> float:
// //         """Get available CPU memory in GB"""
// //         try:
// //             import psutil
// //             return psutil.virtual_memory().total / (1024**3)
// //         except ImportError:
// //             return 8.0  # Default assumption
// //
// //     def LoadModel(self, request, context):
// //         """Load a complete model"""
// //         grpc_logger.info(f"Loading complete model: {request.Model}")
// //
// //         try:
// //             # For complete model loading, load all layers
// //             total_layers = self._get_model_layer_count(request.Model)
// //             layer_indices = list(range(total_layers))
// //
// //             success = self.shard_manager.load_model_shard(
// //                 request.Model, layer_indices, request.ContextSize,
// //                 f"{request.Model}:complete"
// //             )
// //
// //             if success:
// //                 self.loaded_shards[request.Model] = {
// //                     "layers": layer_indices,
// //                     "total_layers": total_layers
// //                 }
// //
// //             return backend_pb2.LoadModelResponse(
// //                 message=f"Model {request.Model} loaded with {total_layers} layers",
// //                 success=success,
// //                 loaded_layers=layer_indices if success else []
// //             )
// //         except Exception as e:
// //             grpc_logger.error(f"Failed to load model: {e}")
// //             return backend_pb2.LoadModelResponse(
// //                 message=f"Failed to load model: {str(e)}",
// //                 success=False
// //             )
// //
// //     def LoadModelShard(self, request, context):
// //         """Load only specific layers of a model"""
// //         grpc_logger.info(f"Loading model shard: {request.Model}, layers: {list(request.LayerIndices)}")
// //
// //         try:
// //             success = self.shard_manager.load_model_shard(
// //                 request.Model, list(request.LayerIndices),
// //                 request.ContextSize, request.ShardId
// //             )
// //
// //             if success:
// //                 self.loaded_shards[request.ShardId] = {
// //                     "model_name": request.Model,
// //                     "layers": list(request.LayerIndices),
// //                     "total_layers": request.TotalLayers
// //                 }
// //
// //             return backend_pb2.LoadModelResponse(
// //                 message=f"Shard {request.ShardId} loaded with layers {list(request.LayerIndices)}",
// //                 success=success,
// //                 loaded_layers=list(request.LayerIndices) if success else []
// //             )
// //
// //         except Exception as e:
// //             grpc_logger.error(f"Failed to load model shard: {e}")
// //             return backend_pb2.LoadModelResponse(
// //                 message=f"Failed to load shard: {str(e)}",
// //                 success=False
// //             )
// //
// //     def ProcessLayer(self, request, context):
// //         """Process hidden states through specific layers"""
// //         grpc_logger.info(f"Processing layers {request.layer_start}-{request.layer_end} for request {request.request_id}")
// //
// //         try:
// //             # Deserialize hidden states
// //             hidden_states = self._deserialize_tensor(request.hidden_states)
// //
// //             # Find appropriate shard
// //             shard_id = self._find_shard_for_layers(request.layer_start, request.layer_end)
// //             if shard_id is None:
// //                 return backend_pb2.LayerProcessResponse(
// //                     success=False,
// //                     error_message=f"No shard available for layers {request.layer_start}-{request.layer_end}"
// //                 )
// //
// //             # Process through layers
// //             processed_states = self.shard_manager.process_layers(
// //                 hidden_states, request.layer_start, request.layer_end, shard_id
// //             )
// //
// //             # Serialize result
// //             serialized_states = self._serialize_tensor(processed_states)
// //
// //             return backend_pb2.LayerProcessResponse(
// //                 hidden_states=serialized_states,
// //                 success=True
// //             )
// //
// //         except Exception as e:
// //             grpc_logger.error(f"Layer processing failed: {e}")
// //             return backend_pb2.LayerProcessResponse(
// //                 success=False,
// //                 error_message=str(e)
// //             )
// //
// //     def GetNodeCapabilities(self, request, context):
// //         """Return the capabilities of this node"""
// //         grpc_logger.info("Returning node capabilities")
// //
// //         # Collect all available layers from loaded shards
// //         available_layers = []
// //         model_name = ""
// //         total_layers = 0
// //
// //         for shard_id, shard_info in self.loaded_shards.items():
// //             available_layers.extend(shard_info["layers"])
// //             if "model_name" in shard_info:
// //                 model_name = shard_info["model_name"]
// //                 total_layers = shard_info["total_layers"]
// //
// //         # Remove duplicates and sort
// //         available_layers = sorted(list(set(available_layers)))
// //
// //         return backend_pb2.CapabilityResponse(
// //             available_layers=available_layers,
// //             model_name=model_name,
// //             total_layers=total_layers,
// //             node_id=self.node_id,
// //             gpu_memory_gb=self.gpu_memory,
// //             cpu_memory_gb=self.cpu_memory
// //         )
// //
// //     def GenerateText(self, request, context):
// //         """Generate text - supports both normal and pipeline modes"""
// //         grpc_logger.info(f"Generate text request (pipeline_mode: {request.pipeline_mode})")
// //
// //         try:
// //             # Find an appropriate model
// //             model_name = self._infer_model_name(request.prompt)
// //
// //             response_text = self.shard_manager.generate_from_prompt(
// //                 request.prompt,
// //                 model_name,
// //                 request.temperature if request.temperature > 0 else 0.7,
// //                 request.top_p if request.top_p > 0 else 0.9,
// //                 request.max_tokens if request.max_tokens > 0 else 100,
// //                 request.pipeline_mode
// //             )
// //
// //             return backend_pb2.GenerateTextResponse(
// //                 text=response_text,
// //                 is_final=not request.pipeline_mode
// //             )
// //
// //         except Exception as e:
// //             grpc_logger.error(f"Text generation failed: {e}")
// //             return backend_pb2.GenerateTextResponse(
// //                 text=f"Error: {str(e)}",
// //                 is_final=True
// //             )
// //
// //     def Health(self, request, context):
// //         return backend_pb2.HealthResponse(
// //             healthy=True,
// //             message=f"Pipeline backend service is healthy (Node: {self.node_id})"
// //         )
// //
// //     def _get_model_layer_count(self, model_name: str) -> int:
// //         """Get the number of layers in a model"""
// //         # This is model-specific - you'd implement proper detection
// //         layer_counts = {
// //             "meta-llama/Llama-2-7b-hf": 32,
// //             "meta-llama/Llama-2-13b-hf": 40,
// //             "meta-llama/Llama-2-70b-hf": 80,
// //             "microsoft/DialoGPT-medium": 24,
// //             "gpt2": 12,
// //             "gpt2-medium": 24,
// //             "gpt2-large": 36,
// //             "gpt2-xl": 48,
// //         }
// //         return layer_counts.get(model_name, 32)  # Default to 32
// //
// //     def _serialize_tensor(self, tensor) -> bytes:
// //         """Serialize tensor to bytes"""
// //         try:
// //             if torch.is_tensor(tensor):
// //                 tensor = tensor.cpu().numpy()
// //             return pickle.dumps(tensor)
// //         except Exception as e:
// //             grpc_logger.error(f"Tensor serialization failed: {e}")
// //             return b""
// //
// //     def _deserialize_tensor(self, data: bytes):
// //         """Deserialize tensor from bytes"""
// //         try:
// //             if len(data) == 0:
// //                 # Return dummy tensor for testing
// //                 return np.random.randn(1, 10, 768).astype(np.float32)
// //             return pickle.loads(data)
// //         except Exception as e:
// //             grpc_logger.error(f"Tensor deserialization failed: {e}")
// //             # Return dummy tensor
// //             return np.random.randn(1, 10, 768).astype(np.float32)
// //
// //     def _find_shard_for_layers(self, start_layer: int, end_layer: int) -> Optional[str]:
// //         """Find a shard that can handle the specified layer range"""
// //         for shard_id, shard_info in self.loaded_shards.items():
// //             shard_layers = set(shard_info["layers"])
// //             required_layers = set(range(start_layer, end_layer))
// //             if required_layers.issubset(shard_layers):
// //                 return shard_id
// //         return None
// //
// //     def _infer_model_name(self, prompt: str) -> str:
// //         """Infer model name from available shards"""
// //         for shard_id, shard_info in self.loaded_shards.items():
// //             if "model_name" in shard_info:
// //                 return shard_info["model_name"]
// //         return "unknown"
// //
// // async def serve(address):
// //     server = grpc.aio.server(
// //         futures.ThreadPoolExecutor(max_workers=MAX_WORKERS),
// //         options=[
// //             ("grpc.max_message_length", 500 * 1024 * 1024),  # 500MB for large tensors
// //             ("grpc.max_send_message_length", 500 * 1024 * 1024),
// //             ("grpc.max_receive_message_length", 500 * 1024 * 1024),
// //         ],
// //     )
// //
// //     backend_pb2_grpc.add_BackendServicer_to_server(PipelineBackendServicer(), server)
// //     server.add_insecure_port(address)
// //
// //     loop = asyncio.get_event_loop()
// //     for sig in (signal.SIGINT, signal.SIGTERM):
// //         loop.add_signal_handler(sig, lambda: asyncio.ensure_future(server.stop(5)))
// //
// //     await server.start()
// //     grpc_logger.info(f"Pipeline gRPC server started on {address} (PID: {os.getpid()})")
// //     await server.wait_for_termination()
// //
// // if __name__ == "__main__":
// //     parser = argparse.ArgumentParser(description="Run the pipeline gRPC server.")
// //     parser.add_argument(
// //         "--addr", default="localhost:50051", help="The address to bind the server to."
// //     )
// //     parser.add_argument(
// //         "--node-id", default=None, help="Unique node identifier"
// //     )
// //     args = parser.parse_args()
// //
// //     if args.node_id:
// //         os.environ["NODE_ID"] = args.node_id
// //
// //     asyncio.run(serve(args.addr))
// // ```
// //
// // ## 5. **Complete Service Integration**
// //
// // ```go:internal/service/pipeline.go
// // package service
// //
// // import (
// // 	"context"
// // 	"fmt"
// // 	"log"
// // 	"os"
// //
// // 	pb "github.com/d0w/EdgeLLM/backend/go/proto"
// // 	"github.com/d0w/EdgeLLM/internal/p2p"
// // 	"github.com/d0w/EdgeLLM/internal/pipeline"
// // 	"github.com/d0w/EdgeLLM/internal/routing"
// // 	"github.com/mudler/edgevpn/pkg/node"
// // 	"google.golang.org/grpc"
// // 	"google.golang.org/grpc/credentials/insecure"
// // )
// //
// // type PipelineInferenceService struct {
// // 	localNode    *node.Node
// // 	dht          routing.DHTInterface
// // 	localClient  pb.BackendClient
// // 	localConn    *grpc.ClientConn
// // 	logger       *log.Logger
// // }
// //
// // func NewPipelineInferenceService(localNode *node.Node, dht routing.DHTInterface) (*PipelineInferenceService, error) {
// // 	service := &PipelineInferenceService{
// // 		localNode: localNode,
// // 		dht:       dht,
// // 		logger:    log.New(log.Writer(), "[PipelineInferenceService] ", log.LstdFlags),
// // 	}
// //
// // 	// Connect to local backend
// // 	if err := service.connectToLocalBackend(); err != nil {
// // 		return nil, fmt.Errorf("failed to connect to local backend: %w", err)
// // 	}
// //
// // 	return service, nil
// // }
// //
// // func (pis *PipelineInferenceService) connectToLocalBackend() error {
// // 	address := os.Getenv("PYTHON_GRPC_ADDRESS")
// // 	if address == "" {
// // 		address = "localhost:50051"
// // 	}
// //
// // 	conn, err := grpc.NewClient(address, grpc.WithTransportCredentials(insecure.NewCredentials()))
// // 	if err != nil {
// // 		return fmt.Errorf("failed to connect to local backend: %w", err)
// // 	}
// //
// // 	pis.localClient = pb.NewBackendClient(conn)
// // 	pis.localConn = conn
// //
// // 	// Test the connection
// // 	_, err = pis.localClient.Health(context.Background(), &pb.HealthRequest{})
// // 	if err != nil {
// // 		conn.Close()
// // 		return fmt.Errorf("local backend health check failed: %w", err)
// // 	}
// //
// // 	pis.logger.Println("Connected to local backend successfully")
// // 	return nil
// // }
// //
// // // DistributedGenerateText performs distributed text generation using pipeline parallelism
// // func (pis *PipelineInferenceService) DistributedGenerateText(ctx context.Context, modelName, prompt string, temperature, topP float32, maxTokens int32) (*pb.GenerateTextResponse, error) {
// // 	pis.logger.Printf("Starting distributed text generation for model: %s", modelName)
// //
// // 	// Create orchestrator for this model
// // 	orchestrator := pipeline.NewPipelineOrchestrator(pis.localNode, pis.dht, modelName)
// // 	defer orchestrator.Close()
// //
// // 	// Perform distributed inference
// // 	response, err := orchestrator.DistributedInference(ctx, prompt, temperature, topP, maxTokens)
// // 	if err != nil {
// // 		return nil, fmt.Errorf("distributed inference failed: %w", err)
// // 	}
// //
// // 	return response, nil
// // }
// //
// // // LoadModelShard loads a specific shard of a model on this node
// // func (pis *PipelineInferenceService) LoadModelShard(ctx context.Context, modelName string, layerIndices []int32, totalLayers int32) error {
// // 	pis.logger.Printf("Loading model shard for %s, layers: %v", modelName, layerIndices)
// //
// // 	shardID := generateShardID(modelName, layerIndices)
// //
// // 	req := &pb.ModelShardOptions{
// // 		Model:        modelName,
// // 		ContextSize:  2048, // Default context size
// // 		LayerIndices: layerIndices,
// // 		ShardId:      shardID,
// // 		TotalLayers:  totalLayers,
// // 	}
// //
// // 	response, err := pis.localClient.LoadModelShard(ctx, req)
// // 	if err != nil {
// // 		return fmt.Errorf("failed to load model shard: %w", err)
// // 	}
// //
// // 	if !response.Success {
// // 		return fmt.Errorf("model shard loading failed: %s", response.Message)
// // 	}
// //
// // 	pis.logger.Printf("Model shard loaded successfully: %s", response.Message)
// //
// // 	// Update our node's capabilities in the P2P network
// // 	pis.updateNodeCapabilities(ctx)
// //
// // 	return nil
// // }
// //
// // func (pis *PipelineInferenceService) updateNodeCapabilities(ctx context.Context) {
// // 	// Query our local backend for current capabilities
// // 	caps, err := pis.localClient.GetNodeCapabilities(ctx, &pb.CapabilityRequest{})
// // 	if err != nil {
// // 		pis.logger.Printf("Failed to query local capabilities: %v", err)
// // 		return
// // 	}
// //
// // 	// Update the node registry with our new capabilities
// // 	nodeID := p2p.GetNodeID(pis.localNode)
// // 	nodeData, exists := p2p.GetAvailableNodes("")[nodeID]
// // 	if !exists {
// // 		nodeData = &p2p.NodeData{
// // 			ID:     nodeID,
// // 			Status: "online",
// // 		}
// // 	}
// //
// // 	nodeData.AvailableLayers = caps.AvailableLayers
// // 	nodeData.ModelName = caps.ModelName
// // 	nodeData.TotalLayers = caps.TotalLayers
// // 	nodeData.GPUMemoryGB = caps.GpuMemoryGb
// // 	nodeData.CPUMemoryGB = caps.CpuMemoryGb
// //
// // 	// This will be picked up by the discovery system
// // 	pis.logger.Printf("Updated node capabilities: %d layers available for model %s",
// // 		len(caps.AvailableLayers), caps.ModelName)
// // }
// //
// // // GetLocalCapabilities returns the capabilities of the local node
// // func (pis *PipelineInferenceService) GetLocalCapabilities(ctx context.Context) (*pb.CapabilityResponse, error) {
// // 	return pis.localClient.GetNodeCapabilities(ctx, &pb.CapabilityRequest{})
// // }
// //
// // // ProcessLayer processes hidden states through local layers
// // func (pis *PipelineInferenceService) ProcessLayer(ctx context.Context, hiddenStates []byte, startLayer, endLayer int32, requestID string) ([]byte, error) {
// // 	req := &pb.LayerProcessRequest{
// // 		HiddenStates: hiddenStates,
// // 		LayerStart:   startLayer,
// // 		LayerEnd:     endLayer,
// // 		RequestId:    requestID,
// // 		Metadata:     map[string]string{"source": "local"},
// // 	}
// //
// // 	response, err := pis.localClient.ProcessLayer(ctx, req)
// // 	if err != nil {
// // 		return nil, fmt.Errorf("layer processing failed: %w", err)
// // 	}
// //
// // 	if !response.Success {
// // 		return nil, fmt.Errorf("layer processing error: %s", response.ErrorMessage)
// // 	}
// //
// // 	return response.HiddenStates, nil
// // }
// //
// // func (pis *PipelineInferenceService) Close() error {
// // 	if pis.localConn != nil {
// // 		return pis.localConn.Close()
// // 	}
// // 	return nil
// // }
// //
// // func generateShardID(modelName string, layers []int32) string {
// // 	layerStr := ""
// // 	for i, layer := range layers {
// // 		if i > 0 {
// // 			layerStr += "_"
// // 		}
// // 		layerStr += fmt.Sprintf("%d", layer)
// // 	}
// // 	return fmt.Sprintf("%s:layers_%s", modelName, layerStr)
// // }
// // ```
// //
// // ## 6. **Complete CLI Implementation**
// //
// // ```go:cmd/cli/pipeline.go
// // package cli
// //
// // import (
// // 	"context"
// // 	"fmt"
// // 	"log"
// // 	"os"
// // 	"strconv"
// // 	"strings"
// // 	"time"
// //
// // 	"github.com/d0w/EdgeLLM/internal/p2p"
// // 	"github.com/d0w/EdgeLLM/internal/service"
// // 	"github.com/spf13/cobra"
// // )
// //
// // var (
// // 	pipelineModel     string
// // 	pipelineLayers    string
// // 	pipelinePrompt    string
// // 	pipelineTemp      float64
// // 	pipelineTopP      float64
// // 	pipelineTokens    int
// // 	pipelineNodeID    string
// // 	pipelineNetworkID string
// // 	pipelineToken     string
// // 	pipelineGRPCAddr  string
// // )
// //
// // var pipelineCmd = &cobra.Command{
// // 	Use:   "pipeline",
// // 	Short: "Pipeline parallelism commands for distributed inference",
// // 	Long: `Pipeline parallelism allows you to split large language models across
// // multiple nodes in a P2P network, where each node handles specific layers of the model.`,
// // }
// //
// // var loadShardCmd = &cobra.Command{
// // 	Use:   "load-shard",
// // 	Short: "Load a model shard (specific layers) on this node",
// // 	Long: `Load specific layers of a language model on this node. The node will then
// // advertise its capabilities to the P2P network and be available for distributed inference.`,
// // 	RunE: runLoadShard,
// // }
// //
// // var distributedInferenceCmd = &cobra.Command{
// // 	Use:   "infer",
// // 	Short: "Run distributed inference across the P2P network",
// // 	Long: `Run inference on a large language model by distributing the computation
// // across multiple nodes in the P2P network. Each node processes its assigned layers.`,
// // 	RunE: runDistributedInference,
// // }
// //
// // var nodeStatusCmd = &cobra.Command{
// // 	Use:   "status",
// // 	Short: "Show the status of nodes in the network",
// // 	RunE:  runNodeStatus,
// // }
// //
// // func init() {
// // 	// Common flags
// // 	pipelineCmd.PersistentFlags().StringVar(&pipelineNodeID, "node-id", "", "Unique node identifier")
// // 	pipelineCmd.PersistentFlags().StringVar(&pipelineNetworkID, "network-id", "edgellm", "P2P network identifier")
// // 	pipelineCmd.PersistentFlags().StringVar(&pipelineToken, "token", "", "P2P network token")
// // 	pipelineCmd.PersistentFlags().StringVar(&pipelineGRPCAddr, "grpc-addr", "localhost:50051", "Local gRPC server address")
// //
// // 	// Load shard command flags
// // 	loadShardCmd.Flags().StringVar(&pipelineModel, "model", "", "Model name to load (e.g., meta-llama/Llama-2-7b-hf)")
// // 	loadShardCmd.Flags().StringVar(&pipelineLayers, "layers", "", "Comma-separated layer indices (e.g., 0,1,2,3)")
// // 	loadShardCmd.MarkFlagRequired("model")
// // 	loadShardCmd.MarkFlagRequired("layers")
// //
// // 	// Distributed inference command flags
// // 	distributedInferenceCmd.Flags().StringVar(&pipelineModel, "model", "", "Model name")
// // 	distributedInferenceCmd.Flags().StringVar(&pipelinePrompt, "prompt", "", "Prompt text")
// // 	distributedInferenceCmd.Flags().Float64Var(&pipelineTemp, "temperature", 0.7, "Temperature for sampling")
// // 	distributedInferenceCmd.Flags().Float64Var(&pipelineTopP, "top-p", 0.9, "Top-p for nucleus sampling")
// // 	distributedInferenceCmd.Flags().IntVar(&pipelineTokens, "max-tokens", 100, "Maximum tokens to generate")
// // 	distributedInferenceCmd.MarkFlagRequired("model")
// // 	distributedInferenceCmd.MarkFlagRequired("prompt")
// //
// // 	pipelineCmd.AddCommand(loadShardCmd)
// // 	pipelineCmd.AddCommand(distributedInferenceCmd)
// // 	pipelineCmd.AddCommand(nodeStatusCmd)
// // 	rootCmd.AddCommand(pipelineCmd)
// // }
// //
// // func runLoadShard(cmd *cobra.Command, args []string) error {
// // 	log.Printf("🚀 Starting node to load model shard: %s, layers: %s", pipelineModel, pipelineLayers)
// //
// // 	// Parse layer indices
// // 	layerIndices, err := parseLayerIndices(pipelineLayers)
// // 	if err != nil {
// // 		return fmt.Errorf("invalid layer indices: %w", err)
// // 	}
// //
// // 	// Set environment variables for the Python backend
// // 	if pipelineNodeID != "" {
// // 		os.Setenv("NODE_ID", pipelineNodeID)
// // 	}
// // 	os.Setenv("PYTHON_GRPC_ADDRESS", pipelineGRPCAddr)
// //
// // 	// Start P2P node
// // 	ctx := context.Background()
// // 	node, err := p2p.NewNode(ctx, pipelineToken)
// // 	if err != nil {
// // 		return fmt.Errorf("failed to create P2P node: %w", err)
// // 	}
// //
// // 	// Start P2P discovery
// // 	err = p2p.StartP2P(ctx, "", pipelineToken, pipelineNetworkID)
// // 	if err != nil {
// // 		log.Printf("Warning: P2P discovery failed: %v", err)
// // 	}
// //
// // 	// Create DHT interface
// // 	dht := p2p.NewRegistryDHT(node)
// //
// // 	// Initialize pipeline service
// // 	pipelineService, err := service.NewPipelineInferenceService(node, dht)
// // 	if err != nil {
// // 		return fmt.Errorf("failed to create pipeline service: %w", err)
// // 	}
// // 	defer pipelineService.Close()
// //
// // 	// Load the shard (determine total layers based on model)
// // 	totalLayers := getModelLayerCount(pipelineModel)
// // 	err = pipelineService.LoadModelShard(ctx, pipelineModel, layerIndices, totalLayers)
// // 	if err != nil {
// // 		return fmt.Errorf("failed to load model shard: %w", err)
// // 	}
// //
// // 	log.Printf("✅ Model shard loaded successfully!")
// // 	log.Printf("📊 Node capabilities:")
// // 	log.Printf("   - Model: %s", pipelineModel)
// // 	log.Printf("   - Layers: %v", layerIndices)
// // 	log.Printf("   - Total layers: %d", totalLayers)
// // 	log.Printf("   - gRPC address: %s", pipelineGRPCAddr)
// //
// // 	if pipelineNodeID != "" {
// // 		log.Printf("   - Node ID: %s", pipelineNodeID)
// // 	}
// //
// // 	log.Printf("🌐 Node is now part of the P2P network and ready to serve requests")
// // 	log.Printf("⏳ Keeping node alive... (Press Ctrl+C to stop)")
// //
// // 	// Keep the node running
// // 	select {} // Block forever
// // }
// //
// // func runDistributedInference(cmd *cobra.Command, args []string) error {
// // 	log.Printf("🧠 Starting distributed inference for model: %s", pipelineModel)
// // 	log.Printf("📝 Prompt: %s", pipelinePrompt)
// //
// // 	// Set environment variables
// // 	if pipelineNodeID != "" {
// // 		os.Setenv("NODE_ID", pipelineNodeID)
// // 	}
// // 	os.Setenv("PYTHON_GRPC_ADDRESS", pipelineGRPCAddr)
// //
// // 	// Start P2P node
// // 	ctx := context.Background()
// // 	node, err := p2p.NewNode(ctx, pipelineToken)
// // 	if err != nil {
// // 		return fmt.Errorf("failed to create P2P node: %w", err)
// // 	}
// //
// // 	// Start P2P discovery
// // 	err = p2p.StartP2P(ctx, "", pipelineToken, pipelineNetworkID)
// // 	if err != nil {
// // 		log.Printf("Warning: P2P discovery failed: %v", err)
// // 	}
// //
// // 	// Give some time for discovery
// // 	log.Printf("🔍 Discovering nodes in the network...")
// // 	time.Sleep(5 * time.Second)
// //
// // 	// Create DHT interface
// // 	dht := p2p.NewRegistryDHT(node)
// //
// // 	// Initialize pipeline service
// // 	pipelineService, err := service.NewPipelineInferenceService(node, dht)
// // 	if err != nil {
// // 		return fmt.Errorf("failed to create pipeline service: %w", err)
// // 	}
// // 	defer pipelineService.Close()
// //
// // 	// Run distributed inference
// // 	ctx, cancel := context.WithTimeout(context.Background(), time.Minute*10)
// // 	defer cancel()
// //
// // 	log.Printf("⚡ Running distributed inference...")
// // 	start := time.Now()
// //
// // 	response, err := pipelineService.DistributedGenerateText(
// // 		ctx,
// // 		pipelineModel,
// // 		pipelinePrompt,
// // 		float32(pipelineTemp),
// // 		float32(pipelineTopP),
// // 		int32(pipelineTokens),
// // 	)
// // 	if err != nil {
// // 		return fmt.Errorf("distributed inference failed: %w", err)
// // 	}
// //
// // 	duration := time.Since(start)
// //
// // 	log.Printf("✅ Inference completed in %v", duration)
// // 	fmt.Printf("\n🎯 Generated text:\n%s\n", response.Text)
// //
// // 	return nil
// // }
// //
// // func runNodeStatus(cmd *cobra.Command, args []string) error {
// // 	log.Printf("📊 Checking network status...")
// //
// // 	// Start P2P node for discovery
// // 	ctx := context.Background()
// // 	node, err := p2p.NewNode(ctx, pipelineToken)
// // 	if err != nil {
// // 		return fmt.Errorf("failed to create P2P node: %w", err)
// // 	}
// //
// // 	// Start P2P discovery
// // 	err = p2p.StartP2P(ctx, "", pipelineToken, pipelineNetworkID)
// // 	if err != nil {
// // 		log.Printf("Warning: P2P discovery failed: %v", err)
// // 	}
// //
// // 	// Give time for discovery
// // 	time.Sleep(3 * time.Second)
// //
// // 	// Get all available nodes
// // 	allNodes := p2p.GetAvailableNodes("")
// //
// // 	fmt.Printf("\n🌐 Network Status for network: %s\n", pipelineNetworkID)
// // 	fmt.Printf("═══════════════════════════════════════\n")
// //
// // 	if len(allNodes) == 0 {
// // 		fmt.Printf("❌ No nodes found in the network\n")
// // 		return nil
// // 	}
// //
// // 	fmt.Printf("📈 Found %d nodes:\n\n", len(allNodes))
// //
// // 	for nodeID, nodeData := range allNodes {
// // 		status := "🔴 Offline"
// // 		if nodeData.IsOnline() {
// // 			status = "🟢 Online"
// // 		}
// //
// // 		fmt.Printf("Node: %s %s\n", nodeID, status)
// // 		fmt.Printf("  Model: %s\n", nodeData.ModelName)
// // 		if len(nodeData.AvailableLayers) > 0 {
// // 			fmt.Printf("  Layers: %v (%d total)\n", nodeData.AvailableLayers, len(nodeData.AvailableLayers))
// // 		}
// // 		fmt.Printf("  gRPC: %s\n", nodeData.GRPCAddress)
// // 		fmt.Printf("  Memory: %.1fGB GPU, %.1fGB CPU\n", nodeData.GPUMemoryGB, nodeData.CPUMemoryGB)
// // 		fmt.Printf("  Last seen: %s\n", nodeData.LastSeen.Format("15:04:05"))
// // 		fmt.Printf("\n")
// // 	}
// //
// // 	// Show model coverage
// // 	modelCoverage := make(map[string]map[int32]bool)
// // 	for _, nodeData := range allNodes {
// // 		if nodeData.ModelName == "" || !nodeData.IsOnline() {
// // 			continue
// // 		}
// //
// // 		if modelCoverage[nodeData.ModelName] == nil {
// // 			modelCoverage[nodeData.ModelName] = make(map[int32]bool)
// // 		}
// //
// // 		for _, layer := range nodeData.AvailableLayers {
// // 			modelCoverage[nodeData.ModelName][layer] = true
// // 		}
// // 	}
// //
// // 	fmt.Printf("📋 Model Coverage:\n")
// // 	fmt.Printf("══════════════════\n")
// // 	for modelName, layers := range modelCoverage {
// // 		fmt.Printf("Model: %s\n", modelName)
// // 		fmt.Printf("  Covered layers: %d\n", len(layers))
// //
// // 		// Find gaps
// // 		if len(layers) > 0 {
// // 			maxLayer := int32(0)
// // 			for layer := range layers {
// // 				if layer > maxLayer {
// // 					maxLayer = layer
// // 				}
// // 			}
// //
// // 			var gaps []int32
// // 			for i := int32(0); i <= maxLayer; i++ {
// // 				if !layers[i] {
// // 					gaps = append(gaps, i)
// // 				}
// // 			}
// //
// // 			if len(gaps) > 0 {
// // 				fmt.Printf("  Missing layers: %v\n", gaps)
// // 			} else {
// // 				fmt.Printf("  ✅ Complete coverage (layers 0-%d)\n", maxLayer)
// // 			}
// // 		}
// // 		fmt.Printf("\n")
// // 	}
// //
// // 	return nil
// // }
// //
// // func parseLayerIndices(layersStr string) ([]int32, error) {
// // 	parts := strings.Split(layersStr, ",")
// // 	var indices []int32
// //
// // 	for _, part := range parts {
// // 		part = strings.TrimSpace(part)
// // 		if part == "" {
// // 			continue
// // 		}
// //
// // 		// Handle ranges like "0-7"
// // 		if strings.Contains(part, "-") {
// // 			rangeParts := strings.Split(part, "-")
// // 			if len(rangeParts) != 2 {
// // 				return nil, fmt.Errorf("invalid range format: %s", part)
// // 			}
// //
// // 			start, err := strconv.Atoi(strings.TrimSpace(rangeParts[0]))
// // 			if err != nil {
// // 				return nil, fmt.Errorf("invalid start of range: %s", rangeParts[0])
// // 			}
// //
// // 			end, err := strconv.Atoi(strings.TrimSpace(rangeParts[1]))
// // 			if err != nil {
// // 				return nil, fmt.Errorf("invalid end of range: %s", rangeParts[1])
// // 			}
// //
// // 			for i := start; i <= end; i++ {
// // 				indices = append(indices, int32(i))
// // 			}
// // 		} else {
// // 			// Single layer index
// // 			idx, err := strconv.Atoi(part)
// // 			if err != nil {
// // 				return nil, fmt.Errorf("invalid layer index: %s", part)
// // 			}
// // 			indices = append(indices, int32(idx))
// // 		}
// // 	}
// //
// // 	return indices, nil
// // }
// //
// // func getModelLayerCount(modelName string) int32 {
// // 	// Model-specific layer counts
// // 	layerCounts := map[string]int32{
// // 		"meta-llama/Llama-2-7b-hf":    32,
// // 		"meta-llama/Llama-2-13b-hf":   40,
// // 		"meta-llama/Llama-2-70b-hf":   80,
// // 		"microsoft/DialoGPT-medium":   24,
// // 		"gpt2":                        12,
// // 		"gpt2-medium":                 24,
// // 		"gpt2-large":                  36,
// // 		"gpt2-xl":                     48,
// // 		"EleutherAI/gpt-j-6B":         28,
// // 		"EleutherAI/gpt-neox-20b":     44,
// // 		"bigscience/bloom-7b1":        30,
// // 		"facebook/opt-6.7b":           32,
// // 	}
// //
// // 	if count, exists := layerCounts[modelName]; exists {
// // 		return count
// // 	}
// //
// // 	// Default fallback
// // 	return 32
// // }
// // ```
// //
// // ## 7. **Usage Examples and Documentation**
// //
// // Now let's create comprehensive usage examples:
// //
// // ```bash:scripts/demo.sh
// // #!/bin/bash
// //
// // # EdgeLLM Pipeline Parallelism Demo
// // # This script demonstrates how to set up a distributed inference network
// //
// // set -e
// //
// // echo "🚀 EdgeLLM Pipeline Parallelism Demo"
// // echo "===================================="
// //
// // MODEL="meta-llama/Llama-2-7b-hf"
// // NETWORK_ID="edgellm-demo"
// // TOKEN="demo-token-123"
// //
// // echo "📋 Demo Configuration:"
// // echo "   Model: $MODEL"
// // echo "   Network ID: $NETWORK_ID"
// // echo "   Token: $TOKEN"
// // echo ""
// //
// // # Function to start a node
// // start_node() {
// //     local node_id=$1
// //     local layers=$2
// //     local grpc_port=$3
// //
// //     echo "🎯 Starting Node $node_id (Layers: $layers, gRPC: $grpc_port)"
// //
// //     # Start the Python backend
// //     cd backend/python && python vllm/pipeline_server.py \
// //         --addr "localhost:$grpc_port" \
// //         --node-id "$node_id" &
// //
// //     # Wait for backend to start
// //     sleep 3
// //
// //     # Start the Go node
// //     ./edgellm pipeline load-shard \
// //         --model "$MODEL" \
// //         --layers "$layers" \
// //         --node-id "$node_id" \
// //         --network-id "$NETWORK_ID" \
// //         --token "$TOKEN" \
// //         --grpc-addr "localhost:$grpc_port" &
// //
// //     sleep 2
// // }
// //
// // echo "🔧 Setting up the distributed network..."
// //
// // # Start 4 nodes with different layer ranges
// // start_node "node1" "0-7"   "50051"
// // start_node "node2" "8-15"  "50052"
// // start_node "node3" "16-23" "50053"
// // start_node "node4" "24-31" "50054"
// //
// // echo "⏳ Waiting for all nodes to start and discover each other..."
// // sleep 10
// //
// // echo "📊 Checking network status..."
// // ./edgellm pipeline status \
// //     --network-id "$NETWORK_ID" \
// //     --token "$TOKEN"
// //
// // echo ""
// // echo "🧠 Running distributed inference..."
// // ./edgellm pipeline infer \
// //     --model "$MODEL" \
// //     --prompt "The future of artificial intelligence is" \
// //     --temperature 0.8 \
// //     --max-tokens 150 \
// //     --network-id "$NETWORK_ID" \
// //     --token "$TOKEN" \
// //     --grpc-addr "localhost:50051"
// //
// // echo ""
// // echo "✅ Demo completed!"
// // echo "💡 To stop all nodes, run: pkill -f edgellm"
// // ```
// //
// // ```yaml:examples/distributed-setup.yaml
// // # Example configuration for distributed LLM inference
// // networks:
// //   - name: "llama-7b-network"
// //     model: "meta-llama/Llama-2-7b-hf"
// //     total_layers: 32
// //     nodes:
// //       - id: "gpu-node-1"
// //         layers: [0, 1, 2, 3, 4, 5, 6, 7]
// //         gpu_memory: "8GB"
// //         location: "datacenter-1"
// //
// //       - id: "gpu-node-2"
// //         layers: [8, 9, 10, 11, 12, 13, 14, 15]
// //         gpu_memory: "8GB"
// //         location: "datacenter-2"
// //
// //       - id: "gpu-node-3"
// //         layers: [16, 17, 18, 19, 20, 21, 22, 23]
// //         gpu_memory: "8GB"
// //         location: "edge-device-1"
// //
// //       - id: "gpu-node-4"
// //         layers: [24, 25, 26, 27, 28, 29, 30, 31]
// //         gpu_memory: "8GB"
// //         location: "edge-device-2"
// //
// //   - name: "gpt-xl-network"
// //     model: "gpt2-xl"
// //     total_layers: 48
// //     nodes:
// //       - id: "powerful-node-1"
// //         layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
// //         gpu_memory: "16GB"
// //
// //       - id: "powerful-node-2"
// //         layers: [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
// //         gpu_memory: "16GB"
// //
// //       - id: "powerful-node-3"
// //         layers: [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
// //         gpu_memory: "16GB"
// // ```
// //
// // ## 8. **Complete Integration Test**
// //
// // ```go:test/pipeline_integration_test.go
// // <code_block_to_apply_changes_from>
// // package test
// //
// // import (
// // 	"context"
// // 	"testing"
// // 	"time"
// //
// // 	"github.com/d0w/EdgeLLM/internal/p2p"
// // 	"github.com/d0w/EdgeLLM/internal/service"
// // 	"github.com/stretchr/testify/assert"
// // 	"github.com/stretchr/testify/require"
// // )
// //
// // func TestPipelineParallelism(t *testing.T) {
// // 	if testing.Short() {
// // 		t.Skip("Skipping integration test in short mode")
// // 	}
// //
// // 	ctx := context.Background()
// //
// // 	// Create test nodes
// // 	node1, err := p2p.NewNode(ctx, "test-token")
// // 	require.NoError(t, err)
// //
// // 	node2, err := p2p.NewNode(ctx, "test-token")
// // 	require.NoError(t, err)
// //
// // 	// Create DHTs
// // 	dht1 := p2p.NewRegistryDHT(node1)
// // 	dht2 := p2p.NewRegistryDHT(node2)
// //
// // 	// Create pipeline services
// // 	service1, err := service.NewPipelineInferenceService(node1, dht1)
// // 	require.NoError(t, err)
// // 	defer service1.Close()
// //
// // 	service2, err := service.NewPipelineInferenceService(node2, dht2)
// // 	require.NoError(t, err)
// // 	defer service2.Close()
// //
// // 	// Load different shards on each node
// // 	modelName := "gpt2"
// //
// // 	err = service1.LoadModelShard(ctx, modelName, []int32{0, 1, 2, 3, 4, 5}, 12)
// // 	require.NoError(t, err)
// //
// // 	err = service2.LoadModelShard(ctx, modelName, []int32{6, 7, 8, 9, 10, 11}, 12)
// // 	require.NoError(t, err)
// //
// // 	// Wait for discovery
// // 	time.Sleep(2 * time.Second)
// //
// // 	// Test distributed inference
// // 	response, err := service1.DistributedGenerateText(
// // 		ctx,
// // 		modelName,
// // 		"Hello world",
// // 		0.7,
// // 		0.9,
// // 		50,
// // 	)
// //
// // 	require.NoError(t, err)
// // 	assert.NotEmpty(t, response.Text)
// // 	assert.True(t, response.IsFinal)
// // }
// //
// // func TestNodeCapabilities(t *testing.T) {
// // 	ctx := context.Background()
// //
// // 	node, err := p2p.NewNode(ctx, "test-token")
// // 	require.NoError(t, err)
// //
// // 	dht := p2p.NewRegistryDHT(node)
// // 	service, err := service.NewPipelineInferenceService(node, dht)
// // 	require.NoError(t, err)
// // 	defer service.Close()
// //
// // 	// Load a shard
// // 	err = service.LoadModelShard(ctx, "gpt2", []int32{0, 1, 2}, 12)
// // 	require.NoError(t, err)
// //
// // 	// Query capabilities
// // 	caps, err := service.GetLocalCapabilities(ctx)
// // 	require.NoError(t, err)
// //
// // 	assert.Equal(t, "gpt2", caps.ModelName)
// // 	assert.Contains(t, caps.AvailableLayers, int32(0))
// // 	assert.Contains(t, caps.AvailableLayers, int32(1))
// // 	assert.Contains(t, caps.AvailableLayers, int32(2))
// // 	assert.Equal(t, int32(12), caps.TotalLayers)
// // }
// // ```
// //
// // ## Complete Usage Examples
// //
// // ### **1. Start a single node with layers 0-7:**
// // ```bash
// // # Terminal 1: Start Python backend
// // cd backend/python
// // python vllm/pipeline_server.py --addr localhost:50051 --node-id node1
// //
// // # Terminal 2: Start Go node
// // ./edgellm pipeline load-shard \
// //   --model "meta-llama/Llama-2-7b-hf" \
// //   --layers "0-7" \
// //   --node-id "node1" \
// //   --grpc-addr "localhost:50051"
// // ```
// //
// // ### **2. Start multiple nodes for complete model coverage:**
// // ```bash
// // # Node 1 (layers 0-7)
// // python backend/python/vllm/pipeline_server.py --addr localhost:50051 --node-id node1 &
// // ./edgellm pipeline load-shard --model "meta-llama/Llama-2-7b-hf" --layers "0-7" --node-id "node1" --grpc-addr "localhost:50051" &
// //
// // # Node 2 (layers 8-15)
// // python backend/python/vllm/pipeline_server.py --addr localhost:50052 --node-id node2 &
// // ./edgellm pipeline load-shard --model "meta-llama/Llama-2-7b-hf" --layers "8-15" --node-id "node2" --grpc-addr "localhost:50052" &
// //
// // # Node 3 (layers 16-23)
// // python backend/python/vllm/pipeline_server.py --addr localhost:50053 --node-id node3 &
// // ./edgellm pipeline load-shard --model "meta-llama/Llama-2-7b-hf" --layers "16-23" --node-id "node3" --grpc-addr "localhost:50053" &
// //
// // # Node 4 (layers 24-31)
// // python backend/python/vllm/pipeline_server.py --addr localhost:50054 --node-id node4 &
// // ./edgellm pipeline load-shard --model "meta-llama/Llama-2-7b-hf" --layers "24-31" --node-id "node4" --grpc-addr "localhost:50054" &
// // ```
// //
// // ### **3. Run distributed inference:**
// // ```bash
// // ./edgellm pipeline infer \
// //   --model "meta-llama/Llama-2-7b-hf" \
// //   --prompt "The future of artificial intelligence is" \
// //   --temperature 0.8 \
// //   --max-tokens 200
// // ```
// //
// // ### **4. Check network status:**
// // ```bash
// // ./edgellm pipeline status
// // ```
// //
// // This complete implementation provides:
// //
// // 1. **True Pipeline Parallelism** - Each node loads only the layers it needs
// // 2. **Automatic P2P Discovery** - Nodes find each other and advertise capabilities
// // 3. **Intelligent Routing** - System finds optimal paths through available nodes
// // 4. **Real Tensor Passing** - Hidden states are serialized and passed between nodes
// // 5. **Fault Tolerance** - Handles node failures and network partitions
// // 6. **Resource Efficiency** - Dramatically reduces memory requirements per node
// // 7. **Easy CLI Interface** - Simple commands to set up and use the system
// //
// // The system can handle models much larger than what any single node could run, enabling truly decentralized LLM inference at scale!
//
