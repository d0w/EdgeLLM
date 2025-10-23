package cli

// import (
// 	"context"
// 	"fmt"
// 	"log"
// 	"os"
// 	"strconv"
// 	"strings"
// 	"time"
//
// 	"github.com/d0w/EdgeLLM/internal/p2p"
// 	"github.com/d0w/EdgeLLM/internal/service"
// 	"github.com/spf13/cobra"
// )
//
// var (
// 	pipelineModel     string
// 	pipelineLayers    string
// 	pipelinePrompt    string
// 	pipelineTemp      float64
// 	pipelineTopP      float64
// 	pipelineTokens    int
// 	pipelineNodeID    string
// 	pipelineNetworkID string
// 	pipelineToken     string
// 	pipelineGRPCAddr  string
// )
//
// var pipelineCmd = &cobra.Command{
// 	Use:   "pipeline",
// 	Short: "Pipeline parallelism commands for distributed inference",
// 	Long: `Pipeline parallelism allows you to split large language models across
// multiple nodes in a P2P network, where each node handles specific layers of the model.`,
// }
//
// var loadShardCmd = &cobra.Command{
// 	Use:   "load-shard",
// 	Short: "Load a model shard (specific layers) on this node",
// 	Long: `Load specific layers of a language model on this node. The node will then
// advertise its capabilities to the P2P network and be available for distributed inference.`,
// 	RunE: runLoadShard,
// }
//
// var distributedInferenceCmd = &cobra.Command{
// 	Use:   "infer",
// 	Short: "Run distributed inference across the P2P network",
// 	Long: `Run inference on a large language model by distributing the computation
// across multiple nodes in the P2P network. Each node processes its assigned layers.`,
// 	RunE: runDistributedInference,
// }
//
// var nodeStatusCmd = &cobra.Command{
// 	Use:   "status",
// 	Short: "Show the status of nodes in the network",
// 	RunE:  runNodeStatus,
// }
//
// func init() {
// 	// Common flags
// 	pipelineCmd.PersistentFlags().StringVar(&pipelineNodeID, "node-id", "", "Unique node identifier")
// 	pipelineCmd.PersistentFlags().StringVar(&pipelineNetworkID, "network-id", "edgellm", "P2P network identifier")
// 	pipelineCmd.PersistentFlags().StringVar(&pipelineToken, "token", "", "P2P network token")
// 	pipelineCmd.PersistentFlags().StringVar(&pipelineGRPCAddr, "grpc-addr", "localhost:50051", "Local gRPC server address")
//
// 	// Load shard command flags
// 	loadShardCmd.Flags().StringVar(&pipelineModel, "model", "", "Model name to load (e.g., meta-llama/Llama-2-7b-hf)")
// 	loadShardCmd.Flags().StringVar(&pipelineLayers, "layers", "", "Comma-separated layer indices (e.g., 0,1,2,3)")
// 	loadShardCmd.MarkFlagRequired("model")
// 	loadShardCmd.MarkFlagRequired("layers")
//
// 	// Distributed inference command flags
// 	distributedInferenceCmd.Flags().StringVar(&pipelineModel, "model", "", "Model name")
// 	distributedInferenceCmd.Flags().StringVar(&pipelinePrompt, "prompt", "", "Prompt text")
// 	distributedInferenceCmd.Flags().Float64Var(&pipelineTemp, "temperature", 0.7, "Temperature for sampling")
// 	distributedInferenceCmd.Flags().Float64Var(&pipelineTopP, "top-p", 0.9, "Top-p for nucleus sampling")
// 	distributedInferenceCmd.Flags().IntVar(&pipelineTokens, "max-tokens", 100, "Maximum tokens to generate")
// 	distributedInferenceCmd.MarkFlagRequired("model")
// 	distributedInferenceCmd.MarkFlagRequired("prompt")
//
// 	pipelineCmd.AddCommand(loadShardCmd)
// 	pipelineCmd.AddCommand(distributedInferenceCmd)
// 	pipelineCmd.AddCommand(nodeStatusCmd)
// 	rootCmd.AddCommand(pipelineCmd)
// }
//
// func runLoadShard(cmd *cobra.Command, args []string) error {
// 	log.Printf("Starting node to load model shard: %s, layers: %s", pipelineModel, pipelineLayers)
//
// 	// Parse layer indices
// 	layerIndices, err := parseLayerIndices(pipelineLayers)
// 	if err != nil {
// 		return fmt.Errorf("invalid layer indices: %w", err)
// 	}
//
// 	// Set environment variables for the Python backend
// 	if pipelineNodeID != "" {
// 		os.Setenv("NODE_ID", pipelineNodeID)
// 	}
// 	os.Setenv("PYTHON_GRPC_ADDRESS", pipelineGRPCAddr)
//
// 	ctx := context.Background()
// 	node, err := p2p.NewNode(ctx, pipelineToken)
// 	if err != nil {
// 		return fmt.Errorf("failed to create P2P node: %w", err)
// 	}
//
// 	// Start P2P discovery
// 	err = p2p.StartP2P(ctx, node, "", pipelineToken, pipelineNetworkID)
// 	if err != nil {
// 		log.Printf("Warning: P2P discovery failed: %v", err)
// 	}
//
// 	dht := p2p.NewRegistryDHT(node)
//
// 	pipelineService, err := service.NewPipelineInferenceService(node, dht)
// 	if err != nil {
// 		return fmt.Errorf("failed to create pipeline service: %w", err)
// 	}
// 	defer pipelineService.Close()
//
// 	// Load the shard (determine total layers based on model)
// 	totalLayers := getModelLayerCount(pipelineModel)
// 	err = pipelineService.LoadModelShard(ctx, pipelineModel, layerIndices, totalLayers)
// 	if err != nil {
// 		return fmt.Errorf("failed to load model shard: %w", err)
// 	}
//
// 	log.Printf("Model shard loaded successfully!")
// 	log.Printf("Node capabilities:")
// 	log.Printf("   - Model: %s", pipelineModel)
// 	log.Printf("   - Layers: %v", layerIndices)
// 	log.Printf("   - Total layers: %d", totalLayers)
// 	log.Printf("   - gRPC address: %s", pipelineGRPCAddr)
//
// 	if pipelineNodeID != "" {
// 		log.Printf("   - Node ID: %s", pipelineNodeID)
// 	}
//
// 	log.Printf("Node is now part of the P2P network and ready to serve requests")
// 	log.Printf("Keeping node alive... (Press Ctrl+C to stop)")
//
// 	// Keep the node running
// 	select {}
// }
//
// func runDistributedInference(cmd *cobra.Command, args []string) error {
// 	log.Printf("Starting distributed inference for model: %s", pipelineModel)
// 	log.Printf("Prompt: %s", pipelinePrompt)
//
// 	// Set environment variables
// 	if pipelineNodeID != "" {
// 		os.Setenv("NODE_ID", pipelineNodeID)
// 	}
// 	os.Setenv("PYTHON_GRPC_ADDRESS", pipelineGRPCAddr)
//
// 	ctx := context.Background()
// 	node, err := p2p.NewNode(ctx, pipelineToken)
// 	if err != nil {
// 		return fmt.Errorf("failed to create P2P node: %w", err)
// 	}
//
// 	err = p2p.StartP2P(ctx, node, "", pipelineToken, pipelineNetworkID)
// 	if err != nil {
// 		log.Printf("Warning: P2P discovery failed: %v", err)
// 	}
//
// 	log.Printf("Discovering nodes in the network...")
// 	time.Sleep(5 * time.Second)
//
// 	dht := p2p.NewRegistryDHT(node)
//
// 	pipelineService, err := service.NewPipelineInferenceService(node, dht)
// 	if err != nil {
// 		return fmt.Errorf("failed to create pipeline service: %w", err)
// 	}
// 	defer pipelineService.Close()
//
// 	// Run distributed inference
// 	ctx, cancel := context.WithTimeout(context.Background(), time.Minute*10)
// 	defer cancel()
//
// 	log.Printf("Running distributed inference...")
// 	start := time.Now()
//
// 	response, err := pipelineService.DistributedGenerateText(
// 		ctx,
// 		pipelineModel,
// 		pipelinePrompt,
// 		float32(pipelineTemp),
// 		float32(pipelineTopP),
// 		int32(pipelineTokens),
// 	)
// 	if err != nil {
// 		return fmt.Errorf("distributed inference failed: %w", err)
// 	}
//
// 	duration := time.Since(start)
//
// 	log.Printf("Inference completed in %v", duration)
// 	fmt.Printf("\nGenerated text:\n%s\n", response.Text)
//
// 	return nil
// }
//
// func runNodeStatus(cmd *cobra.Command, args []string) error {
// 	log.Printf("Checking network status...")
//
// 	// Start P2P node for discovery
// 	ctx := context.Background()
// 	node, err := p2p.NewNode(ctx, pipelineToken)
// 	if err != nil {
// 		return fmt.Errorf("failed to create P2P node: %w", err)
// 	}
//
// 	// Start P2P discovery
// 	err = p2p.StartP2P(ctx, node, "", pipelineToken, pipelineNetworkID)
// 	if err != nil {
// 		log.Printf("Warning: P2P discovery failed: %v", err)
// 	}
//
// 	// Give time for discovery
// 	time.Sleep(3 * time.Second)
//
// 	// Get all available nodes
// 	allNodes := p2p.GetAvailableNodes("")
//
// 	fmt.Printf("\nNetwork Status for network: %s\n", pipelineNetworkID)
// 	fmt.Printf("═══════════════════════════════════════\n")
//
// 	if len(allNodes) == 0 {
// 		fmt.Printf("No nodes found in the network\n")
// 		return nil
// 	}
//
// 	fmt.Printf("Found %d nodes:\n\n", len(allNodes))
//
// 	for nodeID, nodeData := range allNodes {
// 		status := "Offline"
// 		if nodeData.IsOnline() {
// 			status = "Online"
// 		}
//
// 		fmt.Printf("Node: %s %s\n", nodeID, status)
// 		fmt.Printf("  Model: %s\n", nodeData.ModelName)
// 		if len(nodeData.AvailableLayers) > 0 {
// 			fmt.Printf("  Layers: %v (%d total)\n", nodeData.AvailableLayers, len(nodeData.AvailableLayers))
// 		}
// 		fmt.Printf("  gRPC: %s\n", nodeData.GRPCAddress)
// 		fmt.Printf("  Memory: %.1fGB GPU, %.1fGB CPU\n", nodeData.GPUMemoryGB, nodeData.CPUMemoryGB)
// 		fmt.Printf("  Last seen: %s\n", nodeData.LastSeen.Format("15:04:05"))
// 		fmt.Printf("\n")
// 	}
//
// 	// Show model coverage
// 	modelCoverage := make(map[string]map[int32]bool)
// 	for _, nodeData := range allNodes {
// 		if nodeData.ModelName == "" || !nodeData.IsOnline() {
// 			continue
// 		}
//
// 		if modelCoverage[nodeData.ModelName] == nil {
// 			modelCoverage[nodeData.ModelName] = make(map[int32]bool)
// 		}
//
// 		for _, layer := range nodeData.AvailableLayers {
// 			modelCoverage[nodeData.ModelName][layer] = true
// 		}
// 	}
//
// 	fmt.Printf("Model Coverage:\n")
// 	fmt.Printf("══════════════════\n")
// 	for modelName, layers := range modelCoverage {
// 		fmt.Printf("Model: %s\n", modelName)
// 		fmt.Printf("  Covered layers: %d\n", len(layers))
//
// 		// Find gaps
// 		if len(layers) > 0 {
// 			maxLayer := int32(0)
// 			for layer := range layers {
// 				if layer > maxLayer {
// 					maxLayer = layer
// 				}
// 			}
//
// 			var gaps []int32
// 			for i := int32(0); i <= maxLayer; i++ {
// 				if !layers[i] {
// 					gaps = append(gaps, i)
// 				}
// 			}
//
// 			if len(gaps) > 0 {
// 				fmt.Printf("  Missing layers: %v\n", gaps)
// 			} else {
// 				fmt.Printf("  Complete coverage (layers 0-%d)\n", maxLayer)
// 			}
// 		}
// 		fmt.Printf("\n")
// 	}
//
// 	return nil
// }
//
// func parseLayerIndices(layersStr string) ([]int32, error) {
// 	parts := strings.Split(layersStr, ",")
// 	var indices []int32
//
// 	for _, part := range parts {
// 		part = strings.TrimSpace(part)
// 		if part == "" {
// 			continue
// 		}
//
// 		// Handle ranges like "0-7"
// 		if strings.Contains(part, "-") {
// 			rangeParts := strings.Split(part, "-")
// 			if len(rangeParts) != 2 {
// 				return nil, fmt.Errorf("invalid range format: %s", part)
// 			}
//
// 			start, err := strconv.Atoi(strings.TrimSpace(rangeParts[0]))
// 			if err != nil {
// 				return nil, fmt.Errorf("invalid start of range: %s", rangeParts[0])
// 			}
//
// 			end, err := strconv.Atoi(strings.TrimSpace(rangeParts[1]))
// 			if err != nil {
// 				return nil, fmt.Errorf("invalid end of range: %s", rangeParts[1])
// 			}
//
// 			for i := start; i <= end; i++ {
// 				indices = append(indices, int32(i))
// 			}
// 		} else {
// 			// Single layer index
// 			idx, err := strconv.Atoi(part)
// 			if err != nil {
// 				return nil, fmt.Errorf("invalid layer index: %s", part)
// 			}
// 			indices = append(indices, int32(idx))
// 		}
// 	}
//
// 	return indices, nil
// }
//
// func getModelLayerCount(modelName string) int32 {
// 	// Model-specific layer counts
// 	layerCounts := map[string]int32{
// 		"meta-llama/Llama-2-7b-hf":  32,
// 		"meta-llama/Llama-2-13b-hf": 40,
// 		"meta-llama/Llama-2-70b-hf": 80,
// 		"microsoft/DialoGPT-medium": 24,
// 		"gpt2":                      12,
// 		"gpt2-medium":               24,
// 		"gpt2-large":                36,
// 		"gpt2-xl":                   48,
// 		"EleutherAI/gpt-j-6B":       28,
// 		"EleutherAI/gpt-neox-20b":   44,
// 		"bigscience/bloom-7b1":      30,
// 		"facebook/opt-6.7b":         32,
// 	}
//
// 	if count, exists := layerCounts[modelName]; exists {
// 		return count
// 	}
//
// 	// Default fallback
// 	return 32
// }
//
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
// // echo "EdgeLLM Pipeline Parallelism Demo"
// // echo "===================================="
// //
// // MODEL="meta-llama/Llama-2-7b-hf"
// // NETWORK_ID="edgellm-demo"
// // TOKEN="demo-token-123"
// //
// // echo "Demo Configuration:"
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
// //     echo "Starting Node $node_id (Layers: $layers, gRPC: $grpc_port)"
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
// // <code_block_to_apply_changes_from>
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
