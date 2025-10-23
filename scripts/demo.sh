#!/bin/bash

# EdgeLLM Pipeline Parallelism Demo
# This script demonstrates how to set up a distributed inference network

set -e

echo "🚀 EdgeLLM Pipeline Parallelism Demo"
echo "===================================="

MODEL="meta-llama/Llama-2-7b-hf"
NETWORK_ID="edgellm-demo"
TOKEN="demo-token-123"

echo "📋 Demo Configuration:"
echo "   Model: $MODEL"
echo "   Network ID: $NETWORK_ID"
echo "   Token: $TOKEN"
echo ""

# Function to start a node
start_node() {
    local node_id=$1
    local layers=$2
    local grpc_port=$3
    
    echo "🎯 Starting Node $node_id (Layers: $layers, gRPC: $grpc_port)"
    
    # Start the Python backend
    cd backend/python && python vllm/pipeline_server.py \
        --addr "localhost:$grpc_port" \
        --node-id "$node_id" &
    
    # Wait for backend to start
    sleep 3
    
    # Start the Go node
    ./edgellm pipeline load-shard \
        --model "$MODEL" \
        --layers "$layers" \
        --node-id "$node_id" \
        --network-id "$NETWORK_ID" \
        --token "$TOKEN" \
        --grpc-addr "localhost:$grpc_port" &
    
    sleep 2
}

echo "🔧 Setting up the distributed network..."

# Start 4 nodes with different layer ranges
start_node "node1" "0-7"   "50051"
start_node "node2" "8-15"  "50052"
start_node "node3" "16-23" "50053"
start_node "node4" "24-31" "50054"

echo "⏳ Waiting for all nodes to start and discover each other..."
sleep 10

echo "📊 Checking network status..."
./edgellm pipeline status \
    --network-id "$NETWORK_ID" \
    --token "$TOKEN"

echo ""
echo "🧠 Running distributed inference..."
./edgellm pipeline infer \
    --model "$MODEL" \
    --prompt "The future of artificial intelligence is" \
    --temperature 0.8 \
    --max-tokens 150 \
    --network-id "$NETWORK_ID" \
    --token "$TOKEN" \
    --grpc-addr "localhost:50051"

echo ""
echo "✅ Demo completed!"
echo "💡 To stop all nodes, run: pkill -f edgellm" 