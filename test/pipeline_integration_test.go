package test

import (
	"context"
	"testing"
	"time"

	"github.com/d0w/EdgeLLM/internal/p2p"
	"github.com/d0w/EdgeLLM/internal/service"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPipelineParallelism(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	ctx := context.Background()
	
	// Create test nodes
	node1, err := p2p.NewNode(ctx, "test-token")
	require.NoError(t, err)
	
	node2, err := p2p.NewNode(ctx, "test-token") 
	require.NoError(t, err)

	// Create DHTs
	dht1 := p2p.NewRegistryDHT(node1)
	dht2 := p2p.NewRegistryDHT(node2)

	// Create pipeline services
	service1, err := service.NewPipelineInferenceService(node1, dht1)
	require.NoError(t, err)
	defer service1.Close()

	service2, err := service.NewPipelineInferenceService(node2, dht2)
	require.NoError(t, err)
	defer service2.Close()

	// Load different shards on each node
	modelName := "gpt2"
	
	err = service1.LoadModelShard(ctx, modelName, []int32{0, 1, 2, 3, 4, 5}, 12)
	require.NoError(t, err)

	err = service2.LoadModelShard(ctx, modelName, []int32{6, 7, 8, 9, 10, 11}, 12)
	require.NoError(t, err)

	// Wait for discovery
	time.Sleep(2 * time.Second)

	// Test distributed inference
	response, err := service1.DistributedGenerateText(
		ctx, 
		modelName,
		"Hello world", 
		0.7, 
		0.9, 
		50,
	)
	
	require.NoError(t, err)
	assert.NotEmpty(t, response.Text)
	assert.True(t, response.IsFinal)
}

func TestNodeCapabilities(t *testing.T) {
	ctx := context.Background()
	
	node, err := p2p.NewNode(ctx, "test-token")
	require.NoError(t, err)

	dht := p2p.NewRegistryDHT(node)
	service, err := service.NewPipelineInferenceService(node, dht)
	require.NoError(t, err)
	defer service.Close()

	// Load a shard
	err = service.LoadModelShard(ctx, "gpt2", []int32{0, 1, 2}, 12)
	require.NoError(t, err)

	// Query capabilities
	caps, err := service.GetLocalCapabilities(ctx)
	require.NoError(t, err)

	assert.Equal(t, "gpt2", caps.ModelName)
	assert.Contains(t, caps.AvailableLayers, int32(0))
	assert.Contains(t, caps.AvailableLayers, int32(1))
	assert.Contains(t, caps.AvailableLayers, int32(2))
	assert.Equal(t, int32(12), caps.TotalLayers)
}

```

```
