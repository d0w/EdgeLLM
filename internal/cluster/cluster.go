package cluster

import (
	"context"
	"fmt"
	"sync"

	"github.com/mudler/edgevpn/pkg/node"
)

// Cluster represents an ephemeral VPN-based cluster.
type Cluster struct {
	RequesterNode *node.Node
	HeadNode      *node.Node
	WorkerNodes   []*node.Node
	mu            sync.Mutex
	started       bool
}

// NewCluster creates a new Cluster instance.
func NewCluster(requester *node.Node, workers []*node.Node) *Cluster {
	return &Cluster{
		RequesterNode: requester,
		WorkerNodes:   workers,
		HeadNode:      workers[0],
	}
}

// Start brings up all nodes in the cluster.
func (c *Cluster) Start(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.started {
		return fmt.Errorf("cluster already started")
	}
	if err := c.RequesterNode.Start(ctx); err != nil {
		return fmt.Errorf("failed to start requester node: %w", err)
	}
	for i, w := range c.WorkerNodes {
		if err := w.Start(ctx); err != nil {
			return fmt.Errorf("failed to start worker node %d: %w", i, err)
		}
	}
	c.started = true
	return nil
}

// Stop gracefully stops all nodes in the cluster.
func (c *Cluster) Stop() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.started {
		return fmt.Errorf("cluster not started")
	}
	for i, w := range c.WorkerNodes {
		if err := w.Stop(); err != nil {
			return fmt.Errorf("failed to stop worker node %d: %w", i, err)
		}
	}
	if err := c.RequesterNode.Stop(); err != nil {
		return fmt.Errorf("failed to stop requester node: %w", err)
	}
	c.started = false
	return nil
}

// Broadcast sends a message to all nodes in the cluster.
func (c *Cluster) Broadcast(msg []byte) error {
	for i, w := range c.WorkerNodes {
		if err := w.Send(msg); err != nil {
			return fmt.Errorf("failed to send message to worker node %d: %w", i, err)
		}
	}
	return nil
}
