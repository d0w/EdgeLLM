package service

import (
	"context"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	pb "github.com/d0w/EdgeLLM/backend/go/proto"
	"github.com/d0w/EdgeLLM/pkg/logger"
)

type NodeCoordinatorService struct {
	nodes      map[string]*NodeInfo
	nodesMutex sync.RWMutex
	logger     *logger.Logger
	clusterID  string
}

type NodeInfo struct {
	ID        string
	Address   string
	Port      int32
	Capacity  *pb.NodeCapacity
	LastSeen  time.Time
	Status    string
	LoadScore float32
	Models    []string
}

func NewNodeCoordinatorService(logger *logger.Logger) *NodeCoordinatorService {
	return &NodeCoordinatorService{
		nodes:     make(map[string]*NodeInfo),
		logger:    logger,
		clusterID: "cluster-" + fmt.Sprintf("%d", time.Now().Unix()),
	}
}

func (s *NodeCoordinatorService) RegisterNode(ctx context.Context, req *pb.NodeRegistration) (*pb.NodeRegistrationResponse, error) {
	s.nodesMutex.Lock()
	defer s.nodesMutex.Unlock()

	s.logger.Info(fmt.Sprintf("Registering node %s at %s:%d", req.NodeId, req.Address, req.Port))

	nodeInfo := &NodeInfo{
		ID:       req.NodeId,
		Address:  req.Address,
		Port:     req.Port,
		Capacity: req.Capacity,
		LastSeen: time.Now(),
		Status:   "active",
		Models:   req.SupportedModels,
	}

	s.nodes[req.NodeId] = nodeInfo
	s.updateLoadScore(nodeInfo)

	return &pb.NodeRegistrationResponse{
		Success:   true,
		Message:   "Node registered successfully",
		ClusterId: s.clusterID,
	}, nil
}

func (s *NodeCoordinatorService) GetAvailableNodes(ctx context.Context, req *pb.NodesRequest) (*pb.NodesResponse, error) {
	s.nodesMutex.RLock()
	defer s.nodesMutex.RUnlock()

	var availableNodes []*pb.NodeInfo

	for _, node := range s.nodes {
		if s.isNodeHealthy(node) && s.supportsModel(node, req.Model) {
			pbNode := &pb.NodeInfo{
				NodeId:    node.ID,
				Address:   node.Address,
				Port:      node.Port,
				Capacity:  node.Capacity,
				LoadScore: node.LoadScore,
				Status:    node.Status,
			}
			availableNodes = append(availableNodes, pbNode)
		}
	}

	// Sort by load score (ascending - lower is better)
	sort.Slice(availableNodes, func(i, j int) bool {
		return availableNodes[i].LoadScore < availableNodes[j].LoadScore
	})

	// Limit to requested count
	if req.MinNodes > 0 && len(availableNodes) > int(req.MinNodes) {
		availableNodes = availableNodes[:req.MinNodes]
	}

	return &pb.NodesResponse{
		Nodes:          availableNodes,
		TotalAvailable: int32(len(availableNodes)),
	}, nil
}

func (s *NodeCoordinatorService) RouteInference(ctx context.Context, req *pb.InferenceRoutingRequest) (*pb.InferenceRoutingResponse, error) {
	s.nodesMutex.RLock()
	defer s.nodesMutex.RUnlock()

	// If preferred node is specified and available, use it
	if req.PreferredNode != "" {
		if node, exists := s.nodes[req.PreferredNode]; exists && s.isNodeHealthy(node) {
			return &pb.InferenceRoutingResponse{
				AssignedNode:       req.PreferredNode,
				RoutingStrategy:    "preferred",
				EstimatedQueueTime: s.estimateQueueTime(node),
			}, nil
		}
	}

	// Find best available node
	var bestNode *NodeInfo
	bestScore := float32(math.Inf(1))

	for _, node := range s.nodes {
		if s.isNodeHealthy(node) && s.supportsModel(node, req.Request.Model) {
			if node.LoadScore < bestScore {
				bestScore = node.LoadScore
				bestNode = node
			}
		}
	}

	if bestNode == nil {
		return nil, fmt.Errorf("no available nodes for model %s", req.Request.Model)
	}

	return &pb.InferenceRoutingResponse{
		AssignedNode:       bestNode.ID,
		RoutingStrategy:    "load_based",
		EstimatedQueueTime: s.estimateQueueTime(bestNode),
	}, nil
}

func (s *NodeCoordinatorService) ReportCapacity(ctx context.Context, req *pb.CapacityReport) (*pb.CapacityResponse, error) {
	s.nodesMutex.Lock()
	defer s.nodesMutex.Unlock()

	if node, exists := s.nodes[req.NodeId]; exists {
		node.Capacity = req.CurrentCapacity
		node.LastSeen = time.Now()
		s.updateLoadScore(node)

		return &pb.CapacityResponse{
			Acknowledged: true,
			Message:      "Capacity report received",
		}, nil
	}

	return &pb.CapacityResponse{
		Acknowledged: false,
		Message:      "Node not found",
	}, nil
}

func (s *NodeCoordinatorService) updateLoadScore(node *NodeInfo) {
	if node.Capacity == nil {
		node.LoadScore = 100.0
		return
	}

	// Calculate load score based on multiple factors
	memoryUtilization := node.Capacity.MemoryUtilization
	cpuUtilization := node.Capacity.CpuUtilization
	activeRequests := float32(node.Capacity.MaxConcurrentRequests)

	// Weight different factors
	loadScore := (memoryUtilization * 0.4) + (cpuUtilization * 0.3) + (activeRequests * 0.3)
	node.LoadScore = loadScore
}

func (s *NodeCoordinatorService) isNodeHealthy(node *NodeInfo) bool {
	return time.Since(node.LastSeen) < 30*time.Second && node.Status == "active"
}

func (s *NodeCoordinatorService) supportsModel(node *NodeInfo, model string) bool {
	if model == "" {
		return true // No specific model requirement
	}

	for _, supportedModel := range node.Models {
		if supportedModel == model {
			return true
		}
	}
	return false
}

func (s *NodeCoordinatorService) estimateQueueTime(node *NodeInfo) int32 {
	if node.Capacity == nil {
		return 10000 // 10 seconds default
	}

	// Simple estimation based on current load
	baseTime := int32(1000) // 1 second base
	loadMultiplier := node.LoadScore / 100.0

	return int32(float32(baseTime) * (1.0 + loadMultiplier))
}

// Background cleanup of stale nodes
func (s *NodeCoordinatorService) StartCleanupWorker() {
	go func() {
		ticker := time.NewTicker(30 * time.Second)
		defer ticker.Stop()

		for range ticker.C {
			s.cleanupStaleNodes()
		}
	}()
}

func (s *NodeCoordinatorService) cleanupStaleNodes() {
	s.nodesMutex.Lock()
	defer s.nodesMutex.Unlock()

	now := time.Now()
	for nodeID, node := range s.nodes {
		if now.Sub(node.LastSeen) > 60*time.Second {
			s.logger.Info(fmt.Sprintf("Removing stale node: %s", nodeID))
			delete(s.nodes, nodeID)
		}
	}
}

func (s *NodeCoordinatorService) GetClusterStatus() *pb.ClusterStatusResponse {
	s.nodesMutex.RLock()
	defer s.nodesMutex.RUnlock()

	totalNodes := int32(len(s.nodes))
	healthyNodes := int32(0)
	var nodes []*pb.NodeInfo

	for _, node := range s.nodes {
		pbNode := &pb.NodeInfo{
			NodeId:    node.ID,
			Address:   node.Address,
			Port:      node.Port,
			Capacity:  node.Capacity,
			LoadScore: node.LoadScore,
			Status:    node.Status,
		}
		nodes = append(nodes, pbNode)

		if s.isNodeHealthy(node) {
			healthyNodes++
		}
	}

	return &pb.ClusterStatusResponse{
		TotalNodes:   totalNodes,
		HealthyNodes: healthyNodes,
		Nodes:        nodes,
		Metrics: &pb.ClusterMetrics{
			TotalActiveRequests: 0, // Would be calculated from all nodes
		},
	}
}
