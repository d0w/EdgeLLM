package service

// import (
// 	"context"
// 	"fmt"
// 	"math/rand"
// 	"time"
//
// 	pb "github.com/d0w/EdgeLLM/backend/go/proto"
// 	"github.com/d0w/EdgeLLM/pkg/logger"
// 	"google.golang.org/grpc"
// 	"google.golang.org/grpc/credentials/insecure"
// 	"google.golang.org/grpc/keepalive"
// )
//
// type LoadBalancerService struct {
// 	coordinator *NodeCoordinatorService
// 	clients     map[string]pb.BackendClient
// 	connections map[string]*grpc.ClientConn
// 	logger      *logger.Logger
// }
//
// func NewLoadBalancerService(coordinator *NodeCoordinatorService, logger *logger.Logger) *LoadBalancerService {
// 	return &LoadBalancerService{
// 		coordinator: coordinator,
// 		clients:     make(map[string]pb.BackendClient),
// 		connections: make(map[string]*grpc.ClientConn),
// 		logger:      logger,
// 	}
// }
//
// func (s *LoadBalancerService) DistributeInference(ctx context.Context, req *pb.DistributedInferenceRequest) (*pb.DistributedInferenceResponse, error) {
// 	startTime := time.Now()
//
// 	// Get available nodes
// 	nodesReq := &pb.NodesRequest{
// 		Model:    req.Request.Model,
// 		MinNodes: 1,
// 	}
//
// 	nodesResp, err := s.coordinator.GetAvailableNodes(ctx, nodesReq)
// 	if err != nil {
// 		return nil, fmt.Errorf("failed to get available nodes: %w", err)
// 	}
//
// 	if len(nodesResp.Nodes) == 0 {
// 		return nil, fmt.Errorf("no available nodes for model %s", req.Request.Model)
// 	}
//
// 	// Select node based on distribution strategy
// 	var selectedNode *pb.NodeInfo
// 	switch req.DistributionStrategy {
// 	case "round_robin":
// 		selectedNode = s.selectRoundRobin(nodesResp.Nodes)
// 	case "latency_based":
// 		selectedNode = s.selectLowLatency(nodesResp.Nodes)
// 	case "load_based":
// 		fallthrough
// 	default:
// 		selectedNode = s.selectLowLoad(nodesResp.Nodes)
// 	}
//
// 	// Get or create client for the selected node
// 	client, err := s.getNodeClient(selectedNode)
// 	if err != nil {
// 		// Try fallback if enabled
// 		if req.EnableFallback && len(nodesResp.Nodes) > 1 {
// 			s.logger.Warn(fmt.Sprintf("Primary node %s failed, trying fallback", selectedNode.NodeId))
// 			for _, node := range nodesResp.Nodes {
// 				if node.NodeId != selectedNode.NodeId {
// 					if fallbackClient, fallbackErr := s.getNodeClient(node); fallbackErr == nil {
// 						client = fallbackClient
// 						selectedNode = node
// 						break
// 					}
// 				}
// 			}
// 		}
//
// 		if client == nil {
// 			return nil, fmt.Errorf("failed to connect to any available node: %w", err)
// 		}
// 	}
//
// 	// Execute inference
// 	queueStartTime := time.Now()
// 	response, err := client.GenerateText(ctx, req.Request)
// 	if err != nil {
// 		return nil, fmt.Errorf("inference failed on node %s: %w", selectedNode.NodeId, err)
// 	}
//
// 	processingTime := time.Since(startTime).Milliseconds()
// 	queueTime := queueStartTime.Sub(startTime).Milliseconds()
//
// 	return &pb.DistributedInferenceResponse{
// 		Response:         response,
// 		ServingNode:      selectedNode.NodeId,
// 		ProcessingTimeMs: processingTime,
// 		QueueTimeMs:      queueTime,
// 	}, nil
// }
//
// func (s *LoadBalancerService) GetClusterStatus(ctx context.Context, req *pb.ClusterStatusRequest) (*pb.ClusterStatusResponse, error) {
// 	return s.coordinator.GetClusterStatus(), nil
// }
//
// func (s *LoadBalancerService) selectRoundRobin(nodes []*pb.NodeInfo) *pb.NodeInfo {
// 	// Simple random selection for now - in production, maintain state
// 	return nodes[rand.Intn(len(nodes))]
// }
//
// func (s *LoadBalancerService) selectLowLoad(nodes []*pb.NodeInfo) *pb.NodeInfo {
// 	var bestNode *pb.NodeInfo
// 	bestScore := float32(1000.0)
//
// 	for _, node := range nodes {
// 		if node.LoadScore < bestScore {
// 			bestScore = node.LoadScore
// 			bestNode = node
// 		}
// 	}
//
// 	return bestNode
// }
//
// func (s *LoadBalancerService) selectLowLatency(nodes []*pb.NodeInfo) *pb.NodeInfo {
// 	// For now, use load-based selection
// 	// In production, maintain latency metrics per node
// 	return s.selectLowLoad(nodes)
// }
//
// func (s *LoadBalancerService) getNodeClient(node *pb.NodeInfo) (pb.BackendClient, error) {
// 	nodeAddr := fmt.Sprintf("%s:%d", node.Address, node.Port)
//
// 	if client, exists := s.clients[node.NodeId]; exists {
// 		return client, nil
// 	}
//
// 	// Create new connection
// 	conn, err := grpc.NewClient(
// 		nodeAddr,
// 		grpc.WithTransportCredentials(insecure.NewCredentials()),
// 		grpc.WithKeepaliveParams(keepalive.ClientParameters{
// 			Time:                10 * time.Second,
// 			Timeout:             3 * time.Second,
// 			PermitWithoutStream: true,
// 		}),
// 	)
// 	if err != nil {
// 		return nil, fmt.Errorf("failed to connect to node %s: %w", nodeAddr, err)
// 	}
//
// 	client := pb.NewBackendClient(conn)
//
// 	s.clients[node.NodeId] = client
// 	s.connections[node.NodeId] = conn
//
// 	return client, nil
// }
//
// func (s *LoadBalancerService) Close() error {
// 	for nodeID, conn := range s.connections {
// 		if err := conn.Close(); err != nil {
// 			s.logger.Error(fmt.Sprintf("Failed to close connection to node %s: %v", nodeID, err))
// 		}
// 	}
// 	return nil
// }
