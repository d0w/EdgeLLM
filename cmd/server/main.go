// package main
//
// import (
// 	"context"
// 	"fmt"
// 	"net"
// 	"os"
// 	"os/signal"
// 	"syscall"
// 	"time"
//
// 	pb "github.com/d0w/EdgeLLM/backend/go/proto"
// 	"github.com/d0w/EdgeLLM/internal/config"
// 	"github.com/d0w/EdgeLLM/internal/handler"
// 	"github.com/d0w/EdgeLLM/internal/service"
// 	"github.com/d0w/EdgeLLM/pkg/logger"
//
// 	"github.com/gin-gonic/gin"
// 	"google.golang.org/grpc"
// 	"google.golang.org/grpc/keepalive"
// )
//
// func main() {
// 	// Load configuration
// 	cfg := config.Load()
//
// 	// Initialize logger
// 	log := logger.New(cfg.LogLevel)
//
// 	// Initialize services
// 	healthService := service.NewHealthService()
// 	apiService := service.NewAPIService()
//
// 	// Initialize inference service
// 	inferenceService, err := service.NewInferenceService()
// 	if err != nil {
// 		log.Fatal("Failed to initialize inference service: " + err.Error())
// 	}
// 	defer inferenceService.Close()
//
// 	// Initialize distributed services
// 	coordinatorService := service.NewNodeCoordinatorService(log)
// 	coordinatorService.StartCleanupWorker()
//
// 	loadBalancerService := service.NewLoadBalancerService(coordinatorService, log)
// 	defer loadBalancerService.Close()
//
// 	// Start gRPC server for coordination
// 	go func() {
// 		if err := startGRPCServer(coordinatorService, loadBalancerService, log, cfg); err != nil {
// 			log.Fatal("Failed to start gRPC server: " + err.Error())
// 		}
// 	}()
//
// 	// Initialize handlers
// 	healthHandler := handler.NewHealthHandler(healthService, log)
// 	apiHandler := handler.NewAPIHandler(apiService, log)
// 	inferenceHandler := handler.NewInferenceHandler(inferenceService, loadBalancerService, log)
//
// 	// Set up Gin router
// 	r := gin.Default()
//
// 	// Register routes
// 	registerRoutes(r, healthHandler, apiHandler, inferenceHandler)
//
// 	// Start HTTP server
// 	log.Info("Starting HTTP server on port " + cfg.Port)
//
// 	// Graceful shutdown
// 	ctx, cancel := context.WithCancel(context.Background())
// 	defer cancel()
//
// 	go func() {
// 		if err := r.Run(":" + cfg.Port); err != nil {
// 			log.Fatal("Failed to start HTTP server: " + err.Error())
// 		}
// 	}()
//
// 	// Wait for interrupt signal
// 	c := make(chan os.Signal, 1)
// 	signal.Notify(c, syscall.SIGINT, syscall.SIGTERM)
// 	<-c
//
// 	log.Info("Shutting down servers...")
// 	cancel()
// }
//
// func startGRPCServer(coordinator *service.NodeCoordinatorService, loadBalancer *service.LoadBalancerService, log *logger.Logger, cfg *config.Config) error {
// 	coordinatorPort := "50052" // Default coordinator port
//
// 	lis, err := net.Listen("tcp", ":"+coordinatorPort)
// 	if err != nil {
// 		return fmt.Errorf("failed to listen on port %s: %w", coordinatorPort, err)
// 	}
//
// 	s := grpc.NewServer(
// 		grpc.KeepaliveParams(keepalive.ServerParameters{
// 			MaxConnectionIdle: 15 * time.Second,
// 			MaxConnectionAge:  30 * time.Second,
// 			Time:              5 * time.Second,
// 			Timeout:           1 * time.Second,
// 		}),
// 		grpc.KeepaliveEnforcementPolicy(keepalive.EnforcementPolicy{
// 			MinTime:             5 * time.Second,
// 			PermitWithoutStream: true,
// 		}),
// 	)
//
// 	pb.RegisterNodeCoordinatorServer(s, coordinator)
// 	pb.RegisterLoadBalancerServer(s, loadBalancer)
//
// 	log.Info("Starting gRPC coordination server on port " + coordinatorPort)
// 	return s.Serve(lis)
// }
//
// func registerRoutes(r *gin.Engine, healthHandler *handler.HealthHandler, apiHandler *handler.APIHandler, inferenceHandler *handler.InferenceHandler) {
// 	// Health check endpoint
// 	r.GET("/health", healthHandler.Health)
//
// 	// API v1 routes
// 	v1 := r.Group("/api/v1")
// 	{
// 		v1.GET("/hello", apiHandler.Hello)
//
// 		// Inference endpoints
// 		v1.POST("/inference/generate", inferenceHandler.Generate)
// 		v1.POST("/inference/stream", inferenceHandler.GenerateStream)
// 		v1.POST("/models/load", inferenceHandler.LoadModel)
//
// 		// Cluster management
// 		v1.GET("/cluster/status", inferenceHandler.ClusterStatus)
// 		v1.GET("/cluster/nodes", inferenceHandler.GetNodes)
// 	}
// }
