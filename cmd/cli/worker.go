package cli

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/d0w/EdgeLLM/internal/p2p"
	"github.com/d0w/EdgeLLM/internal/runner"
	"github.com/d0w/EdgeLLM/pkg/logger"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var (
	workerModel       string
	workerPort        string
	workerP2PToken    string
	workerNetworkID   string
	workerHFCachePath string
)

var workerCommand = &cobra.Command{
	Use:   "worker [vllm]",
	Short: "Start a worker node for distributed inference",
	Args:  cobra.ExactArgs(1),
	RunE:  startWorker,
}

func init() {
	workerCommand.Flags().BoolVar(&vllmQuiet, "quiet", false, "Suppress output from the vLLM server")
	workerCommand.Flags().StringVar(&workerModel, "model", "", "Model to load (required)")
	workerCommand.Flags().StringVar(&workerPort, "port", "50051", "gRPC port to listen on")
	workerCommand.Flags().StringVar(&workerP2PToken, "token", "", "P2P network token")
	workerCommand.Flags().StringVar(&workerNetworkID, "network", "default", "P2P network ID")
	workerCommand.Flags().StringVar(&workerHFCachePath, "hf-cache", "", "HuggingFace cache path")

	// Mark model as required
	workerCommand.MarkFlagRequired("model")

	rootCmd.AddCommand(workerCommand)
}

func startWorker(cmd *cobra.Command, args []string) error {
	if args[0] != "vllm" {
		return fmt.Errorf("unknown worker type: %s, only 'vllm' is supported", args[0])
	}

	// Set up logger
	workerLogger := logger.New(log.LevelInfo)
	
	// Set up context with cancellation
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	// Determine server address and cache path
	serverAddress := fmt.Sprintf("localhost:%s", workerPort)
	hfCachePath := workerHFCachePath
	if hfCachePath == "" {
		if path, ok := os.LookupEnv("HF_HOME"); ok {
			hfCachePath = path
		} else {
			hfCachePath = os.ExpandEnv("$HOME/.cache/huggingface")
		}
	}

	workerLogger.Info("Starting VLLM worker node...")
	workerLogger.Info(fmt.Sprintf("Model: %s", workerModel))
	workerLogger.Info(fmt.Sprintf("Address: %s", serverAddress))
	workerLogger.Info(fmt.Sprintf("HF Cache: %s", hfCachePath))

	// Create VLLM server instance
	vllmServer := runner.NewVllmServer(runner.VllmWorker, serverAddress, hfCachePath)

	// Start VLLM server
	vllmArgs := []string{
		"--model", workerModel,
		"--port", workerPort,
		"--host", "0.0.0.0",
		"--tensor-parallel-size", "1",
	}

	// Add quiet flag if specified
	if viper.GetBool("quiet") {
		vllmArgs = append(vllmArgs, "--disable-log-requests")
	}

	workerLogger.Info("Starting VLLM server...")
	if err := vllmServer.Start(vllmArgs); err != nil {
		return fmt.Errorf("failed to start VLLM server: %w", err)
	}

	// Start P2P networking
	workerLogger.Info("Joining P2P network...")
	if err := p2p.StartP2P(ctx, serverAddress, workerP2PToken, workerNetworkID); err != nil {
		workerLogger.Error(fmt.Sprintf("Failed to start P2P networking: %v", err))
		// Continue without P2P for now - the worker can still serve local requests
	} else {
		workerLogger.Info("Successfully joined P2P network")
	}

	// Start background health monitoring
	go func() {
		ticker := time.NewTicker(30 * time.Second)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				// Check if VLLM server is still healthy
				if !vllmServer.isReady() {
					workerLogger.Warn("VLLM server health check failed")
				}
			}
		}
	}()

	workerLogger.Info("Worker node is running. Press Ctrl+C to stop.")

	// Wait for shutdown signal
	select {
	case sig := <-sigChan:
		workerLogger.Info(fmt.Sprintf("Received signal %v, shutting down...", sig))
	case <-ctx.Done():
		workerLogger.Info("Context cancelled, shutting down...")
	}

	// Graceful shutdown
	workerLogger.Info("Stopping VLLM server...")
	if err := vllmServer.Stop(); err != nil {
		workerLogger.Error(fmt.Sprintf("Error stopping VLLM server: %v", err))
	}

	workerLogger.Info("Worker node stopped")
	return nil
}


