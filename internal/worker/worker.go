package worker

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/d0w/EdgeLLM/internal/server"
	"github.com/d0w/EdgeLLM/pkg/logger"
)

type Worker struct {
	listener        *server.Server
	inferenceServer string // needs to be an interface for different types of inference servers/runners later
	p2pServer       string // needs to be a a p2p server listene
	logger          *logger.Logger
}

func CreateWorker(
	inferenceRunner string,
	inferenceModel string,
	listenerPort int,
	inferencePort int,
	address string,
	hfCachePath string,
) (*Worker, error) {
	if inferenceRunner != "vllm" {
		return nil, fmt.Errorf("unknown worker type: %s, only 'vllm' is supported", inferenceRunner)
	}

	// Set up logger
	workerLogger := logger.New("debug")

	// Determine server address and cache path
	// serverAddress := fmt.Sprintf("localhost:%s", listenerPort)
	if hfCachePath == "" {
		if path, ok := os.LookupEnv("HF_HOME"); ok {
			hfCachePath = path
		} else {
			hfCachePath = os.ExpandEnv("$HOME/.cache/huggingface")
		}
	}

	workerLogger.Info("Starting worker node...")
	workerLogger.Info(fmt.Sprintf("Model: %s", inferenceModel))
	workerLogger.Info(fmt.Sprintf("Address: %s", address))
	workerLogger.Info(fmt.Sprintf("HF Cache: %s", hfCachePath))

	// Create VLLM server instance. Parameterize to anything later
	// vllmServer := runner.NewVllmServer(runner.VllmWorker, serverAddress, hfCachePath)

	// Start VLLM server
	vllmArgs := []string{
		"--model", inferenceModel,
		"--port", strconv.Itoa(inferencePort),
		"--host", "0.0.0.0",
		"--tensor-parallel-size", "1",
	}

	// Add quiet flag if specified
	// if viper.GetBool("quiet") {
	// 	vllmArgs = append(vllmArgs, "--disable-log-requests")
	// }

	// create inference listerning server
	listener := server.CreateServer(address, listenerPort)

	worker := &Worker{
		listener:        listener,
		inferenceServer: "some server",
		p2pServer:       "some p2p server",
		logger:          workerLogger,
	}

	return worker, nil
}

func (w *Worker) Start() error {
	w.logger.Info("Starting listening server...")

	// Set up context with cancellation
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	// if err := vllmServer.Start(vllmArgs); err != nil {
	// 	return fmt.Errorf("failed to start VLLM server: %w", err)
	// }

	// Start P2P networking
	// workerLogger.Info("Joining P2P network...")
	// if err := p2p.StartP2P(ctx, serverAddress, workerP2PToken, workerNetworkID); err != nil {
	// 	workerLogger.Error(fmt.Sprintf("Failed to start P2P networking: %v", err))
	// 	// Continue without P2P for now - the worker can still serve local requests
	// } else {
	// 	workerLogger.Info("Successfully joined P2P network")
	// }

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
				if ready, err := w.listener.Ready(); !ready {
					w.logger.Warn(fmt.Sprintf("Listening server health check not ready or encountered error: %v", err))
				}
			}
		}
	}()

	w.logger.Info("Worker node is running. Press Ctrl+C to stop.")

	// Wait for shutdown signal
	select {
	case sig := <-sigChan:
		w.logger.Info(fmt.Sprintf("Received signal %v, shutting down...", sig))
	case <-ctx.Done():
		w.logger.Info("Context cancelled, shutting down...")
	}

	// Graceful shutdown
	w.logger.Info("Stopping VLLM server...")
	// if err := w.inferenceServer.Stop(); err != nil {
	// 	w.logger.Error(fmt.Sprintf("Error stopping VLLM server: %v", err))
	// }

	w.logger.Info("Worker node stopped")

	return nil
}
