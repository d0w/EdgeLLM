package worker

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"sync"
	"time"

	"github.com/d0w/EdgeLLM/internal/server"
	"github.com/d0w/EdgeLLM/pkg/logger"
)

type Worker struct {
	listener        *server.ListenerServer
	inferenceServer server.InferenceServer
	p2pServer       string
	logger          *logger.Logger
	wg              sync.WaitGroup
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

	workerLogger := logger.New("debug")

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

	listener := server.NewListenerServer(address, listenerPort, server.InferenceServerConfig{
		Runner:         server.InferenceRunnerVllm,
		Type:           server.ServerTypeHead,
		ContainerImage: "vllm/vllm-openai:latest",
		ContainerName:  fmt.Sprintf("edgellm-vllm-%d", listenerPort),
		RayStartCmd:    "ray start --head --port=6379",
		HFCachePath:    hfCachePath,
		Args:           []string{},
	})

	worker := &Worker{
		listener:        listener,
		inferenceServer: listener.InferenceServer,
		p2pServer:       "some p2p server",
		logger:          workerLogger,
	}

	return worker, nil
}

func (w *Worker) Start() error {
	w.logger.Info("Starting listening server...")

	ctx, cancel := context.WithCancel(context.Background())

	// Set up signal handling for graceful shutdown
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt)

	if err := w.listener.Start(ctx); err != nil {
		return fmt.Errorf("failed to start listener server: %w", err)
	}

	w.wg.Add(1)
	go w.healthMonitor(ctx)

	w.logger.Info("Worker node is running. Press Ctrl+C to stop.")
	w.logger.Info("Inference server will start on-demand when requests are received.")

	sig := <-sigCh
	w.logger.Info(fmt.Sprintf("Received signal %v, initiating graceful shutdown...", sig))

	if cancel != nil {
		cancel()
	}
	if err := w.shutdown(); err != nil {
		return err
	}

	return nil
}

func (w *Worker) healthMonitor(ctx context.Context) {
	defer w.wg.Done()

	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	w.logger.Info("Health monitoring started")

	for {
		select {
		case <-ctx.Done():
			w.logger.Info("Health monitoring stopped")
			return
		case <-ticker.C:
			if ready, err := w.listener.Ready(); !ready {
				w.logger.Warn(fmt.Sprintf("Listener health check failed: %v", err))
			}

			if vllm, ok := w.inferenceServer.(*server.VllmServer); ok && vllm.IsRunning() {
				if health := w.inferenceServer.Health(); health != 0 {
					w.logger.Warn("Inference server is unhealthy")
				} else {
					w.logger.Debug("Inference server is healthy")
				}
			}
		}
	}
}

func (w *Worker) shutdown() error {
	w.logger.Info("Initiating graceful shutdown...")

	// Stop listener server (which will also stop inference server)
	if err := w.listener.Stop(); err != nil {
		w.logger.Error(fmt.Sprintf("Error stopping listener: %v", err))
	}

	// Wait for all goroutines to finish with timeout
	done := make(chan struct{})
	go func() {
		w.wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		w.logger.Info("All goroutines stopped successfully")
	case <-time.After(10 * time.Second):
		w.logger.Warn("Timeout waiting for goroutines to stop")
	}

	w.logger.Info("Worker node stopped successfully")
	return nil
}

func (w *Worker) Stop() error {
	return w.shutdown()
}
