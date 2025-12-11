package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"sync"
	"time"

	"github.com/d0w/EdgeLLM/pkg/logger"
)

// this is the listening server that will listen for inference requests
// upon receiving an inference request, it will join the p2p network (or already be part of it)
// to create a temporary distributed inference service
// perhaps we should make this an interface for p2p server, inference server, and more

type ListenerServer struct {
	BaseServer
	InferenceServer InferenceServer

	httpServer *http.Server
	mu         sync.RWMutex
}

func NewListenerServer(address string, port int, inferenceServerConfig InferenceServerConfig) *ListenerServer {
	server := &ListenerServer{
		BaseServer: BaseServer{
			Port:    port,
			Address: address,
			logger:  logger.New("info"),
		},
		// TODO: Parameterize inference server config
		// InferenceServer: newInferenceServer(InferenceServerConfig{
		// 	Type:            ServerTypeWorker,
		// 	ContainerImage:  "vllm/vllm-openai:latest",
		// 	ContainerName:   "edgellm-vllm",
		// 	HFCachePath:     "./.cache/huggingface",
		// 	HeadNodeAddress: "10.0.0.83",
		// 	Model:           "Qwen/Qwen3-0.6B",
		// 	Args: []string{
		// 		"--tensor-parallel-size=1",
		// 		"--pipeline-parallel-size=1",
		// 		"--port=8010",
		// 		"--gpu-memory-utilization=0.3",
		// 		"--max-model-len=512",
		// 	},
		// }),
	}

	return server
}

func (s *ListenerServer) initializeInferenceServer(serverType InferenceServerType, model string) error {
	// TODO :Add worker set limitations
	cfg := InferenceServerConfig{
		Type:            serverType,
		ContainerImage:  "vllm/vllm-openai:latest",
		ContainerName:   "edgellm-vllm",
		HFCachePath:     "./.cache/huggingface",
		HeadNodeAddress: "10.0.0.83",
		Model:           model,
		Args: []string{
			"--tensor-parallel-size=1",
			"--pipeline-parallel-size=1",
			"--port=8010",
			"--gpu-memory-utilization=0.3",
			"--max-model-len=512",
		},
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.InferenceServer = newInferenceServer(cfg)
	return nil
}

func (s *ListenerServer) removeInferenceServer() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.InferenceServer = nil
	return nil
}

func (s *ListenerServer) Start(ctx context.Context) error {
	serverAddress := fmt.Sprintf("%s:%d", s.Address, s.Port)

	// TODO: Probably use gRPC or similar to work with p2p frameworkinstead
	mux := http.NewServeMux()
	mux.HandleFunc("/health", s.healthHandler)
	mux.HandleFunc("/ready", s.readyHandler)
	mux.HandleFunc("/inference/start", s.startInferenceHandler)
	mux.HandleFunc("/inference/stop", s.stopInferenceHandler)
	mux.HandleFunc("/inference/status", s.inferenceStatusHandler)
	// mux.HandleFunc("/inference/initialize", s.initializeInferenceServerHandler)
	// mux.HandleFunc("/inference/remove", s.removeInferenceServerHandler)

	listenerCtx, listenerCancel := context.WithCancel(ctx)

	httpServer := &http.Server{
		Addr:    serverAddress,
		Handler: mux,
		BaseContext: func(listener net.Listener) context.Context {
			return listenerCtx
		},
	}
	s.httpServer = httpServer

	s.logger.Info("Starting listener server on %s", serverAddress)

	// join inference network

	go func() {
		if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			listenerCancel()
			s.logger.Error("failed to start listener server: %v", err)
		}
	}()

	select {
	case <-ctx.Done():
		s.Stop()
	case <-listenerCtx.Done():
		// do something else
		s.Stop()
	}

	return nil
}

func (s *ListenerServer) Stop() error {
	s.logger.Info("Stopping listener server...")

	if s.InferenceServer != nil {
		if vllm, ok := s.InferenceServer.(*VllmServer); ok && vllm.IsRunning() {
			s.logger.Info("Stopping inference server...")
			if err := s.InferenceServer.Stop(); err != nil {
				s.logger.Error("Error stopping inference server: %v", err)
			}
		}
	}

	if s.httpServer != nil {
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()

		if err := s.httpServer.Shutdown(shutdownCtx); err != nil {
			s.logger.Error("Error shutting down HTTP server: %v", err)
			return err
		}
	}
	s.logger.Info("Inference server stopped.")

	return nil
}

// returns true if the server is ready to accept requests
// returns error if server encounters an error in the readiness check
func (s *ListenerServer) Ready() (bool, error) {
	return true, nil
}

func (s *ListenerServer) Health() int {
	return 0
}

// TODO: Authenticate requester
func (s *ListenerServer) initializeInferenceServerHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	type InitRequest struct {
		ServerType InferenceServerType `json:"serverType"`
		Model      string              `json:"model"`
	}

	var req InitRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	if s.isInferenceServerRunning() {
		w.WriteHeader(http.StatusConflict)
		json.NewEncoder(w).Encode(map[string]any{
			"error": "Inference server already running",
		})
		return
	}

	if err := s.initializeInferenceServer(req.ServerType, req.Model); err != nil {
		s.logger.Error("Failed to initialize inference server: %v", err)
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]any{
			"error": fmt.Sprintf("Failed to initialize inference server: %v", err),
		})
		return
	}

	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]any{
		"message": "Inference server initialized successfully",
	})
}

// TODO: Authenticate requester
func (s *ListenerServer) removeInferenceServerHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	if !s.isInferenceServerRunning() {
		w.WriteHeader(http.StatusConflict)
		json.NewEncoder(w).Encode(map[string]any{
			"error": "Inference server is not running",
		})
		return
	}

	s.removeInferenceServer()

	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]any{
		"message": "Inference server removed successfully",
	})
}

func (s *ListenerServer) healthHandler(w http.ResponseWriter, r *http.Request) {
	health := s.Health()
	if health != 0 {
		w.WriteHeader(http.StatusServiceUnavailable)
		json.NewEncoder(w).Encode(map[string]any{
			"status":            "unhealthy",
			"inference_running": false,
		})
		return
	}

	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]any{
		"status":            "healthy",
		"inference_running": s.isInferenceServerRunning(),
	})
}

func (s *ListenerServer) readyHandler(w http.ResponseWriter, r *http.Request) {
	ready, err := s.Ready()
	if err != nil || !ready {
		w.WriteHeader(http.StatusServiceUnavailable)
		json.NewEncoder(w).Encode(map[string]any{
			"ready": false,
			"error": fmt.Sprintf("%v", err),
		})
		return
	}

	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]any{"ready": true})
}

// TODO: This actually needs to wait for the inference server to be ready
// TODO: Authenticate requester
func (s *ListenerServer) startInferenceHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	type StartRequest struct {
		ServerType InferenceServerType `json:"serverType"`
		Model      string              `json:"model"`
	}

	var req StartRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	// Check if already running
	if s.isInferenceServerRunning() {
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]any{
			"message": "Inference server already running",
			"status":  "running",
		})
		return
	}

	s.logger.Info("Received request to start inference server")
	s.initializeInferenceServer(req.ServerType, req.Model)

	// startCtx, cancel := context.WithTimeout(r.Context(), 3*time.Second)
	inferenceCtx, inferenceCancel := context.WithCancel(s.httpServer.BaseContext(nil))
	// defer cancel()
	defer inferenceCancel()

	// errChan := make(chan error, 1)
	// go func() {
	// TODO: Actually wait for ready and make this synchronous
	s.InferenceServer.Start(inferenceCtx)
	// }()

	// select {
	// case err := <-errChan:
	// 	if err != nil {
	// 		s.logger.Error("Failed to start inference server: %v", err)
	// 		w.WriteHeader(http.StatusInternalServerError)
	// 		json.NewEncoder(w).Encode(map[string]interface{}{
	// 			"error": fmt.Sprintf("Failed to start inference server: %v", err),
	// 		})
	// 		return
	// 	}
	// case <-startCtx.Done():
	// 	s.logger.Error("Inference server startup timed out")
	// 	// Attempt cleanup
	// 	inferenceCancel()
	// 	go func() {
	// 		s.InferenceServer.Stop()
	// 	}()
	// 	w.WriteHeader(http.StatusGatewayTimeout)
	// 	json.NewEncoder(w).Encode(map[string]any{
	// 		"error": "Inference server startup timed out after 5 minutes",
	// 	})
	// 	return
	//
	// }

	s.logger.Info("Inference server started successfully")

	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]any{
		"message": "Inference server started successfully",
		"status":  "running",
	})
}

func (s *ListenerServer) stopInferenceHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.isInferenceServerRunning() {
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]any{
			"message": "Inference server not running",
			"status":  "stopped",
		})
		return
	}

	s.logger.Info("Received request to stop inference server")

	if err := s.InferenceServer.Stop(); err != nil {
		s.logger.Error("Failed to stop inference server: %v", err)
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]any{
			"error": fmt.Sprintf("Failed to stop inference server: %v", err),
		})
		return
	}

	if err := s.removeInferenceServer(); err != nil {
		s.logger.Error("Failed to remove inference server: %v", err)
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]any{
			"error": fmt.Sprintf("Failed to remove inference server: %v", err),
		})
		return
	}

	s.logger.Info("Inference server stopped successfully")

	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]any{
		"message": "Inference server stopped successfully",
		"status":  "stopped",
	})
}

func (s *ListenerServer) inferenceStatusHandler(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	running := s.isInferenceServerRunning()
	health := s.InferenceServer.Health()

	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]any{
		"running": running,
		"healthy": health == 0,
	})
}

func (s *ListenerServer) isInferenceServerRunning() bool {
	if vllm, ok := s.InferenceServer.(*VllmServer); ok {
		return vllm.IsRunning()
	}
	return false
}

// joinInferenceNetwork will add this host to the temporary distributed inference network
func (s *ListenerServer) joinInferenceNetwork() error {
	// TODO: Implement P2P network joining logic
	s.logger.Info("Joining inference network...")
	return nil
}
