package server

import (
	"fmt"
	"net/http"

	"github.com/d0w/EdgeLLM/pkg/logger"
)

// this is the listening server that will listen for inference requests
// upon receiving an inference request, it will join the p2p network (or already be part of it)
// to create a temporary distributed inference service
// perhaps we should make this an interface for p2p server, inference server, and more

type ListenerServer struct {
	BaseServer
	InferenceServer InferenceServer
}

func NewListenerServer(address string, port int, inferenceServerConfig InferenceServerConfig) *ListenerServer {
	server := &ListenerServer{
		BaseServer: BaseServer{
			Port:    port,
			Address: address,
			logger:  logger.New("info"),
		},
		// TODO: Parameterize inference server config
		InferenceServer: newInferenceServer(InferenceServerConfig{
			Type:           ServerTypeHead,
			ContainerImage: "vllm/vllm-openai:latest",
			ContainerName:  "edgellm-vllm",
			RayStartCmd:    "ray start --head",
			HFCachePath:    "/data/huggingface",
			Args:           []string{},
		}),
	}

	return server
}

func (s *ListenerServer) Start() error {
	serverAddress := fmt.Sprintf("%s:%d", s.Address, s.Port)

	mux := http.NewServeMux()
	mux.HandleFunc("/health", s.healthHandler)
	mux.HandleFunc("/ready", s.readyHandler)

	httpServer := &http.Server{
		Addr:    serverAddress,
		Handler: mux,
	}

	s.logger.Info("Starting listener server on %s", serverAddress)

	// join inference network

	if err := httpServer.ListenAndServe(); err != nil {
		return fmt.Errorf("failed to start listener server: %v", err)
	}

	return nil
}

func (s *ListenerServer) Stop() error {
	return nil
}

func (s *ListenerServer) healthHandler(w http.ResponseWriter, r *http.Request) {
	health := s.Health()
	if health != 0 {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte("NOT OK"))
		return
	}

	w.WriteHeader(http.StatusOK)
	w.Write([]byte("OK"))
}

func (s *ListenerServer) readyHandler(w http.ResponseWriter, r *http.Request) {
	_, err := s.Ready()
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte("NOT READY"))
	}

	w.WriteHeader(http.StatusOK)
	w.Write([]byte("READY"))
}

// returns true if the server is ready to accept requests
// returns error if server encounters an error in the readiness check
func (s *ListenerServer) Ready() (bool, error) {
	return true, nil
}

func (s *ListenerServer) Health() int {
	return 0
}

// will add this host to the temporary distributed inference network
func joinInferenceNetwork() error {
	return nil
}
