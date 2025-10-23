package server

// this is the listening server that will listen for inference requests
// upon receiving an inference request, it will join the p2p network to create a
// temporary distributed inference service
// perhaps we should make this an interface for p2p server, inference server, and more

type Server struct {
	Port    int    // incoming request port
	Address string // incoming request address
}

func CreateServer(address string, port int) *Server {
	server := &Server{
		Port:    port,
		Address: address,
	}

	return server
}

func (s *Server) StartServer() error {
	return nil
}

func (s *Server) StopServer() error {
	return nil
}

// returns true if the server is ready to accept requests
// returns error if server encounters an error in the readiness check
func (s *Server) Ready() (bool, error) {
	return true, nil
}

// will add this host to the temporary distributed inference network
func joinInferenceNetwork() error {
	return nil
}
