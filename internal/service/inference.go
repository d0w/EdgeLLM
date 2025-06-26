package service

import (
	"context"
	"os"

	pb "github.com/d0w/EdgeLLM/backend/go/proto"
	"google.golang.org/grpc"
)

type InferenceService struct {
	pythonClient pb.BackendClient
}

func NewInferenceService() (*InferenceService, error) {
	address := os.Getenv("PYTHON_GRPC_ADDRESS")
	if address == "" {
		address = "localhost:50051"
	}
	conn, err := grpc.Dial(address, grpc.WithInsecure())
	if err != nil {
		return nil, err
	}

	client := pb.NewBackendClient(conn)
	return &InferenceService{
		pythonClient: client,
	}, nil
}

func (s *InferenceService) GenerateText(ctx context.Context, prompt string) (*pb.GenerateTextResponse, error) {
	req := &pb.GenerateTextRequest{
		Prompt: prompt,
		// ...
	}

	return s.pythonClient.GenerateText(ctx, req)
}
