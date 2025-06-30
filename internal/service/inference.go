package service

import (
	"context"
	"os"

	pb "github.com/d0w/EdgeLLM/backend/go/proto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type InferenceService struct {
	pythonClient pb.BackendClient
	connection   *grpc.ClientConn
}

func NewInferenceService() (*InferenceService, error) {
	address := os.Getenv("PYTHON_GRPC_ADDRESS")
	if address == "" {
		address = "localhost:50051"
	}
	conn, err := grpc.NewClient(address, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, err
	}

	client := pb.NewBackendClient(conn)
	return &InferenceService{
		pythonClient: client,
		connection:   conn,
	}, nil
}

func (s *InferenceService) Close() error {
	return s.connection.Close()
}

func (s *InferenceService) GenerateText(ctx context.Context, prompt string) (*pb.GenerateTextResponse, error) {
	req := &pb.GenerateTextRequest{
		Prompt: prompt,
		// ...
	}

	return s.pythonClient.GenerateText(ctx, req)
}

func (s *InferenceService) LoadModel(ctx context.Context, modelName string, contextSize int32) (*pb.LoadModelResponse, error) {
	req := &pb.ModelOptions{
		Model:       modelName,
		ContextSize: contextSize,
	}

	return s.pythonClient.LoadModel(ctx, req)
}
