package service

import (
	"github.com/d0w/EdgeLLM/internal/model"
)

// APIService handles API-related business logic
type APIService struct{}

// NewAPIService creates a new API service
func NewAPIService() *APIService {
	return &APIService{}
}

// GetHelloMessage returns a hello message
func (s *APIService) GetHelloMessage() *model.HelloResponse {
	return &model.HelloResponse{
		Message: "Hello from EdgeLLM!",
		Version: "1.0.0",
	}
}

