package service

import (
	"github.com/d0w/EdgeLLM/internal/model"
)

// HealthService handles health-related business logic
type HealthService struct{}

// NewHealthService creates a new health service
func NewHealthService() *HealthService {
	return &HealthService{}
}

// GetHealth returns the current health status
func (s *HealthService) GetHealth() *model.HealthResponse {
	return &model.HealthResponse{
		Status:  "healthy",
		Service: "edgellm",
		Version: "1.0.0",
	}
}

