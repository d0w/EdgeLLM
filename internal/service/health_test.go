package service

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestHealthService_GetHealth(t *testing.T) {
	service := NewHealthService()
	
	health := service.GetHealth()
	
	assert.NotNil(t, health)
	assert.Equal(t, "healthy", health.Status)
	assert.Equal(t, "edgellm", health.Service)
	assert.Equal(t, "1.0.0", health.Version)
} 