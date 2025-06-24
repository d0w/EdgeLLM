package handler

import (
	"net/http"

	"github.com/d0w/EdgeLLM/internal/service"
	"github.com/d0w/EdgeLLM/pkg/logger"
	"github.com/gin-gonic/gin"
)

// HealthHandler handles health-related requests
type HealthHandler struct {
	service *service.HealthService
	logger  *logger.Logger
}

// NewHealthHandler creates a new health handler
func NewHealthHandler(service *service.HealthService, logger *logger.Logger) *HealthHandler {
	return &HealthHandler{
		service: service,
		logger:  logger,
	}
}

// Health returns the health status of the application
func (h *HealthHandler) Health(c *gin.Context) {
	status := h.service.GetHealth()

	h.logger.Info("Health check requested")

	c.JSON(http.StatusOK, status)
}

