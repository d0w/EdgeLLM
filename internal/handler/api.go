package handler

import (
	"net/http"

	"github.com/d0w/EdgeLLM/internal/service"
	"github.com/d0w/EdgeLLM/pkg/logger"
	"github.com/gin-gonic/gin"
)

// APIHandler handles API requests
type APIHandler struct {
	service *service.APIService
	logger  *logger.Logger
}

// NewAPIHandler creates a new API handler
func NewAPIHandler(service *service.APIService, logger *logger.Logger) *APIHandler {
	return &APIHandler{
		service: service,
		logger:  logger,
	}
}

// Hello returns a hello message
func (h *APIHandler) Hello(c *gin.Context) {
	response := h.service.GetHelloMessage()

	h.logger.Info("Hello endpoint requested")

	c.JSON(http.StatusOK, response)
}

