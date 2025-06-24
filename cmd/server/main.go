package main

import (
	"github.com/d0w/EdgeLLM/internal/config"
	"github.com/d0w/EdgeLLM/internal/handler"
	"github.com/d0w/EdgeLLM/internal/service"
	"github.com/d0w/EdgeLLM/pkg/logger"
	"github.com/gin-gonic/gin"
)

func main() {
	// Load configuration
	cfg := config.Load()

	// Initialize logger
	log := logger.New(cfg.LogLevel)

	// Initialize services
	healthService := service.NewHealthService()
	apiService := service.NewAPIService()

	// Initialize handlers
	healthHandler := handler.NewHealthHandler(healthService, log)
	apiHandler := handler.NewAPIHandler(apiService, log)

	// Set up Gin router
	r := gin.Default()

	// Register routes
	registerRoutes(r, healthHandler, apiHandler)

	log.Info("Starting server on port " + cfg.Port)
	if err := r.Run(":" + cfg.Port); err != nil {
		log.Fatal("Failed to start server: " + err.Error())
	}
}

func registerRoutes(r *gin.Engine, healthHandler *handler.HealthHandler, apiHandler *handler.APIHandler) {
	// Health check endpoint
	r.GET("/health", healthHandler.Health)

	// API v1 routes
	v1 := r.Group("/api/v1")
	{
		v1.GET("/hello", apiHandler.Hello)
	}
}

