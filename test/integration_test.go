package test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/d0w/EdgeLLM/internal/config"
	"github.com/d0w/EdgeLLM/internal/handler"
	"github.com/d0w/EdgeLLM/internal/model"
	"github.com/d0w/EdgeLLM/internal/service"
	"github.com/d0w/EdgeLLM/pkg/logger"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
)

func setupRouter() *gin.Engine {
	// Set Gin to test mode
	gin.SetMode(gin.TestMode)

	// Initialize dependencies
	cfg := config.Load()
	log := logger.New(cfg.LogLevel)

	healthService := service.NewHealthService()
	apiService := service.NewAPIService()

	healthHandler := handler.NewHealthHandler(healthService, log)
	apiHandler := handler.NewAPIHandler(apiService, log)

	// Setup router
	r := gin.Default()
	r.GET("/health", healthHandler.Health)

	v1 := r.Group("/api/v1")
	{
		v1.GET("/hello", apiHandler.Hello)
	}

	return r
}

func TestHealthEndpoint(t *testing.T) {
	router := setupRouter()

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/health", nil)
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)

	var response model.HealthResponse
	err := json.Unmarshal(w.Body.Bytes(), &response)
	assert.NoError(t, err)
	assert.Equal(t, "healthy", response.Status)
	assert.Equal(t, "edgellm", response.Service)
	assert.Equal(t, "1.0.0", response.Version)
}

func TestHelloEndpoint(t *testing.T) {
	router := setupRouter()

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/api/v1/hello", nil)
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)

	var response model.HelloResponse
	err := json.Unmarshal(w.Body.Bytes(), &response)
	assert.NoError(t, err)
	assert.Equal(t, "Hello from EdgeLLM!", response.Message)
	assert.Equal(t, "1.0.0", response.Version)
}
