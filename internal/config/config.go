package config

import (
	"os"
)

// Config holds all configuration for the application
type Config struct {
	Port     string
	LogLevel string
	GinMode  string
}

// Load reads configuration from environment variables with defaults
func Load() *Config {
	return &Config{
		Port:     getEnv("PORT", "8080"),
		LogLevel: getEnv("LOG_LEVEL", "info"),
		GinMode:  getEnv("GIN_MODE", "debug"),
	}
}

// getEnv gets an environment variable with a fallback value
func getEnv(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
} 