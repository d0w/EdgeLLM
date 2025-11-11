package server

import (
	"context"

	"github.com/d0w/EdgeLLM/pkg/logger"
)

type Server interface {
	Start(ctx context.Context) error
	Stop() error
	Ready() error
	Health() int
}

type BaseServer struct {
	Port    int
	Address string
	logger  *logger.Logger
}
