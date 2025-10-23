package server

import "github.com/d0w/EdgeLLM/pkg/logger"

type Server interface {
	Start() error
	Stop() error
	Ready() error
	Health() int
}

type BaseServer struct {
	Port    int
	Address string
	logger  *logger.Logger
}
