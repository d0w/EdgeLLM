package server

type InferenceServer interface {
	Server
	Start() error
	Stop() error
}
