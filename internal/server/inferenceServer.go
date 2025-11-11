package server

type InferenceServer interface {
	Server
	// add inference server specific methods here
	GetModelInfo(modelName string) (string, error)
}

type InferenceServerConfig struct {
	Runner          InferenceRunner
	Type            InferenceServerType
	ContainerImage  string
	ContainerName   string
	RayStartCmd     string
	HFCachePath     string
	Args            []string
	HeadNodeAddress string
	// Add other fields as needed
}

type (
	InferenceServerType int
	InferenceRunner     int
)

const (
	ServerTypeWorker InferenceServerType = iota
	ServerTypeHead
)

const (
	InferenceRunnerVllm InferenceRunner = iota
)

func newInferenceServer(cfg InferenceServerConfig) InferenceServer {
	switch cfg.Runner {
	case InferenceRunnerVllm:
		return &VllmServer{
			Type:            cfg.Type,
			ContainerImage:  cfg.ContainerImage,
			ContainerName:   cfg.ContainerName,
			RayStartCmd:     cfg.RayStartCmd,
			HFCachePath:     cfg.HFCachePath,
			HeadNodeAddress: "", // set dynamically
			Args:            cfg.Args,
		}
	// case ServerTypeOther:
	//     return &OtherServer{...}
	default:
		return nil
	}
}
