package server

import (
	"fmt"
	"os"
	"os/exec"
)

type VllmServer struct {
	BaseServer
	Type           InferenceServerType
	ContainerImage string
	ContainerName  string
	RayStartCmd    string
	HFCachePath    string
	Args           []string
	Model          string
}

func (v *VllmServer) Start() error {
	cmdArgs := []string{
		"run",
		"--entrypoint", "/bin/bash",
		"--network", "host",
		"--name", v.ContainerName,
		"--shm-size", "10.24g",
		"--gpus", "all",
		"-v", fmt.Sprintf("%s:/root/.cache/huggingface", v.HFCachePath),
	}
	cmdArgs = append(cmdArgs, v.ContainerImage)
	cmdArgs = append(cmdArgs, "-c", v.RayStartCmd)

	cmd := exec.Command("docker", cmdArgs...)

	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	v.logger.Info("Starting vLLM server with command: docker %v", cmdArgs)

	return cmd.Run()
}

func (v *VllmServer) Stop() error {
	return nil
}

func (v *VllmServer) Ready() error {
	return nil
}

func (v *VllmServer) Health() int {
	return 0
}

func (v *VllmServer) GetModelInfo(modelName string) (string, error) {
	return v.Model, nil
}
