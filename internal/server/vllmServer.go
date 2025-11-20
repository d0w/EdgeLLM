package server

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"strings"
	"sync"
	"syscall"
	"time"
)

type VllmServer struct {
	BaseServer
	Type            InferenceServerType
	ContainerImage  string
	ContainerName   string
	HFCachePath     string
	Args            []string
	Model           string
	HeadNodeAddress string

	cmd       *exec.Cmd
	mu        sync.RWMutex
	isRunning bool
}

// TODO: Make this work for multi-node. Currently uses modified args to replicate multiple nodes on a single machine,.
// Instead the docker images should just run using --host
func (v *VllmServer) Start(ctx context.Context) error {
	v.mu.Lock()

	var vllmHostIP string
	for _, arg := range v.Args {
		if strings.HasPrefix(arg, "VLLM_HOST_IP=") {
			vllmHostIP = strings.TrimPrefix(arg, "VLLM_HOST_IP=")
			break
		}
	}

	rayStartCmd := "ray start --block"
	if v.Type == ServerTypeHead {
		rayStartCmd += " --head --port=6379"
	} else {
		rayStartCmd += " --address=" + v.HeadNodeAddress + ":6379"
	}

	// Build Ray IP environment variables if VLLM_HOST_IP is set
	rayIPVars := []string{}
	if vllmHostIP != "" {
		rayIPVars = append(rayIPVars,
			"-e", fmt.Sprintf("RAY_NODE_IP_ADDRESS=%s", vllmHostIP),
			"-e", fmt.Sprintf("RAY_OVERRIDE_NODE_IP_ADDRESS=%s", vllmHostIP),
		)
	}

	// Add port mappings if node type is head
	ports := []string{}
	if v.Type == ServerTypeHead {
		ports = append(ports, "-p", "6379:6379", "-p", "8010:8010", "-p", "6379:6379")
	}

	// Assemble Docker command arguments
	// cmdArgs := []string{
	// 	"run",
	// 	"--entrypoint", "/bin/bash",
	// 	"--network", "vllmnet",
	// 	"--name", v.ContainerName,
	// 	"--hostname", v.ContainerName,
	// 	"--shm-size", "10.24g",
	// 	"--gpus", "all",
	// }
	cmdArgs := []string{
		"run",
		"--rm",
		"--entrypoint", "/bin/bash",
		"--network", "host",
		"--name", v.ContainerName,
		"--shm-size", "10.24g",
		"--gpus", "all",
	}
	cmdArgs = append(cmdArgs, ports...)
	cmdArgs = append(cmdArgs, "-v", fmt.Sprintf("%s:/root/.cache/huggingface", v.HFCachePath))
	cmdArgs = append(cmdArgs, rayIPVars...)
	// cmdArgs = append(cmdArgs, v.Args...)

	startCmd := fmt.Sprintf("%s & vllm serve %s %s", rayStartCmd, v.Model, strings.Join(v.Args, " "))
	cmdArgs = append(cmdArgs, v.ContainerImage, "-c", startCmd)

	v.cmd = exec.Command("docker", cmdArgs...)
	v.cmd.Stdout = os.Stdout
	v.cmd.Stderr = os.Stderr
	v.cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	v.logger.Info(fmt.Sprintf("Starting vLLM server with command: docker %v", cmdArgs))

	if err := v.cmd.Start(); err != nil {
		v.mu.Unlock()
		return fmt.Errorf("failed to start vLLM server: %w", err)
	}

	// wait for the container to start
	if err := pollContainerReady(ctx, v.ContainerName); err != nil {
		v.mu.Unlock()
		v.Stop()
		return fmt.Errorf("vLLM container did not become ready: %w", err)
	}

	// TODO: Make model path and args dynamic
	// serveArgs := []string{
	// 	"vllm", "serve",
	// 	v.Model,
	// }
	// serveArgs = append(serveArgs, v.Args...)
	//
	// if err := execInContainer(ctx, v.ContainerName, serveArgs); err != nil {
	// 	return fmt.Errorf("failed to start vLLM serve command: %w", err)
	// }

	v.isRunning = true

	// Wait for the server to be ready
	v.mu.Unlock()

	go func() {
		err := v.cmd.Wait()
		if err != nil {
			v.logger.Error("vLLM server process exited unexpectedly: %v", err)
		} else {
			v.logger.Info("vLLM server process stopped")
		}
	}()

	// v.monitorProcess(ctx)
	// if err := v.waitForReady(ctx); err != nil {
	// 	v.Stop()
	// 	return fmt.Errorf("vLLM server did not become ready: %w", err)
	// }

	v.logger.Info("vLLM server started successfully")

	// defer v.Stop()
	// <-ctx.Done()

	return nil
}

func pollContainerReady(ctx context.Context, containerName string) error {
	ticker := time.NewTicker(1 * time.Second)
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			slog.Error(fmt.Sprintf("Timeout waiting for container %s to be ready", containerName))
			return fmt.Errorf("timeout waiting for container %s to be ready", containerName)
		case <-ticker.C:
			slog.Info(fmt.Sprintf("Checking if container %s is running...", containerName))
			out, err := exec.CommandContext(ctx, "docker", "inspect", "-f", "{{.State.Running}}", containerName).Output()
			if err == nil && strings.TrimSpace(string(out)) == "true" {
				slog.Info(fmt.Sprintf("Container %s is running", containerName))
				return nil
			}
		}
	}
}

func execInContainer(ctx context.Context, containerName string, args []string) error {
	fullArgs := append([]string{"exec", containerName}, args...)
	cmd := exec.CommandContext(ctx, "docker", fullArgs...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	// TODO:  Using cmd.Start() instead of Run() for now
	slog.Info(fmt.Sprintf("Executing in container %s: docker %v", containerName, fullArgs))
	return cmd.Run()
}

func (v *VllmServer) monitorProcess(ctx context.Context) {
	err := v.cmd.Wait()

	v.mu.Lock()
	defer v.mu.Unlock()
	v.isRunning = false

	if err != nil {
		if ctx.Err() == nil {
			// Process crashed unexpectedly
			v.logger.Error("vLLM server process exited unexpectedly: %v", err)
		} else {
			// Process was stopped intentionally
			v.logger.Info("vLLM server process stopped")
		}
	}
}

func (v *VllmServer) waitForReady(ctx context.Context) error {
	timeout := 5 * time.Minute
	deadline := time.Now().Add(timeout)

	v.logger.Info("Waiting for vLLM server to be ready (timeout: %v)", timeout)

	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return fmt.Errorf("context cancelled while waiting for ready")
		case <-ticker.C:
			if time.Now().After(deadline) {
				return fmt.Errorf("timeout waiting for vLLM server to be ready")
			}

			if err := v.Ready(); err == nil {
				return nil
			} else {
				v.logger.Error("vLLM failed readiness check: %v", err)
				return fmt.Errorf("vLLM server failed readiness check: %w", err)
			}
		}
	}
}

func (v *VllmServer) Stop() error {
	v.mu.Lock()
	defer v.mu.Unlock()
	if !v.isRunning {
		v.logger.Info("vLLM server is not running")
		return nil
	}

	v.logger.Info("Stopping vLLM server...")

	// if v.cmd != nil && v.cmd.Process != nil {
	// 	pgid, err := syscall.Getpgid(v.cmd.Process.Pid)
	// 	if err == nil {
	// 		syscall.Kill(-pgid, syscall.SIGTERM)
	// 	}
	//
	// 	done := make(chan error, 1)
	// 	go func() {
	// 		done <- v.cmd.Wait()
	// 	}()
	//
	// 	select {
	// 	case <-time.After(30 * time.Second):
	// 		// Force kill if graceful shutdown takes too long
	// 		v.logger.Warn("Graceful shutdown timed out, forcing kill")
	// 		if pgid, err := syscall.Getpgid(v.cmd.Process.Pid); err == nil {
	// 			syscall.Kill(-pgid, syscall.SIGKILL)
	// 		}
	// 		v.cmd.Process.Kill()
	// 	case <-done:
	// 		v.logger.Info("vLLM server stopped gracefully")
	// 	}
	// }

	stopCmd := exec.Command("docker", "stop", v.ContainerName)
	if err := stopCmd.Run(); err != nil {
		v.logger.Warn("Failed to stop Docker container: %v", err)
	}

	v.isRunning = false
	return nil
}

func (v *VllmServer) Ready() error {
	// For now, just check if the container is running
	// In a real implementation, you'd check the Ray/vLLM health endpoint
	cmd := exec.Command("docker", "inspect", "-f", "{{.State.Running}}", v.ContainerName)
	output, err := cmd.Output()
	if err != nil {
		v.logger.Error("Failed to inspect Docker container: %v", err)
		return fmt.Errorf("container not running: %w", err)
	}

	if string(output) != "true\n" {
		return fmt.Errorf("container is not in running state")
	}

	return nil
}

func (v *VllmServer) Health() int {
	v.mu.RLock()
	defer v.mu.RUnlock()

	if !v.isRunning {
		return 1 // Not healthy
	}

	if err := v.Ready(); err != nil {
		return 1 // Not healthy
	}

	return 0 // Healthy
}

func (v *VllmServer) GetModelInfo(modelName string) (string, error) {
	v.mu.RLock()
	defer v.mu.RUnlock()

	if !v.isRunning {
		return "", fmt.Errorf("server is not running")
	}

	return v.Model, nil
}

func (v *VllmServer) IsRunning() bool {
	v.mu.RLock()
	defer v.mu.RUnlock()
	return v.isRunning
}
