package server

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"sync"
	"syscall"
	"time"
)

type VllmServer struct {
	BaseServer
	Type            InferenceServerType
	ContainerImage  string
	ContainerName   string
	RayStartCmd     string
	HFCachePath     string
	Args            []string
	Model           string
	HeadNodeAddress string

	cmd       *exec.Cmd
	mu        sync.RWMutex
	isRunning bool
}

func (v *VllmServer) Start(ctx context.Context) error {
	v.mu.Lock()

	if v.isRunning {
		v.logger.Info("vLLM server is already running")
		v.mu.Unlock()
		return nil
	}

	cmdArgs := []string{
		"run",
		"--rm", // Automatically remove container when it exits
		"--entrypoint", "/bin/bash",
		"--network", "host",
		"--name", v.ContainerName,
		"--shm-size", "10.24g",
		"--gpus", "all",
		"-v", fmt.Sprintf("%s:/root/.cache/huggingface", v.HFCachePath),
	}

	// Add any additional arguments
	cmdArgs = append(cmdArgs, v.Args...)
	cmdArgs = append(cmdArgs, v.ContainerImage)

	// TODO: Set this based on worker or head type
	fullRayStartCmd := v.RayStartCmd + fmt.Sprintf(" --address=%s:6379", v.HeadNodeAddress)
	cmdArgs = append(cmdArgs, "-c", fullRayStartCmd)
	// if head
	// --head --port=6379

	v.cmd = exec.CommandContext(ctx, "docker", cmdArgs...)
	v.cmd.Stdout = os.Stdout
	v.cmd.Stderr = os.Stderr

	// Set process group so we can kill the entire process tree
	v.cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	v.logger.Info("Starting vLLM server with command: docker %v", cmdArgs)

	if err := v.cmd.Start(); err != nil {
		v.mu.Unlock()
		return fmt.Errorf("failed to start vLLM server: %w", err)
	}

	v.isRunning = true

	// Monitor the process in a goroutine
	go v.monitorProcess(ctx)

	// Wait for the server to be ready
	v.mu.Unlock()
	if err := v.waitForReady(ctx); err != nil {
		v.Stop()
		return fmt.Errorf("vLLM server did not become ready: %w", err)
	}

	v.logger.Info("vLLM server started successfully")
	return nil
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
