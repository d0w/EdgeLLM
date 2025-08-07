package runner

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"time"

	// "github.com/briandowns/spinner"
	"github.com/briandowns/spinner"
	"github.com/spf13/viper"
)

type VllmServerType int

const (
	VllmHead VllmServerType = iota
	VllmWorker
)

var serverType = map[VllmServerType]string{
	VllmHead:   "--head",
	VllmWorker: "--worker",
}

type VllmServer struct {
	cmd           *exec.Cmd
	serverAddress string
	serverType    VllmServerType
	hfCachePath   string
	logger        *log.Logger
}

func NewVllmServer(serverType VllmServerType, serverAddress string, hfCachePath string) *VllmServer {
	cyan := "\033[36m"
	reset := "\033[0m"

	var alias string
	if serverType == VllmHead {
		alias = cyan + "[VllmHead]: " + reset
	} else {
		alias = cyan + "[VllmWorker]: " + reset
	}

	var logger *log.Logger
	if viper.GetBool("quiet") {
		logger = log.New(io.Discard, "", 0) // No prefix or flags
	} else {
		logger = log.New(os.Stdout, alias, log.LstdFlags)
	}
	return &VllmServer{
		logger:        logger,
		serverAddress: serverAddress,
		serverType:    serverType,
		hfCachePath:   hfCachePath,
	}
}

func (s *VllmServer) Start(startArgs []string) error {
	// setting server's root path
	var serverDirectory string
	if path, ok := os.LookupEnv("VLLM_SERVER_PATH"); ok {
		serverDirectory = filepath.Join(path)
	} else {
		serverDirectory = filepath.Join("scripts")
	}

	// check if server script exists
	if _, err := os.Stat(serverDirectory); os.IsNotExist(err) {
		s.logger.Fatalf("VLLM server script not found at %s", serverDirectory)
		return fmt.Errorf("VLLM server script not found at %s", serverDirectory)
	}

	// bootstrap command
	s.cmd = exec.Command(
		"vllm_cluster.sh",
		append([]string{
			"vllm/vllm-openai",
			s.serverAddress,
			serverType[s.serverType],
			s.hfCachePath,
			"-e",
			fmt.Sprintf("VLLM_HOST_IP=%s", s.serverAddress),
		},
			startArgs...,
		)...,
	)
	s.cmd.Stdout = s.logger.Writer()
	s.cmd.Stderr = s.logger.Writer()
	s.cmd.Dir = filepath.Dir(serverDirectory)

	// avoids zombie process. must manually kill the process with SIGKILL or SIGTERM
	s.cmd.SysProcAttr = &syscall.SysProcAttr{}

	s.logger.Printf("Starting VLLM server with command: %s", s.cmd.String())

	if err := s.cmd.Start(); err != nil {
		s.logger.Fatalf("Failed to start VLLM server: %v", err)
		return fmt.Errorf("failed to start VLLM server: %w", err)
	}

	return s.waitForReady()
}

// func (s *VllmServer) getPythonCommand() string {
// 	venvPath := viper.GetString("vllm-venv-path")
// 	if venvPath != "" {
// 		pythonPath := filepath.Join(venvPath, "bin", "python3")
//
// 		if _, err := os.Stat(pythonPath); err == nil {
// 			s.logger.Printf("Using Python interpreter from virtual environment: %s", pythonPath)
// 			return pythonPath
// 		}
// 	}
//
// 	// look for venv in root of repo
// 	if _, err := os.Stat(".venv/bin/python3"); err == nil {
// 		s.logger.Println("Using Python interpreter from .venv")
// 		return ".venv/bin/python3"
// 	}
//
// 	s.logger.Println("Using system Python interpreter")
// 	return "python3"
// }

func (s *VllmServer) waitForReady() error {
	timeout := viper.GetDuration("vllm-startup-timeout")
	if timeout <= 0 {
		fmt.Println("Using default timeout of 30 seconds for VLLM server startup")
		timeout = time.Second * 30
	}

	s.logger.Printf("Waiting for VLLM server to be ready (timeout: %s)", timeout)

	indicator := spinner.New(spinner.CharSets[9], 100*time.Millisecond)
	indicator.Start()
	defer indicator.Stop()

	for range int(timeout.Seconds()) {
		if s.isReady() {
			s.logger.Println("vLLM server is ready")
			return nil
		}

		time.Sleep(time.Second * 1)
	}

	return fmt.Errorf("python server did not start within the timeout period of %s", timeout)
}

func (s *VllmServer) isReady() bool {
	// Actual health check via HTTP endpoint
	healthURL := fmt.Sprintf("http://%s/health", s.serverAddress)
	
	client := &http.Client{
		Timeout: 2 * time.Second,
	}
	
	resp, err := client.Get(healthURL)
	if err != nil {
		s.logger.Printf("Health check failed: %v", err)
		return false
	}
	defer resp.Body.Close()
	
	if resp.StatusCode == 200 {
		s.logger.Printf("VLLM server is healthy at %s", healthURL)
		return true
	}
	
	s.logger.Printf("Health check returned status %d", resp.StatusCode)
	return false
}

func (s *VllmServer) Stop() error {
	if s.cmd == nil || s.cmd.Process == nil {
		return nil
	}

	s.logger.Println("Stopping vLLM server...")

	if err := s.cmd.Process.Signal(syscall.SIGTERM); err != nil {
		s.logger.Printf("Failed to send SIGTERM to VLLM server: %v", err)

		if err := s.cmd.Process.Signal(syscall.SIGINT); err != nil {
			s.logger.Printf("Failed to send SIGINT to VLLM server: %v", err)
		}
	}

	shutdownTimeout := time.Second * 10

	done := make(chan error, 1)
	go func() {
		done <- s.cmd.Wait()
	}()

	select {
	case <-time.After(shutdownTimeout):
		syscall.Kill(-s.cmd.Process.Pid, syscall.SIGKILL)

		s.logger.Println("VLLM server did not stop gracefully, sending SIGKILL...")
		if err := s.cmd.Process.Kill(); err != nil {
			s.logger.Printf("Failed to kill VLLM server process: %v", err)
		}
		s.cmd.Wait()

	case err := <-done:
		if err != nil {
			s.logger.Printf("vLLM server stopped with error: %v", err)
		} else {
			s.logger.Println("vLLM server stopped gracefully")
		}
	}

	return nil
}
