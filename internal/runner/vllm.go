package runner

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"time"

	// "github.com/briandowns/spinner"
	"github.com/spf13/viper"
)

type VllmServer struct {
	cmd    *exec.Cmd
	logger *log.Logger
}

func NewVllmServer(alias string) *VllmServer {
	cyan := "\033[36m"
	reset := "\033[0m"
	if alias == "" {
		alias = cyan + "[VllmServer]: " + reset
	}
	return &VllmServer{
		logger: log.New(os.Stdout, alias, log.LstdFlags),
	}
}

func (s *VllmServer) Start(startArgs []string) error {
	// flag variables
	serverAddr := viper.GetString("vllm-server")
	maxWorkers := viper.GetInt("vllm-max-workers")
	if maxWorkers <= 0 {
		maxWorkers = 4 // Default to 4 workers if not set
	}

	// setting server path
	var serverDirectory string
	if path, ok := os.LookupEnv("VLLM_SERVER_PATH"); ok {
		serverDirectory = filepath.Join(path)
	} else {
		serverDirectory = filepath.Join("backend", "python", "vllm")
	}

	// check if server script exists
	if _, err := os.Stat(serverDirectory); os.IsNotExist(err) {
		log.Fatalf("VLLM server script not found at %s", serverDirectory)
		return fmt.Errorf("VLLM server script not found at %s", serverDirectory)
	}

	// bootstrap command
	s.cmd = exec.Command("python3", append(startArgs, "server.py")...)
	s.cmd.Stdout = os.Stdout
	s.cmd.Stderr = os.Stderr
	s.cmd.Dir = filepath.Dir(serverDirectory)
	s.cmd.Env = append(os.Environ(),
		fmt.Sprintf("VLLM_SERVER_ADDR=%s", serverAddr),
		fmt.Sprintf("VLLM_MAX_WORKERS=%d", maxWorkers),
	)

	// avoids zombie process. must manually kill the process though
	s.cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	s.logger.Printf("Starting VLLM server with command: %s", s.cmd.String())
	s.logger.Println("address", serverAddr, "workers", maxWorkers, "path", serverDirectory+"/server.py")

	if err := s.cmd.Start(); err != nil {
		log.Fatalf("Failed to start VLLM server: %v", err)
		return fmt.Errorf("failed to start VLLM server: %w", err)
	}

	return s.waitForReady()
}

func (s *VllmServer) waitForReady() error {
	timeout := viper.GetDuration("vllm-startup-timeout")
	if timeout <= 0 {
		fmt.Println("Using default timeout of 30 seconds for VLLM server startup")
		timeout = time.Second * 30
	}

	s.logger.Printf("Waiting for VLLM server to be ready (timeout: %s)", timeout)

	// indicator := spinner.New(spinner.CharSets[9], 100*time.Millisecond)
	// indicator.Start()
	// defer indicator.Stop()

	for range int(timeout.Seconds()) {
		if s.isReady() {
			s.logger.Println("VLLM server is ready")
			return nil
		}

		time.Sleep(time.Second * 1)
	}

	return fmt.Errorf("python server did not start within the timeout period of %s", timeout)
}

func (s *VllmServer) isReady() bool {
	// TODO: acutal health check

	ready := make(chan bool, 1)

	go func() {
		time.Sleep(5 * time.Second)
		ready <- true
	}()

	select {
	case <-ready:
		return true
	default:
		return false
	}
}

func (s *VllmServer) Stop() error {
	if s.cmd == nil || s.cmd.Process == nil {
		return nil
	}

	s.logger.Println("Stopping VLLM server...")

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
			s.logger.Printf("VLLM server stopped with error: %v", err)
		} else {
			s.logger.Println("VLLM server stopped gracefully")
		}
	}

	return nil
}
