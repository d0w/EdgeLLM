package cli

import (
	"fmt"

	"github.com/d0w/EdgeLLM/internal/worker"
	"github.com/spf13/cobra"
)

var (
	workerModel       string
	workerPort        string
	workerP2PToken    string
	workerNetworkID   string
	workerHFCachePath string
)

var workerCommand = &cobra.Command{
	Use:   "worker [vllm]",
	Short: "Start a worker node for distributed inference",
	Args:  cobra.ExactArgs(1),
	RunE:  startWorker,
}

func init() {
	workerCommand.Flags().BoolVar(&vllmQuiet, "quiet", false, "Suppress output from the vLLM server")
	workerCommand.Flags().StringVar(&workerModel, "model", "", "Model to load (required)")
	workerCommand.Flags().StringVar(&workerPort, "port", "50051", "gRPC port to listen on")
	workerCommand.Flags().StringVar(&workerP2PToken, "token", "", "P2P network token")
	workerCommand.Flags().StringVar(&workerNetworkID, "network", "default", "P2P network ID")
	workerCommand.Flags().StringVar(&workerHFCachePath, "hf-cache", "", "HuggingFace cache path")

	// Mark model as required
	workerCommand.MarkFlagRequired("model")

	rootCmd.AddCommand(workerCommand)
}

func startWorker(cmd *cobra.Command, args []string) error {
	inferenceWorker, err := worker.CreateWorker("vllm", "test-model", 60051, 50051, "localhost:60051", "/tmp/hf_cache")
	if err != nil {
		return fmt.Errorf("failed to create worker: %v", err)
	}

	err = inferenceWorker.Start()
	if err != nil {
		return fmt.Errorf("failed to start worker: %v", err)
	}
	return nil
}
