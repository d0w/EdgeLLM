package cli

import (
	"fmt"

	"github.com/d0w/EdgeLLM/internal/p2p"
	"github.com/d0w/EdgeLLM/internal/worker"
	"github.com/spf13/cobra"
)

var (
	quiet             bool
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
	workerCommand.Flags().BoolVar(&quiet, "quiet", false, "Suppress output from the vLLM server")
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
	ctx := cmd.Context()
	inferenceWorker, err := worker.CreateWorker("vllm", "test-model", 60051, 50051, "localhost", "/tmp/hf_cache")
	if err != nil {
		return fmt.Errorf("failed to create worker: %v", err)
	}

	// setup and attach node
	node, err := p2p.NewP2PNode(ctx, []string{"test", "test2"}, 8000)
	if err != nil {
		return fmt.Errorf("failed to create p2p node: %v", err)
	}

	if err := inferenceWorker.Start(); err != nil {
		return fmt.Errorf("failed to start worker: %v", err)
	}

	return nil
}
