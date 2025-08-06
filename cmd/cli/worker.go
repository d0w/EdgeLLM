package cli

import (
	"fmt"

	"github.com/spf13/cobra"
)

var workerCommand = &cobra.Command{
	Use:   "worker",
	Short: "Start a worker node for distributed inference",
	Args:  cobra.ExactArgs(1),
	RunE:  runGenerate,
}

func init() {
	workerCommand.Flags().BoolVar(&vllmQuiet, "quiet", false, "Suppress output from the vLLM server")

	rootCmd.AddCommand(workerCommand)
}

func startWorker(cmd *cobra.Command, args []string) error {
	// start worker as ray node
	if args[0] == "vllm" {
	}
	return fmt.Errorf("unknown worker type: %s", args[0])
}
