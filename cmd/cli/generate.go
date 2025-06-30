package cli

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/d0w/EdgeLLM/internal/service"
	"github.com/spf13/cobra"
)

var generateCmd = &cobra.Command{
	Use:   "generate <prompt>",
	Short: "Generate text using the LLM",
	Args:  cobra.ExactArgs(1),
	RunE:  runGenerate,
}

func init() {
	rootCmd.AddCommand(generateCmd)
}

func runGenerate(cmd *cobra.Command, args []string) error {
	prompt := args[0]

	if verbose {
		log.Printf("Generating text for prompt: %s", prompt)
	}

	inferenceService, err := service.NewInferenceService()
	if err != nil {
		log.Fatalf("Failed to create inference service: %v", err)
		return fmt.Errorf("failed to create inference service: %w", err)
	}
	defer inferenceService.Close()

	ctx, cancel := context.WithTimeout(context.Background(), time.Second*30)
	defer cancel()

	response, err := inferenceService.GenerateText(ctx, prompt)
	if err != nil {
		log.Fatalf("Failed to generate text: %v", err)
		return fmt.Errorf("failed to generate text: %w", err)
	}

	fmt.Println(response.GetText())
	return nil
}
