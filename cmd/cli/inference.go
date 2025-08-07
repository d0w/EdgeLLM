package cli

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/d0w/EdgeLLM/internal/service"
	pb "github.com/d0w/EdgeLLM/backend/go/proto"
	"github.com/d0w/EdgeLLM/pkg/logger"
	"github.com/spf13/cobra"
)

var (
	inferenceModel       string
	inferencePrompt      string
	inferenceMaxTokens   int32
	inferenceTemperature float32
	inferenceTopP        float32
	inferenceUseP2P      bool
	inferenceStrategy    string
)

var inferenceCmd = &cobra.Command{
	Use:   "inference",
	Short: "Run distributed inference using P2P network",
	RunE:  runDistributedInference,
}

func init() {
	inferenceCmd.Flags().StringVar(&inferenceModel, "model", "", "Model to use for inference (required)")
	inferenceCmd.Flags().StringVar(&inferencePrompt, "prompt", "", "Prompt for text generation (required)")
	inferenceCmd.Flags().Int32Var(&inferenceMaxTokens, "max-tokens", 100, "Maximum tokens to generate")
	inferenceCmd.Flags().Float32Var(&inferenceTemperature, "temperature", 0.7, "Sampling temperature")
	inferenceCmd.Flags().Float32Var(&inferenceTopP, "top-p", 1.0, "Top-p sampling parameter")
	inferenceCmd.Flags().BoolVar(&inferenceUseP2P, "p2p", true, "Use P2P network for distributed inference")
	inferenceCmd.Flags().StringVar(&inferenceStrategy, "strategy", "auto", "Inference strategy: auto, single, sharded, replicated")

	// Mark required flags
	inferenceCmd.MarkFlagRequired("model")
	inferenceCmd.MarkFlagRequired("prompt")

	rootCmd.AddCommand(inferenceCmd)
}

func runDistributedInference(cmd *cobra.Command, args []string) error {
	// Set up logger
	inferenceLogger := logger.New(log.LevelInfo)
	
	// Set up context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	inferenceLogger.Info("Starting distributed inference...")
	inferenceLogger.Info(fmt.Sprintf("Model: %s", inferenceModel))
	inferenceLogger.Info(fmt.Sprintf("Prompt: %s", inferencePrompt))
	inferenceLogger.Info(fmt.Sprintf("Strategy: %s", inferenceStrategy))

	if inferenceUseP2P {
		// Use the new dynamic coordinator for P2P inference
		coordinator := service.NewDynamicCoordinator("client_node", inferenceLogger)
		
		// Create inference request
		request := &pb.GenerateTextRequest{
			Prompt:      inferencePrompt,
			Model:       inferenceModel,
			MaxTokens:   inferenceMaxTokens,
			Temperature: inferenceTemperature,
			TopP:        inferenceTopP,
			RequestId:   fmt.Sprintf("client_req_%d", time.Now().UnixNano()),
		}

		// Initiate distributed inference
		response, err := coordinator.InitiateInference(ctx, request)
		if err != nil {
			return fmt.Errorf("distributed inference failed: %w", err)
		}

		// Display results
		fmt.Println("\n=== Inference Results ===")
		fmt.Printf("Request ID: %s\n", response.RequestId)
		fmt.Printf("Generated Text: %s\n", response.Text)
		fmt.Printf("Finish Reason: %s\n", response.FinishReason)
		fmt.Printf("Prompt Tokens: %d\n", response.PromptTokens)
		fmt.Printf("Completion Tokens: %d\n", response.CompletionTokens)
		fmt.Printf("Total Tokens: %d\n", response.TotalTokens)

		// Show active requests for debugging
		activeRequests := coordinator.GetActiveRequests()
		if len(activeRequests) > 0 {
			fmt.Println("\n=== Active Requests ===")
			for id, req := range activeRequests {
				fmt.Printf("Request %s: %s (participants: %v)\n", 
					id, req.Status, req.ParticipantNodes)
			}
		}

	} else {
		// Fallback to local inference service
		inferenceLogger.Info("Using local inference service...")
		
		inferenceService, err := service.NewInferenceService()
		if err != nil {
			return fmt.Errorf("failed to create inference service: %w", err)
		}
		defer inferenceService.Close()

		response, err := inferenceService.GenerateText(ctx, inferencePrompt)
		if err != nil {
			return fmt.Errorf("local inference failed: %w", err)
		}

		fmt.Println("\n=== Local Inference Results ===")
		fmt.Printf("Generated Text: %s\n", response.GetText())
	}

	return nil
}
