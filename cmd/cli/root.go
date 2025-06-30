package cli

import (
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"

	"github.com/d0w/EdgeLLM/internal/runner"
)

var (
	cfgFile string
	verbose bool

	vllmServer           *runner.VllmServer
	vllmServerAddr       string
	vllmServerMaxWorkers int
	vllmStartupTimeout   time.Duration
)

var rootCmd = &cobra.Command{
	Use:   "edgellm",
	Short: "EdgeLLM - Decentralized P2P LLM inference CLI",
	Long: `EdgeLLM CLI provides commands to interact with the distributed 
LLM inference backend, including text generation, model management, 
and health monitoring.`,
	PersistentPreRun: startBackend,
}

func Execute() error {
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-c
		if vllmServer != nil {
			vllmServer.Stop()
		}
		os.Exit(0)
	}()
	return rootCmd.Execute()
}

func init() {
	cobra.OnInitialize(initConfig)

	rootCmd.PersistentFlags().StringVar(&cfgFile, "config", "", "config file (default is $HOME/.peerllm.yaml)")
	rootCmd.PersistentFlags().StringVar(&vllmServerAddr, "vllm-server", "localhost:50051", "gRPC server address")
	rootCmd.PersistentFlags().IntVar(&vllmServerMaxWorkers, "vllm-server-max-workers", 4, "maximum number of workers for vLLM server")
	rootCmd.PersistentFlags().DurationVar(&vllmStartupTimeout, "vllm-startup-timeout", 30*time.Second, "timeout for vLLM server startup")
	rootCmd.PersistentFlags().BoolVarP(&verbose, "verbose", "v", false, "verbose output")

	viper.BindPFlag("vllm-server", rootCmd.PersistentFlags().Lookup("vllm-server"))
	viper.BindPFlag("verbose", rootCmd.PersistentFlags().Lookup("verbose"))
	viper.BindPFlag("vllm-server-max-workers", rootCmd.PersistentFlags().Lookup("vllm-server-max-workers"))
	viper.BindPFlag("vllm-startup-timeout", rootCmd.PersistentFlags().Lookup("vllm-startup-timeout"))
}

func initConfig() {
	if cfgFile != "" {
		viper.SetConfigFile(cfgFile)
	} else {
		viper.AddConfigPath("$HOME")
		viper.SetConfigName(".peerllm")
	}

	viper.AutomaticEnv()

	if err := viper.ReadInConfig(); err == nil {
		if verbose {
			cobra.CheckErr(err)
		}
	}
}

func startBackend(cmd *cobra.Command, args []string) {
	// can set alias later, for now just VllmServer
	vllmServer = runner.NewVllmServer("")

	var startArgs []string

	// TODO: Add startarg support with viper instead of manually adding in args

	if err := vllmServer.Start(startArgs); err != nil {
		log.Fatalf("Failed to start VLLM server: %v", err)
	}
}
