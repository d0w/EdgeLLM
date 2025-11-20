package client

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"strings"
)

type Client struct {
	ListenerEndpoint  string // endpoint for listener
	InferenceEndpoint string // openapi compatible endpoint
	Model             string
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatRequest struct {
	Model    string        `json:"model"`
	Messages []chatMessage `json:"messages"`
	Stream   bool          `json:"stream"`
}

type streamResponse struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
	} `json:"choices"`
}

// TODO: We wont know the endpoint yet, but for now we are hardcoding it
func NewClient(listenerEndpoint string, model string) *Client {
	return &Client{
		ListenerEndpoint:  listenerEndpoint,
		InferenceEndpoint: "127.0.0.1:8010",
		Model:             model,
	}
}

func (c *Client) Close() error {
	if err := StopInferenceServer(c.ListenerEndpoint); err != nil {
		slog.Error(fmt.Sprintf("Failed to stop inference server: %v", err))
		return err
	}
	return nil
}

func (c *Client) Chat() error {
	if err := StartInferenceServer(c.ListenerEndpoint); err != nil {
		return err
	}
	defer StopInferenceServer(c.ListenerEndpoint)
	reader := bufio.NewReader(os.Stdin)
	var messages []chatMessage

	fmt.Println("Start chatting with the model (type 'exit' to quit):")
	for {
		fmt.Print("> ")
		userInput, err := reader.ReadString('\n')
		if err != nil {
			return err
		}
		userInput = strings.TrimSpace(userInput)
		if userInput == "exit" {
			break
		}
		messages = append(messages, chatMessage{Role: "user", Content: userInput})

		reqBody, err := json.Marshal(chatRequest{
			Model:    c.Model,
			Messages: messages,
			Stream:   true,
		})
		if err != nil {
			return err
		}

		req, err := http.NewRequest("POST", "http://"+c.InferenceEndpoint+"/v1/chat/completions", bytes.NewBuffer(reqBody))
		if err != nil {
			return err
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return err
		}
		defer resp.Body.Close()

		// Stream response
		scanner := bufio.NewScanner(resp.Body)
		fmt.Print("Model: ")
		var assistantMsg strings.Builder
		for scanner.Scan() {
			line := scanner.Text()
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			data := strings.TrimPrefix(line, "data: ")
			if data == "[DONE]" {
				break
			}
			var sr streamResponse
			if err := json.Unmarshal([]byte(data), &sr); err != nil {
				continue
			}
			for _, choice := range sr.Choices {
				token := choice.Delta.Content
				fmt.Print(token)
				assistantMsg.WriteString(token)
			}
		}
		fmt.Println()
		messages = append(messages, chatMessage{Role: "assistant", Content: assistantMsg.String()})
	}
	return nil
}
