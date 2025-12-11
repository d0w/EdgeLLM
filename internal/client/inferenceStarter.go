package client

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/d0w/EdgeLLM/internal/server"
)

// client is responsible for creating and managing cluster metadata
// "Qwen/Qwen-3-0.6B"
func StartInferenceCluster(endpoints []string, model string) error {
	started := []string{}

	// for vllm specifically
	// TODO: Change from RESTful
	for i, endpoint := range endpoints {
		// config is validated by the worker server itself
		serverType := server.ServerTypeWorker

		if i == 0 {
			serverType = server.ServerTypeHead
		}

		err := startInferenceServer(endpoint, model, serverType)
		if err != nil {
			// Rollback: stop all previously started servers
			for _, startedEndpoint := range started {
				_ = stopInferenceServer(startedEndpoint) // ignore rollback errors
			}
			return fmt.Errorf("failed to start inference server at %s: %w", endpoint, err)
		}
		started = append(started, endpoint)
	}
	return nil
}

func StopInferenceCluster(endpoints []string) error {
	stopped := []string{}
	failed := []string{}
	for _, endpoint := range endpoints {
		err := stopInferenceServer(endpoint)
		if err != nil {
			failed = append(failed, endpoint)
		}
		stopped = append(stopped, endpoint)
	}
	return nil
}

func startInferenceServer(endpoint string, model string, serverType server.InferenceServerType) error {
	// TODO: Support no servertype if not using vllm
	if serverType == server.ServerTypeOther {
		return fmt.Errorf("server type cannot be empty or other for vllm")
	}

	// initialize inference server with correct model
	initUrl := fmt.Sprintf("http://%s/inference/initialize", endpoint)

	initBody := struct {
		ServerType server.InferenceServerType `json:"serverType"`
		Model      string                     `json:"model"`
	}{
		ServerType: serverType,
		Model:      model,
	}

	bodyBytes, err := json.Marshal(initBody)
	if err != nil {
		return err
	}

	initReq, err := http.NewRequest("POST", initUrl, bytes.NewBuffer(bodyBytes))
	if err != nil {
		return err
	}
	initReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	intialResp, err := client.Do(initReq)
	if err != nil {
		return err
	}
	defer intialResp.Body.Close()

	if intialResp.StatusCode != http.StatusOK && intialResp.StatusCode != http.StatusAccepted {
		return &httpError{StatusCode: intialResp.StatusCode}
	}

	// start request
	url := fmt.Sprintf("http://%s/inference/start", endpoint)
	req, err := http.NewRequest("POST", url, bytes.NewBuffer([]byte{}))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusAccepted {
		return &httpError{StatusCode: resp.StatusCode}
	}
	return nil
}

type httpError struct {
	StatusCode int
}

func (e *httpError) Error() string {
	return http.StatusText(e.StatusCode)
}

func stopInferenceServer(endpoint string) error {
	url := fmt.Sprintf("http://%s/inference/stop", endpoint)
	req, err := http.NewRequest("POST", url, bytes.NewBuffer([]byte{}))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusAccepted {
		return &httpError{StatusCode: resp.StatusCode}
	}
	return nil
}
