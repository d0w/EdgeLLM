# EdgeLLM

Run decentralized LLMs. Distribute your inference and training requests across a vast peer-to-peer network without a need for a centralized server. EdgeLLM gives users the ability to perform inference on models much larger than they could with their own machine, utilizing a network of volunteers' machines to perform inference on their behalf.

## Goals

Make the backend (model serving) interchangeable and make this project sort of a standalone that can use any inference server as long as they implement the grpc interface. We just need to focus on the inference routing, discovery, etc. and make this something you can just layer on top of an inference server to start supporting decentralized p2p inference.

more details to come...

## Features

- Peer-to-peer decentralized inference
- vLLM support
- more to come...

## Quick Start

TODO: Startup script to automate this

### Prerequisites

- Go 1.21 or later
- Docker and Docker Compose (optional for now)
- Make (optional, for using Makefile commands)
- Python 3.10 or later
- [protoc](https://protobuf.dev/installation/) and the [go extension for protoc](https://grpc.io/docs/languages/go/quickstart/)

### Local Development

1. **Clone and setup dependencies:**

   ```bash
   go mod download
   ```

2. **Setup a local Python environment:**

 ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

   ```bash
   python3 -m pip install -r backend/python/requirements.txt
   ```

3. **Build proto**
    1. Install protoc and go's [extension](https://grpc.io/docs/languages/go/quickstart/) for protoc

    1. Build protobuf

      ```bash
      # For python spec
      python3 -m grpc_tools.protoc -Ibackend --python_out=backend/python/proto --grpc_python_out=backend/python/proto --pyi_out=backend/python/proto backend/backend.proto
      
      # For go spec
      protoc --go_out=. --go-grpc_out=. backend/backend.proto
      ```

4. Run main to view commands

*Make sure you've activated your Python virtual environment in the same terminal session as you're running the go application.*

   ```bash
   go run main.go
   ```

### Docker Development (WIP)

1. **Build and run with Docker Compose:**

   ```bash
   docker-compose up --build
   # or
   make dev
   ```

2. **Stop the containers:**

   ```bash
   docker-compose down
   # or
   make docker-stop
   ```

## API Endpoints (WIP)

- `GET /health` - Health check endpoint
- `GET /api/v1/hello` - Example API endpoint

## Available Make Commands (WIP)

```bash
make help          # Show available commands
make build         # Build the Go application
make run           # Run the application locally
make test          # Run tests
make docker-build  # Build Docker image
make docker-run    # Run with Docker Compose
make docker-stop   # Stop Docker containers
make dev           # Start development environment
make clean         # Clean build artifacts
```

## Environment Variables

- `PORT` - Server port (default: 8080)
- `GIN_MODE` - Gin mode (debug/release, default: debug)

## Deployment (WIP)

1. **Build the Docker image:**

   ```bash
   make docker-build
   ```

2. **Push to registry (configure DOCKER_REGISTRY in Makefile):**

   ```bash
   make docker-push
   ```

3. **Deploy using Docker Compose:**

   ```bash
   docker-compose up -d
   ```

## References

This project takes inspiration and code snippets from several sources:

- [LocalAI](https://github.com/mudler/LocalAI)
  - Main source of inspiration. This project aims to take the distributed inference capabilities of LocalAI and extend them to truly decentralized networks that work on more inference servers than just llama.cpp

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
