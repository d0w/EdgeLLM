# EdgeLLM

Run decentralized LLMs. Distribute your inference and training requests across a vast peer-to-peer network without a need for a centralized server. EdgeLLM gives users the ability to perform inference on models much larger than they could with their own machine, utilizing a network of volunteers' machines to perform inference on their behalf.

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
   python -m pip install -r backend/python/requirements.txt
   ```

3. **Build proto**
    1. Install protoc and go's extension for protoc

    1. Build protobuf

      ```bash
      # For python spec
      python -m grpc_tools.protoc -Ibackend --python_out=backend/python/proto --grpc_python_out=backend/python/proto --pyi_out=backend/python/proto backend/backend.proto
      
      # For go spec
      protoc --go_out=. --go-grpc_out=. backend/backend.proto
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

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

