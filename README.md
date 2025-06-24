# EdgeLLM

A Go-based application with Docker support.

## Features

- RESTful API built with Gin framework
- Docker containerization with multi-stage builds
- Docker Compose for easy development
- Health check endpoints
- Makefile for common tasks

## Quick Start

### Prerequisites

- Go 1.21 or later
- Docker and Docker Compose
- Make (optional, for using Makefile commands)

### Local Development

1. **Clone and setup dependencies:**
   ```bash
   go mod download
   ```

2. **Run locally:**
   ```bash
   go run .
   # or
   make run
   ```

3. **Test the application:**
   ```bash
   curl http://localhost:8080/health
   curl http://localhost:8080/api/v1/hello
   ```

### Docker Development

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

## API Endpoints

- `GET /health` - Health check endpoint
- `GET /api/v1/hello` - Example API endpoint

## Available Make Commands

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

## Project Structure

```
.
├── cmd/
│   └── server/          # Main application entry point
│       └── main.go
├── internal/            # Private application code
│   ├── config/          # Configuration management
│   ├── handler/         # HTTP handlers
│   ├── middleware/      # HTTP middleware
│   ├── model/           # Data models
│   └── service/         # Business logic
├── pkg/                 # Public library code
│   ├── logger/          # Logging utilities
│   └── utils/           # Common utilities
├── api/                 # API definitions
│   └── v1/
│       └── openapi.yaml # OpenAPI specification
├── configs/             # Configuration files
│   └── app.yaml
├── scripts/             # Build and deployment scripts
│   ├── build.sh
│   └── test.sh
├── deployments/         # Deployment configurations
│   ├── docker-compose.prod.yml
│   └── nginx.conf
├── test/                # Integration tests
├── web/                 # Web UI assets
│   └── static/
├── go.mod              # Go module file
├── Dockerfile          # Docker configuration
├── docker-compose.yml  # Docker Compose configuration
├── Makefile           # Build and deployment commands
└── README.md          # This file
```

## Deployment

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