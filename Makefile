# Variables
APP_NAME=edgellm
DOCKER_IMAGE=$(APP_NAME):latest
DOCKER_REGISTRY=your-registry.com

# Default target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  build       - Build the Go application"
	@echo "  run         - Run the application locally"
	@echo "  test        - Run tests"
	@echo "  clean       - Clean build artifacts"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run application in Docker"
	@echo "  docker-stop  - Stop Docker containers"
	@echo "  docker-clean - Clean Docker images and containers"
	@echo "  dev         - Start development environment"

# Go targets
.PHONY: build
build:
	go build -o bin/$(APP_NAME) ./cmd/server

.PHONY: run
run:
	go run ./cmd/server

.PHONY: test
test:
	go test -v ./...

.PHONY: clean
clean:
	rm -rf bin/
	go clean

# Docker targets
.PHONY: docker-build
docker-build:
	docker build -t $(DOCKER_IMAGE) .

.PHONY: docker-run
docker-run:
	docker-compose up -d

.PHONY: docker-stop
docker-stop:
	docker-compose down

.PHONY: docker-clean
docker-clean:
	docker-compose down -v --rmi all --remove-orphans
	docker system prune -f

.PHONY: dev
dev:
	docker-compose up --build

# Production targets
.PHONY: docker-push
docker-push:
	docker tag $(DOCKER_IMAGE) $(DOCKER_REGISTRY)/$(DOCKER_IMAGE)
	docker push $(DOCKER_REGISTRY)/$(DOCKER_IMAGE)

# Dependencies
.PHONY: deps
deps:
	go mod download
	go mod tidy 