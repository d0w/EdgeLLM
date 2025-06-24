#!/bin/bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Running tests...${NC}"

# Run tests with coverage
go test -v -race -coverprofile=coverage.out -covermode=atomic ./...

# Generate coverage report
if [ -f coverage.out ]; then
    echo -e "${GREEN}Generating coverage report...${NC}"
    go tool cover -html=coverage.out -o coverage.html
    echo -e "${GREEN}Coverage report generated: coverage.html${NC}"
    
    # Display coverage summary
    echo -e "${YELLOW}Coverage Summary:${NC}"
    go tool cover -func=coverage.out | tail -1
fi

# Run linting (if golangci-lint is installed)
if command -v golangci-lint &> /dev/null; then
    echo -e "${GREEN}Running linter...${NC}"
    golangci-lint run
else
    echo -e "${YELLOW}golangci-lint not installed, skipping linting...${NC}"
    echo -e "${YELLOW}Install with: go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest${NC}"
fi

echo -e "${GREEN}All tests passed!${NC}" 