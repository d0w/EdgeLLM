#!/bin/bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="edgellm"
VERSION=${VERSION:-"1.0.0"}
BUILD_DIR="build"
MAIN_PATH="cmd/server"

echo -e "${GREEN}Building ${APP_NAME} v${VERSION}...${NC}"

# Clean previous builds
if [ -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}Cleaning previous build...${NC}"
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"

# Build for different platforms
build_for_platform() {
    local os=$1
    local arch=$2
    local output_name="${APP_NAME}"
    
    if [ "$os" = "windows" ]; then
        output_name="${output_name}.exe"
    fi
    
    echo -e "${GREEN}Building for ${os}/${arch}...${NC}"
    
    GOOS=$os GOARCH=$arch go build \
        -ldflags "-X main.Version=${VERSION} -X main.BuildTime=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        -o "${BUILD_DIR}/${os}-${arch}/${output_name}" \
        "./${MAIN_PATH}"
    
    echo -e "${GREEN}✓ Built ${BUILD_DIR}/${os}-${arch}/${output_name}${NC}"
}

# Build for common platforms
build_for_platform "linux" "amd64"
build_for_platform "linux" "arm64"
build_for_platform "darwin" "amd64"
build_for_platform "darwin" "arm64"
build_for_platform "windows" "amd64"

echo -e "${GREEN}Build completed successfully!${NC}"
echo -e "${YELLOW}Artifacts available in: ${BUILD_DIR}/${NC}" 