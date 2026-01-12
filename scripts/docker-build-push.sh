#!/bin/bash
# Build and push infinite-context to Docker Hub
#
# Usage: ./scripts/docker-build-push.sh [username]
#
# Prerequisites:
#   docker login

set -e

USERNAME="${1:-andrewmang}"
IMAGE_NAME="infinite-context"
VERSION="0.1.0"

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  Building Infinite Context Docker Image                       ║"
echo "╠═══════════════════════════════════════════════════════════════╣"
echo "║  Username: $USERNAME"
echo "║  Image:    $USERNAME/$IMAGE_NAME:$VERSION"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo

cd "$(dirname "$0")/.."

echo "Step 1: Building image..."
docker build -t "$USERNAME/$IMAGE_NAME:$VERSION" -t "$USERNAME/$IMAGE_NAME:latest" .

echo
echo "Step 2: Testing image..."
docker run --rm "$USERNAME/$IMAGE_NAME:$VERSION" infinite-context --help

echo
echo "Step 3: Pushing to Docker Hub..."
docker push "$USERNAME/$IMAGE_NAME:$VERSION"
docker push "$USERNAME/$IMAGE_NAME:latest"

echo
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  SUCCESS! Image pushed to Docker Hub                          ║"
echo "╠═══════════════════════════════════════════════════════════════╣"
echo "║  Pull:  docker pull $USERNAME/$IMAGE_NAME"
echo "║  Run:   docker run -it $USERNAME/$IMAGE_NAME"
echo "╚═══════════════════════════════════════════════════════════════╝"
