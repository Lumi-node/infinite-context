#!/bin/bash
# Infinite Context - One-Line Installer
#
# Usage: curl -sSL https://raw.githubusercontent.com/Lumi-node/infinite-context/main/install.sh | bash
#
# This script:
#   1. Detects your OS
#   2. Installs dependencies if needed
#   3. Pulls the Docker image OR builds from source
#   4. Verifies everything works

set -e

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║       INFINITE CONTEXT - One-Line Installer                      ║"
echo "║       Give any local LLM unlimited memory                        ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

success() { echo -e "${GREEN}✓${NC} $1"; }
warn() { echo -e "${YELLOW}!${NC} $1"; }
error() { echo -e "${RED}✗${NC} $1"; exit 1; }

# Detect OS
OS="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
fi

echo "Detected OS: $OS"
echo

# Check for Docker
echo "Checking for Docker..."
if command -v docker &> /dev/null; then
    success "Docker found: $(docker --version)"
    USE_DOCKER=true
else
    warn "Docker not found - will try building from source"
    USE_DOCKER=false
fi

# Check for Ollama
echo "Checking for Ollama..."
if command -v ollama &> /dev/null; then
    success "Ollama found"

    # Check if running
    if curl -s http://localhost:11434/api/tags &> /dev/null; then
        success "Ollama is running"
    else
        warn "Ollama installed but not running"
        echo "  Start it with: ollama serve"
    fi
else
    warn "Ollama not found"
    echo
    echo "  Install Ollama from: https://ollama.com"
    echo "  Or run: curl -fsSL https://ollama.com/install.sh | sh"
    echo
fi

echo

# Method 1: Docker (preferred)
if [ "$USE_DOCKER" = true ]; then
    echo "═══════════════════════════════════════════════════════════════════"
    echo "  Installing via Docker (recommended)"
    echo "═══════════════════════════════════════════════════════════════════"
    echo

    echo "Pulling image..."
    docker pull andrewmang/infinite-context:latest || {
        warn "Docker pull failed - image may not be published yet"
        warn "Falling back to building from source..."
        USE_DOCKER=false
    }

    if [ "$USE_DOCKER" = true ]; then
        success "Docker image ready!"
        echo
        echo "═══════════════════════════════════════════════════════════════════"
        echo "  INSTALLATION COMPLETE!"
        echo "═══════════════════════════════════════════════════════════════════"
        echo
        echo "  Run the demo:"
        echo "    docker run -it --rm andrewmang/infinite-context"
        echo
        echo "  Run benchmark:"
        echo "    docker run -it --rm andrewmang/infinite-context infinite-context bench --chunks 100000"
        echo
        echo "  Chat with Ollama (make sure ollama serve is running):"
        echo "    docker run -it --rm --network host andrewmang/infinite-context infinite-context chat"
        echo
        exit 0
    fi
fi

# Method 2: Build from source
echo "═══════════════════════════════════════════════════════════════════"
echo "  Installing from source"
echo "═══════════════════════════════════════════════════════════════════"
echo

# Check for Rust
if ! command -v cargo &> /dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi
success "Rust found: $(cargo --version)"

# Check for Python
if ! command -v python3 &> /dev/null; then
    error "Python 3 not found. Please install Python 3.9+ first."
fi
success "Python found: $(python3 --version)"

# Clone repo
INSTALL_DIR="$HOME/.infinite-context"
if [ -d "$INSTALL_DIR" ]; then
    echo "Updating existing installation..."
    cd "$INSTALL_DIR"
    git pull
else
    echo "Cloning repository..."
    git clone https://github.com/Lumi-node/infinite-context.git "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# Build Rust CLI
echo "Building Rust CLI..."
cargo build --release

# Create symlink
BINARY="$INSTALL_DIR/target/release/infinite-context"
if [ -f "$BINARY" ]; then
    sudo ln -sf "$BINARY" /usr/local/bin/infinite-context 2>/dev/null || {
        # If sudo fails, try user bin
        mkdir -p "$HOME/.local/bin"
        ln -sf "$BINARY" "$HOME/.local/bin/infinite-context"
        echo "  Added to ~/.local/bin (make sure it's in your PATH)"
    }
    success "CLI installed: infinite-context"
fi

# Install Python package
echo "Installing Python package..."
python3 -m pip install --user sentence-transformers

# Create venv and install
python3 -m venv "$INSTALL_DIR/.venv"
source "$INSTALL_DIR/.venv/bin/activate"
pip install maturin sentence-transformers
maturin develop --release

echo
echo "═══════════════════════════════════════════════════════════════════"
echo "  INSTALLATION COMPLETE!"
echo "═══════════════════════════════════════════════════════════════════"
echo
echo "  Run benchmark:"
echo "    infinite-context bench --chunks 100000"
echo
echo "  Chat with Ollama:"
echo "    infinite-context chat --model gemma3:1b"
echo
echo "  Run the realistic demo:"
echo "    cd $INSTALL_DIR && source .venv/bin/activate"
echo "    python examples/realistic_demo.py"
echo
success "Enjoy unlimited memory!"
