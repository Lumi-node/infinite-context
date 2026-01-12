# Infinite Context - Give any local LLM unlimited memory
#
# Public image: docker pull andrewmang/infinite-context
# Run: docker run -it andrewmang/infinite-context
#
# 100% conversation retrieval | 0.56ms search | 750K+ tokens

FROM python:3.11-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install maturin with patchelf
RUN pip install --no-cache-dir maturin[patchelf]

WORKDIR /build

# Copy source
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY python ./python
COPY pyproject.toml ./
COPY README.md ./

# Build Rust CLI
RUN cargo build --release --no-default-features

# Build Python wheel
RUN maturin build --release --features python -o /wheels

# =============================================================================
# Final image - minimal runtime
# =============================================================================
FROM python:3.11-slim-bookworm

LABEL maintainer="Andrew Young"
LABEL description="Infinite Context - Give any local LLM unlimited memory. 100% retrieval accuracy, sub-millisecond latency."
LABEL version="0.1.0"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Copy Rust CLI binary
COPY --from=builder /build/target/release/infinite-context /usr/local/bin/

# Install Python package with CPU-only PyTorch (saves ~5GB and massive RAM)
# Install torch CPU-only FIRST to prevent pulling CUDA version
COPY --from=builder /wheels/*.whl /tmp/
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir /tmp/*.whl sentence-transformers numpy && \
    rm -rf /tmp/*.whl /root/.cache/pip

# Copy examples for users to try
COPY examples ./examples
COPY docs ./docs
COPY README.md ./

# Pre-download embedding model (~80MB, tiny)
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); print('Model ready!')"

# Environment
ENV OLLAMA_HOST=http://host.docker.internal:11434
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s \
    CMD infinite-context --help > /dev/null || exit 1

# Show help by default
CMD ["sh", "-c", "echo '╔═══════════════════════════════════════════════════════════════╗' && echo '║  INFINITE CONTEXT - Unlimited LLM Memory                      ║' && echo '╠═══════════════════════════════════════════════════════════════╣' && echo '║  Commands:                                                    ║' && echo '║    infinite-context bench --chunks 100000                     ║' && echo '║    infinite-context chat --model gemma3:1b                    ║' && echo '║    python examples/realistic_demo.py                          ║' && echo '╠═══════════════════════════════════════════════════════════════╣' && echo '║  Make sure Ollama is running on your host machine!            ║' && echo '║    ollama serve                                               ║' && echo '╚═══════════════════════════════════════════════════════════════╝' && exec /bin/bash"]
