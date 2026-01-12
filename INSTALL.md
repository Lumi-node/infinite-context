# Installation Guide

Get infinite context running in under 5 minutes.

## Quick Install (From Source)

```bash
git clone https://github.com/Lumi-node/infinite-context
cd infinite-context
pip install maturin sentence-transformers
maturin develop --release
```

## Prerequisites

### 1. Python 3.9+

Check your Python version:
```bash
python3 --version
```

### 2. Ollama (Local LLM Runtime)

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download from [ollama.com](https://ollama.com/download)

### 3. Start Ollama & Pull a Model

```bash
# Start Ollama server
ollama serve

# In another terminal, pull a model (pick one):
ollama pull gemma3:1b     # Small & fast (recommended to start)
ollama pull phi4          # Larger, more capable
ollama pull llama3.2      # Meta's latest
ollama pull mistral       # Good all-rounder
```

## Verify Installation

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags
```

## First Run

```python
from infinite_context import InfiniteContext

# Initialize with your model
ctx = InfiniteContext(model="gemma3:1b")

# Add information to memory
ctx.add("My name is Alex and I work on quantum computing.")

# Chat with infinite memory!
response = ctx.chat("What's my name?")
print(response)
```

---

## Build Rust CLI (Optional - Benchmarks Only)

```bash
cargo build --release
./target/release/infinite-context bench --chunks 100000
```

> **Note:** The Rust CLI is for benchmarking HAT performance. For actual chat with memory, use the Python API above.

## Troubleshooting

### "Cannot connect to Ollama"

Make sure Ollama is running:
```bash
ollama serve
```

### "Model not found"

Pull the model first:
```bash
ollama pull gemma3:1b
```

### "sentence-transformers not found"

Install dependencies:
```bash
pip install sentence-transformers
```

### Slow first query

The embedding model downloads on first use (~90MB). Subsequent queries are fast.

### Out of memory

Try a smaller model:
```bash
ollama pull gemma3:1b  # Uses ~1GB RAM
```

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4 GB | 8+ GB |
| Storage | 2 GB | 10+ GB |
| Python | 3.9 | 3.11+ |
| GPU | Not required | Nice to have |

The embedding model uses CPU by default. For faster embeddings with GPU:
```bash
pip install sentence-transformers[gpu]
```

---

## Docker

```bash
docker run -it --rm --network host andrewmang/infinite-context
```

---

## Next Steps

1. [Quickstart Example](examples/quickstart.py)
2. [Benchmark Demo](examples/benchmark_demo.py)
3. [CLI Reference](README.md#cli-reference)
