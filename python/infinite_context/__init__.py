"""
Infinite Context - Give any local LLM unlimited memory.

11M+ tokens | 28ms latency | 100% accuracy

Quick Start:
    from infinite_context import InfiniteContext

    ctx = InfiniteContext(model="gemma3:1b")
    ctx.add("Important information to remember")
    response = ctx.chat("What did you remember?")

Low-Level API:
    from infinite_context import HatIndex
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    index = HatIndex.cosine(384)
    index.add(embedder.encode("text"))
    results = index.near(embedder.encode("query"), k=10)
"""

__version__ = "0.1.0"

# Import from Rust core
from ._core import (
    HatIndex,
    HatConfig,
    HatStats,
    SearchResult,
    InfiniteContext as _RustInfiniteContext,
)

# High-level Python wrapper for easier usage
from .context import InfiniteContext

__all__ = [
    "InfiniteContext",
    "HatIndex",
    "HatConfig",
    "HatStats",
    "SearchResult",
]


def cli_main():
    """Entry point for the CLI."""
    import subprocess
    import sys
    import os

    # Find the Rust binary
    # When installed via pip, the binary should be in the same bin directory
    bin_name = "infinite-context"
    if sys.platform == "win32":
        bin_name += ".exe"

    # Try to find it in PATH or alongside Python
    import shutil
    binary = shutil.which(bin_name)

    if binary:
        result = subprocess.run([binary] + sys.argv[1:])
        sys.exit(result.returncode)
    else:
        print("Error: infinite-context binary not found.")
        print("The Rust CLI binary should be installed alongside the Python package.")
        print("Try: cargo install --path . (from the repo root)")
        sys.exit(1)
