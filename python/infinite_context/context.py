"""
High-level InfiniteContext API.

This provides a complete, easy-to-use interface that combines:
- HAT indexing (from Rust)
- Sentence-transformers embeddings
- Ollama integration

Example:
    ctx = InfiniteContext(model="gemma3:1b")
    ctx.add("The quantum experiment showed 47% improvement")
    response = ctx.chat("What were the results?")
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

from ._core import HatIndex, InfiniteContext as _RustCtx


class InfiniteContext:
    """
    Give any local LLM unlimited memory.

    This class provides a complete, batteries-included interface for infinite
    context with local LLMs. It handles:
    - Embedding text with sentence-transformers
    - Indexing in HAT for O(log n) retrieval
    - Context injection into Ollama prompts
    - Memory persistence

    Example:
        ctx = InfiniteContext(model="gemma3:1b")

        # Add information
        ctx.add("My name is Alex and I work on quantum computing.")
        ctx.add("The latest experiment showed 47% improvement.")

        # Chat - context is automatically retrieved
        response = ctx.chat("What were the experiment results?")

        # Save for later
        ctx.save("memory.hat")
    """

    def __init__(
        self,
        model: str = "gemma3:1b",
        embedding_model: str = "all-MiniLM-L6-v2",
        ollama_host: str = "http://localhost:11434",
        memory_path: Optional[str] = None,
        beam_width: int = 10,
    ):
        """
        Initialize InfiniteContext.

        Args:
            model: Ollama model name (gemma3:1b, phi4, llama3.2, etc.)
            embedding_model: Sentence-transformers model for embeddings
            ollama_host: Ollama API host
            memory_path: Optional path to load existing memory from
            beam_width: HAT beam width (higher = more thorough search)
        """
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "sentence-transformers is required. Install with:\n"
                "  pip install sentence-transformers"
            )

        self.model = model
        self.ollama_host = ollama_host
        self.memory_path = memory_path

        # Load embedding model
        self.embedder = SentenceTransformer(embedding_model)
        self.dims = self.embedder.get_sentence_embedding_dimension()

        # Initialize or load Rust context
        if memory_path and Path(memory_path).exists():
            self._ctx = _RustCtx.load(memory_path, model, ollama_host)
            self._load_texts(memory_path)
        else:
            self._ctx = _RustCtx(model, self.dims, ollama_host)
            self._texts: List[str] = []

        self._auto_save = memory_path is not None

    def _load_texts(self, path: str):
        """Load texts from sidecar file."""
        texts_path = f"{path}.texts.json"
        if Path(texts_path).exists():
            with open(texts_path) as f:
                self._texts = json.load(f)
        else:
            self._texts = []

    def add(self, text: str, new_session: bool = False, new_topic: bool = False) -> int:
        """
        Add text to memory.

        Args:
            text: The text to remember
            new_session: Start a new conversation session
            new_topic: Start a new topic within current session

        Returns:
            The index of the added text
        """
        if new_session:
            self._ctx.new_session()
        if new_topic:
            self._ctx.new_topic()

        # Embed and add
        embedding = self.embedder.encode(text, normalize_embeddings=True).tolist()
        idx = self._ctx.add(embedding, text)
        self._texts.append(text)

        if self._auto_save and self.memory_path:
            self.save(self.memory_path)

        return idx

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve relevant context for a query.

        Args:
            query: The query text
            k: Number of results to return

        Returns:
            List of (text, similarity_score) tuples
        """
        query_embedding = self.embedder.encode(query, normalize_embeddings=True).tolist()
        return self._ctx.retrieve(query_embedding, k)

    def chat(self, message: str, k: int = 5, include_context: bool = True) -> str:
        """
        Chat with the model using HAT-retrieved context.

        Args:
            message: Your message/question
            k: Number of context chunks to retrieve
            include_context: Whether to include retrieved context

        Returns:
            The model's response
        """
        query_embedding = self.embedder.encode(message, normalize_embeddings=True).tolist()

        if include_context and len(self) > 0:
            response = self._ctx.chat(query_embedding, message, k)
        else:
            response = self._ctx.generate(message)

        # Add the exchange to memory
        self.add(f"User: {message}")
        self.add(f"Assistant: {response}")

        return response

    def generate(self, prompt: str) -> str:
        """Generate a response without memory retrieval."""
        return self._ctx.generate(prompt)

    def save(self, path: Optional[str] = None):
        """
        Save memory to disk.

        Args:
            path: File path (uses initialization path if not specified)
        """
        path = path or self.memory_path
        if not path:
            raise ValueError("No path specified")

        self._ctx.save(path)

        # Save texts sidecar
        texts_path = f"{path}.texts.json"
        with open(texts_path, 'w') as f:
            json.dump(self._texts, f)

    @classmethod
    def load(
        cls,
        path: str,
        model: str = "gemma3:1b",
        embedding_model: str = "all-MiniLM-L6-v2",
        ollama_host: str = "http://localhost:11434",
    ) -> "InfiniteContext":
        """
        Load memory from disk.

        Args:
            path: Path to the .hat memory file
            model: Ollama model to use
            embedding_model: Sentence-transformers model
            ollama_host: Ollama API host

        Returns:
            InfiniteContext with loaded memory
        """
        return cls(
            model=model,
            embedding_model=embedding_model,
            ollama_host=ollama_host,
            memory_path=path,
        )

    def stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        s = self._ctx.stats()
        return {
            'sessions': s.sessions,
            'documents': s.documents,
            'chunks': s.chunks,
            'model': self.model,
        }

    def new_session(self):
        """Start a new conversation session."""
        self._ctx.new_session()

    def new_topic(self):
        """Start a new topic within the current session."""
        self._ctx.new_topic()

    def is_ollama_available(self) -> bool:
        """Check if Ollama is available."""
        return self._ctx.is_ollama_available()

    def list_models(self) -> List[str]:
        """List available Ollama models."""
        return self._ctx.list_models()

    def __len__(self) -> int:
        return len(self._texts)

    def __repr__(self) -> str:
        s = self._ctx.stats()
        return f"InfiniteContext(model='{self.model}', chunks={s.chunks})"
