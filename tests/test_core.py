"""Tests for infinite-context core functionality."""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from infinite_context.core import HATIndex


class TestHATIndex:
    """Tests for the HAT index."""

    def test_create_index(self):
        """Test creating a new index."""
        index = HATIndex(dims=384)
        assert len(index) == 0
        assert index.dims == 384

    def test_add_single(self):
        """Test adding a single item."""
        index = HATIndex(dims=4)
        embedding = np.array([1.0, 0.0, 0.0, 0.0])
        chunk_id = index.add(embedding, "test message")

        assert chunk_id == 0
        assert len(index) == 1

    def test_add_multiple(self):
        """Test adding multiple items."""
        index = HATIndex(dims=4)

        for i in range(10):
            embedding = np.random.randn(4)
            index.add(embedding, f"message {i}")

        assert len(index) == 10

    def test_query_basic(self):
        """Test basic query functionality."""
        index = HATIndex(dims=4, beam_width=5)

        # Add some items
        embeddings = [
            (np.array([1.0, 0.0, 0.0, 0.0]), "about cats"),
            (np.array([0.9, 0.1, 0.0, 0.0]), "more about cats"),
            (np.array([0.0, 1.0, 0.0, 0.0]), "about dogs"),
            (np.array([0.0, 0.9, 0.1, 0.0]), "more about dogs"),
        ]

        for emb, text in embeddings:
            index.add(emb, text)

        # Query for cats
        query = np.array([1.0, 0.0, 0.0, 0.0])
        results = index.query(query, k=2)

        assert len(results) == 2
        assert "cats" in results[0]['text']

    def test_sessions(self):
        """Test session management."""
        index = HATIndex(dims=4)

        index.new_session()
        index.add(np.random.randn(4), "session 1 message 1")
        index.add(np.random.randn(4), "session 1 message 2")

        index.new_session()
        index.add(np.random.randn(4), "session 2 message 1")

        stats = index.stats()
        assert stats['sessions'] == 2
        assert stats['chunks'] == 3

    def test_documents(self):
        """Test document/topic management."""
        index = HATIndex(dims=4)

        index.new_document()
        index.add(np.random.randn(4), "topic 1 message 1")

        index.new_document()
        index.add(np.random.randn(4), "topic 2 message 1")

        stats = index.stats()
        assert stats['documents'] == 2

    def test_save_load(self):
        """Test persistence."""
        index = HATIndex(dims=4)

        # Add some data
        for i in range(5):
            index.add(np.random.randn(4), f"message {i}")

        # Save
        with tempfile.NamedTemporaryFile(suffix='.hat', delete=False) as f:
            path = Path(f.name)

        try:
            index.save(path)

            # Load
            loaded = HATIndex.load(path)

            assert len(loaded) == len(index)
            assert loaded.dims == index.dims
            assert loaded.stats() == index.stats()
        finally:
            path.unlink()

    def test_empty_query(self):
        """Test querying empty index."""
        index = HATIndex(dims=4)
        results = index.query(np.random.randn(4), k=5)
        assert results == []

    def test_retrieval_accuracy(self):
        """Test that retrieval finds the right items."""
        np.random.seed(42)
        index = HATIndex(dims=32, beam_width=10)

        # Create distinct clusters
        clusters = {
            'science': np.array([1.0] + [0.0] * 31),
            'sports': np.array([0.0, 1.0] + [0.0] * 30),
            'music': np.array([0.0, 0.0, 1.0] + [0.0] * 29),
        }

        # Add messages for each cluster
        for topic, base in clusters.items():
            index.new_document()
            for i in range(10):
                # Add noise but keep it close to cluster center
                emb = base + np.random.randn(32) * 0.1
                emb = emb / np.linalg.norm(emb)
                index.add(emb, f"{topic} message {i}")

        # Query each cluster
        for topic, base in clusters.items():
            results = index.query(base, k=5)
            # All top results should be from the same topic
            for r in results:
                assert topic in r['text'], f"Expected {topic} in {r['text']}"


class TestStats:
    """Tests for statistics tracking."""

    def test_token_counting(self):
        """Test token counting."""
        index = HATIndex(dims=4)
        index.add(np.random.randn(4), "one two three four five", token_count=5)
        index.add(np.random.randn(4), "six seven eight", token_count=3)

        stats = index.stats()
        assert stats['tokens'] == 8

    def test_auto_token_estimation(self):
        """Test automatic token estimation from text."""
        index = HATIndex(dims=4)
        index.add(np.random.randn(4), "one two three four five")

        stats = index.stats()
        assert stats['tokens'] == 5  # Word count as estimate


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
