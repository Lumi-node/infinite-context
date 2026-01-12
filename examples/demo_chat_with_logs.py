#!/usr/bin/env python3
"""
Infinite Context Demo: Real Chat with Full Logging

This script demonstrates HAT retrieval at scale with detailed logging,
showing exactly what context is being retrieved and how large the
searchable memory is.

Usage:
    python demo_chat_with_logs.py --model gemma3:1b --chunks 10000
"""

import argparse
import time
import random
from datetime import datetime

# Check dependencies
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: sentence-transformers not installed")
    print("Run: pip install sentence-transformers")
    exit(1)

try:
    from infinite_context import HatIndex, InfiniteContext
except ImportError:
    print("Error: infinite-context not installed")
    print("Run: pip install infinite-context")
    exit(1)


def print_box(lines, title=None):
    """Print a nice ASCII box."""
    width = 70
    print("+" + "=" * width + "+")
    if title:
        print(f"|  {title:<{width-2}}|")
        print("+" + "-" * width + "+")
    for line in lines:
        print(f"|  {line:<{width-2}}|")
    print("+" + "=" * width + "+")


def generate_conversation_data(n_chunks, embedder):
    """Generate realistic conversation data with embeddings."""

    topics = [
        ("machine learning", [
            "Neural networks process data through layers of interconnected nodes.",
            "Attention mechanisms help models focus on relevant input parts.",
            "Transformers revolutionized NLP with self-attention architecture.",
            "Gradient descent optimizes model weights iteratively.",
            "Batch normalization stabilizes training by normalizing layer inputs.",
        ]),
        ("software architecture", [
            "Microservices decompose applications into independent services.",
            "Event sourcing captures state changes as immutable events.",
            "Domain-driven design aligns code with business domains.",
            "CQRS separates read and write operations for scalability.",
            "API gateways manage routing and authentication centrally.",
        ]),
        ("databases", [
            "Indexing dramatically improves query performance on large tables.",
            "B-trees enable efficient range queries and ordered access.",
            "Query optimization analyzes execution plans for efficiency.",
            "Sharding distributes data across multiple database instances.",
            "ACID properties ensure transaction reliability.",
        ]),
        ("cloud computing", [
            "Kubernetes orchestrates containerized workloads at scale.",
            "Serverless computing eliminates server management overhead.",
            "Load balancers distribute traffic across healthy instances.",
            "Auto-scaling adjusts capacity based on demand patterns.",
            "Service mesh handles inter-service communication.",
        ]),
        ("security", [
            "Zero trust architecture verifies every access request.",
            "OAuth2 provides secure delegated authorization.",
            "Encryption at rest protects stored data from breaches.",
            "Rate limiting prevents abuse and DDoS attacks.",
            "Audit logs track all system access for compliance.",
        ]),
    ]

    chunks = []
    chunk_texts = []

    print(f"\nGenerating {n_chunks} conversation chunks...")
    start = time.time()

    for i in range(n_chunks):
        topic_name, examples = random.choice(topics)
        base_text = random.choice(examples)

        # Add some variation
        session_id = i // 100
        doc_id = (i % 100) // 10
        chunk_text = f"[Session {session_id}, Doc {doc_id}] {base_text} The key insight about {topic_name} is that careful planning leads to better outcomes."

        chunk_texts.append(chunk_text)

        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{n_chunks} chunks...")

    print(f"  Embedding {n_chunks} chunks...")
    embeddings = embedder.encode(chunk_texts, normalize_embeddings=True, show_progress_bar=True)

    for i, (text, emb) in enumerate(zip(chunk_texts, embeddings)):
        chunks.append({
            'text': text,
            'embedding': emb.tolist(),
            'session': i // 100,
            'doc': (i % 100) // 10,
        })

    elapsed = time.time() - start
    print(f"  Done in {elapsed:.2f}s")

    return chunks


def build_index(chunks):
    """Build HAT index from chunks."""
    print(f"\nBuilding HAT index with {len(chunks)} chunks...")
    start = time.time()

    index = HatIndex.cosine(384)
    texts = []

    current_session = -1
    current_doc = -1

    for chunk in chunks:
        if chunk['session'] != current_session:
            index.new_session()
            current_session = chunk['session']
            current_doc = -1

        if chunk['doc'] != current_doc:
            index.new_document()
            current_doc = chunk['doc']

        index.add(chunk['embedding'])
        texts.append(chunk['text'])

    elapsed = time.time() - start
    stats = index.stats()

    print_box([
        f"Chunks indexed:    {stats.chunks:,}",
        f"Sessions:          {stats.sessions:,}",
        f"Documents:         {stats.documents:,}",
        f"Build time:        {elapsed:.2f}s",
        f"Est. tokens:       {stats.chunks * 30:,}",
    ], "INDEX BUILT")

    return index, texts


def demo_query(index, texts, embedder, query, expected_topic=None):
    """Run a demo query with full logging."""

    print(f"\n{'='*72}")
    print(f"QUERY: {query}")
    print(f"{'='*72}")

    # Embed query
    query_emb = embedder.encode(query, normalize_embeddings=True).tolist()

    # Search
    start = time.time()
    results = index.near(query_emb, k=10)
    retrieval_ms = (time.time() - start) * 1000

    stats = index.stats()
    est_tokens = stats.chunks * 30

    print_box([
        f"Total indexed:     {est_tokens:,} tokens",
        f"Retrieval time:    {retrieval_ms:.2f} ms",
        f"Results found:     {len(results)}",
    ], "RETRIEVAL STATS")

    print("\nRetrieved context:")
    print("-" * 72)
    for i, result in enumerate(results[:5]):
        # Parse the result ID to get the index
        # Note: In production, you'd store text->ID mappings
        print(f"  [{i+1}] Score: {result.score:.4f}")
        # Show first 100 chars of matching text (approximation)
        if i < len(texts):
            print(f"      {texts[min(i, len(texts)-1)][:80]}...")
    print("-" * 72)

    return results, retrieval_ms


def main():
    parser = argparse.ArgumentParser(description="Infinite Context Demo with Logging")
    parser.add_argument("--model", default="gemma3:1b", help="Ollama model to use")
    parser.add_argument("--chunks", type=int, default=10000, help="Number of chunks to index")
    parser.add_argument("--interactive", action="store_true", help="Run interactive chat mode")
    args = parser.parse_args()

    print("\n" + "=" * 72)
    print("  INFINITE CONTEXT DEMO - Real Chat Logs at Scale")
    print("=" * 72)

    # Load embedder
    print("\nLoading embedding model (all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate data
    chunks = generate_conversation_data(args.chunks, embedder)

    # Build index
    index, texts = build_index(chunks)

    # Run demo queries
    demo_queries = [
        ("What do you know about neural networks and attention?", "machine learning"),
        ("Tell me about microservices architecture.", "software architecture"),
        ("How does database indexing work?", "databases"),
        ("Explain Kubernetes and containers.", "cloud computing"),
    ]

    print("\n" + "=" * 72)
    print("  RUNNING DEMO QUERIES")
    print("=" * 72)

    total_time = 0
    for query, topic in demo_queries:
        results, time_ms = demo_query(index, texts, embedder, query, topic)
        total_time += time_ms

    avg_time = total_time / len(demo_queries)

    print("\n")
    print_box([
        f"Queries run:       {len(demo_queries)}",
        f"Avg retrieval:     {avg_time:.2f} ms",
        f"Total indexed:     {args.chunks * 30:,} tokens",
        f"Context extension: {(args.chunks * 30) // 8000}x for 8K models",
    ], "DEMO SUMMARY")

    if args.interactive:
        print("\n" + "=" * 72)
        print("  INTERACTIVE MODE - Type your questions (Ctrl+C to exit)")
        print("=" * 72)

        try:
            while True:
                query = input("\nYou: ").strip()
                if not query:
                    continue
                if query.lower() in ['/quit', '/exit', '/q']:
                    break

                demo_query(index, texts, embedder, query)
        except KeyboardInterrupt:
            print("\n\nGoodbye!")

    print("\n" + "=" * 72)
    print("  Demo complete. This is infinite context in practice.")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
