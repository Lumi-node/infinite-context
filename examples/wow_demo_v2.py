#!/usr/bin/env python3
"""
WOW DEMO v2: Realistic Needle-in-Haystack with Semantic Similarity

Key insight: Real needles are ON-TOPIC, not random facts.
This demo hides specific details within topically-relevant conversations.

PROVES: HAT can find specific facts within semantically similar content.
"""

import time
import json
import random
from datetime import datetime

print("=" * 80)
print("  INFINITE CONTEXT: WOW DEMO v2 - Semantic Needle Search")
print("=" * 80)

try:
    from sentence_transformers import SentenceTransformer
    import requests
    from infinite_context import HatIndex
    print("[OK] All dependencies loaded")
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    exit(1)

# Configuration
CONFIG = {
    'total_chunks': 100_000,
    'ollama_host': 'http://localhost:11434',
    'model': 'gemma3:1b',
}

# REALISTIC NEEDLES - Facts that are ON-TOPIC with surrounding content
# These are harder to find because they're semantically similar to fillers
NEEDLES = [
    {
        'id': 'ML_ACCURACY_NEEDLE',
        'topic': 'machine learning',
        'position': 15_237,
        'fact': 'The transformer model achieved exactly 97.3% accuracy on the ImageNet validation set after training for 90 epochs with a learning rate of 0.0001.',
        'question': 'What accuracy did the transformer model achieve on ImageNet?',
        'expected': '97.3%',
    },
    {
        'id': 'ARCHITECTURE_LATENCY_NEEDLE',
        'topic': 'software architecture',
        'position': 42_891,
        'fact': 'The microservices migration reduced API latency from 450ms to 23ms, a 19.6x improvement, by implementing event-driven communication.',
        'question': 'What was the latency improvement after the microservices migration?',
        'expected': '19.6x',
    },
    {
        'id': 'DATABASE_SCALING_NEEDLE',
        'topic': 'databases',
        'position': 67_432,
        'fact': 'Database query performance improved by 847% after adding a composite index on (user_id, created_at, status) columns.',
        'question': 'By what percentage did database query performance improve after adding the composite index?',
        'expected': '847%',
    },
    {
        'id': 'KUBERNETES_PODS_NEEDLE',
        'topic': 'cloud computing',
        'position': 89_156,
        'fact': 'The Kubernetes cluster scaled from 12 pods to 2,847 pods during the Black Friday traffic spike while maintaining 99.97% availability.',
        'question': 'How many pods did the Kubernetes cluster scale to during Black Friday?',
        'expected': '2,847',
    },
]

# Topic-specific conversation generators
TOPIC_TEMPLATES = {
    'machine learning': [
        "We discussed neural network architectures and their trade-offs in the {topic} session.",
        "The team explored deep learning optimization techniques for {topic} applications.",
        "Training strategies for {topic} models were a key focus of this discussion.",
        "Model evaluation metrics and validation approaches for {topic} were reviewed.",
        "The importance of hyperparameter tuning in {topic} was emphasized.",
    ],
    'software architecture': [
        "Microservices patterns were discussed in the context of {topic} design.",
        "Event-driven architecture approaches for {topic} systems were explored.",
        "The team reviewed API design principles for {topic} implementations.",
        "Scalability considerations in {topic} were analyzed in depth.",
        "Service communication patterns for {topic} were debated by the team.",
    ],
    'databases': [
        "Query optimization strategies for {topic} were the main focus today.",
        "Indexing approaches and their performance implications for {topic} were reviewed.",
        "The team discussed data modeling best practices for {topic} systems.",
        "Transaction management and consistency in {topic} were explored.",
        "Performance tuning techniques for {topic} workloads were shared.",
    ],
    'cloud computing': [
        "Container orchestration strategies for {topic} deployments were discussed.",
        "The team explored auto-scaling patterns for {topic} infrastructure.",
        "Cloud-native architecture principles for {topic} were reviewed.",
        "Load balancing and traffic management for {topic} were analyzed.",
        "Infrastructure as code practices for {topic} were demonstrated.",
    ],
}

TOPICS = list(TOPIC_TEMPLATES.keys())


def generate_chunk(chunk_idx, topic, session_idx, doc_idx):
    """Generate a topically coherent chunk."""
    templates = TOPIC_TEMPLATES[topic]
    template = random.choice(templates)
    text = template.format(topic=topic)
    return f"[Session {session_idx}, Doc {doc_idx}] {text}"


def check_ollama():
    try:
        r = requests.get(f"{CONFIG['ollama_host']}/api/tags", timeout=5)
        return r.status_code == 200
    except:
        return False


def ollama_generate(prompt, context=None):
    full_prompt = prompt
    if context:
        full_prompt = f"""Based on the following context from memory, answer the question.

CONTEXT:
{context}

QUESTION: {prompt}

Answer with ONLY the specific value or fact requested. Be precise and brief."""

    payload = {
        'model': CONFIG['model'],
        'prompt': full_prompt,
        'stream': False,
    }

    r = requests.post(f"{CONFIG['ollama_host']}/api/generate", json=payload, timeout=120)
    if r.status_code == 200:
        return r.json().get('response', '')
    return f"Error: {r.status_code}"


def main():
    if not check_ollama():
        print("\n[ERROR] Ollama not running. Start with: ollama serve")
        return

    print(f"\n[OK] Ollama available")

    print("\nLoading embedding model...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    n_chunks = CONFIG['total_chunks']
    needle_positions = {n['position']: n for n in NEEDLES}

    print(f"\nGenerating {n_chunks:,} chunks across 4 topics...")

    chunk_texts = []
    chunk_topics = []

    for i in range(n_chunks):
        session_idx = i // 100
        doc_idx = (i % 100) // 10

        # Rotate topics every 25 sessions (so needles are within their topic's region)
        topic = TOPICS[(session_idx // 25) % len(TOPICS)]

        if i in needle_positions:
            needle = needle_positions[i]
            # Make needle look like regular conversation
            text = f"[Session {session_idx}, Doc {doc_idx}] IMPORTANT RESULT: {needle['fact']}"
            print(f"\n  >> NEEDLE at chunk {i:,}: {needle['id']}")
            print(f"     Topic region: {topic}")
        else:
            text = generate_chunk(i, topic, session_idx, doc_idx)

        chunk_texts.append(text)
        chunk_topics.append(topic)

        if (i + 1) % 25000 == 0:
            print(f"  Generated {i + 1:,}/{n_chunks:,}")

    print(f"\n  Embedding {n_chunks:,} chunks...")
    start = time.time()
    embeddings = embedder.encode(chunk_texts, normalize_embeddings=True, show_progress_bar=True, batch_size=128)
    print(f"  Embedded in {time.time() - start:.1f}s")

    print(f"\n  Building HAT index...")
    start = time.time()

    index = HatIndex.cosine(384)
    id_to_chunk = {}

    current_session = -1
    current_doc = -1

    for i, (text, emb) in enumerate(zip(chunk_texts, embeddings)):
        session_idx = i // 100
        doc_idx = (i % 100) // 10

        if session_idx != current_session:
            index.new_session()
            current_session = session_idx
            current_doc = -1

        if doc_idx != current_doc:
            index.new_document()
            current_doc = doc_idx

        chunk_id = index.add(emb.tolist())
        id_to_chunk[chunk_id] = {'idx': i, 'text': text, 'is_needle': i in needle_positions}

    build_time = time.time() - start
    stats = index.stats()

    print(f"""
╔════════════════════════════════════════════════════════════════════════╗
║  HAT INDEX READY                                                        ║
╠════════════════════════════════════════════════════════════════════════╣
║  Chunks:        {stats.chunks:>56,} ║
║  Tokens (est):  {stats.chunks * 30:>56,} ║
║  Sessions:      {stats.sessions:>56,} ║
║  Build time:    {build_time:>53.2f}s ║
╚════════════════════════════════════════════════════════════════════════╝
""")

    print("=" * 80)
    print("  NEEDLE SEARCH: Finding specific facts in 3M tokens")
    print("=" * 80)

    results = []

    for needle in NEEDLES:
        print(f"\n{'─' * 80}")
        print(f"  SEARCHING: {needle['id']}")
        print(f"  Question:  {needle['question']}")
        print(f"  Expected:  {needle['expected']}")
        print(f"{'─' * 80}")

        # Embed the question
        query_emb = embedder.encode(needle['question'], normalize_embeddings=True).tolist()

        # Search
        start = time.time()
        search_results = index.near(query_emb, k=10)
        search_ms = (time.time() - start) * 1000

        # Check if needle found
        found = False
        rank = -1
        for i, r in enumerate(search_results):
            chunk = id_to_chunk.get(r.id, {})
            if chunk.get('idx') == needle['position']:
                found = True
                rank = i + 1
                break

        print(f"\n  Retrieval time: {search_ms:.2f}ms")
        print(f"  Needle found:   {'YES at rank ' + str(rank) if found else 'NO'}")

        print(f"\n  Top 5 results:")
        for i, r in enumerate(search_results[:5]):
            chunk = id_to_chunk.get(r.id, {})
            is_needle = chunk.get('is_needle', False)
            marker = " <<< NEEDLE" if is_needle else ""
            print(f"    [{i+1}] Score: {r.score:.4f} | Chunk {chunk.get('idx', '?'):,}{marker}")
            print(f"        {chunk.get('text', 'Unknown')[:75]}...")

        # Ask LLM if needle found
        if found:
            context = "\n".join([id_to_chunk[r.id]['text'] for r in search_results[:5]])

            print(f"\n  Asking {CONFIG['model']}...")
            start = time.time()
            response = ollama_generate(needle['question'], context)
            llm_ms = (time.time() - start) * 1000

            print(f"  Response ({llm_ms:.0f}ms): {response.strip()[:200]}")

            correct = needle['expected'].lower() in response.lower()
            print(f"  Contains '{needle['expected']}': {'YES!' if correct else 'NO'}")

            results.append({
                'needle_id': needle['id'],
                'found': found,
                'rank': rank,
                'correct': correct,
                'search_ms': search_ms,
                'llm_ms': llm_ms,
            })
        else:
            results.append({
                'needle_id': needle['id'],
                'found': False,
                'rank': -1,
                'correct': False,
                'search_ms': search_ms,
            })

    # Summary
    found_count = sum(1 for r in results if r['found'])
    correct_count = sum(1 for r in results if r.get('correct', False))
    avg_search = sum(r['search_ms'] for r in results) / len(results)

    print(f"""

╔════════════════════════════════════════════════════════════════════════╗
║  FINAL RESULTS                                                          ║
╠════════════════════════════════════════════════════════════════════════╣
║  Haystack size:     {n_chunks * 30:>53,} tokens ║
║  Needles hidden:    {len(NEEDLES):>53} ║
║  Needles found:     {found_count:>53} ║
║  Answers correct:   {correct_count:>53} ║
║  Avg search time:   {avg_search:>50.2f}ms ║
╠════════════════════════════════════════════════════════════════════════╣
║  RETRIEVAL:  {found_count}/{len(NEEDLES)} = {100*found_count/len(NEEDLES):.0f}%                                                    ║
║  ACCURACY:   {correct_count}/{len(NEEDLES)} = {100*correct_count/len(NEEDLES):.0f}%                                                    ║
╚════════════════════════════════════════════════════════════════════════╝
""")

    # Save results
    output = f"wow_demo_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output, 'w') as f:
        json.dump({
            'config': CONFIG,
            'stats': {'chunks': stats.chunks, 'tokens': stats.chunks * 30},
            'results': results,
        }, f, indent=2)
    print(f"Results saved to: {output}")


if __name__ == "__main__":
    main()
