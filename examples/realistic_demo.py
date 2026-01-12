#!/usr/bin/env python3
"""
Realistic Demo: HAT Finding Conversations at Scale

This demo simulates REAL usage:
- User has had hundreds of conversations over months
- Each conversation is about a specific topic with real content
- User asks natural questions like "What did we discuss about X?"
- HAT retrieves the right conversations from massive history

NO tricks. NO impossible needle searches. Just realistic usage.
"""

import time
import json
import random
from datetime import datetime

print("=" * 80)
print("  INFINITE CONTEXT: Realistic Usage Demo")
print("  Finding conversations in massive chat history")
print("=" * 80)

try:
    from sentence_transformers import SentenceTransformer
    import requests
    from infinite_context import HatIndex
    print("[OK] Dependencies loaded")
except ImportError as e:
    print(f"[ERROR] {e}")
    exit(1)

# Configuration
CONFIG = {
    'sessions': 500,           # 500 different conversations
    'docs_per_session': 5,     # 5 topic shifts per conversation
    'chunks_per_doc': 10,      # 10 messages per topic
    'ollama_host': 'http://localhost:11434',
    'model': 'gemma3:1b',
}

# REALISTIC CONVERSATION CONTENT
# Each topic has substantive, varied content that a real user might discuss

CONVERSATIONS = {
    'react_debugging': {
        'topic': 'React debugging session',
        'chunks': [
            "I'm getting a 'Cannot read property of undefined' error in my React component when trying to access user.name",
            "The issue is that the user object is null on first render before the API call completes",
            "We fixed it by adding optional chaining: user?.name and also adding a loading state",
            "Another approach is to initialize the user state with default values instead of null",
            "The component now renders correctly and shows a loading spinner while fetching data",
            "We also added error boundaries to catch any remaining edge cases",
            "Performance improved after we memoized the user selector with useMemo",
            "The final solution uses React Query for data fetching which handles loading states automatically",
            "We should document this pattern for the team to avoid similar issues",
            "Next step is to add unit tests for the loading and error states",
        ],
    },
    'python_optimization': {
        'topic': 'Python performance optimization',
        'chunks': [
            "The data processing script is taking 45 minutes to run on 1 million records",
            "Profiling showed that 80% of time is spent in the pandas merge operation",
            "We switched from iterrows() to vectorized operations which gave 10x speedup",
            "Using numpy arrays instead of lists for numerical data reduced memory by 60%",
            "Implemented multiprocessing with Pool to utilize all 8 CPU cores",
            "The bottleneck was actually disk I/O - switching to parquet format helped significantly",
            "After all optimizations the script now runs in 3 minutes instead of 45",
            "We added caching with joblib to avoid recomputing intermediate results",
            "Memory profiling with memory_profiler showed a leak in the loop that we fixed",
            "Final benchmarks: 15x faster, 70% less memory usage",
        ],
    },
    'kubernetes_deployment': {
        'topic': 'Kubernetes deployment troubleshooting',
        'chunks': [
            "The pods keep crashing with OOMKilled status after about 30 minutes",
            "Memory limits were set to 512Mi but the Java app needs at least 1Gi with GC overhead",
            "Increased memory limits to 1.5Gi and set JVM heap to 1Gi with -Xmx1g flag",
            "Also added liveness and readiness probes to detect unhealthy pods faster",
            "The horizontal pod autoscaler wasn't working because metrics-server wasn't installed",
            "After installing metrics-server, HPA correctly scales from 2 to 10 pods based on CPU",
            "We configured PodDisruptionBudget to ensure at least 2 replicas during updates",
            "Rolling updates now work smoothly with maxSurge=1 and maxUnavailable=0",
            "Added resource quotas to the namespace to prevent runaway resource usage",
            "The cluster is now stable with 99.9% uptime over the past week",
        ],
    },
    'database_migration': {
        'topic': 'PostgreSQL migration project',
        'chunks': [
            "Planning migration from MySQL 5.7 to PostgreSQL 14 for better JSON support",
            "The main challenge is converting AUTO_INCREMENT to SERIAL and handling sequences",
            "Created a migration script using pgloader which handles most type conversions automatically",
            "Had to manually fix ENUM types since PostgreSQL handles them differently",
            "Foreign key constraints needed to be recreated after data migration",
            "Performance testing showed PostgreSQL queries are 30% faster for our JSONB operations",
            "The EXPLAIN ANALYZE output helped us identify missing indexes on the new database",
            "Implemented connection pooling with PgBouncer to handle the increased connection load",
            "Ran parallel testing environment for 2 weeks before the final switchover",
            "Migration completed successfully with only 15 minutes of downtime",
        ],
    },
    'api_design': {
        'topic': 'REST API design discussion',
        'chunks': [
            "Debating between REST and GraphQL for the new customer-facing API",
            "REST makes more sense for our use case since we have well-defined resources",
            "Decided on using OpenAPI 3.0 specification for documentation and code generation",
            "Implementing versioning via URL path: /api/v1/ rather than headers",
            "Rate limiting set to 1000 requests per minute per API key using Redis",
            "Added request validation with JSON Schema to catch malformed requests early",
            "Implemented HATEOAS links for discoverability but keeping it optional",
            "Authentication uses JWT tokens with 1 hour expiry and refresh token rotation",
            "Error responses follow RFC 7807 Problem Details format for consistency",
            "Generated SDKs for Python, JavaScript, and Go from the OpenAPI spec",
        ],
    },
    'machine_learning_model': {
        'topic': 'ML model training discussion',
        'chunks': [
            "Training a sentiment analysis model on 500K customer reviews",
            "Started with BERT-base but it's too slow for production inference",
            "Switched to DistilBERT which is 60% faster with only 3% accuracy drop",
            "Fine-tuning on our domain-specific data improved F1 score from 0.82 to 0.91",
            "Implemented early stopping to prevent overfitting after epoch 5",
            "Using mixed precision training (FP16) reduced training time by 40%",
            "The confusion matrix shows the model struggles with sarcastic reviews",
            "Added data augmentation with back-translation to improve robustness",
            "Deployed the model using TorchServe with batching for throughput",
            "A/B testing shows 15% improvement in customer satisfaction prediction",
        ],
    },
    'security_audit': {
        'topic': 'Security vulnerability fixes',
        'chunks': [
            "Security scan found SQL injection vulnerability in the search endpoint",
            "The issue was string concatenation in raw SQL - switched to parameterized queries",
            "Also found XSS vulnerability in user profile page - now sanitizing all HTML output",
            "Implemented Content-Security-Policy headers to prevent inline script execution",
            "Password hashing upgraded from MD5 to bcrypt with cost factor 12",
            "Added rate limiting on login endpoint to prevent brute force attacks",
            "Enabled HTTPS everywhere and set HSTS header with 1 year max-age",
            "Implemented audit logging for all admin actions using structured JSON",
            "Set up automated dependency scanning with Snyk in the CI pipeline",
            "Passed the penetration test with no critical or high severity findings",
        ],
    },
    'frontend_refactor': {
        'topic': 'Frontend architecture refactoring',
        'chunks': [
            "The frontend codebase has grown to 200K lines and build times are 5 minutes",
            "Implementing module federation to split into micro-frontends",
            "Shared components moved to a separate npm package with Storybook documentation",
            "State management migrated from Redux to Zustand for simpler boilerplate",
            "Replaced Webpack with Vite which reduced dev server startup from 30s to 2s",
            "Implemented lazy loading for routes which cut initial bundle size by 60%",
            "Added TypeScript strict mode and fixed 847 type errors over 2 weeks",
            "Component testing with React Testing Library now covers 80% of UI",
            "CSS modules replaced global styles to prevent naming conflicts",
            "Build time reduced from 5 minutes to 45 seconds after all optimizations",
        ],
    },
}

# Filler conversation topics (less specific, background noise)
FILLER_TOPICS = [
    "general team standup and status updates",
    "code review feedback and suggestions",
    "sprint planning and task estimation",
    "technical documentation updates",
    "infrastructure monitoring alerts",
]


def generate_filler_chunk(session_idx, doc_idx):
    """Generate generic filler content."""
    topic = random.choice(FILLER_TOPICS)
    templates = [
        f"Discussed {topic} during session {session_idx}.",
        f"Team sync about {topic} - no major blockers.",
        f"Routine update on {topic} for the project.",
        f"Quick check-in regarding {topic}.",
        f"Status update: {topic} is progressing as planned.",
    ]
    return random.choice(templates)


def check_ollama():
    try:
        r = requests.get(f"{CONFIG['ollama_host']}/api/tags", timeout=5)
        return r.status_code == 200
    except:
        return False


def ollama_generate(prompt, context):
    full_prompt = f"""You are a helpful assistant with access to conversation history.

RETRIEVED CONVERSATION HISTORY:
{context}

USER QUESTION: {prompt}

Based on the conversation history above, provide a helpful answer. Reference specific details from the conversations."""

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

    print(f"\n[OK] Ollama available at {CONFIG['ollama_host']}")

    print("\nLoading embedding model...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # Calculate totals
    n_sessions = CONFIG['sessions']
    n_docs = CONFIG['docs_per_session']
    n_chunks_per_doc = CONFIG['chunks_per_doc']
    total_chunks = n_sessions * n_docs * n_chunks_per_doc

    print(f"\nGenerating {total_chunks:,} chunks across {n_sessions} conversations...")

    # Place specific conversations at known positions
    conversation_keys = list(CONVERSATIONS.keys())
    conversation_placements = {}

    # Spread the specific conversations across the session range
    for i, key in enumerate(conversation_keys):
        # Place each specific conversation at evenly spaced sessions
        session_idx = (i * n_sessions) // len(conversation_keys) + random.randint(0, 20)
        session_idx = min(session_idx, n_sessions - 1)
        conversation_placements[session_idx] = key

    print(f"  Placed {len(conversation_placements)} specific conversations at sessions: {list(conversation_placements.keys())}")

    # Generate all chunks
    chunk_texts = []
    chunk_meta = []

    for session_idx in range(n_sessions):
        for doc_idx in range(n_docs):
            if session_idx in conversation_placements and doc_idx == 0:
                # This session has a specific conversation
                conv_key = conversation_placements[session_idx]
                conv = CONVERSATIONS[conv_key]
                for chunk_idx, chunk_text in enumerate(conv['chunks']):
                    full_text = f"[Conversation {session_idx}] {chunk_text}"
                    chunk_texts.append(full_text)
                    chunk_meta.append({
                        'session': session_idx,
                        'doc': doc_idx,
                        'conv_key': conv_key,
                        'is_specific': True,
                    })
            else:
                # Filler content
                for chunk_idx in range(n_chunks_per_doc):
                    full_text = f"[Session {session_idx}, Topic {doc_idx}] {generate_filler_chunk(session_idx, doc_idx)}"
                    chunk_texts.append(full_text)
                    chunk_meta.append({
                        'session': session_idx,
                        'doc': doc_idx,
                        'conv_key': None,
                        'is_specific': False,
                    })

        if (session_idx + 1) % 100 == 0:
            print(f"  Generated sessions: {session_idx + 1}/{n_sessions}")

    print(f"\n  Embedding {len(chunk_texts):,} chunks...")
    start = time.time()
    embeddings = embedder.encode(chunk_texts, normalize_embeddings=True, show_progress_bar=True, batch_size=128)
    print(f"  Embedded in {time.time() - start:.1f}s")

    print(f"\n  Building HAT index...")
    start = time.time()

    index = HatIndex.cosine(384)
    id_to_meta = {}

    current_session = -1
    current_doc = -1

    for i, (text, meta, emb) in enumerate(zip(chunk_texts, chunk_meta, embeddings)):
        if meta['session'] != current_session:
            index.new_session()
            current_session = meta['session']
            current_doc = -1

        if meta['doc'] != current_doc:
            index.new_document()
            current_doc = meta['doc']

        chunk_id = index.add(emb.tolist())
        id_to_meta[chunk_id] = {'idx': i, 'text': text, 'meta': meta}

    build_time = time.time() - start
    stats = index.stats()

    est_tokens = stats.chunks * 30

    print(f"""
╔════════════════════════════════════════════════════════════════════════════╗
║  CONVERSATION HISTORY INDEXED                                               ║
╠════════════════════════════════════════════════════════════════════════════╣
║  Total conversations:    {stats.sessions:>50,} ║
║  Topic segments:         {stats.documents:>50,} ║
║  Total messages:         {stats.chunks:>50,} ║
║  Estimated tokens:       {est_tokens:>50,} ║
║  Build time:             {build_time:>47.2f}s ║
╚════════════════════════════════════════════════════════════════════════════╝
""")

    # REALISTIC QUESTIONS - things a user would actually ask
    QUESTIONS = [
        {
            'question': "What did we do to fix the React error about undefined properties?",
            'expected_conv': 'react_debugging',
            'expected_detail': 'optional chaining',
        },
        {
            'question': "How much did we speed up the Python data processing script?",
            'expected_conv': 'python_optimization',
            'expected_detail': '15x faster',
        },
        {
            'question': "What was causing the Kubernetes pods to crash?",
            'expected_conv': 'kubernetes_deployment',
            'expected_detail': 'OOMKilled',
        },
        {
            'question': "How long was the downtime during the PostgreSQL migration?",
            'expected_conv': 'database_migration',
            'expected_detail': '15 minutes',
        },
        {
            'question': "What rate limit did we set for the new API?",
            'expected_conv': 'api_design',
            'expected_detail': '1000 requests',
        },
        {
            'question': "What F1 score did the sentiment model achieve after fine-tuning?",
            'expected_conv': 'machine_learning_model',
            'expected_detail': '0.91',
        },
        {
            'question': "What password hashing algorithm did we upgrade to?",
            'expected_conv': 'security_audit',
            'expected_detail': 'bcrypt',
        },
        {
            'question': "How much did we reduce the frontend build time?",
            'expected_conv': 'frontend_refactor',
            'expected_detail': '45 seconds',
        },
    ]

    print("=" * 80)
    print("  REALISTIC QUESTION ANSWERING TEST")
    print("=" * 80)

    results = []

    for q in QUESTIONS:
        print(f"\n{'─' * 80}")
        print(f"  Q: {q['question']}")
        print(f"  Expected topic: {q['expected_conv']}")
        print(f"{'─' * 80}")

        # Embed and search
        query_emb = embedder.encode(q['question'], normalize_embeddings=True).tolist()

        start = time.time()
        search_results = index.near(query_emb, k=10)
        search_ms = (time.time() - start) * 1000

        # Check if we found the right conversation
        found_correct = False
        found_rank = -1
        retrieved_texts = []

        for rank, r in enumerate(search_results):
            meta = id_to_meta[r.id]
            retrieved_texts.append(meta['text'])
            if meta['meta'].get('conv_key') == q['expected_conv']:
                if not found_correct:
                    found_correct = True
                    found_rank = rank + 1

        print(f"\n  Search time: {search_ms:.2f}ms across {est_tokens:,} tokens")
        print(f"  Found correct conversation: {'YES at rank ' + str(found_rank) if found_correct else 'NO'}")

        print(f"\n  Top 3 retrieved:")
        for i, text in enumerate(retrieved_texts[:3]):
            meta = id_to_meta[search_results[i].id]['meta']
            is_target = meta.get('conv_key') == q['expected_conv']
            marker = " <<<" if is_target else ""
            print(f"    [{i+1}] Score: {search_results[i].score:.3f}{marker}")
            print(f"        {text[:80]}...")

        # Ask LLM
        if found_correct:
            context = "\n".join(retrieved_texts[:5])

            print(f"\n  Asking {CONFIG['model']}...")
            start = time.time()
            response = ollama_generate(q['question'], context)
            llm_ms = (time.time() - start) * 1000

            # Check if expected detail is in response
            has_detail = q['expected_detail'].lower() in response.lower()

            print(f"\n  LLM Response ({llm_ms:.0f}ms):")
            print(f"  {response.strip()[:300]}...")
            print(f"\n  Contains '{q['expected_detail']}': {'YES' if has_detail else 'NO'}")

            results.append({
                'question': q['question'],
                'found_conv': True,
                'rank': found_rank,
                'has_detail': has_detail,
                'search_ms': search_ms,
            })
        else:
            results.append({
                'question': q['question'],
                'found_conv': False,
                'rank': -1,
                'has_detail': False,
                'search_ms': search_ms,
            })

    # Summary
    found_count = sum(1 for r in results if r['found_conv'])
    detail_count = sum(1 for r in results if r.get('has_detail', False))
    avg_search = sum(r['search_ms'] for r in results) / len(results)
    avg_rank = sum(r['rank'] for r in results if r['rank'] > 0) / max(found_count, 1)

    print(f"""

╔════════════════════════════════════════════════════════════════════════════╗
║  FINAL RESULTS                                                              ║
╠════════════════════════════════════════════════════════════════════════════╣
║  Questions asked:          {len(QUESTIONS):>49} ║
║  Correct conversation:     {found_count}/{len(QUESTIONS)} ({100*found_count//len(QUESTIONS)}%)                                        ║
║  Correct details:          {detail_count}/{len(QUESTIONS)} ({100*detail_count//len(QUESTIONS)}%)                                        ║
║  Avg retrieval rank:       {avg_rank:>49.1f} ║
║  Avg search time:          {avg_search:>46.2f}ms ║
║  Total tokens searched:    {est_tokens:>49,} ║
╚════════════════════════════════════════════════════════════════════════════╝
""")

    if found_count == len(QUESTIONS):
        print("  ALL CONVERSATIONS FOUND - HAT successfully retrieves context at scale!")
    else:
        print(f"  {found_count}/{len(QUESTIONS)} conversations found - some questions need refinement")

    # Save results
    output = f"realistic_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output, 'w') as f:
        json.dump({
            'config': CONFIG,
            'total_tokens': est_tokens,
            'results': results,
            'summary': {
                'found_rate': found_count / len(QUESTIONS),
                'detail_rate': detail_count / len(QUESTIONS),
                'avg_search_ms': avg_search,
            }
        }, f, indent=2)
    print(f"\n  Results saved to: {output}")


if __name__ == "__main__":
    main()
