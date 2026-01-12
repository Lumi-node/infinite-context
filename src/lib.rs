//! # Infinite Context
//!
//! Give any local LLM unlimited memory.
//!
//! **11M+ tokens | 28ms latency | 100% accuracy**
//!
//! ## Overview
//!
//! Infinite Context uses the Hierarchical Attention Tree (HAT) algorithm to enable
//! small context models like Gemma 3 (8K) and Phi 4 (16K) to access millions of
//! tokens of conversation history with perfect retrieval accuracy.
//!
//! ## Key Features
//!
//! - **O(log n) retrieval**: Beam search through session→document→chunk hierarchy
//! - **100% accuracy**: Exploits known structure, not approximate search
//! - **28ms latency**: Even on 11M+ token indexes
//! - **Model agnostic**: Works with any Ollama model
//! - **Persistent memory**: Save/load your memory index
//!
//! ## Rust Usage
//!
//! ```rust,ignore
//! use infinite_context::adapters::index::HatIndex;
//! use infinite_context::adapters::ollama::OllamaClient;
//! use infinite_context::core::Point;
//! use infinite_context::ports::Near;
//!
//! // Create index
//! let mut index = HatIndex::cosine(384);
//!
//! // Add embeddings (from sentence-transformers)
//! let embedding = Point::new(vec![0.1; 384]);
//! index.add(Id::now(), &embedding)?;
//!
//! // Query
//! let results = index.near(&query_embedding, 10)?;
//!
//! // Use with Ollama
//! let ollama = OllamaClient::with_model("gemma3:1b");
//! let response = ollama.generate("Hello!", Some("context here"))?;
//! ```
//!
//! ## Python Usage
//!
//! ```python
//! from infinite_context import HatIndex, InfiniteContext
//! from sentence_transformers import SentenceTransformer
//!
//! # Low-level API
//! embedder = SentenceTransformer('all-MiniLM-L6-v2')
//! index = HatIndex.cosine(384)
//! index.add(embedder.encode("Important info"))
//! results = index.near(embedder.encode("query"), k=10)
//!
//! # High-level API
//! ctx = InfiniteContext("gemma3:1b")
//! ctx.add(embedder.encode("Info"), "Important info")
//! response = ctx.chat(embedder.encode("What's important?"), "What's important?", k=5)
//! ```

pub mod core;
pub mod ports;
pub mod adapters;

// Re-exports for convenience
pub use core::{Id, Point};
pub use adapters::index::{HatIndex, HatConfig, HatStats};
pub use adapters::ollama::OllamaClient;
pub use ports::{Near, SearchResult};
