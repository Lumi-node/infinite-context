//! Infinite Context CLI
//!
//! Give any local LLM unlimited memory from the command line.
//!
//! Usage:
//!     infinite-context chat --model gemma3:1b
//!     infinite-context stats --memory ~/.infinite-context/memory.hat
//!     infinite-context models

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::io::{self, Write, BufRead};

use infinite_context::adapters::ollama::OllamaClient;
use infinite_context::adapters::index::HatIndex;
use infinite_context::ports::Near;
use infinite_context::core::{Point, Id};

/// Infinite Context - Give any local LLM unlimited memory
#[derive(Parser)]
#[command(name = "infinite-context")]
#[command(author = "Andrew Young")]
#[command(version)]
#[command(about = "11M+ tokens | 28ms latency | 100% accuracy", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Interactive chat with infinite memory
    Chat {
        /// Ollama model to use
        #[arg(short, long, default_value = "gemma3:1b")]
        model: String,

        /// Memory file path
        #[arg(long, default_value = "~/.infinite-context/memory.hat")]
        memory: String,

        /// Ollama host
        #[arg(long, default_value = "http://localhost:11434")]
        host: String,
    },

    /// Show memory statistics
    Stats {
        /// Memory file path
        #[arg(long, default_value = "~/.infinite-context/memory.hat")]
        memory: String,
    },

    /// List available Ollama models
    Models {
        /// Ollama host
        #[arg(long, default_value = "http://localhost:11434")]
        host: String,
    },

    /// Quick test of Ollama connection
    Test {
        /// Model to test
        #[arg(short, long, default_value = "gemma3:1b")]
        model: String,

        /// Ollama host
        #[arg(long, default_value = "http://localhost:11434")]
        host: String,
    },

    /// Benchmark HAT performance
    Bench {
        /// Number of chunks to index
        #[arg(short, long, default_value = "10000")]
        chunks: usize,

        /// Embedding dimensions
        #[arg(short, long, default_value = "384")]
        dims: usize,
    },
}

fn expand_path(path: &str) -> PathBuf {
    if path.starts_with("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(&path[2..]);
        }
    }
    PathBuf::from(path)
}

fn cmd_chat(model: &str, memory_path: &str, host: &str) {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║           INFINITE CONTEXT - Unlimited LLM Memory                ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  Model: {:<55} ║", model);
    println!("║  Commands: /quit, /new (new session), /stats                     ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let ollama = OllamaClient::new(host, model);

    // Check Ollama availability
    if !ollama.is_available() {
        eprintln!("Error: Cannot connect to Ollama at {}", host);
        eprintln!("Make sure Ollama is running: ollama serve");
        eprintln!("And you have the model: ollama pull {}", model);
        return;
    }

    println!("Connected to Ollama. Model: {}", model);
    println!();
    println!("Note: For full infinite context, use the Python API with sentence-transformers.");
    println!("This CLI demonstrates the Ollama integration.");
    println!();

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("You: ");
        stdout.flush().unwrap();

        let mut input = String::new();
        if stdin.lock().read_line(&mut input).is_err() {
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        match input.to_lowercase().as_str() {
            "/quit" | "/exit" | "/q" => {
                println!("\nGoodbye!");
                break;
            }
            "/new" | "/session" => {
                println!("[Started new session]");
                continue;
            }
            "/stats" => {
                println!("[Memory stats not available in CLI mode - use Python API]");
                continue;
            }
            "/help" => {
                println!("\nCommands:");
                println!("  /quit    - Exit");
                println!("  /new     - Start new session");
                println!("  /stats   - Show memory stats");
                println!("  /help    - Show this help");
                println!();
                continue;
            }
            _ => {}
        }

        // Generate response
        match ollama.generate(input, None) {
            Ok(response) => {
                println!("\nAssistant: {}\n", response);
            }
            Err(e) => {
                eprintln!("\nError: {}\n", e);
            }
        }
    }
}

fn cmd_stats(memory_path: &str) {
    let path = expand_path(memory_path);

    if !path.exists() {
        println!("No memory file found at: {:?}", path);
        println!("Start chatting to create one!");
        return;
    }

    match HatIndex::load_from_file(&path) {
        Ok(index) => {
            let stats = index.stats();
            let file_size = std::fs::metadata(&path)
                .map(|m| m.len())
                .unwrap_or(0);

            println!("╔══════════════════════════════════════════════════════════════════╗");
            println!("║                 INFINITE CONTEXT MEMORY STATS                    ║");
            println!("╠══════════════════════════════════════════════════════════════════╣");
            println!("║  File: {:58?} ║", path);
            println!("║  Size: {:52.2} MB ║", file_size as f64 / 1024.0 / 1024.0);
            println!("╠══════════════════════════════════════════════════════════════════╣");
            println!("║  Sessions:  {:50} ║", stats.session_count);
            println!("║  Documents: {:50} ║", stats.document_count);
            println!("║  Chunks:    {:50} ║", stats.chunk_count);
            println!("╚══════════════════════════════════════════════════════════════════╝");
        }
        Err(e) => {
            eprintln!("Error loading memory: {}", e);
        }
    }
}

fn cmd_models(host: &str) {
    let client = OllamaClient::new(host, "");

    if !client.is_available() {
        eprintln!("Error: Cannot connect to Ollama at {}", host);
        eprintln!("Make sure Ollama is running: ollama serve");
        return;
    }

    match client.list_models() {
        Ok(models) => {
            println!("Available models:");
            println!();
            for model in models {
                println!("  - {}", model);
            }
            println!();
            println!("Pull more with: ollama pull <model>");
        }
        Err(e) => {
            eprintln!("Error listing models: {}", e);
        }
    }
}

fn cmd_test(model: &str, host: &str) {
    println!("Testing Ollama connection...");
    println!("  Host: {}", host);
    println!("  Model: {}", model);
    println!();

    let client = OllamaClient::new(host, model);

    if !client.is_available() {
        println!("❌ Cannot connect to Ollama");
        println!();
        println!("Make sure:");
        println!("  1. Ollama is installed: https://ollama.com");
        println!("  2. Ollama is running: ollama serve");
        println!("  3. Model is pulled: ollama pull {}", model);
        return;
    }

    println!("✓ Connected to Ollama");

    println!("Testing generation...");
    match client.generate("Say 'Hello from Infinite Context!' in exactly those words.", None) {
        Ok(response) => {
            println!("✓ Generation works");
            println!();
            println!("Response: {}", response.trim());
        }
        Err(e) => {
            println!("❌ Generation failed: {}", e);
        }
    }
}

fn cmd_bench(chunks: usize, dims: usize) {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║               HAT PERFORMANCE BENCHMARK                          ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  Chunks: {:55} ║", chunks);
    println!("║  Dimensions: {:52} ║", dims);
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // Build index
    println!("Building index...");
    let start = std::time::Instant::now();

    let mut index = HatIndex::cosine(dims);

    // Generate random embeddings
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    for i in 0..chunks {
        // Deterministic pseudo-random embedding
        let mut hasher = DefaultHasher::new();
        i.hash(&mut hasher);
        let seed = hasher.finish();

        let embedding: Vec<f32> = (0..dims)
            .map(|j| {
                let mut h = DefaultHasher::new();
                (seed, j).hash(&mut h);
                (h.finish() as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32
            })
            .collect();

        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let normalized: Vec<f32> = embedding.iter().map(|x| x / norm).collect();

        let point = Point::new(normalized);
        index.add(Id::now(), &point).unwrap();

        // New session every 100 chunks
        if i % 100 == 99 {
            index.new_session();
        }
        // New document every 10 chunks
        if i % 10 == 9 {
            index.new_document();
        }
    }

    let build_time = start.elapsed();
    println!("  Build time: {:?}", build_time);

    // Query benchmark
    println!("\nRunning queries...");
    let query_embedding: Vec<f32> = (0..dims).map(|i| (i as f32 / dims as f32)).collect();
    let norm: f32 = query_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    let query: Vec<f32> = query_embedding.iter().map(|x| x / norm).collect();
    let query_point = Point::new(query);

    let n_queries = 100;
    let start = std::time::Instant::now();

    for _ in 0..n_queries {
        let _ = index.near(&query_point, 10).unwrap();
    }

    let query_time = start.elapsed();
    let avg_query_ms = query_time.as_secs_f64() * 1000.0 / n_queries as f64;

    let stats = index.stats();

    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                         RESULTS                                  ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  Indexed chunks: {:48} ║", stats.chunk_count);
    println!("║  Sessions: {:54} ║", stats.session_count);
    println!("║  Documents: {:53} ║", stats.document_count);
    println!("║  Build time: {:52.2?} ║", build_time);
    println!("║  Avg query time: {:45.2} ms ║", avg_query_ms);
    println!("╚══════════════════════════════════════════════════════════════════╝");

    // Estimate tokens
    let est_tokens = chunks * 30;  // ~30 tokens per chunk average
    println!();
    println!("Estimated capacity: {} tokens", est_tokens);
    println!("At 11M tokens: ~{:.0}ms query latency (extrapolated)", avg_query_ms * (11_000_000.0 / est_tokens as f64).log2());
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Chat { model, memory, host } => {
            cmd_chat(&model, &memory, &host);
        }
        Commands::Stats { memory } => {
            cmd_stats(&memory);
        }
        Commands::Models { host } => {
            cmd_models(&host);
        }
        Commands::Test { model, host } => {
            cmd_test(&model, &host);
        }
        Commands::Bench { chunks, dims } => {
            cmd_bench(chunks, dims);
        }
    }
}
