//! Ollama Integration
//!
//! Client for Ollama's local LLM API, providing:
//! - Text generation with context injection
//! - Embedding generation (for models that support it)
//!
//! # Example
//! ```rust,ignore
//! let client = OllamaClient::new("http://localhost:11434", "gemma3:1b");
//! let response = client.generate("What is HAT?", Some("HAT is a hierarchical attention tree..."))?;
//! ```

use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

/// Ollama API client for local LLM inference
pub struct OllamaClient {
    host: String,
    model: String,
    client: reqwest::blocking::Client,
    temperature: f32,
    max_tokens: u32,
}

/// Generation request to Ollama
#[derive(Serialize)]
struct GenerateRequest {
    model: String,
    prompt: String,
    stream: bool,
    options: GenerateOptions,
}

#[derive(Serialize)]
struct GenerateOptions {
    temperature: f32,
    num_predict: u32,
}

/// Generation response from Ollama
#[derive(Deserialize)]
struct GenerateResponse {
    response: String,
    #[serde(default)]
    done: bool,
}

/// Embedding request to Ollama
#[derive(Serialize)]
struct EmbedRequest {
    model: String,
    input: String,
}

/// Embedding response from Ollama
#[derive(Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

/// Result of a generation with timing info
pub struct GenerationResult {
    pub response: String,
    pub duration_ms: u64,
}

/// Ollama client errors
#[derive(Debug, thiserror::Error)]
pub enum OllamaError {
    #[error("Connection error: {0}")]
    Connection(String),

    #[error("Request failed: {0}")]
    Request(String),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Timeout after {0}ms")]
    Timeout(u64),
}

impl OllamaClient {
    /// Create a new Ollama client
    ///
    /// # Arguments
    /// * `host` - Ollama API host (e.g., "http://localhost:11434")
    /// * `model` - Model name (e.g., "gemma3:1b", "phi4", "llama3.2")
    pub fn new(host: &str, model: &str) -> Self {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            host: host.trim_end_matches('/').to_string(),
            model: model.to_string(),
            client,
            temperature: 0.3,
            max_tokens: 500,
        }
    }

    /// Create with default host (localhost:11434)
    pub fn with_model(model: &str) -> Self {
        Self::new("http://localhost:11434", model)
    }

    /// Set generation temperature (0.0 = deterministic, 1.0 = creative)
    pub fn temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    /// Set maximum tokens to generate
    pub fn max_tokens(mut self, max: u32) -> Self {
        self.max_tokens = max;
        self
    }

    /// Generate a response, optionally with retrieved context
    ///
    /// # Arguments
    /// * `prompt` - The user's message/question
    /// * `context` - Optional retrieved context from HAT
    ///
    /// # Returns
    /// The generated response text
    pub fn generate(&self, prompt: &str, context: Option<&str>) -> Result<String, OllamaError> {
        let result = self.generate_with_timing(prompt, context)?;
        Ok(result.response)
    }

    /// Generate a response with timing information
    pub fn generate_with_timing(&self, prompt: &str, context: Option<&str>) -> Result<GenerationResult, OllamaError> {
        let full_prompt = if let Some(ctx) = context {
            format!(
                "You are a helpful assistant with access to your memory.\n\n\
                RETRIEVED MEMORY:\n{}\n\n\
                USER MESSAGE:\n{}\n\n\
                Respond based on your memory and the user's message. Be specific and reference relevant information.",
                ctx, prompt
            )
        } else {
            prompt.to_string()
        };

        let request = GenerateRequest {
            model: self.model.clone(),
            prompt: full_prompt,
            stream: false,
            options: GenerateOptions {
                temperature: self.temperature,
                num_predict: self.max_tokens,
            },
        };

        let start = Instant::now();

        let response = self.client
            .post(format!("{}/api/generate", self.host))
            .json(&request)
            .send()
            .map_err(|e| {
                if e.is_connect() {
                    OllamaError::Connection(format!(
                        "Cannot connect to Ollama at {}. Is it running? Try: ollama serve",
                        self.host
                    ))
                } else if e.is_timeout() {
                    OllamaError::Timeout(120000)
                } else {
                    OllamaError::Request(e.to_string())
                }
            })?;

        let duration_ms = start.elapsed().as_millis() as u64;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().unwrap_or_default();
            if status.as_u16() == 404 {
                return Err(OllamaError::ModelNotFound(format!(
                    "Model '{}' not found. Try: ollama pull {}",
                    self.model, self.model
                )));
            }
            return Err(OllamaError::Request(format!("Status {}: {}", status, text)));
        }

        let gen_response: GenerateResponse = response.json()
            .map_err(|e| OllamaError::Request(format!("Invalid response: {}", e)))?;

        Ok(GenerationResult {
            response: gen_response.response,
            duration_ms,
        })
    }

    /// Generate embeddings for text (requires embedding-capable model)
    ///
    /// Note: Not all models support embeddings. Use models like nomic-embed-text
    /// or check if your model supports the /api/embed endpoint.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, OllamaError> {
        let request = EmbedRequest {
            model: self.model.clone(),
            input: text.to_string(),
        };

        let response = self.client
            .post(format!("{}/api/embed", self.host))
            .json(&request)
            .send()
            .map_err(|e| OllamaError::Request(e.to_string()))?;

        if !response.status().is_success() {
            return Err(OllamaError::Request(format!(
                "Embedding failed: {}. Model may not support embeddings.",
                response.status()
            )));
        }

        let embed_response: EmbedResponse = response.json()
            .map_err(|e| OllamaError::Request(format!("Invalid embedding response: {}", e)))?;

        embed_response.embeddings.into_iter().next()
            .ok_or_else(|| OllamaError::Request("No embedding returned".to_string()))
    }

    /// Check if Ollama is available and the model is loaded
    pub fn is_available(&self) -> bool {
        self.client
            .get(format!("{}/api/tags", self.host))
            .timeout(Duration::from_secs(5))
            .send()
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }

    /// List available models
    pub fn list_models(&self) -> Result<Vec<String>, OllamaError> {
        #[derive(Deserialize)]
        struct TagsResponse {
            models: Vec<ModelInfo>,
        }

        #[derive(Deserialize)]
        struct ModelInfo {
            name: String,
        }

        let response = self.client
            .get(format!("{}/api/tags", self.host))
            .send()
            .map_err(|e| OllamaError::Connection(e.to_string()))?;

        let tags: TagsResponse = response.json()
            .map_err(|e| OllamaError::Request(e.to_string()))?;

        Ok(tags.models.into_iter().map(|m| m.name).collect())
    }

    /// Get the current model name
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Get the host URL
    pub fn host(&self) -> &str {
        &self.host
    }
}

impl Default for OllamaClient {
    fn default() -> Self {
        Self::with_model("gemma3:1b")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = OllamaClient::new("http://localhost:11434", "gemma3:1b");
        assert_eq!(client.model(), "gemma3:1b");
        assert_eq!(client.host(), "http://localhost:11434");
    }

    #[test]
    fn test_builder_pattern() {
        let client = OllamaClient::with_model("phi4")
            .temperature(0.5)
            .max_tokens(1000);

        assert_eq!(client.model(), "phi4");
        assert_eq!(client.temperature, 0.5);
        assert_eq!(client.max_tokens, 1000);
    }
}
