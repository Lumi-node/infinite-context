//! Adapters - Swappable implementations of port interfaces

pub mod index;

#[cfg(feature = "python")]
pub mod python;

pub mod ollama;
