//! Decoder architectures for experimentation.
//!
//! `llama.rs` hosts the reference implementation that mirrors Meta's Llama
//! decoder structure. To build a new idea, create `my_model.rs` next to this
//! file, expose it here, and point the trainer to it.

pub mod llama;

pub use llama::LlamaModel;
