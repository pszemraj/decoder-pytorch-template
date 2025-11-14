use anyhow::{Context, Result};
use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
};
use rand::Rng;
use std::{fs, path::Path};

use crate::train::TextBatch;

/// Character-level dataset
pub struct CharDataset {
    data: Vec<u8>,
    sequence_length: usize,
}

impl CharDataset {
    /// Load dataset from text file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let data = fs::read(path.as_ref())
            .with_context(|| format!("Failed to read file: {:?}", path.as_ref()))?;

        Ok(Self {
            data,
            sequence_length: 512, // default
        })
    }

    /// Create from string
    pub fn from_text(text: &str, sequence_length: usize) -> Self {
        Self {
            data: text.bytes().collect(),
            sequence_length,
        }
    }

    /// Set sequence length
    pub fn with_sequence_length(mut self, sequence_length: usize) -> Self {
        self.sequence_length = sequence_length;
        self
    }

    /// Get vocabulary size (always 256 for byte-level)
    pub fn vocab_size(&self) -> usize {
        256
    }
}

impl Dataset<TextItem> for CharDataset {
    fn get(&self, _index: usize) -> Option<TextItem> {
        // Ensure we have enough data for a sequence
        let max_start = self.data.len().saturating_sub(self.sequence_length + 1);
        if max_start == 0 {
            return None;
        }

        // Random sampling similar to the PyTorch template
        let mut rng = rand::thread_rng();
        let start = rng.gen_range(0..max_start);
        let end = start + self.sequence_length + 1;

        // Extract sequence
        let sequence = &self.data[start..end];

        // Input is all but last token, target is all but first token
        let input: Vec<i64> = sequence[..self.sequence_length]
            .iter()
            .map(|&b| b as i64)
            .collect();

        let target: Vec<i64> = sequence[1..].iter().map(|&b| b as i64).collect();

        Some(TextItem { input, target })
    }

    fn len(&self) -> usize {
        // Number of possible sequences
        self.data.len().saturating_sub(self.sequence_length)
    }
}

/// Single text sequence item
#[derive(Clone, Debug)]
pub struct TextItem {
    pub input: Vec<i64>,
    pub target: Vec<i64>,
}

/// Batcher for text data
#[derive(Clone)]
pub struct TextBatcher {
    sequence_length: usize,
}

impl TextBatcher {
    pub fn new(sequence_length: usize) -> Self {
        Self { sequence_length }
    }
}

impl<B: Backend> Batcher<B, TextItem, TextBatch<B>> for TextBatcher {
    fn batch(&self, mut items: Vec<TextItem>, device: &B::Device) -> TextBatch<B> {
        let batch_size = items.len();

        // Flatten all inputs and targets
        let mut input_data = Vec::with_capacity(batch_size * self.sequence_length);
        let mut target_data = Vec::with_capacity(batch_size * self.sequence_length);

        for item in items.iter_mut() {
            if item.input.len() < self.sequence_length {
                item.input.resize(self.sequence_length, 0);
            }
            if item.target.len() < self.sequence_length {
                item.target.resize(self.sequence_length, 0);
            }

            input_data.extend_from_slice(&item.input[..self.sequence_length]);
            target_data.extend_from_slice(&item.target[..self.sequence_length]);
        }

        // Create tensors
        let tokens = Tensor::<B, 2, Int>::from_data(
            TensorData::new(input_data, [batch_size, self.sequence_length]),
            device,
        );

        let targets = Tensor::<B, 2, Int>::from_data(
            TensorData::new(target_data, [batch_size, self.sequence_length]),
            device,
        );

        TextBatch { tokens, targets }
    }
}

/// Wikipedia dataset loader (enwik8 style)
pub struct WikiDataset {
    train_data: Vec<u8>,
    val_data: Vec<u8>,
}

impl WikiDataset {
    /// Load enwik8 or similar dataset
    pub fn load_enwik8<P: AsRef<Path>>(path: P) -> Result<Self> {
        use flate2::read::GzDecoder;
        use std::io::Read;

        let file = fs::File::open(path.as_ref())
            .with_context(|| format!("Failed to open: {:?}", path.as_ref()))?;

        let mut decoder = GzDecoder::new(file);
        let mut data = Vec::new();
        decoder
            .read_to_end(&mut data)
            .context("Failed to decompress data")?;

        // Split 90/10 for train/val
        let split_idx = (data.len() as f32 * 0.9) as usize;
        let train_data = data[..split_idx].to_vec();
        let val_data = data[split_idx..].to_vec();

        Ok(Self {
            train_data,
            val_data,
        })
    }

    pub fn train_dataset(&self, sequence_length: usize) -> CharDataset {
        CharDataset {
            data: self.train_data.clone(),
            sequence_length,
        }
    }

    pub fn val_dataset(&self, sequence_length: usize) -> CharDataset {
        CharDataset {
            data: self.val_data.clone(),
            sequence_length,
        }
    }
}

/// Text tokenizer trait
pub trait Tokenizer {
    fn encode(&self, text: &str) -> Vec<i64>;
    fn decode(&self, tokens: &[i64]) -> String;
    fn vocab_size(&self) -> usize;
}

/// Simple byte-level tokenizer
pub struct ByteTokenizer;

impl Tokenizer for ByteTokenizer {
    fn encode(&self, text: &str) -> Vec<i64> {
        text.bytes().map(|b| b as i64).collect()
    }

    fn decode(&self, tokens: &[i64]) -> String {
        let bytes: Vec<u8> = tokens
            .iter()
            .filter_map(|&t| {
                if (0..256).contains(&t) {
                    Some(t as u8)
                } else {
                    None
                }
            })
            .collect();

        String::from_utf8_lossy(&bytes).to_string()
    }

    fn vocab_size(&self) -> usize {
        256
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_tokenizer() {
        let tokenizer = ByteTokenizer;
        let text = "Hello, World!";

        let tokens = tokenizer.encode(text);
        let decoded = tokenizer.decode(&tokens);

        assert_eq!(text, decoded);
        assert_eq!(tokenizer.vocab_size(), 256);
    }

    #[test]
    fn test_char_dataset() {
        let text = "The quick brown fox jumps over the lazy dog.";
        let dataset = CharDataset::from_text(text, 10);

        assert!(dataset.len() > 0);

        if let Some(item) = dataset.get(0) {
            assert_eq!(item.input.len(), 10);
            assert_eq!(item.target.len(), 10);
        }
    }
}
