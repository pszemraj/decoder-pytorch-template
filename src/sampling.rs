//! Unified sampling module for text generation.
//!
//! Provides configurable sampling strategies including temperature, min-p, top-k,
//! top-p (nucleus), and repetition penalty.

use burn::tensor::{backend::Backend, Tensor};
use rand::random;

/// Sampling parameters for text generation.
#[derive(Clone, Debug)]
pub struct SamplingParams {
    /// Temperature for logit scaling (higher = more random)
    pub temperature: f32,
    /// Min-p filtering threshold (0.0 = disabled)
    pub min_p: f32,
    /// Top-k filtering (0 = disabled)
    pub top_k: usize,
    /// Top-p (nucleus) filtering (1.0 = disabled)
    pub top_p: f32,
    /// Repetition penalty (1.0 = no penalty)
    pub repetition_penalty: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            min_p: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
        }
    }
}

impl SamplingParams {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn with_min_p(mut self, min_p: f32) -> Self {
        self.min_p = min_p;
        self
    }

    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }

    pub fn with_repetition_penalty(mut self, repetition_penalty: f32) -> Self {
        self.repetition_penalty = repetition_penalty;
        self
    }
}

/// Sample the next token from logits using the specified sampling parameters.
///
/// # Arguments
/// * `logits` - Raw logits tensor of shape [vocab_size]
/// * `params` - Sampling configuration
/// * `generated_tokens` - Previously generated tokens (for repetition penalty)
///
/// # Returns
/// The sampled token index
pub fn sample_next_token<B: Backend>(
    logits: Tensor<B, 1>,
    params: &SamplingParams,
    generated_tokens: &[i64],
) -> anyhow::Result<i64> {
    let data = logits.to_data();
    let mut values: Vec<f32> = data.iter::<f32>().collect();

    // Apply repetition penalty
    if params.repetition_penalty != 1.0 && !generated_tokens.is_empty() {
        apply_repetition_penalty(&mut values, generated_tokens, params.repetition_penalty);
    }

    // Apply temperature
    let temp = if params.temperature <= 0.0 {
        1e-5
    } else {
        params.temperature
    };
    for v in &mut values {
        *v /= temp;
    }

    // Convert to probabilities via softmax
    let probs = softmax_vec(&values);

    // Build sorted (index, probability) pairs
    let mut ranked: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Apply top-k filtering
    let ranked = if params.top_k > 0 && params.top_k < ranked.len() {
        ranked[..params.top_k].to_vec()
    } else {
        ranked
    };

    // Apply top-p (nucleus) filtering
    let ranked = apply_top_p_filter(ranked, params.top_p);

    // Apply min-p filtering
    let filtered = apply_min_p_filter(ranked, params.min_p);

    // Sample from filtered distribution
    sample_from_distribution(&filtered)
}

/// Sample next token using simple temperature + min-p (legacy interface).
pub fn sample_next_token_simple<B: Backend>(
    logits: Tensor<B, 1>,
    temperature: f32,
    min_p: f32,
) -> anyhow::Result<i64> {
    let params = SamplingParams::new()
        .with_temperature(temperature)
        .with_min_p(min_p);
    sample_next_token::<B>(logits, &params, &[])
}

/// Apply repetition penalty to logits.
fn apply_repetition_penalty(logits: &mut [f32], tokens: &[i64], penalty: f32) {
    for &token in tokens {
        if let Some(logit) = logits.get_mut(token as usize) {
            // If logit is positive, divide by penalty; if negative, multiply
            if *logit > 0.0 {
                *logit /= penalty;
            } else {
                *logit *= penalty;
            }
        }
    }
}

/// Compute softmax over a vector of logits.
fn softmax_vec(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();
    logits
        .iter()
        .map(|&x| (x - max_logit).exp() / exp_sum)
        .collect()
}

/// Apply top-p (nucleus) filtering to sorted probabilities.
fn apply_top_p_filter(ranked: Vec<(usize, f32)>, top_p: f32) -> Vec<(usize, f32)> {
    if top_p >= 1.0 || top_p <= 0.0 {
        return ranked;
    }

    let mut filtered = Vec::new();
    let mut cumulative = 0.0;
    for (idx, prob) in ranked {
        cumulative += prob;
        filtered.push((idx, prob));
        if cumulative >= top_p {
            break;
        }
    }
    filtered
}

/// Apply min-p filtering: keep tokens with prob >= min_p * max_prob.
fn apply_min_p_filter(ranked: Vec<(usize, f32)>, min_p: f32) -> Vec<(usize, f32)> {
    if min_p <= 0.0 || min_p >= 1.0 || ranked.is_empty() {
        return ranked;
    }

    let max_prob = ranked[0].1;
    let threshold = min_p * max_prob;
    ranked
        .into_iter()
        .filter(|(_, prob)| *prob >= threshold)
        .collect()
}

/// Sample from a filtered probability distribution.
fn sample_from_distribution(filtered: &[(usize, f32)]) -> anyhow::Result<i64> {
    if filtered.is_empty() {
        return Ok(0);
    }

    let sum: f32 = filtered.iter().map(|(_, p)| *p).sum();
    if sum == 0.0 {
        return Ok(filtered[0].0 as i64);
    }

    let mut draw = random::<f32>() * sum;
    for &(idx, prob) in filtered {
        draw -= prob;
        if draw <= 0.0 {
            return Ok(idx as i64);
        }
    }

    // Fallback to first token
    Ok(filtered[0].0 as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_vec() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax_vec(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_repetition_penalty() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0];
        apply_repetition_penalty(&mut logits, &[1, 3], 2.0);
        assert!((logits[0] - 1.0).abs() < 1e-5);
        assert!((logits[1] - 1.0).abs() < 1e-5); // 2.0 / 2.0
        assert!((logits[2] - 3.0).abs() < 1e-5);
        assert!((logits[3] - 2.0).abs() < 1e-5); // 4.0 / 2.0
    }

    #[test]
    fn test_top_p_filter() {
        let ranked = vec![(0, 0.5), (1, 0.3), (2, 0.15), (3, 0.05)];
        let filtered = apply_top_p_filter(ranked, 0.9);
        assert_eq!(filtered.len(), 3); // 0.5 + 0.3 + 0.15 >= 0.9
    }

    #[test]
    fn test_min_p_filter() {
        let ranked = vec![(0, 0.5), (1, 0.3), (2, 0.1), (3, 0.05)];
        let filtered = apply_min_p_filter(ranked, 0.2);
        // threshold = 0.2 * 0.5 = 0.1, so keep probs >= 0.1
        assert_eq!(filtered.len(), 3);
    }
}
