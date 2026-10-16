//! Word retrieval using a neural embedding learned from real and synthetic swipes.
use crate::retrieval_encoder::{Encoder, DIM};
use crate::{Prediction, ScoreBreakdown};
use std::collections::HashSet;
use std::path::Path;

struct Word {
    text: String,
    popularity: f32,
}

/// Native neural recognizer. Load matching exported encoder and word-bank files.
///
/// The encoder maps a gesture to a unit vector. Word vectors are cached at
/// export time, so inference needs one encoder pass and a vocabulary scan.
pub struct RetrievalEngine {
    encoder: Encoder,
    words: Vec<Word>,
    vectors: Vec<[f32; DIM]>,
    prior_weight: f32,
}

impl RetrievalEngine {
    pub fn from_files(directory: impl AsRef<Path>) -> Result<Self, String> {
        let encoder =
            std::fs::read(directory.as_ref().join("encoder.bin")).map_err(|e| e.to_string())?;
        let bank =
            std::fs::read(directory.as_ref().join("words.bin")).map_err(|e| e.to_string())?;
        Self::from_bytes(&encoder, &bank)
    }

    pub fn from_bytes(model: &[u8], bank: &[u8]) -> Result<Self, String> {
        let encoder = Encoder::from_bytes(model)?;
        if bank.get(..8) != Some(b"SWPBK001") || bank.len() < 24 {
            return Err("unsupported route word-bank format".into());
        }
        let tag = u64::from_le_bytes(bank[8..16].try_into().unwrap());
        if tag != fingerprint(model) {
            return Err("word bank belongs to a different route encoder".into());
        }
        let count = u32::from_le_bytes(bank[16..20].try_into().unwrap()) as usize;
        let dimension = u32::from_le_bytes(bank[20..24].try_into().unwrap()) as usize;
        if dimension != DIM || count == 0 || count > bank.len() / (2 + 4 + DIM * 4) {
            return Err("invalid route word-bank dimensions".into());
        }
        let mut position = 24;
        let mut words = Vec::with_capacity(count);
        let mut vectors = Vec::with_capacity(count);
        let mut seen = HashSet::new();
        for _ in 0..count {
            let length = bank
                .get(position..position + 2)
                .ok_or("truncated word-bank entry")?;
            let length = u16::from_le_bytes(length.try_into().unwrap()) as usize;
            position += 2;
            let text = bank
                .get(position..position + length)
                .ok_or("truncated word-bank word")?;
            let text = std::str::from_utf8(text)
                .map_err(|e| e.to_string())?
                .to_string();
            if text.is_empty()
                || !text.bytes().all(|b| b.is_ascii_lowercase())
                || !seen.insert(text.clone())
            {
                return Err("invalid or duplicate word-bank word".into());
            }
            position += length;
            let values = bank
                .get(position..position + (DIM + 1) * 4)
                .ok_or("truncated word-bank vector")?;
            let values: Vec<_> = values
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect();
            if values.iter().any(|v| !v.is_finite()) || !(0.0..=1.0).contains(&values[0]) {
                return Err("invalid word-bank values".into());
            }
            let vector: [f32; DIM] = values[1..].try_into().unwrap();
            let norm = vector.iter().map(|v| v * v).sum::<f32>();
            if !(0.99..=1.01).contains(&norm) {
                return Err("word-bank vector is not normalized".into());
            }
            words.push(Word {
                text,
                popularity: values[0],
            });
            vectors.push(vector);
            position += (DIM + 1) * 4;
        }
        if position != bank.len() {
            return Err("word bank has trailing data".into());
        }
        Ok(Self {
            encoder,
            words,
            vectors,
            prior_weight: 0.025,
        })
    }

    pub fn word_count(&self) -> usize {
        self.words.len()
    }

    /// Restrict the frequency-ordered vocabulary, for paired benchmarks.
    pub fn truncate_vocabulary(&mut self, count: usize) {
        self.words.truncate(count);
        self.vectors.truncate(count);
    }

    pub fn set_prior_weight(&mut self, weight: f32) {
        if weight.is_finite() {
            self.prior_weight = weight.clamp(0.0, 0.5);
        }
    }

    /// Expose the normalized encoder output for cross-runtime verification.
    pub fn embedding(&self, path: &[(f64, f64)]) -> Option<[f32; DIM]> {
        self.encoder.embed(path)
    }

    pub fn predict_from_path(&self, path: &[(f64, f64)], limit: usize) -> Vec<Prediction> {
        if limit == 0 {
            return Vec::new();
        }
        let Some(query) = self.encoder.embed(path) else {
            return Vec::new();
        };
        let mut ranked: Vec<_> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(index, vector)| {
                let similarity: f32 = query.iter().zip(vector).map(|(a, b)| a * b).sum();
                let distance = (1.0 - similarity).max(0.0);
                (
                    distance - self.prior_weight * self.words[index].popularity,
                    index,
                    distance,
                )
            })
            .collect();
        let compare = |a: &(f32, usize, f32), b: &(f32, usize, f32)| {
            a.0.total_cmp(&b.0)
                .then(self.words[a.1].text.cmp(&self.words[b.1].text))
        };
        let count = limit.min(ranked.len());
        if count < ranked.len() {
            ranked.select_nth_unstable_by(count, compare);
        }
        ranked.truncate(count);
        ranked.sort_by(compare);
        ranked
            .into_iter()
            .map(|(rating, index, raw)| Prediction {
                word: self.words[index].text.clone(),
                raw_score: raw as f64,
                popularity: self.words[index].popularity as f64,
                rating: rating as f64,
                breakdown: ScoreBreakdown::default(),
            })
            .collect()
    }
}

fn fingerprint(data: &[u8]) -> u64 {
    data.iter().fold(0xcbf29ce484222325, |value, byte| {
        (value ^ *byte as u64).wrapping_mul(0x100000001b3)
    })
}
