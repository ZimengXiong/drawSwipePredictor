//! Word retrieval using a neural embedding learned from real and synthetic swipes.
use crate::retrieval_encoder::{dot, Encoder, DIM};
use crate::{Prediction, ScoreBreakdown};
use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::rerank::{Candidate, Geometry, Query, Ranker};

struct Word {
    text: String,
    popularity: f32,
    geometry: Geometry,
}

/// Native neural recognizer. Load matching exported encoder and word-bank files.
///
/// The encoder maps a gesture to a unit vector. Word vectors are cached at
/// export time, so inference needs one encoder pass and a vocabulary scan.
pub struct RetrievalEngine {
    encoder: Encoder,
    words: Vec<Word>,
    word_ids: HashMap<String, usize>,
    vectors: Vec<[f32; DIM]>,
    prior_weight: f32,
    tta_degrees: f32,
    adaptive_prior: bool,
    ranker: Option<Ranker>,
}

impl RetrievalEngine {
    /// Load the validated model and word vectors shipped with this crate.
    pub fn bundled() -> Result<Self, String> {
        let mut engine = Self::from_bytes(
            include_bytes!("../models/retrieval/encoder.bin"),
            include_bytes!("../models/retrieval/words.bin"),
        )?;
        engine.configure(include_str!("../models/retrieval/manifest.json"))?;
        Ok(engine)
    }

    pub fn from_files(directory: impl AsRef<Path>) -> Result<Self, String> {
        let encoder =
            std::fs::read(directory.as_ref().join("encoder.bin")).map_err(|e| e.to_string())?;
        let bank =
            std::fs::read(directory.as_ref().join("words.bin")).map_err(|e| e.to_string())?;
        let mut engine = Self::from_bytes(&encoder, &bank)?;
        match std::fs::read_to_string(directory.as_ref().join("manifest.json")) {
            Ok(json) => engine.configure(&json)?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.to_string()),
        }
        let ranker = directory.as_ref().join("reranker.json");
        match std::fs::read_to_string(ranker) {
            Ok(json) => engine.ranker = Some(Ranker::parse(&json)?),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.to_string()),
        }
        Ok(engine)
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
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| f32::from_le_bytes(*v))
                .collect();
            if values.iter().any(|v| !v.is_finite()) || !(0.0..=1.0).contains(&values[0]) {
                return Err("invalid word-bank values".into());
            }
            let vector: [f32; DIM] = values[1..].try_into().unwrap();
            let norm = vector.iter().map(|v| v * v).sum::<f32>();
            if !(0.99..=1.01).contains(&norm) {
                return Err("word-bank vector is not normalized".into());
            }
            if text.bytes().all(|b| b == text.as_bytes()[0]) {
                return Err("stationary word-bank route".into());
            }
            words.push(Word {
                geometry: Geometry::new(&text),
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
            word_ids: words
                .iter()
                .enumerate()
                .map(|(i, w)| (w.text.clone(), i))
                .collect(),
            words,
            vectors,
            prior_weight: 0.025,
            tta_degrees: 0.0,
            adaptive_prior: false,
            ranker: None,
        })
    }

    /// Apply the validation-selected prior and orientation ensemble settings.
    pub fn configure(&mut self, json: &str) -> Result<(), String> {
        #[derive(serde::Deserialize)]
        struct Settings {
            prior_weight: f32,
            #[serde(default)]
            tta_degrees: f32,
            #[serde(default)]
            adaptive_prior: bool,
        }
        let settings: Settings = serde_json::from_str(json).map_err(|e| e.to_string())?;
        if !(0.0..=2.0).contains(&settings.prior_weight)
            || !(0.0..=24.0).contains(&settings.tta_degrees)
        {
            return Err("invalid route ranking settings".into());
        }
        self.prior_weight = settings.prior_weight;
        self.tta_degrees = settings.tta_degrees;
        self.adaptive_prior = settings.adaptive_prior;
        Ok(())
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
            self.prior_weight = weight.clamp(0.0, 2.0);
        }
    }

    /// Expose the normalized encoder output for cross-runtime verification.
    pub fn embedding(&self, path: &[(f64, f64)]) -> Option<[f32; DIM]> {
        self.encoder.embed(path)
    }

    /// Extract the same candidate features used during deployed reranking.
    pub fn ranking_features(&self, path: &[(f64, f64)], limit: usize) -> Vec<Candidate> {
        let predictions = self.retrieve(path, limit);
        let Some(query) = Query::new(path) else {
            return Vec::new();
        };
        let best = predictions.first().map_or(0.0, |p| p.rating as f32);
        predictions
            .into_iter()
            .map(|p| {
                let index = self.word_ids[&p.word];
                Candidate {
                    features: query.features(
                        &self.words[index].geometry,
                        [
                            p.raw_score as f32,
                            p.rating as f32,
                            p.popularity as f32,
                            p.rating as f32 - best,
                        ],
                    ),
                    word: p.word,
                }
            })
            .collect()
    }

    pub fn predict_from_path(&self, path: &[(f64, f64)], limit: usize) -> Vec<Prediction> {
        let Some(ranker) = &self.ranker else {
            return self.retrieve(path, limit);
        };
        if limit == 0 {
            return Vec::new();
        }
        let mut candidates = self.ranking_features(path, ranker.shortlist.max(limit));
        candidates.sort_by(|a, b| {
            ranker
                .score(&a.features)
                .total_cmp(&ranker.score(&b.features))
                .then(a.word.cmp(&b.word))
        });
        candidates
            .into_iter()
            .take(limit)
            .map(|c| Prediction {
                word: c.word,
                raw_score: c.features[0] as f64,
                popularity: c.features[2] as f64,
                rating: ranker.score(&c.features) as f64,
                breakdown: ScoreBreakdown::default(),
            })
            .collect()
    }

    fn retrieve(&self, path: &[(f64, f64)], limit: usize) -> Vec<Prediction> {
        if limit == 0 {
            return Vec::new();
        }
        let Some(mut query) = self.encoder.embed(path) else {
            return Vec::new();
        };
        if self.tta_degrees > 0.0 {
            for sign in [-1.0_f64, 1.0] {
                let angle = sign * (self.tta_degrees as f64).to_radians();
                let (sin, cos) = angle.sin_cos();
                let origin = path[0];
                let rotated: Vec<_> = path
                    .iter()
                    .map(|p| {
                        let x = p.0 - origin.0;
                        let y = p.1 - origin.1;
                        (cos * x - sin * y, sin * x + cos * y)
                    })
                    .collect();
                let Some(vector) = self.encoder.embed(&rotated) else {
                    return Vec::new();
                };
                for (a, b) in query.iter_mut().zip(vector) {
                    *a += b;
                }
            }
            let norm = query.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            for value in &mut query {
                *value /= norm;
            }
        }
        let mut ranked: Vec<_> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(index, vector)| {
                let similarity = dot(&query, vector);
                let distance = (1.0 - similarity).max(0.0);
                let popularity = self.words[index].popularity;
                let rating = if self.adaptive_prior {
                    distance + self.prior_weight * (1.0 - popularity) * distance.sqrt()
                } else {
                    distance - self.prior_weight * popularity
                };
                (rating, index, distance)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_model_matches_pytorch_and_recognizes_reference_routes() {
        #[derive(serde::Deserialize)]
        struct Case {
            word: String,
            points: Vec<(f64, f64)>,
            embedding: Vec<f32>,
        }
        let engine = RetrievalEngine::bundled().unwrap();
        assert_eq!(engine.word_count(), 60000);
        let cases: Vec<Case> =
            serde_json::from_str(include_str!("../models/retrieval/verification.json")).unwrap();
        for case in cases {
            let actual = engine.embedding(&case.points).unwrap();
            let error = actual
                .iter()
                .zip(&case.embedding)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(error < 0.0001, "native embedding differs for {}", case.word);
            let predictions = engine.predict_from_path(&case.points, 5);
            assert_eq!(predictions[0].word, case.word);
            assert!(predictions.iter().all(|p| p.rating.is_finite()));
        }
    }

    #[test]
    fn validates_bank_identity_and_values() {
        let mut model = vec![0_u8; 8 + 405072 * 4];
        model[..8].copy_from_slice(b"SWPRT001");
        let mut bank = b"SWPBK001".to_vec();
        bank.extend(fingerprint(&model).to_le_bytes());
        bank.extend(1_u32.to_le_bytes());
        bank.extend((DIM as u32).to_le_bytes());
        bank.extend(5_u16.to_le_bytes());
        bank.extend(b"hello");
        bank.extend(0.5_f32.to_le_bytes());
        bank.extend(1.0_f32.to_le_bytes());
        bank.extend(vec![0_u8; (DIM - 1) * 4]);
        let mut engine = RetrievalEngine::from_bytes(&model, &bank).unwrap();
        assert_eq!(engine.word_count(), 1);
        assert!(engine.predict_from_path(&[(1.0, 1.0); 5], 5).is_empty());
        assert!(engine
            .configure(r#"{"prior_weight":0.1,"tta_degrees":10}"#)
            .is_ok());
        assert!(engine.configure(r#"{"prior_weight":-1}"#).is_err());
        let mut mismatched = bank.clone();
        mismatched[8] ^= 1;
        assert!(RetrievalEngine::from_bytes(&model, &mismatched).is_err());
        assert!(RetrievalEngine::from_bytes(&model, &bank[..bank.len() - 1]).is_err());
        let start = bank.len() - DIM * 4;
        bank[start..start + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(RetrievalEngine::from_bytes(&model, &bank).is_err());
    }
}
