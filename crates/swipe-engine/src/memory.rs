//! Recognition of freehand gestures without a visible keyboard.
//!
//! Each word has sharp and rounded motor templates. Retrieval scans cached
//! geometry; a shortlist receives joint position-and-direction alignment.
use crate::memory_geometry::{
    distance, elastic, key_vertices, normalize, prepare, resample, smooth, Shape, N,
};
use crate::{Prediction, ScoreBreakdown};
use std::collections::HashSet;

const ANGLES: [f32; 5] = [-24.0, -12.0, 0.0, 12.0, 24.0];
const SHORTLIST: usize = 192;

struct Template {
    word: String,
    popularity: f64,
    shapes: [Shape; 2],
}

/// Cached, training-free recognizer that treats drawing geometry as uncertain.
///
/// Load a frequency dictionary once, then reuse the engine across gestures.
/// Scores are distances, not calibrated probabilities. Lower scores are better.
pub struct MemoryEngine {
    templates: Vec<Template>,
    prior_weight: f32,
}

impl Default for MemoryEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryEngine {
    pub fn new() -> Self {
        Self {
            templates: Vec::new(),
            prior_weight: 0.025,
        }
    }

    /// Replace the vocabulary. Ignore malformed entries and duplicate words.
    pub fn load_dictionary(&mut self, text: &str) {
        let mut seen = HashSet::new();
        let entries: Vec<_> = text
            .lines()
            .filter_map(|line| {
                let (word, count) = line.split_once('\t')?;
                let word = word.trim().to_ascii_lowercase();
                let count: f64 = count.trim().parse().ok()?;
                if word.len() < 2
                    || !word.bytes().all(|c| c.is_ascii_lowercase())
                    || !count.is_finite()
                    || count <= 0.0
                    || !seen.insert(word.clone())
                {
                    return None;
                }
                Some((word, count.ln_1p()))
            })
            .collect();
        let max_log = entries.iter().map(|e| e.1).fold(1.0, f64::max);
        self.templates = entries
            .into_iter()
            .filter_map(|(word, log_count)| {
                let vertices = key_vertices(&word);
                let mut path = resample(&vertices, 96);
                smooth(&mut path, 2);
                let sharp = normalize(&path, 0.0)?;
                smooth(&mut path, 20);
                let rounded = normalize(&path, 0.0)?;
                Some(Template {
                    word,
                    popularity: log_count / max_log,
                    shapes: [sharp, rounded],
                })
            })
            .collect();
    }

    pub fn word_count(&self) -> usize {
        self.templates.len()
    }

    /// Control the weak word-frequency prior. Zero gives geometry-only ranking.
    pub fn set_prior_weight(&mut self, weight: f32) {
        if weight.is_finite() {
            self.prior_weight = weight.clamp(0.0, 0.2);
        }
    }

    pub fn predict_from_path(&self, raw: &[(f64, f64)], limit: usize) -> Vec<Prediction> {
        if limit == 0 {
            return Vec::new();
        }
        let Some(path) = prepare(raw) else {
            return Vec::new();
        };
        let queries: Vec<_> = ANGLES
            .iter()
            .filter_map(|a| normalize(&path, a.to_radians()))
            .collect();
        if queries.len() != ANGLES.len() {
            return Vec::new();
        }
        // Keep all candidates until retrieval. Hard aspect or turn filters can
        // exclude precisely the words that a memory gesture distorts most.
        let mut coarse: Vec<_> = self
            .templates
            .iter()
            .enumerate()
            .map(|(index, t)| {
                let mut best = (f32::INFINITY, 0);
                for (rotation, query) in queries.iter().enumerate() {
                    for shape in &t.shapes {
                        let mut score = 0.0;
                        for i in (0..N).step_by(3) {
                            score += distance(query[i], shape[i]);
                        }
                        score /= 11.0;
                        if score < best.0 {
                            best = (score, rotation);
                        }
                    }
                }
                (
                    best.0 - self.prior_weight * t.popularity as f32,
                    index,
                    best.1,
                )
            })
            .collect();
        let count = SHORTLIST.max(limit).min(coarse.len());
        if count == 0 {
            return Vec::new();
        }
        if count < coarse.len() {
            coarse.select_nth_unstable_by(count, |a, b| a.0.total_cmp(&b.0));
        }
        coarse.truncate(count);
        let mut predictions: Vec<_> = coarse
            .into_iter()
            .map(|(_, index, rotation)| {
                let template = &self.templates[index];
                let mut best = f32::INFINITY;
                // Refine neighboring rotations too: coarse pointwise alignment can
                // choose a different angle from the elastic alignment.
                for (r, query) in queries.iter().enumerate() {
                    if r.abs_diff(rotation) > 1 {
                        continue;
                    }
                    for shape in &template.shapes {
                        let score = elastic(query, shape) + 0.002 * (ANGLES[r] / 24.0).powi(2);
                        best = best.min(score);
                    }
                }
                Prediction {
                    word: template.word.clone(),
                    raw_score: best as f64,
                    popularity: template.popularity,
                    rating: best as f64 - self.prior_weight as f64 * template.popularity,
                    breakdown: ScoreBreakdown {
                        shape: best as f64,
                        ..Default::default()
                    },
                }
            })
            .collect();
        predictions.sort_by(|a, b| a.rating.total_cmp(&b.rating).then(a.word.cmp(&b.word)));
        predictions.truncate(limit);
        predictions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> MemoryEngine {
        let mut e = MemoryEngine::new();
        e.load_dictionary("lunchbox\t100\nalpacas\t100\nkeyboard\t100\nworld\t100\nhello\t100\n");
        e
    }

    #[test]
    fn recognizes_translated_stretched_and_rotated_word() {
        let e = engine();
        let (s, c) = 0.28_f64.sin_cos();
        let path: Vec<_> = key_vertices("lunchbox")
            .iter()
            .map(|p| {
                let x = p[0] as f64 * 80.0;
                let y = p[1] as f64 * 140.0;
                (500.0 + c * x - s * y, 100.0 + s * x + c * y)
            })
            .collect();
        assert_eq!(e.predict_from_path(&path, 5)[0].word, "lunchbox");
    }

    #[test]
    fn validates_dictionary_and_input() {
        let mut e = engine();
        e.load_dictionary("hello\t1\nhello\t2\nbad\tNaN\nno\t-1\n!\t1\naa\t3\n");
        assert_eq!(e.word_count(), 1); // "aa" has no moving gesture.
        for raw in [
            vec![],
            vec![(0.0, 0.0); 3],
            vec![(f64::INFINITY, 1.0), (1.0, 2.0)],
        ] {
            assert!(e.predict_from_path(&raw, 10).is_empty());
        }
        assert!(e.predict_from_path(&[(0.0, 0.0), (1.0, 1.0)], 0).is_empty());
    }

    #[test]
    fn ranking_does_not_depend_on_dictionary_order() {
        let a = engine();
        let mut b = MemoryEngine::new();
        b.load_dictionary("hello\t100\nworld\t100\nkeyboard\t100\nalpacas\t100\nlunchbox\t100\n");
        let raw: Vec<_> = key_vertices("alpacas")
            .iter()
            .map(|p| (p[0] as f64, p[1] as f64))
            .collect();
        let words = |e: &MemoryEngine| {
            e.predict_from_path(&raw, 5)
                .into_iter()
                .map(|p| p.word)
                .collect::<Vec<_>>()
        };
        assert_eq!(words(&a), words(&b));
    }
}
