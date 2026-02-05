//! Neural network inference for blind swipe typing
//!
//! This module provides ONNX-based neural network inference for higher accuracy
//! blind swipe recognition. The model is trained in Python and exported to ONNX.

#[cfg(feature = "neural")]
use crate::types::{Point, Prediction, ScoreBreakdown};

#[cfg(feature = "neural")]
use crate::keyboard::{euclidean_dist, simplify_path};

#[cfg(feature = "neural")]
use std::collections::HashMap;

#[cfg(feature = "neural")]
use std::sync::Mutex;

#[cfg(feature = "neural")]
use ort::session::Session;

#[cfg(feature = "neural")]
use ort::value::Tensor;

/// Resample a path to N equidistant points
#[cfg(feature = "neural")]
fn resample_path(path: &[Point], n: usize) -> Vec<Point> {
    if path.len() < 2 || n < 2 {
        return path.to_vec();
    }

    // Calculate total path length
    let mut total_len = 0.0;
    for i in 1..path.len() {
        total_len += euclidean_dist(&path[i - 1], &path[i]);
    }

    if total_len < 1e-9 {
        return vec![path[0]; n];
    }

    let interval = total_len / (n - 1) as f64;
    let mut resampled = vec![path[0]];
    let mut accumulated = 0.0;
    let mut j = 1;

    for _ in 1..n {
        let target = resampled.len() as f64 * interval;

        while j < path.len() {
            let seg_len = euclidean_dist(&path[j - 1], &path[j]);
            if accumulated + seg_len >= target {
                let t = (target - accumulated) / seg_len;
                let new_pt = Point {
                    x: path[j - 1].x + t * (path[j].x - path[j - 1].x),
                    y: path[j - 1].y + t * (path[j].y - path[j - 1].y),
                };
                resampled.push(new_pt);
                break;
            }
            accumulated += seg_len;
            j += 1;
        }

        if j >= path.len() {
            resampled.push(*path.last().unwrap());
        }
    }

    while resampled.len() < n {
        resampled.push(*path.last().unwrap());
    }
    resampled.truncate(n);
    resampled
}

/// Normalize a path for neural network input
#[cfg(feature = "neural")]
fn normalize_path(path: &[Point]) -> Vec<Point> {
    if path.len() < 2 {
        return vec![];
    }

    // Calculate total path length
    let mut total_len = 0.0;
    for i in 1..path.len() {
        total_len += euclidean_dist(&path[i - 1], &path[i]);
    }

    if total_len < 1e-9 {
        total_len = 1.0;
    }

    // Calculate centroid
    let cx = path.iter().map(|p| p.x).sum::<f64>() / path.len() as f64;
    let cy = path.iter().map(|p| p.y).sum::<f64>() / path.len() as f64;

    // Normalize: center at origin, scale by path length
    path.iter()
        .map(|p| Point {
            x: (p.x - cx) / total_len,
            y: (p.y - cy) / total_len,
        })
        .collect()
}

/// Neural network based swipe predictor
#[cfg(feature = "neural")]
pub struct NeuralPredictor {
    session: Mutex<Session>,
    vocabulary: Vec<String>,
    n_points: usize,
    word_freq: HashMap<String, f64>,
    pop_weight: f64,
}

#[cfg(feature = "neural")]
impl NeuralPredictor {
    /// Create a new neural predictor from ONNX model and vocabulary
    pub fn new(
        model_path: &str,
        vocab_path: &str,
        freq_text: Option<&str>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Load ONNX model using ort 2.0 API
        let session = Session::builder()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        // Load vocabulary
        let vocab_str = std::fs::read_to_string(vocab_path)?;
        let vocabulary: Vec<String> = serde_json::from_str(&vocab_str)?;

        // Load word frequencies if provided
        let mut word_freq = HashMap::new();
        if let Some(freq_text) = freq_text {
            let mut max_freq: f64 = 0.0;
            let lines: Vec<&str> = freq_text.lines().collect();

            // First pass: find max frequency
            for line in &lines {
                if let Some((_, count_str)) = line.split_once('\t') {
                    if let Ok(count) = count_str.parse::<f64>() {
                        max_freq = max_freq.max(count);
                    }
                }
            }

            // Second pass: store normalized log frequencies
            for line in &lines {
                if let Some((word, count_str)) = line.split_once('\t') {
                    let word = word.trim().to_lowercase();
                    if let Ok(count) = count_str.parse::<f64>() {
                        let log_freq = (count.ln() - 1.0) / max_freq.ln();
                        word_freq.insert(word, log_freq.max(0.0));
                    }
                }
            }
        }

        Ok(Self {
            session: Mutex::new(session),
            vocabulary,
            n_points: 32,
            word_freq,
            pop_weight: 0.25,
        })
    }

    /// Set the popularity weight
    pub fn set_pop_weight(&mut self, weight: f64) {
        self.pop_weight = weight;
    }

    /// Preprocess a raw path into model input features
    fn preprocess(&self, raw_path: &[(f64, f64)]) -> Option<Vec<f32>> {
        if raw_path.len() < 2 {
            return None;
        }

        // Convert to Points
        let input_points: Vec<Point> = raw_path
            .iter()
            .map(|(x, y)| Point { x: *x, y: *y })
            .collect();

        let simplified = simplify_path(&input_points);
        if simplified.len() < 2 {
            return None;
        }

        // Resample and normalize
        let resampled = resample_path(&simplified, self.n_points);
        let normalized = normalize_path(&resampled);

        // Flatten to feature vector
        let mut features: Vec<f32> = Vec::with_capacity(self.n_points * 2);
        for p in &normalized {
            features.push(p.x as f32);
            features.push(p.y as f32);
        }

        Some(features)
    }

    /// Predict words from a raw drawn path
    pub fn predict(&self, raw_path: &[(f64, f64)], limit: usize) -> Vec<Prediction> {
        let features = match self.preprocess(raw_path) {
            Some(f) => f,
            None => return vec![],
        };

        // Create input tensor with shape [1, n_points * 2]
        // Use (shape, data) form to avoid ndarray version mismatches.
        let input_tensor = Tensor::from_array(([1usize, self.n_points * 2], features)).unwrap();

        // Run inference
        let mut session = match self.session.lock() {
            Ok(s) => s,
            Err(_) => return vec![],
        };

        let outputs = match session.run(ort::inputs![input_tensor]) {
            Ok(o) => o,
            Err(_) => return vec![],
        };

        // Get output - first output, extract as f32 array
        let output_tensor = outputs.get("output").unwrap_or(&outputs[0]);

        let (_shape, logits_slice) = match output_tensor.try_extract_tensor::<f32>() {
            Ok(v) => v,
            Err(_) => return vec![],
        };

        // Apply softmax
        let max_logit = logits_slice
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits_slice.iter().map(|x| (x - max_logit).exp()).sum();
        let probs: Vec<f64> = logits_slice
            .iter()
            .map(|x| ((x - max_logit).exp() / exp_sum) as f64)
            .collect();

        // Create predictions with word frequency adjustment
        let mut candidates: Vec<(String, f64, f64)> = self
            .vocabulary
            .iter()
            .enumerate()
            .filter_map(|(i, word)| {
                if i >= probs.len() {
                    return None;
                }
                let prob = probs[i];
                let popularity = *self.word_freq.get(word).unwrap_or(&0.0);
                // Convert probability to score (lower = better to match existing API)
                let raw_score = 1.0 - prob;
                Some((word.clone(), raw_score, popularity))
            })
            .collect();

        // Sort by adjusted score
        let pop_weight = self.pop_weight;
        candidates.sort_by(|a, b| {
            let score_a = a.1 - (pop_weight * a.2);
            let score_b = b.1 - (pop_weight * b.2);
            score_a
                .partial_cmp(&score_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        candidates
            .into_iter()
            .take(limit)
            .map(|(word, raw_score, popularity)| {
                let rating = raw_score - (pop_weight * popularity);
                Prediction {
                    word,
                    raw_score,
                    popularity,
                    rating,
                    breakdown: ScoreBreakdown::default(),
                }
            })
            .collect()
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocabulary.len()
    }
}

/// Stub implementation when neural feature is disabled
#[cfg(not(feature = "neural"))]
pub struct NeuralPredictor;

#[cfg(not(feature = "neural"))]
impl NeuralPredictor {
    pub fn new(
        _model_path: &str,
        _vocab_path: &str,
        _freq_text: Option<&str>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Err("Neural feature not enabled. Compile with --features neural".into())
    }
}
