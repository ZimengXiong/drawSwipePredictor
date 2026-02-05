//! CTC (Connectionist Temporal Classification) swipe decoding.
//!
//! This predictor runs an ONNX CTC model that outputs per-timestep log probabilities
//! over the alphabet: blank (idx 0) + a-z (idx 1..=26).

use crate::types::Prediction;

#[cfg(feature = "neural")]
use crate::keyboard::{euclidean_dist, get_keyboard_layout, get_word_path, simplify_path};

#[cfg(feature = "neural")]
use crate::types::{Point, ScoreBreakdown};

#[cfg(feature = "neural")]
use std::collections::HashMap;

#[cfg(feature = "neural")]
use std::sync::Mutex;

#[cfg(feature = "neural")]
use ort::session::Session;

#[cfg(feature = "neural")]
use ort::value::Tensor;

#[cfg(feature = "neural")]
use serde::Deserialize;

#[cfg(feature = "neural")]
const LOG_ZERO: f32 = -1.0e30;

#[cfg(feature = "neural")]
const TRIE_NONE: u32 = u32::MAX;

#[cfg(feature = "neural")]
#[derive(Debug, Clone, Copy)]
pub struct LexiconConfig {
    pub max_words: usize,
    pub min_count: u64,
}

#[cfg(feature = "neural")]
impl Default for LexiconConfig {
    fn default() -> Self {
        Self {
            // A smaller lexicon avoids very rare/odd outputs.
            max_words: 30_000,
            min_count: 0,
        }
    }
}

#[cfg(feature = "neural")]
#[derive(Debug, Clone, Deserialize)]
struct CtcCharsFile {
    chars: Vec<String>,
    blank_idx: usize,
}

/// Resample a path to N equidistant points.
#[cfg(feature = "neural")]
fn resample_path(path: &[Point], n: usize) -> Vec<Point> {
    if path.len() < 2 || n < 2 {
        return path.to_vec();
    }

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
                resampled.push(Point {
                    x: path[j - 1].x + t * (path[j].x - path[j - 1].x),
                    y: path[j - 1].y + t * (path[j].y - path[j - 1].y),
                });
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

/// Normalize a path: center at centroid and scale by total path length.
#[cfg(feature = "neural")]
fn normalize_path(path: &[Point]) -> Vec<Point> {
    if path.len() < 2 {
        return vec![];
    }

    let mut total_len = 0.0;
    for i in 1..path.len() {
        total_len += euclidean_dist(&path[i - 1], &path[i]);
    }
    if total_len < 1e-9 {
        total_len = 1.0;
    }

    let cx = path.iter().map(|p| p.x).sum::<f64>() / path.len() as f64;
    let cy = path.iter().map(|p| p.y).sum::<f64>() / path.len() as f64;

    path.iter()
        .map(|p| Point {
            x: (p.x - cx) / total_len,
            y: (p.y - cy) / total_len,
        })
        .collect()
}

// Greedy CTC decode was used for early bring-up; lexicon beam search is used now.

/// CTC model based swipe predictor.
///
/// Enabled with the `neural` feature.
#[cfg(feature = "neural")]
pub struct CtcPredictor {
    session: Mutex<Session>,
    n_points: usize,
    blank_idx: usize,
    idx_to_char: Vec<char>,
    out_to_trie: [u8; 27],
    trie: LexiconTrie,
    vocabulary: Vec<String>,
    vocabulary_popularity: Vec<f64>,
    layout: HashMap<char, Point>,
    pop_weight: f64,
    beam_width: usize,
    expand_topk: usize,
    direction_penalty: f64,
    try_reverse: bool,
}

#[cfg(feature = "neural")]
#[derive(Clone, Copy)]
struct BeamProb {
    pb: f32,
    pnb: f32,
}

#[cfg(feature = "neural")]
impl BeamProb {
    fn total(&self) -> f32 {
        log_add(self.pb, self.pnb)
    }
}

#[cfg(feature = "neural")]
#[inline]
fn log_add(a: f32, b: f32) -> f32 {
    if a <= LOG_ZERO {
        return b;
    }
    if b <= LOG_ZERO {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

#[cfg(feature = "neural")]
#[derive(Clone)]
struct TrieNode {
    children: [u32; 26],
    terminal_word_id: u32,
    last_trie_idx: u8,
}

#[cfg(feature = "neural")]
#[derive(Clone)]
struct LexiconTrie {
    nodes: Vec<TrieNode>,
}

#[cfg(feature = "neural")]
impl LexiconTrie {
    fn new() -> Self {
        let root = TrieNode {
            children: [TRIE_NONE; 26],
            terminal_word_id: TRIE_NONE,
            last_trie_idx: 255,
        };
        Self { nodes: vec![root] }
    }

    fn insert(&mut self, word: &str, word_id: u32) {
        let mut node = 0u32;
        for b in word.bytes() {
            if !(b'a'..=b'z').contains(&b) {
                return;
            }
            let i = (b - b'a') as usize;
            let next = self.nodes[node as usize].children[i];
            let next = if next == TRIE_NONE {
                let next_id = self.nodes.len() as u32;
                self.nodes.push(TrieNode {
                    children: [TRIE_NONE; 26],
                    terminal_word_id: TRIE_NONE,
                    last_trie_idx: i as u8,
                });
                self.nodes[node as usize].children[i] = next_id;
                next_id
            } else {
                next
            };
            node = next;
        }
        self.nodes[node as usize].terminal_word_id = word_id;
    }
}

#[cfg(feature = "neural")]
impl CtcPredictor {
    /// Create a new CTC predictor from ONNX model and character mapping JSON.
    ///
    /// `chars_path` is expected to be `{"chars": ["a", ...], "blank_idx": 0}`.
    pub fn new(
        model_path: &str,
        chars_path: &str,
        freq_text: Option<&str>,
        n_points: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_with_lexicon(
            model_path,
            chars_path,
            freq_text,
            n_points,
            LexiconConfig::default(),
        )
    }

    pub fn new_with_lexicon(
        model_path: &str,
        chars_path: &str,
        freq_text: Option<&str>,
        n_points: usize,
        lexicon: LexiconConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let session = Session::builder()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        let chars_str = std::fs::read_to_string(chars_path)?;
        let chars_file: CtcCharsFile = serde_json::from_str(&chars_str)?;
        if chars_file.chars.len() != 26 {
            return Err(format!(
                "expected 26 chars in mapping, got {}",
                chars_file.chars.len()
            )
            .into());
        }

        let mut idx_to_char = vec!['?'; 27];
        idx_to_char[chars_file.blank_idx] = '_';
        for (i, s) in chars_file.chars.iter().enumerate() {
            let c = s.chars().next().ok_or("empty char string")?;
            idx_to_char[i + 1] = c;
        }

        let mut out_to_trie = [255u8; 27];
        for (i, c) in idx_to_char.iter().enumerate() {
            if i == chars_file.blank_idx {
                continue;
            }
            if ('a'..='z').contains(c) {
                out_to_trie[i] = (*c as u8) - b'a';
            }
        }

        let mut trie = LexiconTrie::new();
        let mut vocabulary: Vec<String> = Vec::new();
        let mut vocabulary_popularity: Vec<f64> = Vec::new();

        if let Some(freq_text) = freq_text {
            let mut max_count: u64 = 0;
            let mut entries: Vec<(String, u64)> = Vec::new();

            for line in freq_text.lines() {
                let (word, count_str) = match line.split_once('\t') {
                    Some(v) => v,
                    None => continue,
                };
                let word = word.trim().to_lowercase();
                if word.is_empty() {
                    continue;
                }
                let count = match count_str.trim().parse::<u64>() {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                max_count = max_count.max(count);
                entries.push((word, count));
            }

            entries.sort_by(|a, b| b.1.cmp(&a.1));
            let denom = (max_count.max(1) as f64).ln();

            for (word, count) in entries.into_iter() {
                if vocabulary.len() >= lexicon.max_words {
                    break;
                }
                if count < lexicon.min_count {
                    continue;
                }
                if !word.bytes().all(|b| (b'a'..=b'z').contains(&b)) {
                    continue;
                }

                let popularity = if count > 0 {
                    (((count as f64).ln() - 1.0) / denom).max(0.0)
                } else {
                    0.0
                };

                let word_id = vocabulary.len() as u32;
                trie.insert(&word, word_id);
                vocabulary.push(word);
                vocabulary_popularity.push(popularity);
            }
        }

        Ok(Self {
            session: Mutex::new(session),
            n_points,
            blank_idx: chars_file.blank_idx,
            idx_to_char,
            out_to_trie,
            trie,
            vocabulary,
            vocabulary_popularity,
            layout: get_keyboard_layout(),
            pop_weight: 0.25,
            beam_width: 256,
            expand_topk: 16,
            direction_penalty: 6.0,
            try_reverse: true,
        })
    }

    pub fn set_direction_penalty(&mut self, penalty: f64) {
        self.direction_penalty = penalty;
    }

    pub fn set_try_reverse(&mut self, enabled: bool) {
        self.try_reverse = enabled;
    }

    pub fn set_pop_weight(&mut self, weight: f64) {
        self.pop_weight = weight;
    }

    fn preprocess_points(&self, raw_path: &[(f64, f64)]) -> Option<Vec<Point>> {
        if raw_path.len() < 2 {
            return None;
        }
        let input_points: Vec<Point> = raw_path
            .iter()
            .map(|(x, y)| Point { x: *x, y: *y })
            .collect();
        let simplified = simplify_path(&input_points);
        if simplified.len() < 2 {
            return None;
        }

        let resampled = resample_path(&simplified, self.n_points);
        Some(normalize_path(&resampled))
    }

    fn flatten_points(points: &[Point]) -> Vec<f32> {
        let mut out = Vec::with_capacity(points.len() * 2);
        for p in points {
            out.push(p.x as f32);
            out.push(p.y as f32);
        }
        out
    }

    fn template_direction(&self, word: &str) -> Option<(f64, f64)> {
        let raw = get_word_path(word, &self.layout);
        if raw.len() < 2 {
            return None;
        }
        let simplified = simplify_path(&raw);
        if simplified.len() < 2 {
            return None;
        }
        let resampled = resample_path(&simplified, self.n_points);
        let norm = normalize_path(&resampled);
        if norm.len() < 2 {
            return None;
        }
        let dx = norm.last().unwrap().x - norm.first().unwrap().x;
        let dy = norm.last().unwrap().y - norm.first().unwrap().y;
        Some((dx, dy))
    }

    fn direction_penalty_for(&self, input_norm: &[Point], word: &str) -> f64 {
        if self.direction_penalty <= 0.0 {
            return 0.0;
        }
        if input_norm.len() < 2 {
            return 0.0;
        }
        let (tdx, tdy) = match self.template_direction(word) {
            Some(v) => v,
            None => return 0.0,
        };
        let idx = input_norm.last().unwrap().x - input_norm.first().unwrap().x;
        let idy = input_norm.last().unwrap().y - input_norm.first().unwrap().y;

        let in_norm = (idx * idx + idy * idy).sqrt();
        let t_norm = (tdx * tdx + tdy * tdy).sqrt();
        if in_norm < 1e-9 || t_norm < 1e-9 {
            return 0.0;
        }
        let dot = (idx * tdx + idy * tdy) / (in_norm * t_norm);
        if dot >= 0.0 {
            0.0
        } else {
            (-dot) * self.direction_penalty
        }
    }

    fn decode_word_ids(&self, features: Vec<f32>) -> Option<(usize, Vec<(usize, f32)>)> {
        // Model expects [batch, seq_len, 2] based on training/export.
        let input_tensor = Tensor::from_array(([1usize, self.n_points, 2], features)).ok()?;

        let mut session = self.session.lock().ok()?;
        let outputs = session.run(ort::inputs![input_tensor]).ok()?;

        let out_any = outputs.get("log_probs").unwrap_or(&outputs[0]);
        let (shape, data) = out_any.try_extract_tensor::<f32>().ok()?;
        if shape.len() != 3 {
            return None;
        }
        let seq_len = shape[0] as usize;
        let batch = shape[1] as usize;
        let classes = shape[2] as usize;
        if batch != 1 || classes != self.idx_to_char.len() {
            return None;
        }

        let blank = self.blank_idx;
        let mut beams: HashMap<u32, BeamProb> = HashMap::new();
        beams.insert(
            0,
            BeamProb {
                pb: 0.0,
                pnb: LOG_ZERO,
            },
        );

        for t in 0..seq_len {
            let base = t * (batch * classes);
            let step = &data[base..base + classes];

            let mut best: Vec<(usize, f32)> = Vec::new();
            best.reserve(self.expand_topk + 1);
            for i in 0..classes {
                if i == blank {
                    continue;
                }
                if self.out_to_trie[i] == 255 {
                    continue;
                }
                let v = step[i];
                if best.len() < self.expand_topk {
                    best.push((i, v));
                    if best.len() == self.expand_topk {
                        best.sort_by(|a, b| {
                            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                    }
                } else if let Some(last) = best.last() {
                    if v > last.1 {
                        best.pop();
                        best.push((i, v));
                        best.sort_by(|a, b| {
                            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                    }
                }
            }

            let blank_lp = step[blank];
            let mut next: HashMap<u32, BeamProb> = HashMap::new();

            for (&node_id, prob) in beams.iter() {
                let total = prob.total();

                {
                    let e = next.entry(node_id).or_insert(BeamProb {
                        pb: LOG_ZERO,
                        pnb: LOG_ZERO,
                    });
                    e.pb = log_add(e.pb, total + blank_lp);
                }

                let last_trie = self.trie.nodes[node_id as usize].last_trie_idx;
                for (out_idx, lp) in best.iter().copied() {
                    let trie_idx = self.out_to_trie[out_idx];

                    if last_trie != 255 && trie_idx == last_trie {
                        let e = next.entry(node_id).or_insert(BeamProb {
                            pb: LOG_ZERO,
                            pnb: LOG_ZERO,
                        });
                        e.pnb = log_add(e.pnb, prob.pnb + lp);
                    }

                    let child = self.trie.nodes[node_id as usize].children[trie_idx as usize];
                    if child == TRIE_NONE {
                        continue;
                    }

                    let add = if last_trie != 255 && trie_idx == last_trie {
                        prob.pb + lp
                    } else {
                        total + lp
                    };
                    let e = next.entry(child).or_insert(BeamProb {
                        pb: LOG_ZERO,
                        pnb: LOG_ZERO,
                    });
                    e.pnb = log_add(e.pnb, add);
                }
            }

            let mut scored: Vec<(u32, BeamProb, f32)> = next
                .into_iter()
                .map(|(k, v)| {
                    let s = v.total();
                    (k, v, s)
                })
                .collect();
            scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(self.beam_width);

            beams.clear();
            for (k, v, _) in scored {
                beams.insert(k, v);
            }
        }

        let mut out: Vec<(usize, f32)> = Vec::new();
        for (&node_id, prob) in beams.iter() {
            let wid = self.trie.nodes[node_id as usize].terminal_word_id;
            if wid == TRIE_NONE {
                continue;
            }
            let wid_usize = wid as usize;
            if wid_usize >= self.vocabulary.len() {
                continue;
            }
            out.push((wid_usize, prob.total()));
        }
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out.dedup_by(|a, b| a.0 == b.0);
        Some((seq_len, out))
    }

    /// Predict a decoded word (CTC) from a raw drawn path.
    pub fn predict_word(&self, raw_path: &[(f64, f64)]) -> Option<String> {
        self.predict(raw_path, 1).first().map(|p| p.word.clone())
    }

    /// Return a `Prediction` list compatible with the UI.
    ///
    /// For now we return the single decoded string plus a few popularity-boosted
    /// variants (same decoded token), to keep the UI stable.
    pub fn predict(&self, raw_path: &[(f64, f64)], limit: usize) -> Vec<Prediction> {
        if self.vocabulary.is_empty() {
            return vec![];
        }

        let norm_fwd = match self.preprocess_points(raw_path) {
            Some(v) => v,
            None => return vec![],
        };
        let features_fwd = Self::flatten_points(&norm_fwd);

        let mut best_by_word: HashMap<String, Prediction> = HashMap::new();

        if let Some((seq_len, cands)) = self.decode_word_ids(features_fwd) {
            for (wid, logp) in cands {
                let word = self.vocabulary[wid].clone();
                let raw_score = (-(logp as f64)).max(0.0);
                let popularity = *self.vocabulary_popularity.get(wid).unwrap_or(&0.0);
                let dir_pen = self.direction_penalty_for(&norm_fwd, &word);
                let rating = raw_score - (self.pop_weight * popularity * seq_len as f64) + dir_pen;
                let pred = Prediction {
                    word: word.clone(),
                    raw_score,
                    popularity,
                    rating,
                    breakdown: ScoreBreakdown::default(),
                };
                match best_by_word.get(&word) {
                    Some(existing) if existing.rating <= pred.rating => {}
                    _ => {
                        best_by_word.insert(word, pred);
                    }
                }
            }
        }

        if self.try_reverse {
            let mut raw_rev: Vec<(f64, f64)> = raw_path.to_vec();
            raw_rev.reverse();
            if let Some(norm_rev) = self.preprocess_points(&raw_rev) {
                let features_rev = Self::flatten_points(&norm_rev);
                if let Some((seq_len, cands)) = self.decode_word_ids(features_rev) {
                    for (wid, logp) in cands {
                        let word = self.vocabulary[wid].clone();
                        let raw_score = (-(logp as f64)).max(0.0);
                        let popularity = *self.vocabulary_popularity.get(wid).unwrap_or(&0.0);
                        let dir_pen = self.direction_penalty_for(&norm_rev, &word);
                        let rating =
                            raw_score - (self.pop_weight * popularity * seq_len as f64) + dir_pen;
                        let pred = Prediction {
                            word: word.clone(),
                            raw_score,
                            popularity,
                            rating,
                            breakdown: ScoreBreakdown::default(),
                        };
                        match best_by_word.get(&word) {
                            Some(existing) if existing.rating <= pred.rating => {}
                            _ => {
                                best_by_word.insert(word, pred);
                            }
                        }
                    }
                }
            }
        }

        let mut out: Vec<Prediction> = best_by_word.into_values().collect();
        out.sort_by(|a, b| {
            a.rating
                .partial_cmp(&b.rating)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        out.truncate(limit.max(1));
        out
    }
}

/// Stub implementation when neural feature is disabled.
#[cfg(not(feature = "neural"))]
pub struct CtcPredictor;

#[cfg(not(feature = "neural"))]
impl CtcPredictor {
    pub fn new(
        _model_path: &str,
        _chars_path: &str,
        _freq_text: Option<&str>,
        _n_points: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Err("Neural feature not enabled. Compile with --features neural".into())
    }

    pub fn predict_word(&self, _raw_path: &[(f64, f64)]) -> Option<String> {
        None
    }

    pub fn predict(&self, _raw_path: &[(f64, f64)], _limit: usize) -> Vec<Prediction> {
        vec![]
    }
}
