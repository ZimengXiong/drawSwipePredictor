pub mod ctc;
pub mod dtw;
pub mod keyboard;
pub mod neural;
pub mod types;

pub use ctc::CtcPredictor;

#[cfg(feature = "neural")]
pub use ctc::LexiconConfig;
pub use neural::NeuralPredictor;

use dtw::dtw_distance_fast;
use keyboard::{euclidean_dist, get_keyboard_layout, get_word_path, simplify_path};
use std::collections::HashMap;
use types::{Dictionary, Point};

fn resample_path(path: &[Point], n: usize) -> Vec<Point> {
    if path.len() < 2 || n < 2 {
        return path.to_vec();
    }

    let mut total_len = 0.0;
    for i in 1..path.len() {
        total_len += euclidean_dist(&path[i - 1], &path[i]);
    }

    if total_len < 1e-9 {
        return vec![path[0]];
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

fn normalize_to_shape(path: &[Point], num_points: usize) -> Vec<Point> {
    if path.len() < 2 {
        return vec![];
    }

    let resampled = resample_path(path, num_points);

    let mut total_len = 0.0;
    for i in 1..resampled.len() {
        total_len += euclidean_dist(&resampled[i - 1], &resampled[i]);
    }

    if total_len < 1e-9 {
        return vec![Point { x: 0.0, y: 0.0 }];
    }

    let cx = resampled.iter().map(|p| p.x).sum::<f64>() / resampled.len() as f64;
    let cy = resampled.iter().map(|p| p.y).sum::<f64>() / resampled.len() as f64;

    resampled
        .iter()
        .map(|p| Point {
            x: (p.x - cx) / total_len,
            y: (p.y - cy) / total_len,
        })
        .collect()
}

#[derive(Clone, Debug)]
struct PathFeatures {
    shape: Vec<Point>,
    angles: Vec<f64>,
    displacements: Vec<(f64, f64)>,
    turn_angles: Vec<f64>,
    direction_histogram: [f64; 8],
    end_vector: (f64, f64),
    aspect_ratio: f64,
    straightness: f64,
    total_turning: f64,
    start_angle: f64,
    _end_angle: f64,
    segment_ratios: Vec<f64>,
}

fn extract_features(path: &[Point], num_shape_points: usize) -> Option<PathFeatures> {
    if path.len() < 2 {
        return None;
    }

    let shape = normalize_to_shape(path, num_shape_points);
    if shape.is_empty() {
        return None;
    }

    let min_x = path.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
    let max_x = path.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
    let min_y = path.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
    let max_y = path.iter().map(|p| p.y).fold(f64::NEG_INFINITY, f64::max);

    let width = max_x - min_x;
    let height = max_y - min_y;
    let aspect_ratio = if height > 1e-9 { width / height } else { 10.0 };

    let mut total_len = 0.0;
    for i in 1..path.len() {
        total_len += euclidean_dist(&path[i - 1], &path[i]);
    }
    if total_len < 1e-9 {
        total_len = 1.0;
    }

    let start = &path[0];
    let end = path.last().unwrap();
    let direct_dist = euclidean_dist(start, end);
    let straightness = direct_dist / total_len;

    let end_vector = ((end.x - start.x) / total_len, (end.y - start.y) / total_len);

    let detail_points = resample_path(path, num_shape_points.min(path.len().max(2)));

    let mut angles = Vec::new();
    let mut displacements = Vec::new();
    let mut turn_angles = Vec::new();
    let mut segment_ratios = Vec::new();
    let mut direction_histogram = [0.0f64; 8];
    let mut total_turning = 0.0;
    let mut prev_angle: Option<f64> = None;

    for i in 1..detail_points.len() {
        let dx = detail_points[i].x - detail_points[i - 1].x;
        let dy = detail_points[i].y - detail_points[i - 1].y;
        let seg_len = (dx * dx + dy * dy).sqrt();

        displacements.push((dx / total_len, dy / total_len));
        segment_ratios.push(seg_len / total_len);

        let angle = dy.atan2(dx);
        angles.push(angle);

        let bin = ((angle + std::f64::consts::PI) / (std::f64::consts::PI / 4.0)) as usize % 8;
        direction_histogram[bin] += seg_len / total_len;

        if let Some(pa) = prev_angle {
            let mut turn = angle - pa;
            while turn > std::f64::consts::PI {
                turn -= 2.0 * std::f64::consts::PI;
            }
            while turn < -std::f64::consts::PI {
                turn += 2.0 * std::f64::consts::PI;
            }
            turn_angles.push(turn);
            total_turning += turn.abs();
        }
        prev_angle = Some(angle);
    }

    let start_angle = angles.first().copied().unwrap_or(0.0);
    let end_angle = angles.last().copied().unwrap_or(0.0);

    Some(PathFeatures {
        shape,
        angles,
        displacements,
        turn_angles,
        direction_histogram,
        end_vector,
        aspect_ratio,
        straightness,
        total_turning,
        start_angle,
        _end_angle: end_angle,
        segment_ratios,
    })
}

fn angle_sequence_distance(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 1.0;
    }

    let len = a.len().max(b.len());
    let mut total = 0.0;

    for i in 0..len {
        let angle_a = a.get(i).copied().unwrap_or(0.0);
        let angle_b = b.get(i).copied().unwrap_or(0.0);

        let diff = (angle_a - angle_b).abs();
        let diff = if diff > std::f64::consts::PI {
            2.0 * std::f64::consts::PI - diff
        } else {
            diff
        };
        total += diff;
    }

    total / len as f64
}

fn displacement_sequence_distance(a: &[(f64, f64)], b: &[(f64, f64)]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 1.0;
    }

    let len = a.len().max(b.len());
    let mut total = 0.0;

    for i in 0..len {
        let (ax, ay) = a.get(i).copied().unwrap_or((0.0, 0.0));
        let (bx, by) = b.get(i).copied().unwrap_or((0.0, 0.0));

        let dx = ax - bx;
        let dy = ay - by;
        total += (dx * dx + dy * dy).sqrt();
    }

    total / len as f64
}

fn turn_sequence_distance(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    if a.is_empty() || b.is_empty() {
        return 1.0;
    }

    let len = a.len().max(b.len());
    let mut total = 0.0;

    for i in 0..len {
        let turn_a = a.get(i).copied().unwrap_or(0.0);
        let turn_b = b.get(i).copied().unwrap_or(0.0);
        total += (turn_a - turn_b).abs();
    }

    total / len as f64
}

fn histogram_distance(a: &[f64; 8], b: &[f64; 8]) -> f64 {
    let mut dist = 0.0;
    for i in 0..8 {
        let sum = a[i] + b[i];
        if sum > 1e-9 {
            let diff = a[i] - b[i];
            dist += (diff * diff) / sum;
        }
    }
    dist / 2.0
}

fn angle_distance(a: f64, b: f64) -> f64 {
    let mut diff = (a - b).abs();
    if diff > std::f64::consts::PI {
        diff = 2.0 * std::f64::consts::PI - diff;
    }
    diff / std::f64::consts::PI
}

fn location_distance(input: &[Point], word_path: &[Point], num_points: usize) -> f64 {
    if input.len() < 2 || word_path.len() < 2 {
        return 1.0;
    }

    let input_resampled = resample_path(input, num_points);
    let word_resampled = resample_path(word_path, num_points);

    let input_min_x = input_resampled
        .iter()
        .map(|p| p.x)
        .fold(f64::INFINITY, f64::min);
    let input_max_x = input_resampled
        .iter()
        .map(|p| p.x)
        .fold(f64::NEG_INFINITY, f64::max);
    let input_min_y = input_resampled
        .iter()
        .map(|p| p.y)
        .fold(f64::INFINITY, f64::min);
    let input_max_y = input_resampled
        .iter()
        .map(|p| p.y)
        .fold(f64::NEG_INFINITY, f64::max);

    let word_min_x = word_resampled
        .iter()
        .map(|p| p.x)
        .fold(f64::INFINITY, f64::min);
    let word_max_x = word_resampled
        .iter()
        .map(|p| p.x)
        .fold(f64::NEG_INFINITY, f64::max);
    let word_min_y = word_resampled
        .iter()
        .map(|p| p.y)
        .fold(f64::INFINITY, f64::min);
    let word_max_y = word_resampled
        .iter()
        .map(|p| p.y)
        .fold(f64::NEG_INFINITY, f64::max);

    let input_width = (input_max_x - input_min_x).max(0.001);
    let input_height = (input_max_y - input_min_y).max(0.001);
    let word_width = (word_max_x - word_min_x).max(0.001);
    let word_height = (word_max_y - word_min_y).max(0.001);

    let mut total = 0.0;
    for i in 0..num_points {
        let norm_x = (input_resampled[i].x - input_min_x) / input_width;
        let norm_y = (input_resampled[i].y - input_min_y) / input_height;

        let scaled_x = word_min_x + norm_x * word_width;
        let scaled_y = word_min_y + norm_y * word_height;

        let dx = scaled_x - word_resampled[i].x;
        let dy = scaled_y - word_resampled[i].y;
        total += (dx * dx + dy * dy).sqrt();
    }

    total / (num_points as f64 * 2.0)
}

fn segment_ratio_distance(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    if a.is_empty() || b.is_empty() {
        return 1.0;
    }

    let len = a.len().max(b.len());
    let mut total = 0.0;

    for i in 0..len {
        let ra = a.get(i).copied().unwrap_or(0.0);
        let rb = b.get(i).copied().unwrap_or(0.0);
        total += (ra - rb).abs();
    }

    total / len as f64
}

pub use dtw::{dtw_distance, dtw_distance_fast as dtw_fast};
pub use keyboard::{
    euclidean_dist as euclidean_distance, get_keyboard_layout as keyboard_layout,
    get_word_path as word_path, simplify_path as path_simplify,
};
pub use types::{Point as PointType, Prediction, ScoreBreakdown};

pub struct SwipeEngine {
    dictionary: Dictionary,
    layout: HashMap<char, Point>,
    pop_weight: f64,
}

impl SwipeEngine {
    pub fn new() -> Self {
        Self {
            dictionary: Dictionary::new(),
            layout: get_keyboard_layout(),
            pop_weight: 0.25,
        }
    }

    pub fn set_pop_weight(&mut self, weight: f64) {
        self.pop_weight = weight;
    }

    pub fn load_dictionary(&mut self, freq_text: &str) {
        self.dictionary.load_from_text(freq_text);
    }

    pub fn word_count(&self) -> usize {
        self.dictionary.words.len()
    }

    pub fn predict(&self, swipe_input: &str, limit: usize) -> Vec<Prediction> {
        let raw_input_path = get_word_path(swipe_input, &self.layout);

        if raw_input_path.is_empty() {
            return vec![];
        }

        let input_path = simplify_path(&raw_input_path);
        let input_len = input_path.len() as f64;

        let first_char = match swipe_input.chars().next() {
            Some(c) => c.to_ascii_lowercase(),
            None => return vec![],
        };
        let last_char = swipe_input.chars().last().unwrap().to_ascii_lowercase();
        let last_char_pt = self
            .layout
            .get(&last_char)
            .cloned()
            .unwrap_or(Point { x: 0.0, y: 0.0 });

        let window = (input_path.len() / 2).max(10);
        let mut best_score = f64::INFINITY;

        let mut candidates: Vec<(String, f64, f64)> = self
            .dictionary
            .words
            .iter()
            .filter(|w| !w.is_empty() && w.starts_with(first_char))
            .filter_map(|w| {
                let word_last_char = w.chars().last().unwrap();
                let mut end_penalty = 0.0;

                if word_last_char != last_char {
                    if let Some(word_last_pt) = self.layout.get(&word_last_char) {
                        end_penalty = euclidean_dist(&last_char_pt, word_last_pt) * 5.0;
                    } else {
                        end_penalty = 50.0;
                    }
                }

                let cutoff = best_score * input_len;
                let word_path = get_word_path(w, &self.layout);
                let dist = dtw_distance_fast(&input_path, &word_path, window, cutoff);

                if dist == f64::INFINITY {
                    return None;
                }

                let score = (dist + end_penalty) / input_len;
                if score < best_score {
                    best_score = score;
                }

                let word_freq = *self.dictionary.freq.get(w.as_str()).unwrap_or(&0.0);
                Some((w.clone(), score, word_freq))
            })
            .collect();

        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        candidates
            .into_iter()
            .take(limit)
            .map(|(word, raw_score, popularity)| Prediction {
                word,
                raw_score,
                popularity,
                rating: raw_score,
                breakdown: ScoreBreakdown::default(),
            })
            .collect()
    }

    pub fn predict_from_path(&self, raw_path: &[(f64, f64)], limit: usize) -> Vec<Prediction> {
        if raw_path.len() < 2 {
            return vec![];
        }

        let input_points: Vec<Point> = raw_path
            .iter()
            .map(|(x, y)| Point { x: *x, y: *y })
            .collect();

        let simplified = simplify_path(&input_points);

        let num_points = 32;
        let input_features = match extract_features(&simplified, num_points) {
            Some(f) => f,
            None => return vec![],
        };

        let window = 8;
        let mut best_score = f64::INFINITY;

        let mut candidates: Vec<(String, f64, f64, ScoreBreakdown)> = self
            .dictionary
            .words
            .iter()
            .filter(|w| w.len() >= 2)
            .filter_map(|w| {
                let word_path = get_word_path(w, &self.layout);
                if word_path.len() < 2 {
                    return None;
                }

                let word_features = extract_features(&word_path, num_points)?;

                let ar_ratio = if word_features.aspect_ratio > 1e-9 {
                    input_features.aspect_ratio / word_features.aspect_ratio
                } else {
                    1.0
                };
                if ar_ratio < 0.33 || ar_ratio > 3.0 {
                    return None;
                }

                let straight_diff =
                    (input_features.straightness - word_features.straightness).abs();
                if straight_diff > 0.5 {
                    return None;
                }

                let turn_ratio = if word_features.total_turning > 0.1 {
                    input_features.total_turning / word_features.total_turning
                } else if input_features.total_turning > 0.1 {
                    10.0
                } else {
                    1.0
                };
                if turn_ratio < 0.3 || turn_ratio > 3.0 {
                    return None;
                }

                let end_dot = input_features.end_vector.0 * word_features.end_vector.0
                    + input_features.end_vector.1 * word_features.end_vector.1;
                let input_mag = (input_features.end_vector.0.powi(2)
                    + input_features.end_vector.1.powi(2))
                .sqrt();
                let word_mag = (word_features.end_vector.0.powi(2)
                    + word_features.end_vector.1.powi(2))
                .sqrt();
                if input_mag > 0.1 && word_mag > 0.1 {
                    let cos_sim = end_dot / (input_mag * word_mag);
                    if cos_sim < -0.3 {
                        return None;
                    }
                }

                let start_diff =
                    angle_distance(input_features.start_angle, word_features.start_angle);
                if start_diff > 0.5 {
                    return None;
                }

                let cutoff = best_score * num_points as f64;
                let shape_dist =
                    dtw_distance_fast(&input_features.shape, &word_features.shape, window, cutoff);

                if shape_dist == f64::INFINITY {
                    return None;
                }

                let shape_score = shape_dist / num_points as f64;

                let location_dist = location_distance(&simplified, &word_path, 16);

                let angle_dist =
                    angle_sequence_distance(&input_features.angles, &word_features.angles);
                let disp_dist = displacement_sequence_distance(
                    &input_features.displacements,
                    &word_features.displacements,
                );
                let turn_dist =
                    turn_sequence_distance(&input_features.turn_angles, &word_features.turn_angles);
                let hist_dist = histogram_distance(
                    &input_features.direction_histogram,
                    &word_features.direction_histogram,
                );

                let end_dist = ((input_features.end_vector.0 - word_features.end_vector.0).powi(2)
                    + (input_features.end_vector.1 - word_features.end_vector.1).powi(2))
                .sqrt();

                let start_angle_dist =
                    angle_distance(input_features.start_angle, word_features.start_angle);

                let seg_ratio_dist = segment_ratio_distance(
                    &input_features.segment_ratios,
                    &word_features.segment_ratios,
                );

                let raw_score = shape_score * 0.15
                    + location_dist * 0.15
                    + angle_dist * 0.12
                    + disp_dist * 0.12
                    + turn_dist * 0.08
                    + start_angle_dist * 0.15
                    + seg_ratio_dist * 0.08
                    + hist_dist * 0.05
                    + end_dist * 0.10;

                if raw_score < best_score {
                    best_score = raw_score;
                }

                let popularity = *self.dictionary.freq.get(w.as_str()).unwrap_or(&0.0);

                let breakdown = ScoreBreakdown {
                    shape: shape_score,
                    location: location_dist,
                    angles: angle_dist,
                    displacements: disp_dist,
                    turns: turn_dist,
                    histogram: hist_dist,
                    end_vector: end_dist,
                    start_angle: start_angle_dist,
                    segment_ratios: seg_ratio_dist,
                };

                Some((w.clone(), raw_score, popularity, breakdown))
            })
            .collect();

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
            .map(|(word, raw_score, popularity, breakdown)| {
                let rating = raw_score - (pop_weight * popularity);
                Prediction {
                    word,
                    raw_score,
                    popularity,
                    rating,
                    breakdown,
                }
            })
            .collect()
    }

    pub fn get_layout(&self) -> &HashMap<char, Point> {
        &self.layout
    }
}

impl Default for SwipeEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let engine = SwipeEngine::new();
        assert_eq!(engine.word_count(), 0);
    }

    #[test]
    fn test_dictionary_loading() {
        let mut engine = SwipeEngine::new();
        engine.load_dictionary("hello\t1000\nworld\t500\n");
        assert_eq!(engine.word_count(), 2);
    }

    #[test]
    fn test_prediction() {
        let mut engine = SwipeEngine::new();
        engine.load_dictionary("hello\t1000\nhello\t1000\nhelp\t800\nhell\t600\n");

        let predictions = engine.predict("hello", 5);
        assert!(!predictions.is_empty());
        assert!(predictions.iter().any(|p| p.word == "hello"));
    }
}
