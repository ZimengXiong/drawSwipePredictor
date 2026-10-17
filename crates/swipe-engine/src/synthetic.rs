//! Reproducible gestures from a latent writer and a distorted mental keyboard.
//!
//! This generator uses quadratic corner rounding rather than the recognizer's
//! template smoothing. Separate writer seeds when creating evaluation splits.
use crate::memory_geometry::key_vertices;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Difficulty {
    Clean,
    Loose,
    Memory,
}

impl Difficulty {
    fn strength(self) -> f64 {
        match self {
            Self::Clean => 0.0,
            Self::Loose => 0.55,
            Self::Memory => 1.0,
        }
    }
}

/// Small deterministic PRNG. Seeds are recorded with generated datasets.
pub struct Random(u64);

impl Random {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1_u64 << 53) as f64
    }

    pub fn signed(&mut self) -> f64 {
        2.0 * self.unit() - 1.0
    }

    pub fn index(&mut self, len: usize) -> usize {
        (self.unit() * len as f64) as usize
    }
}

/// A writer has consistent key-placement errors, row offsets, and motor style.
/// Create a new writer for each held-out identity, then draw multiple words.
pub struct SyntheticWriter {
    rng: Random,
    key_error: [[f64; 2]; 26],
    row_shift: [f64; 3],
    angle: f64,
    aspect: f64,
    slant: f64,
    rounding: f64,
}

fn mix(a: [f64; 2], b: [f64; 2], t: f64) -> [f64; 2] {
    [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
}

impl SyntheticWriter {
    pub fn new(seed: u64) -> Self {
        let mut rng = Random::new(seed);
        let mut key_error = [[0.0; 2]; 26];
        for p in &mut key_error {
            *p = [rng.signed() * 0.38, rng.signed() * 0.28];
        }
        let row_shift = [rng.signed() * 0.4, rng.signed() * 0.4, rng.signed() * 0.4];
        let angle = rng.signed() * 28.0_f64.to_radians();
        let aspect = (rng.signed() * 0.65).exp();
        let slant = rng.signed() * 0.25;
        let rounding = 0.1 + rng.unit() * 0.35;
        Self {
            rng,
            key_error,
            row_shift,
            angle,
            aspect,
            slant,
            rounding,
        }
    }

    /// Draw one lowercase ASCII word. Returns screen-space points.
    ///
    /// Repeated consecutive letters share a waypoint: their spelling cannot be
    /// recovered from geometry alone. Nonmoving words return an empty gesture.
    pub fn draw(&mut self, word: &str, difficulty: Difficulty) -> Vec<(f64, f64)> {
        if !word.bytes().all(|c| c.is_ascii_lowercase()) {
            return Vec::new();
        }
        let strength = difficulty.strength();
        let layout = crate::keyboard::get_keyboard_layout();
        let mut keys = Vec::new();
        let mut previous = None;
        for c in word.chars() {
            if previous == Some(c) {
                continue;
            }
            previous = Some(c);
            let p = layout[&c];
            let error = self.key_error[c as usize - 'a' as usize];
            let row = p.y as usize;
            keys.push([
                p.x + strength * (error[0] + self.row_shift[row]),
                p.y + strength * error[1],
            ]);
        }
        if keys.len() < 2 || key_vertices(word).len() < 2 {
            return Vec::new();
        }
        // A remembered route can omit a weak intermediate waypoint. Preserve
        // endpoints and avoid deleting every turn from a short gesture.
        if keys.len() > 4 && self.rng.unit() < strength * 0.35 {
            let i = 1 + self.rng.index(keys.len() - 2);
            keys.remove(i);
        }
        let last = keys.len() - 1;
        for i in [0, last] {
            keys[i][0] += strength * self.rng.signed() * 0.3;
            keys[i][1] += strength * self.rng.signed() * 0.25;
        }
        // Construct rounded corners directly in key space. Each corner uses a
        // quadratic Bezier, with asymmetric entry and exit distances.
        let mut curve = vec![keys[0]];
        for i in 1..last {
            let amount = self.rounding * strength;
            let entry = mix(keys[i], keys[i - 1], amount);
            let exit = mix(
                keys[i],
                keys[i + 1],
                amount * (0.65 + 0.7 * self.rng.unit()),
            );
            let start = *curve.last().unwrap();
            for j in 1..=8 {
                curve.push(mix(start, entry, j as f64 / 8.0));
            }
            for j in 1..=10 {
                let t = j as f64 / 10.0;
                curve.push(mix(mix(entry, keys[i], t), mix(keys[i], exit, t), t));
            }
        }
        let start = *curve.last().unwrap();
        for j in 1..=12 {
            curve.push(mix(start, keys[last], j as f64 / 12.0));
        }

        let scale = 30.0 + self.rng.unit() * 100.0;
        let offset = [self.rng.signed() * 600.0, self.rng.signed() * 400.0];
        let angle = strength * (self.angle + self.rng.signed() * 4.0_f64.to_radians());
        let (sin, cos) = angle.sin_cos();
        let aspect = self.aspect.powf(strength);
        let drift = [self.rng.signed() * 0.45, self.rng.signed() * 0.4];
        let phase = self.rng.unit() * std::f64::consts::TAU;
        let speed = (self.rng.signed() * 0.6).exp();
        // Nonuniform acquisition changes point density independently of shape.
        let count = 45 + self.rng.index(140);
        (0..count)
            .map(|i| {
                let t = (i as f64 / (count - 1) as f64).powf(speed);
                let pos = t * (curve.len() - 1) as f64;
                let k = (pos as usize).min(curve.len() - 2);
                let p = mix(curve[k], curve[k + 1], pos - k as f64);
                let wave = (std::f64::consts::TAU * t + phase).sin();
                let x = p[0] + strength * (self.slant * p[1] + drift[0] * t + 0.08 * wave);
                let y = p[1] * aspect + strength * (drift[1] * t + 0.06 * wave);
                let x = x + strength * self.rng.signed() * 0.025;
                let y = y + strength * self.rng.signed() * 0.025;
                (
                    offset[0] + scale * (cos * x - sin * y),
                    offset[1] + scale * (sin * x + cos * y),
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_is_reproducible_and_writer_specific() {
        let a = SyntheticWriter::new(42).draw("lunchbox", Difficulty::Memory);
        let b = SyntheticWriter::new(42).draw("lunchbox", Difficulty::Memory);
        let c = SyntheticWriter::new(43).draw("lunchbox", Difficulty::Memory);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert!(a.iter().all(|p| p.0.is_finite() && p.1.is_finite()));
    }

    #[test]
    fn nonmoving_or_invalid_words_have_no_gesture() {
        let mut writer = SyntheticWriter::new(7);
        for word in ["", "a", "aaa", "hello!", "世界"] {
            assert!(writer.draw(word, Difficulty::Loose).is_empty());
        }
        assert!(!writer.draw("hello", Difficulty::Clean).is_empty());
    }
}
