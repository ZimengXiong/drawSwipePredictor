//! Export unrounded waypoint targets without changing the stroke generator.
use serde::{Deserialize, Serialize};
use std::io::{BufRead, Write};
use swipe_engine::keyboard::get_keyboard_layout;
use swipe_engine::synthetic::{Difficulty, Random, SyntheticWriter};

#[derive(Deserialize)]
struct Request {
    word: String,
    writer: u64,
    difficulty: Difficulty,
}

#[derive(Serialize)]
struct TrainingSample {
    points: Vec<(f64, f64)>,
    waypoints: Vec<[f64; 2]>,
    visited_keys: String,
    arc_positions: Vec<f64>,
}

fn annotate(request: &Request) -> Result<TrainingSample, String> {
    let points = SyntheticWriter::new(request.writer).draw(&request.word, request.difficulty);
    if points.is_empty() {
        return Err("waypoint labels require a moving lowercase ASCII word".into());
    }
    let strength = match request.difficulty {
        Difficulty::Clean => 0.0,
        Difficulty::Loose => 0.55,
        Difficulty::Memory => 1.0,
    };
    let mut rng = Random::new(request.writer);
    let mut errors = [[0.0; 2]; 26];
    for error in &mut errors {
        *error = [rng.signed() * 0.38, rng.signed() * 0.28];
    }
    let shifts = [rng.signed() * 0.4, rng.signed() * 0.4, rng.signed() * 0.4];
    let angle = rng.signed() * 28.0_f64.to_radians();
    let aspect = (rng.signed() * 0.65).exp();
    let slant = rng.signed() * 0.25;
    let _rounding = 0.1 + rng.unit() * 0.35;
    let layout = get_keyboard_layout();
    let mut keys = Vec::new();
    let mut previous = None;
    for letter in request.word.chars() {
        if previous == Some(letter) {
            continue;
        }
        previous = Some(letter);
        let point = layout[&letter];
        let error = errors[letter as usize - 'a' as usize];
        keys.push((
            letter,
            [
                point.x + strength * (error[0] + shifts[point.y as usize]),
                point.y + strength * error[1],
            ],
        ));
    }
    if keys.len() > 4 && rng.unit() < strength * 0.35 {
        let index = 1 + rng.index(keys.len() - 2);
        keys.remove(index);
    }
    let last = keys.len() - 1;
    for index in [0, last] {
        keys[index].1[0] += strength * rng.signed() * 0.3;
        keys[index].1[1] += strength * rng.signed() * 0.25;
    }
    // Each rounded interior corner consumes one asymmetric-exit draw.
    for _ in 1..last {
        rng.unit();
    }
    let scale = 30.0 + rng.unit() * 100.0;
    let offset = [rng.signed() * 600.0, rng.signed() * 400.0];
    let angle = strength * (angle + rng.signed() * 4.0_f64.to_radians());
    let (sin, cos) = angle.sin_cos();
    let aspect = aspect.powf(strength);
    let drift = [rng.signed() * 0.45, rng.signed() * 0.4];
    let phase = rng.unit() * std::f64::consts::TAU;
    let speed = (rng.signed() * 0.6).exp();
    let count = 45 + rng.index(140);
    if points.len() != count {
        return Err("waypoint RNG replay differs from the unchanged generator".into());
    }
    let denominator = (18 * (keys.len() - 2) + 12) as f64;
    let waypoints = keys
        .iter()
        .enumerate()
        .map(|(index, (_, point))| {
            // Use the midpoint of the corner's quadratic segment for its drift
            // time. Endpoints use their exact acquisition times. Per-point jitter
            // is observation noise and is excluded from the target.
            let time = if index == 0 {
                0.0
            } else if index == last {
                1.0
            } else {
                (18 * (index - 1) + 13) as f64 / denominator
            };
            let wave = (std::f64::consts::TAU * time + phase).sin();
            let x = point[0] + strength * (slant * point[1] + drift[0] * time + 0.08 * wave);
            let y = point[1] * aspect + strength * (drift[1] * time + 0.06 * wave);
            [
                offset[0] + scale * (cos * x - sin * y),
                offset[1] + scale * (sin * x + cos * y),
            ]
        })
        .collect();
    let mut cumulative = vec![0.0];
    for pair in points.windows(2) {
        cumulative.push(
            cumulative.last().unwrap() + (pair[1].0 - pair[0].0).hypot(pair[1].1 - pair[0].1),
        );
    }
    let total = *cumulative.last().unwrap();
    if total <= 0.0 {
        return Err("arc supervision requires a moving observed stroke".into());
    }
    let arc_positions = (0..keys.len())
        .map(|index| {
            let time = if index == 0 {
                0.0
            } else if index == last {
                1.0
            } else {
                (18 * (index - 1) + 13) as f64 / denominator
            };
            // Invert acquisition timing, then express the corner midpoint in the
            // observed polyline's arc coordinates. Timing remains supervision only.
            let position = time.powf(speed.recip()) * (count - 1) as f64;
            let lower = (position as usize).min(count - 2);
            let fraction = position - lower as f64;
            (cumulative[lower] + fraction * (cumulative[lower + 1] - cumulative[lower])) / total
        })
        .collect();
    Ok(TrainingSample {
        points,
        arc_positions,
        waypoints,
        visited_keys: keys.iter().map(|(letter, _)| *letter).collect(),
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut output = std::io::BufWriter::new(std::io::stdout().lock());
    for line in std::io::stdin().lock().lines() {
        let request: Request = serde_json::from_str(&line?)?;
        serde_json::to_writer(&mut output, &annotate(&request)?)?;
        writeln!(output)?;
    }
    output.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_strokes_remain_identical_and_endpoint_targets_exclude_only_jitter() {
        for writer in 0..80 {
            for difficulty in [Difficulty::Clean, Difficulty::Loose, Difficulty::Memory] {
                for word in ["bookkeeper", "abaca", "lunchbox", "read", "za"] {
                    let sample = annotate(&Request {
                        word: word.into(),
                        writer,
                        difficulty,
                    })
                    .unwrap();
                    let raw = SyntheticWriter::new(writer).draw(word, difficulty);
                    assert_eq!(sample.points, raw);
                    assert_eq!(sample.waypoints.len(), sample.visited_keys.len());
                    assert_eq!(sample.arc_positions.len(), sample.waypoints.len());
                    assert_eq!(sample.arc_positions[0], 0.0);
                    assert_eq!(*sample.arc_positions.last().unwrap(), 1.0);
                    assert!(sample
                        .arc_positions
                        .windows(2)
                        .all(|pair| pair[0] <= pair[1]));
                    let maximum = if matches!(difficulty, Difficulty::Clean) {
                        1e-9
                    } else {
                        130.0 * 0.025 * 2.0_f64.sqrt() + 1e-9
                    };
                    for (point, target) in [
                        (raw[0], sample.waypoints[0]),
                        (*raw.last().unwrap(), *sample.waypoints.last().unwrap()),
                    ] {
                        assert!((point.0 - target[0]).hypot(point.1 - target[1]) <= maximum);
                    }
                }
            }
        }
    }

    #[test]
    fn clean_corner_arc_position_matches_physical_segment_lengths() {
        let expected = 1.25_f64.sqrt() / (1.25_f64.sqrt() + 2.0_f64.sqrt());
        for writer in 0..80 {
            let sample = annotate(&Request {
                word: "qaz".into(),
                writer,
                difficulty: Difficulty::Clean,
            })
            .unwrap();
            assert!((sample.arc_positions[1] - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn clean_targets_preserve_the_keyboard_geometry_and_collapsed_visits() {
        let sample = annotate(&Request {
            word: "bookkeeper".into(),
            writer: 71,
            difficulty: Difficulty::Clean,
        })
        .unwrap();
        assert_eq!(sample.visited_keys, "bokeper");
        let layout = get_keyboard_layout();
        let first = layout[&'b'];
        let last = layout[&'r'];
        let observed = sample.waypoints.last().unwrap()[0] - sample.waypoints[0][0];
        let scale = observed / (last.x - first.x);
        for (letter, point) in sample.visited_keys.chars().zip(&sample.waypoints) {
            let key = layout[&letter];
            assert!((point[0] - sample.waypoints[0][0] - scale * (key.x - first.x)).abs() < 1e-9);
            assert!((point[1] - sample.waypoints[0][1] - scale * (key.y - first.y)).abs() < 1e-9);
        }
    }
}
