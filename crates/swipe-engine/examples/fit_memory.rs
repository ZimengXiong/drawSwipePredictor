//! Fit a small ranking metric using synthetic positives and retrieved negatives.
use std::time::Instant;
use swipe_engine::memory::MemoryCandidate;
use swipe_engine::synthetic::{Difficulty, Random, SyntheticWriter};
use swipe_engine::MemoryEngine;

const WORDS: &str = include_str!("../../../word_freq.txt");
const INITIAL: [f32; 6] = [1.0, 0.0, 0.0, 0.002, 0.025, 0.0];

struct Attempt {
    target: String,
    candidates: Vec<MemoryCandidate>,
}

fn generate(engine: &MemoryEngine, targets: &[&str], count: usize, seed: u64) -> Vec<Attempt> {
    let mut random = Random::new(seed);
    (0..count)
        .map(|i| {
            let target = targets[random.index(targets.len())];
            let difficulty = match i % 5 {
                0 => Difficulty::Clean,
                1 | 2 => Difficulty::Loose,
                _ => Difficulty::Memory,
            };
            let writer_seed = seed.wrapping_mul(1_000_003).wrapping_add(i as u64);
            let raw = SyntheticWriter::new(writer_seed).draw(target, difficulty);
            let candidates = engine.ranking_candidates(&raw, 192);
            if (i + 1) % 500 == 0 {
                eprintln!("seed {seed}: generated {} attempts", i + 1);
            }
            Attempt {
                target: target.to_string(),
                candidates,
            }
        })
        .collect()
}

fn accuracy(data: &[Attempt], weights: &[f32; 6]) -> (f64, f64, f64) {
    let mut top1 = 0;
    let mut top5 = 0;
    let mut retrieved = 0;
    for attempt in data {
        let Some(correct) = attempt.candidates.iter().find(|c| c.word == attempt.target) else {
            continue;
        };
        retrieved += 1;
        let score = correct.score(weights);
        let rank = attempt
            .candidates
            .iter()
            .filter(|c| {
                c.score(weights) < score || (c.score(weights) == score && c.word < correct.word)
            })
            .count();
        top1 += usize::from(rank == 0);
        top5 += usize::from(rank < 5);
    }
    let n = data.len() as f64;
    (top1 as f64 / n, top5 as f64 / n, retrieved as f64 / n)
}

fn objective(data: &[Attempt], weights: &[f32; 6]) -> f64 {
    let (top1, top5, _) = accuracy(data, weights);
    top1 + 0.15 * top5
}

fn arg(args: &[String], flag: &str, default: usize) -> Result<usize, Box<dyn std::error::Error>> {
    let Some(i) = args.iter().position(|a| a == flag) else {
        return Ok(default);
    };
    Ok(args.get(i + 1).ok_or("missing argument value")?.parse()?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let samples = arg(&args, "--samples", 1500)?;
    let vocab = arg(&args, "--vocab", 10000)?;
    let seed = arg(&args, "--seed", 101)? as u64;
    if samples < 10 || vocab < 10 {
        return Err("use at least 10 samples and 10 words".into());
    }
    let output = args
        .iter()
        .position(|a| a == "--output")
        .and_then(|i| args.get(i + 1))
        .ok_or("--output is required")?;
    let dictionary: Vec<_> = WORDS
        .lines()
        .filter(|line| {
            let Some((w, _)) = line.split_once('\t') else {
                return false;
            };
            w.len() >= 2
                && w.bytes().all(|c| c.is_ascii_lowercase())
                && w.bytes().any(|c| c != w.as_bytes()[0])
        })
        .take(vocab)
        .collect();
    let targets: Vec<_> = dictionary
        .iter()
        .filter_map(|line| {
            let word = line.split_once('\t')?.0;
            (4..=12).contains(&word.len()).then_some(word)
        })
        .collect();
    if targets.is_empty() {
        return Err("no eligible target words".into());
    }
    let mut engine = MemoryEngine::new();
    engine.load_dictionary(&dictionary.join("\n"));
    engine.set_ranking_weights(INITIAL)?;
    let start = Instant::now();
    let train = generate(&engine, &targets, samples, seed);
    let validation = generate(&engine, &targets, samples / 3, seed.wrapping_add(1));
    let mut weights = INITIAL;
    let mut best_validation = weights;
    let mut best_validation_score = objective(&validation, &weights);
    for _ in 0..3 {
        for feature in 1..6 {
            let mut best_score = objective(&train, &weights);
            let mut best_weight = weights[feature];
            for weight in [0.0, 0.001, 0.003, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8] {
                let mut trial = weights;
                trial[feature] = weight;
                let score = objective(&train, &trial);
                if score > best_score {
                    best_score = score;
                    best_weight = weight;
                }
            }
            weights[feature] = best_weight;
            let score = objective(&validation, &weights);
            if score > best_validation_score {
                best_validation_score = score;
                best_validation = weights;
            }
        }
    }
    // Validation chooses a checkpoint. Final benchmark seeds must be different.
    let report = serde_json::json!({
        "schema_version": 1, "weights": best_validation,
        "features": ["elastic", "aspect_strain", "complexity_strain", "rotation_strain", "negative_popularity", "pointwise"],
        "training_seed": seed, "validation_seed": seed.wrapping_add(1),
        "training_samples": samples, "validation_samples": samples / 3,
        "vocabulary": dictionary.len(), "synthetic_only": true,
        "initial_train": accuracy(&train, &INITIAL),
        "fitted_train": accuracy(&train, &best_validation),
        "initial_validation": accuracy(&validation, &INITIAL),
        "fitted_validation": accuracy(&validation, &best_validation),
        "metric_order": ["top1", "top5", "shortlist_recall"],
        "seconds": start.elapsed().as_secs_f64(),
    });
    std::fs::write(output, serde_json::to_string_pretty(&report)? + "\n")?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
