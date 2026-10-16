//! Compare recognizers on identical gestures or export labeled synthetic data.
use serde::Serialize;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;
use swipe_engine::synthetic::{Difficulty, Random, SyntheticWriter};
use swipe_engine::{MemoryEngine, RetrievalEngine, SwipeEngine};

const WORDS: &str = include_str!("../../../word_freq.txt");

#[derive(Serialize)]
struct Sample<'a> {
    word: &'a str,
    writer_seed: u64,
    difficulty: Difficulty,
    points: Vec<(f64, f64)>,
}

#[derive(Default, Serialize)]
struct Metrics {
    top1: usize,
    top5: usize,
    milliseconds: Vec<f64>,
}

impl Metrics {
    fn add(&mut self, target: &str, predictions: &[swipe_engine::Prediction], elapsed: f64) {
        self.top1 += usize::from(predictions.first().is_some_and(|p| p.word == target));
        self.top5 += usize::from(predictions.iter().take(5).any(|p| p.word == target));
        self.milliseconds.push(elapsed);
    }

    fn summary(&self, samples: usize) -> serde_json::Value {
        let mut times = self.milliseconds.clone();
        times.sort_by(f64::total_cmp);
        serde_json::json!({
            "top1": self.top1 as f64 / samples as f64,
            "top5": self.top5 as f64 / samples as f64,
            "median_ms": times[times.len() / 2],
            "p95_ms": times[((times.len() - 1) as f64 * 0.95) as usize],
        })
    }
}

fn value(args: &[String], name: &str, default: usize) -> Result<usize, Box<dyn std::error::Error>> {
    let Some(i) = args.iter().position(|a| a == name) else {
        return Ok(default);
    };
    Ok(args.get(i + 1).ok_or("missing argument value")?.parse()?)
}

fn filename<'a>(
    args: &'a [String],
    name: &str,
) -> Result<Option<&'a str>, Box<dyn std::error::Error>> {
    match args.iter().position(|a| a == name) {
        Some(i) => Ok(Some(args.get(i + 1).ok_or("missing filename")?)),
        None => Ok(None),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.iter().any(|a| a == "--help") {
        println!("memory_bench [--vocab N] [--target-vocab N] [--target-offset N] [--samples N] [--seed N] [--output report.json] [--dataset samples.jsonl] [--export-only]");
        return Ok(());
    }
    let vocab = value(&args, "--vocab", 10_000)?;
    let samples = value(&args, "--samples", 100)?;
    let target_vocab = value(&args, "--target-vocab", 10000)?;
    let target_offset = value(&args, "--target-offset", 0)?;
    let seed = value(&args, "--seed", 20261015)? as u64;
    if vocab < 10 || samples == 0 {
        return Err("use --vocab >= 10 and --samples > 0".into());
    }
    let dictionary: Vec<_> = WORDS
        .lines()
        .filter(|line| {
            let Some((word, _)) = line.split_once('\t') else {
                return false;
            };
            word.len() >= 2
                && word.bytes().all(|c| c.is_ascii_lowercase())
                && word.bytes().any(|c| c != word.as_bytes()[0])
        })
        .take(vocab)
        .collect();
    let targets: Vec<_> = dictionary
        .iter()
        .skip(target_offset)
        .take(target_vocab)
        .filter_map(|line| {
            let word = line.split_once('\t')?.0;
            (4..=12).contains(&word.len()).then_some(word)
        })
        .collect();
    if targets.is_empty() {
        return Err("no eligible targets".into());
    }
    let dataset_path = filename(&args, "--dataset")?;
    let export_only = args.iter().any(|a| a == "--export-only");
    if export_only && dataset_path.is_none() {
        return Err("--export-only requires --dataset".into());
    }
    let mut dataset = dataset_path
        .map(File::create)
        .transpose()?
        .map(BufWriter::new);
    let text = dictionary.join("\n");
    let mut baseline = SwipeEngine::new();
    let mut memory = MemoryEngine::new();
    let start = Instant::now();
    if !export_only {
        baseline.load_dictionary(&text);
        memory.load_dictionary(&text);
        eprintln!(
            "Cached {} words in {:.2}s",
            memory.word_count(),
            start.elapsed().as_secs_f64()
        );
    }
    let mut neural = filename(&args, "--model")?
        .map(RetrievalEngine::from_files)
        .transpose()?;
    if let Some(engine) = &mut neural {
        if engine.word_count() < dictionary.len() {
            return Err("neural bank does not cover the benchmark vocabulary".into());
        }
        engine.truncate_vocabulary(dictionary.len());
    }
    let mut results = Vec::new();
    for difficulty in [Difficulty::Clean, Difficulty::Loose, Difficulty::Memory] {
        // Reuse targets and writer identities across difficulty levels so the
        // difficulty comparison is paired. Each attempt has an independent seed.
        let mut random = Random::new(seed);
        let mut old_metrics = Metrics::default();
        let mut new_metrics = Metrics::default();
        let mut neural_metrics = Metrics::default();
        let mut failures = Vec::new();
        for i in 0..samples {
            let target = targets[random.index(targets.len())];
            let writer_seed = seed.wrapping_mul(1_000_003).wrapping_add(i as u64);
            let points = SyntheticWriter::new(writer_seed).draw(target, difficulty);
            if !export_only {
                let start = Instant::now();
                let old = baseline.predict_from_path(&points, 5);
                old_metrics.add(target, &old, start.elapsed().as_secs_f64() * 1000.0);
                let start = Instant::now();
                let new = memory.predict_from_path(&points, 5);
                new_metrics.add(target, &new, start.elapsed().as_secs_f64() * 1000.0);
                if let Some(engine) = &neural {
                    let start = Instant::now();
                    let learned = engine.predict_from_path(&points, 5);
                    neural_metrics.add(target, &learned, start.elapsed().as_secs_f64() * 1000.0);
                }
                if new.first().is_none_or(|p| p.word != target) && failures.len() < 12 {
                    failures.push(serde_json::json!({"target": target, "predicted": new.iter().map(|p| &p.word).collect::<Vec<_>>() }));
                }
            }
            if let Some(file) = &mut dataset {
                serde_json::to_writer(
                    &mut *file,
                    &Sample {
                        word: target,
                        writer_seed,
                        difficulty,
                        points,
                    },
                )?;
                writeln!(file)?;
            }
            if !export_only && (i + 1) % 50 == 0 {
                eprintln!("{difficulty:?}: {}/{samples}", i + 1);
            }
        }
        if !export_only {
            let result = serde_json::json!({
                "difficulty": difficulty, "baseline": old_metrics.summary(samples),
                "memory": new_metrics.summary(samples),
                "neural": neural.as_ref().map(|_| neural_metrics.summary(samples)), "example_errors": failures,
            });
            eprintln!("{}", serde_json::to_string(&result)?);
            results.push(result);
        }
    }
    if let Some(file) = &mut dataset {
        file.flush()?;
    }
    let report = serde_json::json!({
        "seed": seed, "samples_per_difficulty": samples, "vocabulary": dictionary.len(),
        "target_policy": "uniform over 4–12 letter words in target vocabulary",
        "target_vocabulary_limit": target_vocab, "target_offset": target_offset,
        "eligible_targets": targets.len(),
        "synthetic_only": true, "neural_model": filename(&args, "--model")?, "results": results,
    });
    if let Some(path) = filename(&args, "--output")? {
        std::fs::write(path, serde_json::to_string_pretty(&report)? + "\n")?;
    }
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
