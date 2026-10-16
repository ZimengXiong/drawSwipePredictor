//! Evaluate real recordings and a separately labeled memory-transfer stress test.
use serde::Deserialize;
use std::collections::HashSet;
use std::io::{BufRead, BufReader};
use std::time::Instant;
use swipe_engine::synthetic::Random;
use swipe_engine::{MemoryEngine, RetrievalEngine, SwipeEngine};

#[derive(Deserialize)]
struct Sample {
    word: String,
    points: Vec<(f64, f64)>,
    session: String,
}

fn option<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn transfer(sample: &Sample) -> Vec<(f64, f64)> {
    let seed = sample
        .session
        .bytes()
        .fold(9003_u64, |h, b| h.wrapping_mul(31).wrapping_add(b as u64));
    let mut random = Random::new(seed);
    let angle = random.signed() * 24.0_f64.to_radians();
    let stretch = (random.signed() * 0.6).exp();
    let slant = random.signed() * 0.2;
    let (sin, cos) = angle.sin_cos();
    let origin = sample.points[0];
    let extent = sample
        .points
        .iter()
        .map(|p| (p.0 - origin.0).hypot(p.1 - origin.1))
        .fold(0.0_f64, f64::max);
    let drift = [
        random.signed() * extent * 0.06,
        random.signed() * extent * 0.06,
    ];
    sample
        .points
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let t = i as f64 / (sample.points.len() - 1).max(1) as f64;
            let x = p.0 - origin.0 + slant * (p.1 - origin.1) + drift[0] * t;
            let y = (p.1 - origin.1) * stretch + drift[1] * t;
            (500.0 + cos * x - sin * y, 200.0 + sin * x + cos * y)
        })
        .collect()
}

fn summary(top1: usize, top5: usize, times: &mut [f64]) -> serde_json::Value {
    times.sort_by(f64::total_cmp);
    let n = times.len();
    serde_json::json!({
        "top1": top1 as f64 / n as f64, "top5": top5 as f64 / n as f64,
        "median_ms": times[n / 2], "p95_ms": times[(n - 1) * 95 / 100],
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let dataset = option(&args, "--dataset").ok_or("--dataset JSONL is required")?;
    let output = option(&args, "--output").ok_or("--output report.json is required")?;
    let samples: usize = option(&args, "--samples").unwrap_or("300").parse()?;
    let vocab: usize = option(&args, "--vocab").unwrap_or("10000").parse()?;
    if samples == 0 || vocab == 0 {
        return Err("samples and vocabulary must be positive".into());
    }
    let dictionary: Vec<_> = include_str!("../../../word_freq.txt")
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
    let vocabulary: HashSet<_> = dictionary
        .iter()
        .map(|line| line.split_once('\t').unwrap().0)
        .collect();
    let mut data = Vec::new();
    let mut skipped = 0;
    for line in BufReader::new(std::fs::File::open(dataset)?).lines() {
        let sample: Sample = serde_json::from_str(&line?)?;
        if !vocabulary.contains(sample.word.as_str()) {
            skipped += 1;
            continue;
        }
        if sample.points.len() < 2
            || sample
                .points
                .iter()
                .any(|p| !p.0.is_finite() || !p.1.is_finite())
        {
            return Err("dataset contains an invalid gesture".into());
        }
        data.push(sample);
        if data.len() >= samples {
            break;
        }
    }
    if data.is_empty() {
        return Err("no in-vocabulary samples".into());
    }
    let mut old = SwipeEngine::new();
    old.load_dictionary(&dictionary.join("\n"));
    let mut memory = MemoryEngine::new();
    memory.load_dictionary(&dictionary.join("\n"));
    if let Some(path) = option(&args, "--metric") {
        let metric: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        memory.set_ranking_weights(serde_json::from_value(metric["weights"].clone())?)?;
    }
    let mut neural = option(&args, "--model")
        .map(RetrievalEngine::from_files)
        .transpose()?;
    if let Some(engine) = &mut neural {
        engine.truncate_vocabulary(vocab);
    }
    let backends = if neural.is_some() { 3 } else { 2 };
    let mut results = Vec::new();
    for mode in ["recorded", "memory_transfer"] {
        let mut counts = [[0_usize; 2]; 3];
        let mut times = [Vec::new(), Vec::new(), Vec::new()];
        let mut errors = Vec::new();
        for (i, sample) in data.iter().enumerate() {
            let points = if mode == "recorded" {
                sample.points.clone()
            } else {
                transfer(sample)
            };
            for backend in 0..backends {
                let start = Instant::now();
                let predictions = if backend == 0 {
                    old.predict_from_path(&points, 5)
                } else if backend == 1 {
                    memory.predict_from_path(&points, 5)
                } else {
                    neural.as_ref().unwrap().predict_from_path(&points, 5)
                };
                times[backend].push(start.elapsed().as_secs_f64() * 1000.0);
                counts[backend][0] +=
                    usize::from(predictions.first().is_some_and(|p| p.word == sample.word));
                counts[backend][1] +=
                    usize::from(predictions.iter().any(|p| p.word == sample.word));
                if backend == 1
                    && predictions.first().is_none_or(|p| p.word != sample.word)
                    && errors.len() < 12
                {
                    errors.push(serde_json::json!({"target": sample.word, "predicted": predictions.iter().map(|p| &p.word).collect::<Vec<_>>() }));
                }
            }
            if (i + 1) % 100 == 0 {
                eprintln!("{mode}: {}/{}", i + 1, data.len());
            }
        }
        let result = serde_json::json!({
            "mode": mode, "baseline": summary(counts[0][0], counts[0][1], &mut times[0]),
            "memory": summary(counts[1][0], counts[1][1], &mut times[1]),
            "neural": if backends == 3 { Some(summary(counts[2][0], counts[2][1], &mut times[2])) } else { None },
            "example_errors": errors,
        });
        eprintln!("{}", serde_json::to_string(&result)?);
        results.push(result);
    }
    let report = serde_json::json!({
        "dataset": dataset, "samples": data.len(), "vocabulary": dictionary.len(), "skipped_oov": skipped,
        "sessions": data.iter().map(|s| &s.session).collect::<HashSet<_>>().len(),
        "metric_override": option(&args, "--metric"), "neural_model": option(&args, "--model"), "results": results,
        "limitations": "Recorded data uses visible keyboards. Memory transfer adds synthetic distortions to real traces and does not measure freehand memory accuracy.",
    });
    std::fs::write(output, serde_json::to_string_pretty(&report)? + "\n")?;
    Ok(())
}
