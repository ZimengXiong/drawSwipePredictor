//! Predict from JSONL on stdin, or verify native inference against PyTorch.
use serde::Deserialize;
use std::io::{BufRead, BufReader};
use swipe_engine::RetrievalEngine;

#[derive(Deserialize)]
struct Verification {
    word: String,
    points: Vec<(f64, f64)>,
    embedding: Vec<f32>,
}

#[derive(Deserialize)]
struct CharacterVerification {
    word: String,
    points: Vec<(f64, f64)>,
    log_probabilities: Vec<[f32; 27]>,
    predictions: Vec<ExpectedPrediction>,
}

#[derive(Deserialize)]
struct ExpectedPrediction {
    word: String,
    rating: f64,
}

fn option<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let directory = option(&args, "--model").ok_or("--model directory is required")?;
    let mut engine = RetrievalEngine::from_files(directory)?;
    if let Some(count) = option(&args, "--vocab") {
        engine.truncate_vocabulary(count.parse()?);
    }
    if let Some(path) = option(&args, "--verify-character") {
        let cases: Vec<CharacterVerification> =
            serde_json::from_str(&std::fs::read_to_string(path)?)?;
        for case in cases {
            let actual = engine
                .character_probabilities(&case.points)
                .ok_or("invalid character input")?;
            if actual.len() != case.log_probabilities.len() {
                return Err("character verification dimensions differ".into());
            }
            let error = actual
                .iter()
                .flatten()
                .zip(case.log_probabilities.iter().flatten())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            if error > 0.0002 {
                return Err(format!("character probabilities differ by {error}").into());
            }
            let predictions = engine.predict_from_path(&case.points, case.predictions.len());
            if predictions.len() != case.predictions.len() {
                return Err("missing character predictions".into());
            }
            for (actual, expected) in predictions.iter().zip(&case.predictions) {
                if actual.word != expected.word || (actual.rating - expected.rating).abs() > 0.002 {
                    return Err(format!(
                        "character ranking differs for {}: {} versus {}",
                        case.word, actual.word, expected.word
                    )
                    .into());
                }
            }
            println!("{}: maximum character log-probability error {error:.8}; exact top-five ranking matches", case.word);
        }
        return Ok(());
    }
    if let Some(path) = option(&args, "--verify") {
        let cases: Vec<Verification> = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        for case in cases {
            let actual = engine
                .embedding(&case.points)
                .ok_or("invalid verification path")?;
            if actual.len() != case.embedding.len() {
                return Err("verification dimensions differ".into());
            }
            let error = actual
                .iter()
                .zip(case.embedding)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            println!("{}: maximum embedding error {error:.8}", case.word);
            if error > 0.0001 {
                return Err("native inference does not match PyTorch".into());
            }
        }
        return Ok(());
    }
    for line in BufReader::new(std::io::stdin()).lines() {
        let value: serde_json::Value = serde_json::from_str(&line?)?;
        let points: Vec<(f64, f64)> = serde_json::from_value(value["points"].clone())?;
        let start = std::time::Instant::now();
        if args.iter().any(|a| a == "--features") {
            let count = option(&args, "--candidates").unwrap_or("32").parse()?;
            println!(
                "{}",
                serde_json::json!({"word": value["word"], "candidates": engine.ranking_features(&points, count)})
            );
            continue;
        }
        let predictions = engine.predict_from_path(&points, 5);
        println!(
            "{}",
            serde_json::json!({"predictions": predictions, "milliseconds": start.elapsed().as_secs_f64() * 1000.0})
        );
    }
    Ok(())
}
