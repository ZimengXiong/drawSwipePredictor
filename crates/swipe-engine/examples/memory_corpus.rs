//! Generate fresh synthetic writers into a compact, auditable training corpus.
use std::fs::File;
use std::io::{BufWriter, Write};
use swipe_engine::synthetic::{Difficulty, Random, SyntheticWriter};

fn option<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn resample(raw: &[(f64, f64)]) -> [[f32; 2]; 64] {
    let mut cumulative = vec![0.0];
    for pair in raw.windows(2) {
        cumulative.push(
            cumulative.last().unwrap() + (pair[1].0 - pair[0].0).hypot(pair[1].1 - pair[0].1),
        );
    }
    let mut result = [[0.0_f64; 2]; 64];
    let mut segment = 1;
    for (i, point) in result.iter_mut().enumerate() {
        let target = cumulative.last().unwrap() * i as f64 / 63.0;
        while segment < raw.len() - 1 && cumulative[segment] < target {
            segment += 1;
        }
        let t = (target - cumulative[segment - 1])
            / (cumulative[segment] - cumulative[segment - 1]).max(1e-12);
        point[0] = raw[segment - 1].0 + t * (raw[segment].0 - raw[segment - 1].0);
        point[1] = raw[segment - 1].1 + t * (raw[segment].1 - raw[segment - 1].1);
    }
    let mean = [
        result.iter().map(|p| p[0]).sum::<f64>() / 64.0,
        result.iter().map(|p| p[1]).sum::<f64>() / 64.0,
    ];
    let rms = (result
        .iter()
        .map(|p| (p[0] - mean[0]).powi(2) + (p[1] - mean[1]).powi(2))
        .sum::<f64>()
        / 64.0)
        .sqrt()
        .max(1e-12);
    result.map(|p| {
        [
            ((p[0] - mean[0]) / rms) as f32,
            ((p[1] - mean[1]) / rms) as f32,
        ]
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let output = option(&args, "--output").ok_or("--output is required")?;
    let samples: u64 = option(&args, "--samples").unwrap_or("1000000").parse()?;
    let seed: u64 = option(&args, "--seed")
        .ok_or("--seed is required")?
        .parse()?;
    let vocab: usize = option(&args, "--vocab").unwrap_or("60000").parse()?;
    if samples == 0 || samples >= 1_000_003 || seed > u64::MAX / 1_000_003 - 1 {
        return Err("use 1 to 1,000,002 writers and a valid seed block".into());
    }
    let words: Vec<_> = include_str!("../../../word_freq.txt")
        .lines()
        .filter_map(|line| {
            let word = line.split_once('\t')?.0;
            (word.len() >= 2
                && word.bytes().all(|b| b.is_ascii_lowercase())
                && word.bytes().any(|b| b != word.as_bytes()[0]))
            .then_some(word)
        })
        .take(vocab)
        .collect();
    let eligible: Vec<_> = words
        .iter()
        .enumerate()
        .filter(|(_, w)| (4..=12).contains(&w.len()))
        .collect();
    if words.len() != vocab || eligible.is_empty() {
        return Err("requested vocabulary is unavailable".into());
    }
    let mut file = BufWriter::new(File::create(output)?);
    // Header: magic, record count, vocabulary size, generator seed. Each record
    // contains word id, difficulty id, writer id, and 64 normalized 2D points.
    file.write_all(b"SWPCOR01")?;
    file.write_all(&samples.to_le_bytes())?;
    file.write_all(&(vocab as u32).to_le_bytes())?;
    file.write_all(&seed.to_le_bytes())?;
    let mut verification = Vec::new();
    let mut chooser = Random::new(seed);
    for i in 0..samples {
        let (id, word) = eligible[chooser.index(eligible.len())];
        let difficulty_id = if i % 10 == 0 {
            0_u32
        } else if i % 10 < 3 {
            1
        } else {
            2
        };
        let difficulty =
            [Difficulty::Clean, Difficulty::Loose, Difficulty::Memory][difficulty_id as usize];
        let writer = seed * 1_000_003 + i;
        let points = SyntheticWriter::new(writer).draw(word, difficulty);
        file.write_all(&(id as u32).to_le_bytes())?;
        file.write_all(&difficulty_id.to_le_bytes())?;
        file.write_all(&writer.to_le_bytes())?;
        let normalized = resample(&points);
        if option(&args, "--verification").is_some() && i < 8 {
            verification.push(serde_json::json!({"word": word, "points": points, "normalized": normalized.to_vec()}));
        }
        for point in normalized {
            for value in point {
                file.write_all(&value.to_le_bytes())?;
            }
        }
    }
    file.flush()?;
    if let Some(path) = option(&args, "--verification") {
        std::fs::write(path, serde_json::to_vec(&verification)?)?;
    }
    let report = serde_json::json!({"samples": samples, "vocabulary": vocab, "seed": seed,
        "writer_start": seed * 1_000_003, "writer_end_exclusive": seed * 1_000_003 + samples,
        "target_policy": "uniform over all 4–12 letter words in the full deployment vocabulary",
        "difficulty_mix": {"clean": 0.1, "loose": 0.2, "memory": 0.7},
        "generator": "unchanged SyntheticWriter; no labels or writer ids are inference features"});
    std::fs::write(
        format!("{output}.json"),
        serde_json::to_string_pretty(&report)? + "\n",
    )?;
    println!("{report}");
    Ok(())
}
