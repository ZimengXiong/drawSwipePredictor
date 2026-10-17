//! Export visited-key training labels for first draws from independent writers.
use serde::Deserialize;
use std::io::{BufRead, Write};
use swipe_engine::synthetic::{Difficulty, Random};

#[derive(Deserialize)]
struct Request {
    word: String,
    writer: u64,
    difficulty: Difficulty,
}

fn visited_route(request: &Request) -> Result<String, String> {
    if !request.word.bytes().all(|c| c.is_ascii_lowercase()) {
        return Err("route labels require lowercase ASCII words".into());
    }
    let strength = match request.difficulty {
        Difficulty::Clean => 0.0,
        Difficulty::Loose => 0.55,
        Difficulty::Memory => 1.0,
    };
    let mut keys = Vec::new();
    for letter in request.word.chars() {
        if keys.last() != Some(&letter) {
            keys.push(letter);
        }
    }
    if keys.len() < 2 {
        return Err("route labels require a moving word".into());
    }
    let mut rng = Random::new(request.writer);
    // Match SyntheticWriter::new: 52 key offsets, three row offsets, and
    // four style parameters. This exporter never changes the stroke generator.
    for _ in 0..59 {
        rng.unit();
    }
    if keys.len() > 4 && rng.unit() < strength * 0.35 {
        let index = 1 + rng.index(keys.len() - 2);
        keys.remove(index);
    }
    // Removing a key can create adjacent duplicates; retain both visits.
    Ok(keys.into_iter().collect())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut output = std::io::BufWriter::new(std::io::stdout().lock());
    for line in std::io::stdin().lock().lines() {
        let request: Request = serde_json::from_str(&line?)?;
        serde_json::to_writer(
            &mut output,
            &serde_json::json!({"visited_keys": visited_route(&request)?}),
        )?;
        writeln!(output)?;
    }
    output.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_routes_collapse_spelling_duplicates_but_never_drop_keys() {
        for writer in 0..100 {
            let request = Request {
                word: "bookkeeper".into(),
                writer,
                difficulty: Difficulty::Clean,
            };
            assert_eq!(visited_route(&request).unwrap(), "bokeper");
        }
    }

    #[test]
    fn memory_omissions_preserve_endpoints_and_new_adjacent_duplicates() {
        let mut dropped = 0;
        let mut repeated = false;
        for writer in 0..1000 {
            let request = Request {
                word: "abacada".into(),
                writer,
                difficulty: Difficulty::Memory,
            };
            let keys = visited_route(&request).unwrap();
            assert!(keys.starts_with('a') && keys.ends_with('a'));
            assert!(keys.len() == 6 || keys.len() == 7);
            dropped += usize::from(keys.len() == 6);
            repeated |= keys.contains("aa");
        }
        assert!((280..420).contains(&dropped));
        assert!(repeated);
    }
}
