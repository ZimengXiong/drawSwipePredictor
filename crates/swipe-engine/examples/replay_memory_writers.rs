//! Recover original raw training strokes without passing labels to inference.
use serde::Deserialize;
use std::io::{BufRead, Write};
use swipe_engine::synthetic::{Difficulty, SyntheticWriter};

#[derive(Deserialize)]
struct Request {
    word: String,
    writer: u64,
    difficulty: Difficulty,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = std::io::stdin();
    let mut output = std::io::BufWriter::new(std::io::stdout().lock());
    for line in input.lock().lines() {
        let request: Request = serde_json::from_str(&line?)?;
        let points = SyntheticWriter::new(request.writer).draw(&request.word, request.difficulty);
        if points.is_empty() {
            return Err("writer request must describe a moving lowercase word".into());
        }
        serde_json::to_writer(&mut output, &serde_json::json!({"points": points}))?;
        writeln!(output)?;
    }
    output.flush()?;
    Ok(())
}
