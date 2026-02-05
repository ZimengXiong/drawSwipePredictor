use swipe_engine::SwipeEngine;

const DICT_TEXT: &str = include_str!("../../../../word_freq.txt");

fn main() {
    let mut engine = SwipeEngine::new();
    engine.load_dictionary(DICT_TEXT);

    println!("Analyzing word distinctiveness...\n");

    // Test words with various characteristics
    let test_words = [
        // Top row only (very distinctive)
        "typewriter",
        "pretty",
        "quote",
        "power",
        "write",
        "tree",
        "two",
        "top",
        // Bottom row heavy
        "vim",
        "zinc",
        "vex",
        "van",
        "man",
        "can",
        // Diagonal patterns
        "lazy",
        "quartz",
        "pixel",
        // Long words
        "beautiful",
        "something",
        "everything",
        "keyboard",
        // Short common words
        "the",
        "and",
        "for",
        "you",
        "are",
        "was",
        "his",
        "her",
        // Unique patterns
        "jump",
        "quick",
        "fox",
        "zap",
        "jazz",
        "quiz",
        // Middle row heavy
        "flash",
        "dash",
        "hash",
        "glad",
        "flag",
        "slag",
    ];

    println!("=== Word Path Analysis ===\n");

    println!("TOP ROW ONLY (q-p, very distinctive):");
    println!("  typewriter, pretty, quote, power, write, tree, top, two, row, root\n");

    println!("BOTTOM ROW HEAVY (z-m):");
    println!("  vim, zinc, cab, van, man, ban\n");

    println!("LONG DIAGONAL:");
    println!("  quartz (q→z), lazy, pixel, jazz\n");

    println!("=== Testing predictions for distinctive words ===\n");

    // For each word, see how well it predicts itself
    for word in &test_words {
        let predictions = engine.predict(word, 5);
        if let Some(first) = predictions.first() {
            let rank = predictions
                .iter()
                .position(|p| p.word == *word)
                .map(|i| i + 1);
            let rank_str = rank
                .map(|r| format!("#{}", r))
                .unwrap_or("not in top 5".to_string());
            println!(
                "{:15} -> top prediction: {:15} (score: {:.4}) | self rank: {}",
                word, first.word, first.raw_score, rank_str
            );
        }
    }

    println!("\n=== Most Distinctive Words (unique shapes) ===\n");
    println!("These words should be easiest to recognize:\n");

    let distinctive = [
        ("typewriter", "Pure top row - very unique horizontal path"),
        ("quiz", "q→u→i→z - covers all rows, unique diagonal"),
        ("jump", "j→u→m→p - bounces up and down"),
        ("lazy", "l→a→z→y - sweeps across keyboard"),
        ("vex", "Bottom-left to right, short and unique"),
        ("jog", "j→o→g - middle to top to middle"),
        ("zip", "z→i→p - bottom-top-top, distinctive"),
        ("box", "b→o→x - unique triangle shape"),
        ("fog", "f→o→g - middle-top-middle"),
        ("yup", "y→u→p - top row, right side"),
    ];

    for (word, reason) in &distinctive {
        let predictions = engine.predict(word, 1);
        if let Some(first) = predictions.first() {
            let correct = if first.word == *word { "✓" } else { "✗" };
            println!("{} {:12} - {} ", correct, word, reason);
        }
    }

    println!("\n=== Recommendations for Testing ===\n");
    println!("EASY (very distinctive paths):");
    println!("  typewriter, quiz, jump, lazy, box, zip, jog, vex, yup\n");

    println!("MEDIUM (somewhat distinctive):");
    println!("  quick, power, write, flash, ghost, plant\n");

    println!("HARD (common patterns, many similar words):");
    println!("  the, and, for, was, are, his, her, has\n");
}
