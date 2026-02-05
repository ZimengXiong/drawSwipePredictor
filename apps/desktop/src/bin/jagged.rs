use std::collections::HashMap;

const DICT_TEXT: &str = include_str!("../../../../word_freq.txt");

fn get_layout() -> HashMap<char, (f64, f64)> {
    let mut layout = HashMap::new();
    let rows = [
        ("qwertyuiop", 0.0, 0.0),
        ("asdfghjkl", 0.5, 1.0),
        ("zxcvbnm", 1.5, 2.0),
    ];
    for (chars, x_offset, y) in rows {
        for (i, c) in chars.chars().enumerate() {
            layout.insert(c, (i as f64 + x_offset, y));
        }
    }
    layout
}

fn analyze_word(word: &str, layout: &HashMap<char, (f64, f64)>) -> Option<(usize, f64, f64)> {
    let points: Vec<(f64, f64)> = word.chars()
        .filter_map(|c| layout.get(&c.to_ascii_lowercase()).copied())
        .collect();

    if points.len() < 3 {
        return None;
    }

    let mut turns = 0;
    let mut total_angle_change = 0.0;
    let mut total_distance = 0.0;

    for i in 1..points.len() {
        let dx = points[i].0 - points[i-1].0;
        let dy = points[i].1 - points[i-1].1;
        total_distance += (dx*dx + dy*dy).sqrt();
    }

    for i in 2..points.len() {
        let dx1 = points[i-1].0 - points[i-2].0;
        let dy1 = points[i-1].1 - points[i-2].1;
        let dx2 = points[i].0 - points[i-1].0;
        let dy2 = points[i].1 - points[i-1].1;

        let angle1 = dy1.atan2(dx1);
        let angle2 = dy2.atan2(dx2);

        let mut turn = (angle2 - angle1).abs();
        if turn > std::f64::consts::PI {
            turn = 2.0 * std::f64::consts::PI - turn;
        }

        total_angle_change += turn;

        // Count as a turn if > 30 degrees
        if turn > std::f64::consts::PI / 6.0 {
            turns += 1;
        }
    }

    Some((turns, total_angle_change, total_distance))
}

fn main() {
    let layout = get_layout();

    // Parse dictionary
    let words: Vec<&str> = DICT_TEXT.lines()
        .filter_map(|line| line.split('\t').next())
        .filter(|w| w.len() >= 5 && w.len() <= 12) // Medium-long words
        .filter(|w| w.chars().all(|c| c.is_ascii_lowercase()))
        .collect();

    println!("Analyzing {} words for jaggedness...\n", words.len());

    // Analyze all words
    let mut analyzed: Vec<(&str, usize, f64, f64)> = words.iter()
        .filter_map(|w| {
            analyze_word(w, &layout).map(|(turns, angle, dist)| (*w, turns, angle, dist))
        })
        .collect();

    // Sort by turns per character (normalized jaggedness)
    analyzed.sort_by(|a, b| {
        let score_a = a.1 as f64 / a.0.len() as f64;
        let score_b = b.1 as f64 / b.0.len() as f64;
        score_b.partial_cmp(&score_a).unwrap()
    });

    println!("=== MOST JAGGED WORDS (Most turns per letter) ===\n");
    println!("{:<15} {:>6} {:>8} {:>10}", "Word", "Turns", "Angle", "TurnsRatio");
    println!("{}", "-".repeat(45));

    for (word, turns, angle, _dist) in analyzed.iter().take(40) {
        let ratio = *turns as f64 / word.len() as f64;
        println!("{:<15} {:>6} {:>8.2} {:>10.3}", word, turns, angle, ratio);
    }

    // Also find words with high absolute turns (longer jagged words)
    analyzed.sort_by(|a, b| b.1.cmp(&a.1));

    println!("\n=== LONGEST JAGGED WORDS (Most total turns) ===\n");
    println!("{:<15} {:>6} {:>8} {:>6}", "Word", "Turns", "Angle", "Len");
    println!("{}", "-".repeat(40));

    for (word, turns, angle, _dist) in analyzed.iter().take(30) {
        println!("{:<15} {:>6} {:>8.2} {:>6}", word, turns, angle, word.len());
    }

    // Specific recommendations
    println!("\n=== RECOMMENDED FOR TESTING (jagged + distinctive) ===\n");

    let recommendations = [
        "lunchbox",    // l-u-n-c-h-b-o-x: lots of vertical bouncing
        "pumpkin",     // p-u-m-p-k-i-n: bouncy
        "jumping",     // j-u-m-p-i-n-g: very bouncy
        "bumpy",       // b-u-m-p-y: up-down-up-down
        "jumpy",       // j-u-m-p-y: bouncy
        "kayak",       // k-a-y-a-k: left-right-left-right
        "puppy",       // p-u-p-p-y: bouncy
        "yoyo",        // y-o-y-o: repetitive bounce
        "kickback",    // k-i-c-k-b-a-c-k: lots of back-and-forth
        "backpack",    // lots of direction changes
        "hijack",      // h-i-j-a-c-k: bouncy
        "polka",       // p-o-l-k-a: zig-zag
        "zombie",      // z-o-m-b-i-e: lots of vertical movement
        "olympic",     // o-l-y-m-p-i-c: bouncy
        "maximum",     // m-a-x-i-m-u-m: lots of changes
    ];

    println!("{:<12} {:>6} {:>8} Description", "Word", "Turns", "Angle");
    println!("{}", "-".repeat(50));

    for word in &recommendations {
        if let Some((turns, angle, _)) = analyze_word(word, &layout) {
            let desc = match *word {
                "lunchbox" => "vertical bouncing across all rows",
                "jumping" => "j↑u↓m↑p→i↓n middle-top-bottom-top",
                "pumpkin" => "p↑u↓m↑p↓k↑i↓n bouncy pattern",
                "bumpy" => "b↑u↓m↑p↓y extreme vertical",
                "jumpy" => "j↑u↓m↑p↓y bouncy",
                "kayak" => "k←a→y←a→k palindrome zigzag",
                "zombie" => "z↑o↓m↓b↑i↓e covers full keyboard",
                "kickback" => "lots of back-and-forth motion",
                "backpack" => "b-a-c-k repeated pattern",
                "hijack" => "h↑i↓j←a↓c→k bouncy + horizontal",
                "maximum" => "m-a-x-i-m-u-m extreme zigzag",
                _ => "",
            };
            println!("{:<12} {:>6} {:>8.2} {}", word, turns, angle, desc);
        }
    }

    println!("\n=== TOP PICKS FOR BLIND SWIPE TESTING ===\n");
    println!("1. jumping   - very bouncy, 7 letters, distinctive");
    println!("2. lunchbox  - 8 letters, lots of vertical movement");
    println!("3. zombie    - covers full keyboard height");
    println!("4. pumpkin   - 7 letters, bouncy");
    println!("5. maximum   - 7 letters, extreme zigzag");
    println!("6. kayak     - palindrome, unique left-right pattern");
    println!("7. bumpy     - 5 letters, extreme vertical bouncing");
}
