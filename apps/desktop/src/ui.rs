use macroquad::prelude::*;
use swipe_engine::Prediction;

pub trait Predictor {
    fn predict(&mut self, path: &[(f64, f64)], limit: usize) -> Vec<Prediction>;
}

pub async fn run_app<P: Predictor + Send>(mut predictor: P, backend_name: &str) {
    let mut path: Vec<(f64, f64)> = Vec::new();
    let mut drawing = false;
    let mut predictions: Vec<Prediction> = Vec::new();
    let mut selected: usize = 0;

    loop {
        clear_background(Color::from_rgba(30, 30, 35, 255));

        let (mouse_x, mouse_y) = mouse_position();

        if is_mouse_button_pressed(MouseButton::Left) {
            drawing = true;
            path.clear();
            predictions.clear();
            selected = 0;
        }

        if drawing && is_mouse_button_down(MouseButton::Left) {
            let should_add = path.last().map_or(true, |(lx, ly)| {
                let dx = mouse_x as f64 - lx;
                let dy = mouse_y as f64 - ly;
                (dx * dx + dy * dy).sqrt() > 3.0
            });
            if should_add {
                path.push((mouse_x as f64, mouse_y as f64));
            }
        }

        if drawing && is_mouse_button_released(MouseButton::Left) {
            drawing = false;
            if path.len() >= 2 {
                predictions = predictor.predict(&path, 10);
            }
        }

        if is_key_pressed(KeyCode::Space) {
            path.clear();
            predictions.clear();
            selected = 0;
        }
        if is_key_pressed(KeyCode::Up) && selected > 0 {
            selected -= 1;
        }
        if is_key_pressed(KeyCode::Down) && selected < predictions.len().saturating_sub(1) {
            selected += 1;
        }

        if path.len() >= 2 {
            for i in 1..path.len() {
                let (x1, y1) = path[i - 1];
                let (x2, y2) = path[i];
                draw_line(x1 as f32, y1 as f32, x2 as f32, y2 as f32, 4.0, WHITE);
            }
            for (x, y) in &path {
                draw_circle(*x as f32, *y as f32, 3.0, Color::from_rgba(100, 200, 255, 200));
            }
        }

        if !predictions.is_empty() {
            draw_predictions_panel(&predictions, selected);
            if let Some(pred) = predictions.get(selected) {
                draw_breakdown_panel(pred, backend_name);
            }
        }

        let instructions = if drawing {
            "Drawing... release to predict"
        } else if path.is_empty() {
            "Draw a swipe pattern | Space=clear | Up/Down=select"
        } else {
            "Space=clear | Up/Down=select | Draw again for new word"
        };
        draw_text(instructions, 20.0, screen_height() - 20.0, 18.0, GRAY);
        draw_text(&format!("Backend: {}", backend_name), 20.0, screen_height() - 40.0, 16.0,
            Color::from_rgba(120, 180, 220, 255));

        draw_keyboard_ghost();

        next_frame().await
    }
}

fn draw_predictions_panel(predictions: &[Prediction], selected: usize) {
    let box_x = 20.0;
    let box_y = 20.0;
    let box_w = 340.0;
    let row_h = 24.0;
    let box_h = 30.0 + predictions.len() as f32 * row_h;

    draw_rectangle(box_x, box_y, box_w, box_h, Color::from_rgba(40, 40, 50, 240));
    draw_rectangle_lines(box_x, box_y, box_w, box_h, 2.0, Color::from_rgba(80, 80, 100, 255));

    draw_text("Word", box_x + 10.0, box_y + 18.0, 16.0, GRAY);
    draw_text("Score", box_x + 120.0, box_y + 18.0, 16.0, GRAY);
    draw_text("Pop", box_x + 190.0, box_y + 18.0, 16.0, GRAY);
    draw_text("Rating", box_x + 260.0, box_y + 18.0, 16.0, GRAY);

    for (i, pred) in predictions.iter().enumerate() {
        let y = box_y + 38.0 + i as f32 * row_h;

        if i == selected {
            draw_rectangle(
                box_x + 2.0,
                y - 14.0,
                box_w - 4.0,
                row_h - 2.0,
                Color::from_rgba(60, 80, 100, 200),
            );
        }

        let color = if i == selected {
            Color::from_rgba(100, 255, 150, 255)
        } else if i == 0 {
            Color::from_rgba(200, 255, 200, 255)
        } else {
            WHITE
        };

        draw_text(&format!("{}. {}", i + 1, pred.word), box_x + 10.0, y, 18.0, color);
        draw_text(&format!("{:.3}", pred.raw_score), box_x + 120.0, y, 16.0,
            score_color(pred.raw_score));
        draw_text(&format!("{:.2}", pred.popularity), box_x + 190.0, y, 16.0,
            pop_color(pred.popularity));
        draw_text(&format!("{:.3}", pred.rating), box_x + 260.0, y, 16.0,
            rating_color(pred.rating));
    }
}

fn draw_breakdown_panel(pred: &Prediction, backend: &str) {
    let box_x = 20.0;
    let box_y = 300.0;
    let box_w = 360.0;
    let box_h = 200.0;

    draw_rectangle(box_x, box_y, box_w, box_h, Color::from_rgba(40, 40, 50, 240));
    draw_rectangle_lines(box_x, box_y, box_w, box_h, 2.0, Color::from_rgba(80, 80, 100, 255));

    draw_text(&format!("{}: \"{}\"", backend, pred.word), box_x + 10.0, box_y + 22.0, 18.0,
        Color::from_rgba(100, 255, 150, 255));

    if backend == "DTW" {
        let b = &pred.breakdown;
        let items = [
            ("Procrustes (15%)", b.location, 0.15),
            ("Shape DTW (10%)", b.shape, 0.10),
            ("Angles (10%)", b.angles, 0.10),
            ("Displacements (8%)", b.displacements, 0.08),
            ("Start Angle (6%)", b.start_angle, 0.06),
            ("Turns (10%)", b.turns, 0.10),
            ("Seg Ratios (7%)", b.segment_ratios, 0.07),
            ("Histogram (6%)", b.histogram, 0.06),
            ("End Vector (6%)", b.end_vector, 0.06),
        ];

        for (i, (name, value, weight)) in items.iter().enumerate() {
            let y = box_y + 44.0 + i as f32 * 18.0;
            let contribution = value * weight;

            draw_text(name, box_x + 10.0, y, 14.0, GRAY);
            draw_text(&format!("{:.3}", value), box_x + 150.0, y, 14.0, score_color(*value));

            let bar_w = (contribution * 500.0).min(80.0) as f32;
            draw_rectangle(box_x + 210.0, y - 9.0, bar_w, 10.0,
                Color::from_rgba(100, 150, 200, 200));
            draw_text(&format!("{:.4}", contribution), box_x + 295.0, y, 12.0, WHITE);
        }
    } else {
        draw_text("CTC decodes sequence directly; breakdown unavailable.",
            box_x + 10.0, box_y + 48.0, 14.0, GRAY);
    }
}

fn score_color(score: f64) -> Color {
    if score < 0.1 {
        Color::from_rgba(100, 255, 100, 255)
    } else if score < 0.3 {
        Color::from_rgba(200, 255, 100, 255)
    } else if score < 0.5 {
        Color::from_rgba(255, 255, 100, 255)
    } else {
        Color::from_rgba(255, 150, 100, 255)
    }
}

fn pop_color(pop: f64) -> Color {
    if pop > 0.7 {
        Color::from_rgba(100, 200, 255, 255)
    } else if pop > 0.4 {
        Color::from_rgba(150, 200, 220, 255)
    } else {
        Color::from_rgba(180, 180, 180, 255)
    }
}

fn rating_color(rating: f64) -> Color {
    if rating < 0.1 {
        Color::from_rgba(50, 255, 100, 255)
    } else if rating < 0.2 {
        Color::from_rgba(100, 255, 150, 255)
    } else if rating < 0.4 {
        Color::from_rgba(200, 255, 100, 255)
    } else {
        Color::from_rgba(255, 200, 100, 255)
    }
}

fn draw_keyboard_ghost() {
    let kb_x = screen_width() - 320.0;
    let kb_y = screen_height() - 120.0;
    let key_w = 28.0;
    let key_h = 28.0;
    let gap = 2.0;

    let rows = [
        ("qwertyuiop", 0.0),
        ("asdfghjkl", 0.5),
        ("zxcvbnm", 1.5),
    ];

    draw_rectangle(kb_x - 10.0, kb_y - 10.0, 320.0, 110.0, Color::from_rgba(40, 40, 50, 200));

    for (row_idx, (chars, x_offset)) in rows.iter().enumerate() {
        for (col, c) in chars.chars().enumerate() {
            let x = kb_x + (col as f32 + x_offset) * (key_w + gap);
            let y = kb_y + row_idx as f32 * (key_h + gap);

            draw_rectangle(x, y, key_w, key_h, Color::from_rgba(60, 60, 70, 180));
            draw_text(&c.to_string(), x + 8.0, y + 20.0, 20.0, Color::from_rgba(150, 150, 160, 200));
        }
    }
}
