use macroquad::prelude::*;
use serde::Serialize;

use swipe_engine::dtw::dtw_distance_fast;
use swipe_engine::keyboard::{euclidean_dist, get_keyboard_layout, get_word_path};
use swipe_engine::types::Point;
use swipe_engine::SwipeEngine;

const DICT_TEXT: &str = include_str!("../../../../word_freq.txt");

const N_POINTS: usize = 64;
const TEMPLATE_STEP: f64 = 0.3;
const N_TEMPLATES: usize = 16;

const RAW_SAMPLE_MIN_DIST: f64 = 3.0;

const LOG_DIR: &str = "logs";
const LOG_PREFIX: &str = "compare_session";

fn resample_path(path: &[Point], n: usize) -> Vec<Point> {
    if path.len() < 2 || n < 2 {
        return path.to_vec();
    }

    let mut dists: Vec<f64> = Vec::with_capacity(path.len());
    dists.push(0.0);
    for i in 1..path.len() {
        let last = *dists.last().unwrap();
        dists.push(last + euclidean_dist(&path[i - 1], &path[i]));
    }
    let total_len = *dists.last().unwrap();
    if total_len < 1e-9 {
        return vec![path[0]; n];
    }

    let mut out: Vec<Point> = Vec::with_capacity(n);
    for i in 0..n {
        let target = (i as f64 / (n - 1) as f64) * total_len;
        let mut j = 1;
        while j < dists.len() {
            if dists[j] >= target {
                let denom = (dists[j] - dists[j - 1]).max(1e-9);
                let t = (target - dists[j - 1]) / denom;
                out.push(Point {
                    x: path[j - 1].x + t * (path[j].x - path[j - 1].x),
                    y: path[j - 1].y + t * (path[j].y - path[j - 1].y),
                });
                break;
            }
            j += 1;
        }
        if j >= dists.len() {
            out.push(*path.last().unwrap());
        }
    }

    out
}

fn normalize_path(path: &[Point]) -> Vec<Point> {
    if path.len() < 2 {
        return vec![];
    }

    let mut total_len = 0.0;
    for i in 1..path.len() {
        total_len += euclidean_dist(&path[i - 1], &path[i]);
    }
    if total_len < 1e-9 {
        total_len = 1.0;
    }

    let cx = path.iter().map(|p| p.x).sum::<f64>() / path.len() as f64;
    let cy = path.iter().map(|p| p.y).sum::<f64>() / path.len() as f64;

    path.iter()
        .map(|p| Point {
            x: (p.x - cx) / total_len,
            y: (p.y - cy) / total_len,
        })
        .collect()
}

fn interpolate_path(path: &[Point], step: f64) -> Vec<Point> {
    if path.len() < 2 {
        return path.to_vec();
    }
    let mut out: Vec<Point> = Vec::new();
    out.push(path[0]);
    for i in 1..path.len() {
        let p1 = path[i - 1];
        let p2 = path[i];
        let dist = euclidean_dist(&p1, &p2);
        if dist > step {
            let n_steps = (dist / step) as usize;
            for j in 1..=n_steps {
                let t = j as f64 / (n_steps + 1) as f64;
                out.push(Point {
                    x: p1.x + t * (p2.x - p1.x),
                    y: p1.y + t * (p2.y - p1.y),
                });
            }
        }
        out.push(p2);
    }
    out
}

fn template_raw_for_word(word: &str) -> Option<Vec<Point>> {
    let layout = get_keyboard_layout();
    let raw = get_word_path(word, &layout);
    if raw.len() < 2 {
        return None;
    }
    Some(interpolate_path(&raw, TEMPLATE_STEP))
}

fn template_variants_for_word(word: &str, n_templates: usize) -> Vec<(Vec<Point>, TemplateVariantLog)> {
    let Some(base) = template_raw_for_word(word) else {
        return vec![];
    };

    let mut out: Vec<(Vec<Point>, TemplateVariantLog)> = Vec::new();

    // Template 0: ideal
    let ideal = normalize_path(&resample_path(&base, N_POINTS));
    if !ideal.is_empty() {
        let points_norm: Vec<(f64, f64)> = ideal.iter().map(|p| (p.x, p.y)).collect();
        out.push((
            ideal,
            TemplateVariantLog {
                idx: 0,
                kind: "ideal".to_string(),
                points_norm,
                rot: None,
                sx: None,
                sy: None,
                jitter: None,
                drift_step: None,
                smooth_win: None,
                dtw_norm: None,
                mean_l2: None,
                dir_cos: None,
            },
        ));
    }

    while out.len() < n_templates {
        let (aug, params) = augment_template_path(&base);
        let norm = normalize_path(&resample_path(&aug, N_POINTS));
        if !norm.is_empty() {
            let points_norm: Vec<(f64, f64)> = norm.iter().map(|p| (p.x, p.y)).collect();
            let idx = out.len();
            out.push((
                norm,
                TemplateVariantLog {
                    idx,
                    kind: "sloppy".to_string(),
                    points_norm,
                    rot: Some(params.rot),
                    sx: Some(params.sx),
                    sy: Some(params.sy),
                    jitter: Some(params.jitter),
                    drift_step: Some(params.drift_step),
                    smooth_win: Some(params.smooth_win),
                    dtw_norm: None,
                    mean_l2: None,
                    dir_cos: None,
                },
            ));
        }
    }

    out
}

fn normalize_drawn_path(raw: &[(f64, f64)]) -> Option<Vec<Point>> {
    if raw.len() < 2 {
        return None;
    }
    let pts: Vec<Point> = raw.iter().map(|(x, y)| Point { x: *x, y: *y }).collect();
    let resampled = resample_path(&pts, N_POINTS);
    Some(normalize_path(&resampled))
}

fn cosine_dir(p: &[Point]) -> Option<(f64, f64, f64)> {
    if p.len() < 2 {
        return None;
    }
    let dx = p.last().unwrap().x - p.first().unwrap().x;
    let dy = p.last().unwrap().y - p.first().unwrap().y;
    let mag = (dx * dx + dy * dy).sqrt();
    if mag < 1e-9 {
        return None;
    }
    Some((dx / mag, dy / mag, mag))
}

fn draw_overlay(panel: Rect, templates: &[(Vec<Point>, TemplateVariantLog)], drawn: &[Point], best_idx: Option<usize>) {
    let center = vec2(panel.x + panel.w * 0.5, panel.y + panel.h * 0.5);
    let scale = panel.w.min(panel.h) * 0.40;

    let to_screen = |p: &Point| -> Vec2 { vec2(center.x + (p.x as f32) * scale, center.y + (p.y as f32) * scale) };

    // Templates (ideal + sloppy variants)
    for (ti, (t, _)) in templates.iter().enumerate() {
        if t.len() < 2 {
            continue;
        }
        let is_ideal = ti == 0;
        let is_best = best_idx.is_some_and(|b| b == ti);

        let (thickness, color) = if is_ideal {
            (4.0, Color::from_rgba(120, 255, 160, 210))
        } else if is_best {
            (4.0, Color::from_rgba(255, 235, 120, 200))
        } else {
            (2.0, Color::from_rgba(120, 255, 160, 60))
        };

        for i in 1..t.len() {
            let a = to_screen(&t[i - 1]);
            let b = to_screen(&t[i]);
            draw_line(a.x, a.y, b.x, b.y, thickness, color);
        }
    }

    // Drawn
    for i in 1..drawn.len() {
        let a = to_screen(&drawn[i - 1]);
        let b = to_screen(&drawn[i]);
        draw_line(a.x, a.y, b.x, b.y, 4.0, Color::from_rgba(100, 200, 255, 220));
    }

    if let Some(best) = best_idx.and_then(|i| templates.get(i)).map(|(p, _)| p) {
        if let (Some(t0), Some(tn)) = (best.first(), best.last()) {
            let a = to_screen(t0);
            let b = to_screen(tn);
            draw_circle(a.x, a.y, 6.0, Color::from_rgba(255, 235, 120, 255));
            draw_circle(b.x, b.y, 6.0, Color::from_rgba(255, 150, 120, 255));
        }
    } else if let Some((ideal, _)) = templates.first() {
        if let (Some(t0), Some(tn)) = (ideal.first(), ideal.last()) {
            let a = to_screen(t0);
            let b = to_screen(tn);
            draw_circle(a.x, a.y, 6.0, Color::from_rgba(80, 255, 120, 255));
            draw_circle(b.x, b.y, 6.0, Color::from_rgba(255, 120, 120, 255));
        }
    }
    if let (Some(d0), Some(dn)) = (drawn.first(), drawn.last()) {
        let a = to_screen(d0);
        let b = to_screen(dn);
        draw_circle(a.x, a.y, 6.0, Color::from_rgba(80, 180, 255, 255));
        draw_circle(b.x, b.y, 6.0, Color::from_rgba(255, 180, 80, 255));
    }
}

fn sample_words_from_freq_text(freq_text: &str, max_words: usize) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for line in freq_text.lines() {
        let Some((w, _)) = line.split_once('\t') else {
            continue;
        };
        let w = w.trim().to_lowercase();
        if w.len() < 2 {
            continue;
        }
        if !w.bytes().all(|b| (b'a'..=b'z').contains(&b)) {
            continue;
        }
        out.push(w);
        if out.len() >= max_words {
            break;
        }
    }
    out
}

fn choose_target(words: &[String]) -> String {
    if words.is_empty() {
        return "hello".to_string();
    }
    let i = (macroquad::rand::gen_range(0, words.len() as i32) as usize).min(words.len() - 1);
    words[i].clone()
}

#[macroquad::main("Swipe Compare (Template Overlay)")]
async fn main() {
    let words = sample_words_from_freq_text(DICT_TEXT, 5000);

    let mut engine = SwipeEngine::new();
    engine.load_dictionary(DICT_TEXT);

    ensure_log_dir();
    let started_unix_ms = now_unix_ms();
    let log_path = make_log_path(started_unix_ms);

    let mut target = choose_target(&words);
    let mut templates = template_variants_for_word(&target, N_TEMPLATES);

    let session_meta = SessionMeta {
        started_unix_ms,
        target: target.clone(),
        n_points: N_POINTS,
        n_templates: N_TEMPLATES,
        template_step: TEMPLATE_STEP,
    };
    append_jsonl(&log_path, &serde_json::json!({"type":"session_start","meta":session_meta}));

    let mut path: Vec<(f64, f64)> = Vec::new();
    let mut drawing = false;
    let mut draw_started_unix_ms: Option<i64> = None;
    let mut attempt_idx: u64 = 0;
    let mut revealed = false;
    let mut drawn_norm: Option<Vec<Point>> = None;
    let mut last_score: Option<(f64, f64, f64)> = None; // (dtw, mean_l2, dir_cos)
    let mut last_best_idx: Option<usize> = None;
    let mut last_classic: Option<String> = None;
    let mut pending_record: Option<ComparisonLog> = None;
    let mut saved_flash_until: Option<f64> = None;

    loop {
        clear_background(Color::from_rgba(25, 26, 30, 255));

        let (mx, my) = mouse_position();

        if is_key_pressed(KeyCode::N) {
            target = choose_target(&words);
            templates = template_variants_for_word(&target, N_TEMPLATES);
            path.clear();
            drawn_norm = None;
            last_score = None;
            last_best_idx = None;
            last_classic = None;
            revealed = false;
            draw_started_unix_ms = None;
            pending_record = None;

            append_jsonl(
                &log_path,
                &serde_json::json!({"type":"new_target","unix_ms":now_unix_ms(),"target":target.clone()}),
            );
        }
        if is_key_pressed(KeyCode::Space) {
            path.clear();
            drawn_norm = None;
            last_score = None;
            last_best_idx = None;
            last_classic = None;
            revealed = false;
            draw_started_unix_ms = None;
            pending_record = None;

            append_jsonl(&log_path, &serde_json::json!({"type":"clear","unix_ms":now_unix_ms()}));
        }

        if is_key_pressed(KeyCode::Escape) {
            append_jsonl(&log_path, &serde_json::json!({"type":"session_end","unix_ms":now_unix_ms()}));
            break;
        }

        if is_mouse_button_pressed(MouseButton::Left) {
            drawing = true;
            path.clear();
            drawn_norm = None;
            last_score = None;
            last_best_idx = None;
            last_classic = None;
            revealed = false;
            draw_started_unix_ms = Some(now_unix_ms());
            pending_record = None;
        }

        if drawing && is_mouse_button_down(MouseButton::Left) {
            let should_add = path.last().map_or(true, |(lx, ly)| {
                let dx = mx as f64 - lx;
                let dy = my as f64 - ly;
                (dx * dx + dy * dy).sqrt() > RAW_SAMPLE_MIN_DIST
            });
            if should_add {
                path.push((mx as f64, my as f64));
            }
        }

        if drawing && is_mouse_button_released(MouseButton::Left) {
            drawing = false;
            let draw_ended_unix_ms = Some(now_unix_ms());
            if let (Some(dn), true) = (normalize_drawn_path(&path), !templates.is_empty()) {
                drawn_norm = Some(dn);
                let (best_idx, best_score, per_template) = score_against_templates(drawn_norm.as_ref().unwrap(), &templates);
                last_best_idx = best_idx;
                last_score = best_score;
                revealed = best_score.is_some();

                let preds = engine.predict_from_path(&path, 1);
                last_classic = preds.first().map(|p| p.word.clone());

                // Write full comparison details.
                let drawn_pairs: Vec<(f64, f64)> = drawn_norm
                    .as_ref()
                    .unwrap()
                    .iter()
                    .map(|p| (p.x, p.y))
                    .collect();

                let mut template_logs: Vec<TemplateVariantLog> = templates.iter().map(|(_, l)| l.clone()).collect();
                for (i, (dtw, mean, dir)) in per_template.into_iter().enumerate() {
                    if let Some(tl) = template_logs.get_mut(i) {
                        tl.dtw_norm = Some(dtw);
                        tl.mean_l2 = Some(mean);
                        tl.dir_cos = Some(dir);
                    }
                }

                let best = best_score.map(|(dtw_norm, mean_l2, dir_cos)| ComparisonScore {
                    dtw_norm,
                    mean_l2,
                    dir_cos,
                });

                let record = ComparisonLog {
                    unix_ms: now_unix_ms(),
                    target: target.clone(),
                    attempt_idx,
                    draw_started_unix_ms,
                    draw_ended_unix_ms,
                    saved_unix_ms: None,
                    best_idx,
                    best,
                    raw_path: path.clone(),
                    drawn_norm: drawn_pairs,
                    templates: template_logs,
                    classic_top1: last_classic.clone(),
                };

                // Only persist when user explicitly saves.
                pending_record = Some(record);
            }
        }

        // Save current comparison (after drawing)
        if is_key_pressed(KeyCode::S) {
            if let Some(mut rec) = pending_record.clone() {
                let saved_ms = now_unix_ms();
                rec.saved_unix_ms = Some(saved_ms);
                append_jsonl(&log_path, &serde_json::json!({"type":"comparison", "record": rec}));
                attempt_idx = attempt_idx.saturating_add(1);
                pending_record = None;
                saved_flash_until = Some(get_time() + 1.2);
            }
        }

        // Raw drawn path
        if path.len() >= 2 {
            for i in 1..path.len() {
                let (x1, y1) = path[i - 1];
                let (x2, y2) = path[i];
                draw_line(x1 as f32, y1 as f32, x2 as f32, y2 as f32, 3.0, WHITE);
            }
        }

        // Overlay panel
        let panel = Rect::new(20.0, 80.0, screen_width() - 40.0, screen_height() - 120.0);
        draw_rectangle(panel.x, panel.y, panel.w, panel.h, Color::from_rgba(38, 40, 48, 220));
        draw_rectangle_lines(panel.x, panel.y, panel.w, panel.h, 2.0, Color::from_rgba(80, 85, 100, 255));

        if revealed {
            if let Some(dn) = drawn_norm.as_deref() {
                draw_overlay(panel, &templates, dn, last_best_idx);
            }
        } else {
            let msg = if drawing {
                "Release to reveal overlay"
            } else {
                "Draw from memory; overlay reveals on release"
            };
            draw_text(msg, panel.x + 16.0, panel.y + 28.0, 18.0, Color::from_rgba(160, 165, 180, 255));
        }

        let title = format!("Target: {target}");
        draw_text(&title, 20.0, 34.0, 28.0, Color::from_rgba(235, 235, 240, 255));

        let help = "Draw the word | N=new target | Space=clear | S=save";
        draw_text(help, 20.0, 60.0, 18.0, Color::from_rgba(160, 165, 180, 255));

        if let Some((dtw, mean_l2, dir_cos)) = last_score {
            let line1 = format!("DTW(norm): {dtw:.4}    MeanL2: {mean_l2:.4}    DirCos: {dir_cos:.3}");
            draw_text(&line1, 24.0, panel.y + panel.h - 14.0, 18.0, Color::from_rgba(210, 210, 220, 255));
        }

        if let Some(i) = last_best_idx {
            let line = if i == 0 {
                "Best template: ideal".to_string()
            } else {
                format!("Best template: sloppy #{i}")
            };
            draw_text(&line, 24.0, panel.y + panel.h - 58.0, 18.0, Color::from_rgba(235, 220, 170, 255));
        }

        if let Some(w) = last_classic.as_deref() {
            let line2 = format!("Classic top-1: {w}");
            draw_text(&line2, 24.0, panel.y + panel.h - 36.0, 18.0, Color::from_rgba(200, 255, 210, 255));
        }

        if pending_record.is_some() && !drawing {
            draw_text(
                "Unsaved: press S to log",
                24.0,
                panel.y + panel.h - 80.0,
                18.0,
                Color::from_rgba(255, 200, 120, 255),
            );
        }

        if let Some(until) = saved_flash_until {
            if get_time() < until {
                draw_text(
                    "Saved",
                    panel.x + panel.w - 84.0,
                    panel.y + 28.0,
                    20.0,
                    Color::from_rgba(200, 255, 210, 255),
                );
            } else {
                saved_flash_until = None;
            }
        }

        draw_keyboard_ghost();

        next_frame().await;
    }
}

fn score_against_templates(
    drawn_norm: &[Point],
    templates: &[(Vec<Point>, TemplateVariantLog)],
) -> (Option<usize>, Option<(f64, f64, f64)>, Vec<(f64, f64, f64)>) {
    if drawn_norm.len() < 2 || templates.is_empty() {
        return (None, None, vec![]);
    }

    let window = 16;
    let cutoff = 1e9;

    let mut best_idx: Option<usize> = None;
    let mut best_dtw = f64::INFINITY;
    let mut per: Vec<(f64, f64, f64)> = Vec::with_capacity(templates.len());

    for (i, (t, _)) in templates.iter().enumerate() {
        if t.len() != drawn_norm.len() || t.len() < 2 {
            continue;
        }
        let dtw = dtw_distance_fast(drawn_norm, t, window, cutoff) / N_POINTS as f64;
        let mean_l2 = drawn_norm
            .iter()
            .zip(t.iter())
            .map(|(a, b)| euclidean_dist(a, b))
            .sum::<f64>()
            / N_POINTS as f64;

        let dir_cos = match (cosine_dir(drawn_norm), cosine_dir(t)) {
            (Some((dx1, dy1, _)), Some((dx2, dy2, _))) => dx1 * dx2 + dy1 * dy2,
            _ => 0.0,
        };

        per.push((dtw, mean_l2, dir_cos));
        if dtw < best_dtw {
            best_dtw = dtw;
            best_idx = Some(i);
        }
    }

    let Some(i) = best_idx else {
        return (None, None, per);
    };

    let (_, mean_l2, dir_cos) = per[i];

    (Some(i), Some((best_dtw, mean_l2, dir_cos)), per)
}

fn augment_template_path(base: &[Point]) -> (Vec<Point>, AugParams) {
    if base.len() < 2 {
        return (
            base.to_vec(),
            AugParams {
                rot: 0.0,
                sx: 1.0,
                sy: 1.0,
                jitter: 0.0,
                drift_step: 0.0,
                smooth_win: 0,
            },
        );
    }

    let rot = gen_f64(-0.35, 0.35);
    let sx = gen_f64(0.80, 1.25);
    let sy = gen_f64(0.80, 1.25);
    let jitter = gen_f64(0.0, 0.10);
    let drift_step = gen_f64(0.0, 0.05);
    let smooth_win = macroquad::rand::gen_range(0, 4); // 0..=3 => win 0/3/5/7

    let mut p = transform_about_centroid(base, rot, sx, sy);
    p = apply_drift(&p, drift_step);
    p = apply_jitter(&p, jitter);
    p = if smooth_win == 0 {
        p
    } else {
        moving_average(&p, 2 * smooth_win as usize + 1)
    };

    (
        p,
        AugParams {
            rot,
            sx,
            sy,
            jitter,
            drift_step,
            smooth_win: if smooth_win == 0 { 0 } else { 2 * smooth_win as usize + 1 },
        },
    )
}

fn transform_about_centroid(path: &[Point], rot: f64, sx: f64, sy: f64) -> Vec<Point> {
    let cx = path.iter().map(|p| p.x).sum::<f64>() / path.len() as f64;
    let cy = path.iter().map(|p| p.y).sum::<f64>() / path.len() as f64;
    let (cr, sr) = (rot.cos(), rot.sin());
    path.iter()
        .map(|p| {
            let mut x = p.x - cx;
            let mut y = p.y - cy;
            x *= sx;
            y *= sy;
            let xr = x * cr - y * sr;
            let yr = x * sr + y * cr;
            Point { x: xr + cx, y: yr + cy }
        })
        .collect()
}

fn apply_drift(path: &[Point], step: f64) -> Vec<Point> {
    if step <= 0.0 {
        return path.to_vec();
    }
    let mut ox = 0.0;
    let mut oy = 0.0;
    let mut out: Vec<Point> = Vec::with_capacity(path.len());
    for p in path {
        ox += randn() * step;
        oy += randn() * step;
        out.push(Point { x: p.x + ox, y: p.y + oy });
    }
    out
}

fn apply_jitter(path: &[Point], sigma: f64) -> Vec<Point> {
    if sigma <= 0.0 {
        return path.to_vec();
    }
    path.iter()
        .map(|p| Point {
            x: p.x + randn() * sigma,
            y: p.y + randn() * sigma,
        })
        .collect()
}

fn moving_average(path: &[Point], win: usize) -> Vec<Point> {
    if win < 3 || path.len() < 3 {
        return path.to_vec();
    }
    let half = win / 2;
    let mut out: Vec<Point> = Vec::with_capacity(path.len());
    for i in 0..path.len() {
        let i0 = i.saturating_sub(half);
        let i1 = (i + half).min(path.len() - 1);
        let mut sx = 0.0;
        let mut sy = 0.0;
        let mut n = 0.0;
        for j in i0..=i1 {
            sx += path[j].x;
            sy += path[j].y;
            n += 1.0;
        }
        out.push(Point { x: sx / n, y: sy / n });
    }
    out
}

fn gen_f64(lo: f64, hi: f64) -> f64 {
    macroquad::rand::gen_range(lo as f32, hi as f32) as f64
}

fn randn() -> f64 {
    // Approximate N(0,1) using sum of uniforms.
    let mut s = 0.0;
    for _ in 0..12 {
        s += macroquad::rand::gen_range(0.0f32, 1.0f32) as f64;
    }
    s - 6.0
}

fn draw_keyboard_ghost() {
    // Copied from modeldraw.rs to keep UX consistent.
    let kb_x = screen_width() - 320.0;
    let kb_y = screen_height() - 120.0;
    let key_w = 28.0;
    let key_h = 28.0;
    let gap = 2.0;

    let rows = [("qwertyuiop", 0.0), ("asdfghjkl", 0.5), ("zxcvbnm", 1.5)];

    draw_rectangle(kb_x - 10.0, kb_y - 10.0, 320.0, 110.0, Color::from_rgba(40, 40, 50, 200));

    for (row_idx, (chars, x_offset)) in rows.iter().enumerate() {
        for (col, c) in chars.chars().enumerate() {
            let x = kb_x + (col as f32 + x_offset) * (key_w + gap);
            let y = kb_y + row_idx as f32 * (key_h + gap);

            draw_rectangle(x, y, key_w, key_h, Color::from_rgba(60, 60, 70, 180));
            draw_text(
                &c.to_string(),
                x + 8.0,
                y + 20.0,
                20.0,
                Color::from_rgba(150, 150, 160, 200),
            );
        }
    }
}

#[derive(Clone, Serialize)]
struct SessionMeta {
    started_unix_ms: i64,
    target: String,
    n_points: usize,
    n_templates: usize,
    template_step: f64,
}

#[derive(Clone, Serialize)]
struct TemplateVariantLog {
    idx: usize,
    kind: String, // "ideal" | "sloppy"
    points_norm: Vec<(f64, f64)>,
    // Params used for the augmentation (None for ideal)
    rot: Option<f64>,
    sx: Option<f64>,
    sy: Option<f64>,
    jitter: Option<f64>,
    drift_step: Option<f64>,
    smooth_win: Option<usize>,
    // Scores vs drawn (filled on release)
    dtw_norm: Option<f64>,
    mean_l2: Option<f64>,
    dir_cos: Option<f64>,
}

#[derive(Clone, Serialize)]
struct ComparisonLog {
    unix_ms: i64,
    target: String,
    attempt_idx: u64,
    draw_started_unix_ms: Option<i64>,
    draw_ended_unix_ms: Option<i64>,
    saved_unix_ms: Option<i64>,
    best_idx: Option<usize>,
    best: Option<ComparisonScore>,
    raw_path: Vec<(f64, f64)>,
    drawn_norm: Vec<(f64, f64)>,
    templates: Vec<TemplateVariantLog>,
    classic_top1: Option<String>,
}

#[derive(Clone, Serialize)]
struct ComparisonScore {
    dtw_norm: f64,
    mean_l2: f64,
    dir_cos: f64,
}

struct AugParams {
    rot: f64,
    sx: f64,
    sy: f64,
    jitter: f64,
    drift_step: f64,
    smooth_win: usize,
}

fn now_unix_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let d = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    (d.as_millis() as i128).min(i64::MAX as i128) as i64
}

fn make_log_path(started_unix_ms: i64) -> std::path::PathBuf {
    let mut p = std::path::PathBuf::from(LOG_DIR);
    p.push(format!("{LOG_PREFIX}_{started_unix_ms}.jsonl"));
    p
}

fn ensure_log_dir() {
    let _ = std::fs::create_dir_all(LOG_DIR);
}

fn append_jsonl<T: Serialize>(path: &std::path::Path, value: &T) {
    let Ok(line) = serde_json::to_string(value) else {
        return;
    };
    let mut s = line;
    s.push('\n');
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        use std::io::Write;
        let _ = f.write_all(s.as_bytes());
    }
}
