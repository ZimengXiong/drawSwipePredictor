//! Geometry shared by the memory recognizer and its synthetic-data tools.
use crate::types::Point;

pub(crate) const N: usize = 32;
pub(crate) type Shape = [[f32; 2]; N];

/// Resample by traveled distance, ignoring repeated points and sampling speed.
pub(crate) fn resample(path: &[[f32; 2]], count: usize) -> Vec<[f32; 2]> {
    if path.is_empty() || count < 2 {
        return Vec::new();
    }
    let mut lengths = vec![0.0];
    for pair in path.windows(2) {
        let d = (pair[1][0] - pair[0][0]).hypot(pair[1][1] - pair[0][1]);
        lengths.push(lengths.last().unwrap() + d);
    }
    let total = *lengths.last().unwrap();
    if total < 1e-6 {
        return vec![path[0]; count];
    }
    let mut segment = 1;
    (0..count)
        .map(|i| {
            let target = total * i as f32 / (count - 1) as f32;
            while segment < path.len() - 1 && lengths[segment] < target {
                segment += 1;
            }
            let t = (target - lengths[segment - 1])
                / (lengths[segment] - lengths[segment - 1]).max(1e-9);
            let a = path[segment - 1];
            let b = path[segment];
            [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
        })
        .collect()
}

pub(crate) fn smooth(path: &mut [[f32; 2]], passes: usize) {
    for _ in 0..passes {
        let old = path.to_vec();
        for i in 1..path.len().saturating_sub(1) {
            for axis in 0..2 {
                path[i][axis] = (old[i - 1][axis] + 2.0 * old[i][axis] + old[i + 1][axis]) * 0.25;
            }
        }
    }
}

/// Remove translation and axis stretch. A floor preserves nearly straight paths.
pub(crate) fn normalize(path: &[[f32; 2]], angle: f32) -> Option<Shape> {
    if path.len() < 2 || path.iter().flatten().any(|x| !x.is_finite()) {
        return None;
    }
    let (sin, cos) = angle.sin_cos();
    let rotated: Vec<_> = path
        .iter()
        .map(|p| [cos * p[0] - sin * p[1], sin * p[0] + cos * p[1]])
        .collect();
    let mut mean = [0.0; 2];
    for p in &rotated {
        for a in 0..2 {
            mean[a] += p[a] / rotated.len() as f32;
        }
    }
    let mut variance = [0.0; 2];
    for p in &rotated {
        for a in 0..2 {
            variance[a] += (p[a] - mean[a]).powi(2) / rotated.len() as f32;
        }
    }
    let rms = (variance[0] + variance[1]).sqrt();
    if rms < 1e-6 {
        return None;
    }
    let scale = variance.map(|v| v.sqrt().max(rms * 0.35));
    let centered: Vec<_> = rotated
        .iter()
        .map(|p| [(p[0] - mean[0]) / scale[0], (p[1] - mean[1]) / scale[1]])
        .collect();
    resample(&centered, N).try_into().ok()
}

pub(crate) fn prepare(path: &[(f64, f64)]) -> Option<Vec<[f32; 2]>> {
    if path.len() < 2 || path.iter().any(|p| !p.0.is_finite() || !p.1.is_finite()) {
        return None;
    }
    // Center in f64 before conversion so large screen offsets retain precision.
    let origin = path[0];
    let points: Vec<_> = path
        .iter()
        .map(|p| [(p.0 - origin.0) as f32, (p.1 - origin.1) as f32])
        .collect();
    if points.iter().flatten().any(|x| !x.is_finite()) {
        return None;
    }
    let mut points = resample(&points, 96);
    smooth(&mut points, 2);
    normalize(&points, 0.0)?;
    Some(points)
}

pub(crate) fn distance(a: [f32; 2], b: [f32; 2]) -> f32 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)
}

pub(crate) fn tangent(shape: &Shape, i: usize) -> [f32; 2] {
    let a = shape[i.saturating_sub(2)];
    let b = shape[(i + 2).min(N - 1)];
    [b[0] - a[0], b[1] - a[1]]
}

/// Align position and direction under the same monotone correspondence.
pub(crate) fn elastic(a: &Shape, b: &Shape) -> f32 {
    let mut prev = [f32::INFINITY; N + 1];
    prev[0] = 0.0;
    for i in 0..N {
        let mut curr = [f32::INFINITY; N + 1];
        for j in i.saturating_sub(7)..=(i + 7).min(N - 1) {
            let cost = distance(a[i], b[j]) + 0.3 * distance(tangent(a, i), tangent(b, j));
            // Charge for warping so loops cannot collapse onto a single point.
            curr[j + 1] = cost + prev[j].min(prev[j + 1] + 0.025).min(curr[j] + 0.025);
        }
        prev = curr;
    }
    prev[N] / N as f32 + 0.06 * (distance(a[0], b[0]) + distance(a[N - 1], b[N - 1]))
}

pub(crate) fn key_vertices(word: &str) -> Vec<[f32; 2]> {
    let layout = crate::keyboard::get_keyboard_layout();
    let mut points = Vec::new();
    for c in word.chars() {
        if let Some(Point { x, y }) = layout.get(&c) {
            let p = [*x as f32, *y as f32];
            if points.last() != Some(&p) {
                points.push(p);
            }
        }
    }
    points
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resampling_preserves_endpoints_with_duplicate_points() {
        let points = resample(&[[0.0, 0.0], [0.0, 0.0], [4.0, 0.0]], 5);
        assert_eq!(
            points,
            vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]
        );
    }

    #[test]
    fn normalization_removes_translation_and_scale() {
        let a = normalize(&[[0.0, 0.0], [1.0, 2.0], [3.0, 0.0]], 0.0).unwrap();
        let b = normalize(&[[50.0, -20.0], [60.0, 0.0], [80.0, -20.0]], 0.0).unwrap();
        assert!(elastic(&a, &b) < 1e-10);
    }

    #[test]
    fn rejects_invalid_and_stationary_input() {
        assert!(prepare(&[(0.0, 0.0); 4]).is_none());
        assert!(prepare(&[(0.0, 0.0), (f64::NAN, 1.0)]).is_none());
    }
}
