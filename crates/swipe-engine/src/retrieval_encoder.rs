//! Native convolutional inference for compact and residual route encoders.
const POINTS: usize = 64;
pub(crate) const DIM: usize = 128;

pub(crate) struct Layer {
    pub(crate) weights: Vec<f32>,
    pub(crate) bias: Vec<f32>,
    pub(crate) inputs: usize,
    pub(crate) outputs: usize,
    pub(crate) kernel: usize,
    pub(crate) stride: usize,
    pub(crate) dilation: usize,
}

pub(crate) struct EncoderContext {
    pub(crate) features: Vec<f32>,
    pub(crate) sequence: Vec<f32>,
    pub(crate) embedding: [f32; DIM],
}

pub(crate) struct Encoder {
    convolutions: Vec<Layer>,
    residual: Vec<[Layer; 2]>,
    downsample: Option<Layer>,
    dense: Vec<Layer>,
}

impl Encoder {
    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let large = match bytes.get(..8) {
            Some(b"SWPRT001") => false,
            Some(b"SWPRT002") | Some(b"SWPRT003") | Some(b"SWPRT004") => true,
            _ => return Err("unsupported route encoder format".into()),
        };
        let giant = bytes.get(..8) == Some(b"SWPRT004");
        let input_channels = if giant || bytes.get(..8) == Some(b"SWPRT003") {
            16
        } else {
            8
        };
        let mut position = 8;
        let mut layer = |inputs: usize,
                         outputs: usize,
                         kernel: usize,
                         stride: usize,
                         dilation: usize|
         -> Result<Layer, String> {
            let count = inputs * outputs * kernel + outputs;
            let end = position + count * 4;
            let data = bytes.get(position..end).ok_or("truncated route encoder")?;
            let values: Vec<_> = data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            if values.iter().any(|v| !v.is_finite()) {
                return Err("route encoder contains nonfinite weights".into());
            }
            position = end;
            let split = count - outputs;
            Ok(Layer {
                weights: values[..split].to_vec(),
                bias: values[split..].to_vec(),
                inputs,
                outputs,
                kernel,
                stride,
                dilation,
            })
        };
        let specification = if large {
            vec![(input_channels, 48, 5, 1), (48, 64, 5, 2), (64, 128, 3, 1)]
        } else {
            vec![
                (8, 48, 5, 1),
                (48, 64, 5, 2),
                (64, 96, 5, 2),
                (96, 128, 5, 2),
            ]
        };
        let convolutions = specification
            .into_iter()
            .map(|(i, o, k, s)| layer(i, o, k, s, 1))
            .collect::<Result<_, _>>()?;
        let mut residual = Vec::new();
        if large {
            let dilations = if giant {
                [1, 2, 4, 8].repeat(4)
            } else {
                vec![1, 2, 4, 8, 1, 2]
            };
            let hidden = if giant { 512 } else { 192 };
            for dilation in dilations {
                residual.push([
                    layer(128, hidden, 5, 1, dilation)?,
                    layer(hidden, 128, 3, 1, dilation)?,
                ]);
            }
        }
        let downsample = if large {
            Some(layer(128, 192, 3, 2, 1)?)
        } else {
            None
        };
        let dimensions = if large {
            [(3072, 768), (768, DIM)]
        } else {
            [(1024, 256), (256, DIM)]
        };
        let dense = dimensions
            .into_iter()
            .map(|(i, o)| layer(i, o, 1, 1, 1))
            .collect::<Result<_, _>>()?;
        if position != bytes.len() {
            return Err("route encoder has trailing data".into());
        }
        Ok(Self {
            convolutions,
            residual,
            downsample,
            dense,
        })
    }

    pub(crate) fn embed(&self, raw: &[(f64, f64)]) -> Option<[f32; DIM]> {
        self.context(raw).map(|context| context.embedding)
    }

    pub(crate) fn context(&self, raw: &[(f64, f64)]) -> Option<EncoderContext> {
        let mut values = prepare_features(raw)?;
        if self.convolutions[0].inputs == 16 {
            append_intrinsic(&mut values);
        }
        let features = values.clone();
        let mut length = POINTS;
        for layer in &self.convolutions {
            let next = convolve(&values, length, layer, true);
            values = next.0;
            length = next.1;
        }
        for [first, second] in &self.residual {
            let hidden = convolve(&values, length, first, true).0;
            let update = convolve(&hidden, length, second, false).0;
            for (value, delta) in values.iter_mut().zip(update) {
                *value = (*value + delta).max(0.0);
            }
        }
        let sequence = values.clone();
        if let Some(layer) = &self.downsample {
            values = convolve(&values, length, layer, true).0;
        }
        for (stage, layer) in self.dense.iter().enumerate() {
            let mut output = layer.bias.clone();
            for (channel, sum) in output.iter_mut().enumerate() {
                *sum += dot(
                    &values,
                    &layer.weights[channel * layer.inputs..(channel + 1) * layer.inputs],
                );
                if stage == 0 {
                    *sum = sum.max(0.0);
                }
            }
            values = output;
        }
        let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        let vector: Vec<_> = values.into_iter().map(|v| v / norm).collect();
        if vector.iter().any(|v| !v.is_finite()) {
            return None;
        }
        Some(EncoderContext {
            features,
            sequence,
            embedding: vector.try_into().ok()?,
        })
    }
}

pub(crate) fn convolve(
    values: &[f32],
    length: usize,
    layer: &Layer,
    relu: bool,
) -> (Vec<f32>, usize) {
    let next_length = (length - 1) / layer.stride + 1;
    let mut output = vec![0.0; layer.outputs * next_length];
    let width = layer.inputs * layer.kernel;
    let mut patch = vec![0.0; width];
    for t in 0..next_length {
        for input in 0..layer.inputs {
            for k in 0..layer.kernel {
                let position = (t * layer.stride + k * layer.dilation) as isize
                    - (layer.kernel / 2 * layer.dilation) as isize;
                patch[input * layer.kernel + k] = if position >= 0 && (position as usize) < length {
                    values[input * length + position as usize]
                } else {
                    0.0
                };
            }
        }
        for channel in 0..layer.outputs {
            let sum = layer.bias[channel]
                + dot(
                    &patch,
                    &layer.weights[channel * width..(channel + 1) * width],
                );
            output[channel * next_length + t] = if relu { sum.max(0.0) } else { sum };
        }
    }
    (output, next_length)
}

/// Append a whitened, stroke-anchored frame and local turn measurements.
fn append_intrinsic(features: &mut Vec<f32>) {
    let floor = (features[..POINTS * 2].iter().map(|v| v * v).sum::<f32>() / POINTS as f32) * 0.025;
    let a = features[..POINTS].iter().map(|v| v * v).sum::<f32>() / POINTS as f32 + floor;
    let d = features[POINTS..POINTS * 2]
        .iter()
        .map(|v| v * v)
        .sum::<f32>()
        / POINTS as f32
        + floor;
    let b = (0..POINTS)
        .map(|i| features[i] * features[POINTS + i])
        .sum::<f32>()
        / POINTS as f32;
    let root = (a * d - b * b).max(1e-8).sqrt();
    let scale = (a + d + 2.0 * root).sqrt();
    let denominator = ((a + root) * (d + root) - b * b).max(1e-8);
    let mut whitened = [[0.0_f32; 2]; POINTS];
    for (i, point) in whitened.iter_mut().enumerate() {
        let (x, y) = (features[i], features[POINTS + i]);
        *point = [
            scale * ((d + root) * x - b * y) / denominator,
            scale * ((a + root) * y - b * x) / denominator,
        ];
    }
    let mut anchor = [
        whitened[POINTS - 1][0] - whitened[0][0],
        whitened[POINTS - 1][1] - whitened[0][1],
    ];
    if anchor[0].powi(2) + anchor[1].powi(2) <= 0.0225 {
        anchor = [
            whitened[8][0] - whitened[0][0],
            whitened[8][1] - whitened[0][1],
        ];
    }
    let norm = anchor[0].hypot(anchor[1]).max(1e-6);
    anchor = anchor.map(|v| v / norm);
    let frame = whitened.map(|p| {
        [
            p[0] * anchor[0] + p[1] * anchor[1],
            p[1] * anchor[0] - p[0] * anchor[1],
        ]
    });
    let mut deltas = [[0.0_f32; 2]; POINTS];
    let mut directions = [[0.0_f32; 2]; POINTS];
    for i in 0..POINTS {
        let (lo, hi) = (i.saturating_sub(1), (i + 1).min(POINTS - 1));
        deltas[i] = [frame[hi][0] - frame[lo][0], frame[hi][1] - frame[lo][1]];
        let norm = deltas[i][0].hypot(deltas[i][1]).max(1e-6);
        directions[i] = deltas[i].map(|v| v / norm);
    }
    features.resize(16 * POINTS, 0.0);
    let mut area = 0.0;
    for i in 0..POINTS {
        for axis in 0..2 {
            features[(8 + axis) * POINTS + i] = frame[i][axis];
            features[(10 + axis) * POINTS + i] = deltas[i][axis];
        }
        let previous = directions[i.saturating_sub(1)];
        features[12 * POINTS + i] = frame[i][0].hypot(frame[i][1]);
        features[13 * POINTS + i] = previous[0] * directions[i][0] + previous[1] * directions[i][1];
        features[14 * POINTS + i] = previous[0] * directions[i][1] - previous[1] * directions[i][0];
        if i > 0 {
            area += frame[i - 1][0] * frame[i][1] - frame[i - 1][1] * frame[i][0];
        }
        features[15 * POINTS + i] = area / 4.0;
    }
}

/// Independent accumulators allow vectorization without architecture-specific code.
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let (a_blocks, a_tail) = a.as_chunks::<4>();
    let (b_blocks, b_tail) = b.as_chunks::<4>();
    let mut sums = [0.0; 4];
    for (a, b) in a_blocks.iter().zip(b_blocks) {
        for i in 0..4 {
            sums[i] += a[i] * b[i];
        }
    }
    sums.iter().sum::<f32>() + a_tail.iter().zip(b_tail).map(|(a, b)| a * b).sum::<f32>()
}

fn prepare_features(raw: &[(f64, f64)]) -> Option<Vec<f32>> {
    if raw.len() < 2 || raw.iter().any(|p| !p.0.is_finite() || !p.1.is_finite()) {
        return None;
    }
    let origin = raw[0];
    let centered: Vec<_> = raw
        .iter()
        .map(|p| [p.0 - origin.0, p.1 - origin.1])
        .collect();
    let scale = centered
        .iter()
        .flatten()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);
    if scale == 0.0 || !scale.is_finite() {
        return None;
    }
    // Match training's f64 arc-length resampling before converting to f32.
    // Early rounding can amplify through the character decoder's residual layers.
    let scaled: Vec<_> = centered
        .iter()
        .map(|p| [p[0] / scale, p[1] / scale])
        .collect();
    let points: Vec<_> = scaled
        .iter()
        .enumerate()
        .filter(|(i, p)| {
            *i == 0 || (p[0] - scaled[i - 1][0]).hypot(p[1] - scaled[i - 1][1]) * scale > 1e-10
        })
        .map(|(_, p)| *p)
        .collect();
    if points.len() < 2 {
        return None;
    }
    let mut lengths = vec![0.0_f64];
    for pair in points.windows(2) {
        let distance = (pair[1][0] - pair[0][0]).hypot(pair[1][1] - pair[0][1]);
        lengths.push(lengths.last()? + distance);
    }
    let total = *lengths.last()?;
    let mut segment = 1;
    let mut precise: Vec<[f64; 2]> = (0..POINTS)
        .map(|i| {
            let target = total * i as f64 / (POINTS - 1) as f64;
            while segment < points.len() - 1 && lengths[segment] < target {
                segment += 1;
            }
            let amount =
                (target - lengths[segment - 1]) / (lengths[segment] - lengths[segment - 1]);
            std::array::from_fn(|a| {
                points[segment - 1][a] + amount * (points[segment][a] - points[segment - 1][a])
            })
        })
        .collect();
    let center: [f64; 2] =
        std::array::from_fn(|a| precise.iter().map(|p| p[a]).sum::<f64>() / POINTS as f64);
    for point in &mut precise {
        for a in 0..2 {
            point[a] -= center[a];
        }
    }
    let radius = (precise
        .iter()
        .map(|p| p[0] * p[0] + p[1] * p[1])
        .sum::<f64>()
        / POINTS as f64)
        .sqrt();
    let mut sampled: Vec<[f32; 2]> = precise
        .iter()
        .map(|p| [(p[0] / radius) as f32, (p[1] / radius) as f32])
        .collect();
    let mean = [
        sampled.iter().map(|p| p[0]).sum::<f32>() / POINTS as f32,
        sampled.iter().map(|p| p[1]).sum::<f32>() / POINTS as f32,
    ];
    for p in &mut sampled {
        p[0] -= mean[0];
        p[1] -= mean[1];
    }
    let variance = [
        sampled.iter().map(|p| p[0].powi(2)).sum::<f32>() / POINTS as f32,
        sampled.iter().map(|p| p[1].powi(2)).sum::<f32>() / POINTS as f32,
    ];
    let rms = (variance[0] + variance[1]).sqrt().max(1e-6);
    let axis = variance.map(|v| v.sqrt().max(rms * 0.35));
    let mut features = vec![0.0; 8 * POINTS];
    for i in 0..POINTS {
        for a in 0..2 {
            features[a * POINTS + i] = sampled[i][a] / rms;
            features[(2 + a) * POINTS + i] = sampled[i][a] / axis[a];
            let delta = sampled[(i + 1).min(POINTS - 1)][a] - sampled[i.saturating_sub(1)][a];
            features[(4 + a) * POINTS + i] = delta / axis[a];
            features[(6 + a) * POINTS + i] = delta / rms;
        }
    }
    Some(features)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn features_match_training_resampling_precision() {
        let path = [
            (0.0, 0.0),
            (0.7, 2.1),
            (3.2, -0.4),
            (3.2, -0.4),
            (-1.6, 1.3),
            (4.9, 0.8),
        ];
        // Reference values from retrieval_model.features(curve(path)).
        let reference = [
            (0, -0.80029976_f32),
            (9, -0.25944304),
            (21, 1.1261655),
            (39, -1.7202265),
            (63, 2.1827266),
            (76, 0.30472907),
            (127, -0.016108382),
            (135, -0.45507714),
            (191, 2.3156624),
            (255, -0.04602395),
            (319, 0.17745042),
            (383, -0.036761202),
            (447, 0.16726351),
            (511, -0.012866421),
        ];
        for offset in [0.0, 1_000_000.0] {
            let shifted: Vec<_> = path.iter().map(|p| (p.0 + offset, p.1 - offset)).collect();
            let actual = prepare_features(&shifted).unwrap();
            for (i, expected) in reference {
                assert!(
                    (actual[i] - expected).abs() < 3e-7,
                    "feature {i}: {} vs {expected}",
                    actual[i]
                );
            }
        }
    }

    #[test]
    fn rejects_invalid_model_and_input() {
        assert!(Encoder::from_bytes(b"invalid").is_err());
        assert!(Encoder::from_bytes(b"SWPRT001").is_err());
        assert!(prepare_features(&[(1.0, 1.0); 4]).is_none());
        assert!(prepare_features(&[(0.0, 0.0), (f64::NAN, 1.0)]).is_none());
    }
}
