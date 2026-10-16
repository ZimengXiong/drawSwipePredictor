//! Small native inference runtime for the exported convolutional route encoder.
//! The fixed architecture keeps deployment independent of Python and ONNX.
use crate::memory_geometry::resample;

const POINTS: usize = 64;
pub(crate) const DIM: usize = 128;

struct Layer {
    weights: Vec<f32>,
    bias: Vec<f32>,
    inputs: usize,
    outputs: usize,
}

pub(crate) struct Encoder {
    convolutions: Vec<Layer>,
    dense: Vec<Layer>,
}

impl Encoder {
    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.get(..8) != Some(b"SWPRT001") {
            return Err("unsupported route encoder format".into());
        }
        let mut position = 8;
        let mut layer = |inputs: usize, outputs: usize, kernel: usize| -> Result<Layer, String> {
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
            })
        };
        let convolutions = [(8, 48), (48, 64), (64, 96), (96, 128)]
            .into_iter()
            .map(|(i, o)| layer(i, o, 5))
            .collect::<Result<_, _>>()?;
        let dense = [(1024, 256), (256, DIM)]
            .into_iter()
            .map(|(i, o)| layer(i, o, 1))
            .collect::<Result<_, _>>()?;
        if position != bytes.len() {
            return Err("route encoder has trailing data".into());
        }
        Ok(Self {
            convolutions,
            dense,
        })
    }

    pub(crate) fn embed(&self, raw: &[(f64, f64)]) -> Option<[f32; DIM]> {
        let mut values = prepare_features(raw)?;
        let mut length = POINTS;
        for (stage, layer) in self.convolutions.iter().enumerate() {
            let stride = if stage == 0 { 1 } else { 2 };
            let next_length = (length + 4 - 5) / stride + 1;
            let mut output = vec![0.0; layer.outputs * next_length];
            for channel in 0..layer.outputs {
                for t in 0..next_length {
                    let mut sum = layer.bias[channel];
                    for input in 0..layer.inputs {
                        for k in 0..5 {
                            let position = (t * stride + k) as isize - 2;
                            if position >= 0 && (position as usize) < length {
                                let w = layer.weights[(channel * layer.inputs + input) * 5 + k];
                                sum += w * values[input * length + position as usize];
                            }
                        }
                    }
                    output[channel * next_length + t] = sum.max(0.0);
                }
            }
            values = output;
            length = next_length;
        }
        for (stage, layer) in self.dense.iter().enumerate() {
            let mut output = layer.bias.clone();
            for (channel, sum) in output.iter_mut().enumerate() {
                for (value, weight) in values
                    .iter()
                    .zip(&layer.weights[channel * layer.inputs..(channel + 1) * layer.inputs])
                {
                    *sum += value * weight;
                }
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
        vector.try_into().ok()
    }
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
    let points: Vec<_> = centered
        .iter()
        .map(|p| [(p[0] / scale) as f32, (p[1] / scale) as f32])
        .collect();
    let mut sampled = resample(&points, POINTS);
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
    fn rejects_invalid_model_and_input() {
        assert!(Encoder::from_bytes(b"invalid").is_err());
        assert!(Encoder::from_bytes(b"SWPRT001").is_err());
        assert!(prepare_features(&[(1.0, 1.0); 4]).is_none());
        assert!(prepare_features(&[(0.0, 0.0), (f64::NAN, 1.0)]).is_none());
    }
}
