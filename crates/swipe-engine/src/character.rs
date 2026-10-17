//! Character evidence and latent missing-key likelihoods for retrieved words.
use crate::retrieval_encoder::{convolve, EncoderContext, Layer};

pub(crate) struct CharacterModel {
    pub(crate) encoder_tag: u64,
    pub(crate) bank_tag: u64,
    pub(crate) proxy_weight: f32,
    pub(crate) deletion_probability: f32,
    pub(crate) length_power: f32,
    channels: usize,
    stem: Layer,
    blocks: Vec<[Layer; 2]>,
    output: Layer,
}

impl CharacterModel {
    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.get(..8) != Some(b"SWPCH001") || bytes.len() < 40 {
            return Err("unsupported character decoder format".into());
        }
        let encoder_tag = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let bank_tag = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        let channels = u32::from_le_bytes(bytes[24..28].try_into().unwrap()) as usize;
        let proxy_weight = f32::from_le_bytes(bytes[28..32].try_into().unwrap());
        let deletion_probability = f32::from_le_bytes(bytes[32..36].try_into().unwrap());
        let length_power = f32::from_le_bytes(bytes[36..40].try_into().unwrap());
        if ![8, 16].contains(&channels)
            || !proxy_weight.is_finite()
            || !(0.0..=16.0).contains(&proxy_weight)
            || !deletion_probability.is_finite()
            || !(0.0..1.0).contains(&deletion_probability)
            || !length_power.is_finite()
            || !(0.0..=2.0).contains(&length_power)
        {
            return Err("invalid character decoder configuration".into());
        }
        let mut position = 40;
        let mut read = |inputs, outputs, kernel, dilation| -> Result<Layer, String> {
            let count: usize = inputs * outputs * kernel + outputs;
            let end = position + count * 4;
            let data = bytes
                .get(position..end)
                .ok_or("truncated character decoder")?;
            let values: Vec<_> = data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            if values.iter().any(|v| !v.is_finite()) {
                return Err("character decoder contains nonfinite weights".into());
            }
            position = end;
            let split = count - outputs;
            Ok(Layer {
                inputs,
                outputs,
                kernel,
                stride: 1,
                dilation,
                weights: values[..split].to_vec(),
                bias: values[split..].to_vec(),
            })
        };
        let stem = read(256 + channels, 192, 1, 1)?;
        let mut blocks = Vec::new();
        for dilation in [1, 2, 4, 8] {
            blocks.push([read(192, 256, 3, dilation)?, read(256, 192, 3, dilation)?]);
        }
        let output = read(192, 27, 1, 1)?;
        if position != bytes.len() {
            return Err("character decoder has trailing data".into());
        }
        Ok(Self {
            encoder_tag,
            bank_tag,
            channels,
            proxy_weight,
            deletion_probability,
            length_power,
            stem,
            blocks,
            output,
        })
    }

    pub(crate) fn log_probabilities(&self, context: &EncoderContext) -> Option<Vec<[f32; 27]>> {
        if context.features.len() != self.channels * 64
            || context.sequence.len() != 128 * 32
            || context.embedding.len() != 128
        {
            return None;
        }
        let mut input = vec![0.0; (256 + self.channels) * 64];
        for channel in 0..128 {
            for t in 0..64 {
                let position = t as f32 * 31.0 / 63.0;
                let lo = (position as usize).min(30);
                let fraction = position - lo as f32;
                input[channel * 64 + t] = context.sequence[channel * 32 + lo] * (1.0 - fraction)
                    + context.sequence[channel * 32 + lo + 1] * fraction;
                input[(128 + self.channels + channel) * 64 + t] = context.embedding[channel];
            }
        }
        input[128 * 64..(128 + self.channels) * 64].copy_from_slice(&context.features);
        let mut values = convolve(&input, 64, &self.stem, true).0;
        for [first, second] in &self.blocks {
            let hidden = convolve(&values, 64, first, true).0;
            let update = convolve(&hidden, 64, second, false).0;
            for (value, delta) in values.iter_mut().zip(update) {
                *value = (*value + delta).max(0.0);
            }
        }
        let logits = convolve(&values, 64, &self.output, false).0;
        let mut probabilities = vec![[0.0; 27]; 64];
        for (t, row) in probabilities.iter_mut().enumerate() {
            let largest = (0..27)
                .map(|c| logits[c * 64 + t])
                .fold(f32::NEG_INFINITY, f32::max);
            let normalizer = largest
                + (0..27)
                    .map(|c| (logits[c * 64 + t] - largest).exp())
                    .sum::<f32>()
                    .ln();
            for (c, probability) in row.iter_mut().enumerate() {
                *probability = logits[c * 64 + t] - normalizer;
            }
        }
        probabilities
            .iter()
            .flatten()
            .all(|v| v.is_finite())
            .then_some(probabilities)
    }

    pub(crate) fn cost(&self, probabilities: &[[f32; 27]], word: &str) -> f32 {
        let keys: Vec<_> = word
            .bytes()
            .enumerate()
            .filter_map(|(i, c)| {
                (i == 0 || c != word.as_bytes()[i - 1]).then_some((c - b'a' + 1) as usize)
            })
            .collect();
        let ordinary = ctc_likelihood(probabilities, &keys);
        let p = if keys.len() > 4 {
            self.deletion_probability
        } else {
            0.0
        };
        let likelihood = if p == 0.0 {
            ordinary
        } else {
            let mut deleted = f32::NEG_INFINITY;
            for skip in 1..keys.len() - 1 {
                let retained: Vec<_> = keys
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &c)| (i != skip).then_some(c))
                    .collect();
                deleted = log_add(deleted, ctc_likelihood(probabilities, &retained));
            }
            log_add(
                ordinary + (1.0 - p).ln(),
                deleted + p.ln() - ((keys.len() - 2) as f32).ln(),
            )
        };
        -likelihood / (keys.len() as f32).powf(self.length_power)
    }
}

fn log_add(a: f32, b: f32) -> f32 {
    let largest = a.max(b);
    if largest == f32::NEG_INFINITY {
        return largest;
    }
    largest + ((a - largest).exp() + (b - largest).exp()).ln()
}

fn ctc_likelihood(probabilities: &[[f32; 27]], keys: &[usize]) -> f32 {
    let mut states = vec![f32::NEG_INFINITY; 2 * keys.len() + 1];
    states[0] = 0.0;
    for row in probabilities {
        let mut next = vec![f32::NEG_INFINITY; states.len()];
        for i in 0..states.len() {
            let symbol = if i % 2 == 0 { 0 } else { keys[i / 2] };
            let mut value = states[i];
            if i > 0 {
                value = log_add(value, states[i - 1]);
            }
            if i > 1 && i % 2 == 1 && keys[i / 2] != keys[i / 2 - 1] {
                value = log_add(value, states[i - 2]);
            }
            next[i] = value + row[symbol];
        }
        states = next;
    }
    log_add(states[states.len() - 1], states[states.len() - 2])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ctc_matches_exhaustive_paths_and_repeated_letter_rules() {
        let rows = [[0.2_f32, 0.5, 0.3], [0.4, 0.2, 0.4], [0.1, 0.6, 0.3]];
        let probabilities: Vec<_> = rows
            .iter()
            .map(|row| {
                let mut values = [f32::NEG_INFINITY; 27];
                for i in 0..3 {
                    values[i] = row[i].ln();
                }
                values
            })
            .collect();
        for target in [vec![1, 2], vec![1, 1], vec![2, 1]] {
            let mut expected = 0.0;
            for a in 0..3 {
                for b in 0..3 {
                    for c in 0..3 {
                        let mut collapsed = Vec::new();
                        let mut previous = usize::MAX;
                        for symbol in [a, b, c] {
                            if symbol != previous && symbol != 0 {
                                collapsed.push(symbol);
                            }
                            previous = symbol;
                        }
                        if collapsed == target {
                            expected += rows[0][a] * rows[1][b] * rows[2][c];
                        }
                    }
                }
            }
            assert!((ctc_likelihood(&probabilities, &target).exp() - expected).abs() < 1e-6);
        }
        assert_eq!(
            ctc_likelihood(&probabilities[..2], &[1, 1]),
            f32::NEG_INFINITY
        );
    }

    #[test]
    fn decoder_rejects_invalid_configuration_and_truncated_weights() {
        let mut bytes = b"SWPCH001".to_vec();
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        for value in [1.0_f32, 0.3, 0.3] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        assert!(CharacterModel::from_bytes(&bytes)
            .err()
            .unwrap()
            .contains("truncated"));
        bytes[28..32].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(CharacterModel::from_bytes(&bytes)
            .err()
            .unwrap()
            .contains("configuration"));
    }
}
