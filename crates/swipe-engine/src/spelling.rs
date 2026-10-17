//! Full-word spelling likelihoods conditioned on an independently bound encoder.
use crate::retrieval_encoder::{dot, EncoderContext};

const WIDTH: usize = 192;
const HEADS: usize = 6;
const MEMORY: usize = 32;

struct Linear {
    inputs: usize,
    outputs: usize,
    weights: Vec<f32>,
    bias: Vec<f32>,
}

impl Linear {
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        self.project(input, 0, self.outputs)
    }

    fn project(&self, input: &[f32], start: usize, outputs: usize) -> Vec<f32> {
        #[cfg(target_os = "macos")]
        {
            let rows = input.len() / self.inputs;
            let mut result = vec![0.0; rows * outputs];
            // SAFETY: All matrices are contiguous f32 slices with the dimensions
            // passed below. Fixed model widths and word lengths fit in i32.
            unsafe {
                cblas_sgemm(
                    101,
                    111,
                    112,
                    rows as i32,
                    outputs as i32,
                    self.inputs as i32,
                    1.0,
                    input.as_ptr(),
                    self.inputs as i32,
                    self.weights.as_ptr().add(start * self.inputs),
                    self.inputs as i32,
                    0.0,
                    result.as_mut_ptr(),
                    outputs as i32,
                );
            }
            for row in result.chunks_exact_mut(outputs) {
                for (value, bias) in row.iter_mut().zip(&self.bias[start..start + outputs]) {
                    *value += bias;
                }
            }
            result
        }
        #[cfg(not(target_os = "macos"))]
        {
            let mut result = Vec::with_capacity(input.len() / self.inputs * outputs);
            for row in input.chunks_exact(self.inputs) {
                for output in start..start + outputs {
                    result.push(
                        dot(
                            row,
                            &self.weights[output * self.inputs..(output + 1) * self.inputs],
                        ) + self.bias[output],
                    );
                }
            }
            result
        }
    }
}

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemm(
        order: i32,
        transpose_a: i32,
        transpose_b: i32,
        rows: i32,
        columns: i32,
        inner: i32,
        alpha: f32,
        a: *const f32,
        stride_a: i32,
        b: *const f32,
        stride_b: i32,
        beta: f32,
        result: *mut f32,
        stride_result: i32,
    );
}

struct Norm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl Norm {
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(input.len());
        let width = self.weight.len();
        for row in input.chunks_exact(width) {
            let mean = row.iter().map(|&x| f64::from(x)).sum::<f64>() / width as f64;
            let variance = row
                .iter()
                .map(|&x| (f64::from(x) - mean).powi(2))
                .sum::<f64>()
                / width as f64;
            let scale = (variance + 1e-5).sqrt().recip() as f32;
            for (i, &value) in row.iter().enumerate() {
                result.push((value - mean as f32) * scale * self.weight[i] + self.bias[i]);
            }
        }
        result
    }
}

struct Attention {
    projection: Linear,
    output: Linear,
}

impl Attention {
    fn keys(&self, memory: &[f32]) -> [Vec<f32>; 2] {
        let width = self.projection.inputs;
        [
            self.projection.project(memory, width, width),
            self.projection.project(memory, 2 * width, width),
        ]
    }

    fn forward(&self, query: &[f32], keys: &[Vec<f32>; 2], causal: bool) -> Vec<f32> {
        let width = self.projection.inputs;
        let query = self.projection.project(query, 0, width);
        let count = query.len() / width;
        let key_count = keys[0].len() / width;
        let head_width = width / HEADS;
        let mut joined = vec![0.0; query.len()];
        for t in 0..count {
            for head in 0..HEADS {
                let offset = head * head_width;
                let q = &query[t * width + offset..t * width + offset + head_width];
                let end = if causal { t + 1 } else { key_count };
                let mut scores: Vec<_> = (0..end)
                    .map(|k| {
                        dot(
                            q,
                            &keys[0][k * width + offset..k * width + offset + head_width],
                        ) / (head_width as f32).sqrt()
                    })
                    .collect();
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut total = 0.0;
                for score in &mut scores {
                    *score = (*score - maximum).exp();
                    total += *score;
                }
                for (k, score) in scores.into_iter().enumerate() {
                    for d in 0..head_width {
                        joined[t * width + offset + d] +=
                            score / total * keys[1][k * width + offset + d];
                    }
                }
            }
        }
        self.output.forward(&joined)
    }
}

struct Block {
    norms: [Norm; 3],
    self_attention: Attention,
    cross_attention: Attention,
    hidden: Linear,
    output: Linear,
}

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl Reader<'_> {
    fn values(&mut self, count: usize) -> Result<Vec<f32>, String> {
        let end = self.position + count * 4;
        let bytes = self
            .bytes
            .get(self.position..end)
            .ok_or("truncated spelling decoder")?;
        let values: Vec<_> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect();
        if values.iter().any(|v| !v.is_finite()) {
            return Err("spelling decoder contains nonfinite weights".into());
        }
        self.position = end;
        Ok(values)
    }

    fn linear(&mut self, inputs: usize, outputs: usize) -> Result<Linear, String> {
        Ok(Linear {
            inputs,
            outputs,
            weights: self.values(inputs * outputs)?,
            bias: self.values(outputs)?,
        })
    }

    fn norm(&mut self, width: usize) -> Result<Norm, String> {
        Ok(Norm {
            weight: self.values(width)?,
            bias: self.values(width)?,
        })
    }

    fn attention(&mut self, width: usize) -> Result<Attention, String> {
        Ok(Attention {
            projection: self.linear(width, 3 * width)?,
            output: self.linear(width, width)?,
        })
    }
}

pub(crate) struct SpellingMemory {
    keys: Vec<[Vec<f32>; 2]>,
}

pub(crate) struct SpellingModel {
    pub(crate) context_tag: u64,
    pub(crate) encoder_tag: u64,
    pub(crate) bank_tag: u64,
    pub(crate) proxy_weight: f32,
    pub(crate) length_power: f32,
    channels: usize,
    width: usize,
    memory: Linear,
    memory_positions: Vec<f32>,
    symbols: Vec<f32>,
    positions: Vec<f32>,
    blocks: Vec<Block>,
    norm: Norm,
    output: Linear,
}

impl SpellingModel {
    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let version = bytes.get(..8);
        if bytes.len() < 44
            || (version != Some(b"SWPSP001")
                && version != Some(b"SWPSP002")
                && version != Some(b"SWPSP003"))
        {
            return Err("unsupported spelling decoder format".into());
        }
        let (layers, header) = if version != Some(b"SWPSP001") {
            if bytes.len() < 48 {
                return Err("truncated spelling decoder header".into());
            }
            let layers = u32::from_le_bytes(bytes[44..48].try_into().unwrap()) as usize;
            if !(1..=12).contains(&layers) {
                return Err("invalid spelling decoder depth".into());
            }
            (layers, 48)
        } else {
            (3, 44)
        };
        let (width, header) = if version == Some(b"SWPSP003") {
            if bytes.len() < 52 {
                return Err("truncated spelling width header".into());
            }
            let width = u32::from_le_bytes(bytes[48..52].try_into().unwrap()) as usize;
            if ![192, 384].contains(&width) {
                return Err("invalid spelling decoder width".into());
            }
            (width, 52)
        } else {
            (WIDTH, header)
        };
        let tag = |start| u64::from_le_bytes(bytes[start..start + 8].try_into().unwrap());
        let channels = u32::from_le_bytes(bytes[32..36].try_into().unwrap()) as usize;
        let proxy_weight = f32::from_le_bytes(bytes[36..40].try_into().unwrap());
        let length_power = f32::from_le_bytes(bytes[40..44].try_into().unwrap());
        if ![8, 16].contains(&channels)
            || !proxy_weight.is_finite()
            || !(0.0..=16.0).contains(&proxy_weight)
            || !length_power.is_finite()
            || !(0.0..=2.0).contains(&length_power)
        {
            return Err("invalid spelling decoder configuration".into());
        }
        let mut reader = Reader {
            bytes,
            position: header,
        };
        let memory = reader.linear(256 + channels, width)?;
        let memory_positions = reader.values(MEMORY * width)?;
        let symbols = reader.values(29 * width)?;
        let positions = reader.values(64 * width)?;
        let mut blocks = Vec::new();
        for _ in 0..layers {
            blocks.push(Block {
                norms: [
                    reader.norm(width)?,
                    reader.norm(width)?,
                    reader.norm(width)?,
                ],
                self_attention: reader.attention(width)?,
                cross_attention: reader.attention(width)?,
                hidden: reader.linear(width, 4 * width)?,
                output: reader.linear(4 * width, width)?,
            });
        }
        let norm = reader.norm(width)?;
        let output = reader.linear(width, 29)?;
        if reader.position != bytes.len() {
            return Err("spelling decoder has trailing data".into());
        }
        Ok(Self {
            context_tag: tag(8),
            encoder_tag: tag(16),
            bank_tag: tag(24),
            proxy_weight,
            length_power,
            channels,
            width,
            memory,
            memory_positions,
            symbols,
            positions,
            blocks,
            norm,
            output,
        })
    }

    pub(crate) fn prepare(&self, context: &EncoderContext) -> Option<SpellingMemory> {
        if context.features.len() != self.channels * 64
            || context.sequence.len() != 128 * MEMORY
            || context.embedding.len() != 128
        {
            return None;
        }
        let mut input = Vec::with_capacity(MEMORY * (256 + self.channels));
        for t in 0..MEMORY {
            input.extend((0..128).map(|c| context.sequence[c * MEMORY + t]));
            let position = t as f32 * 63.0 / 31.0;
            let lo = (position as usize).min(62);
            let fraction = position - lo as f32;
            input.extend((0..self.channels).map(|c| {
                context.features[c * 64 + lo] * (1.0 - fraction)
                    + context.features[c * 64 + lo + 1] * fraction
            }));
            input.extend_from_slice(&context.embedding);
        }
        let mut memory = self.memory.forward(&input);
        add(&mut memory, &self.memory_positions);
        Some(SpellingMemory {
            keys: self
                .blocks
                .iter()
                .map(|b| b.cross_attention.keys(&memory))
                .collect(),
        })
    }

    pub(crate) fn cost(&self, memory: &SpellingMemory, word: &str) -> Option<f32> {
        if word.is_empty() || word.len() >= 64 || !word.bytes().all(|c| c.is_ascii_lowercase()) {
            return None;
        }
        let letters: Vec<_> = word.bytes().map(|c| (c - b'a' + 1) as usize).collect();
        let input: Vec<_> = std::iter::once(27).chain(letters.iter().copied()).collect();
        let mut values = Vec::with_capacity(input.len() * self.width);
        for (t, symbol) in input.into_iter().enumerate() {
            values.extend((0..self.width).map(|d| {
                self.symbols[symbol * self.width + d] + self.positions[t * self.width + d]
            }));
        }
        for (block, keys) in self.blocks.iter().zip(&memory.keys) {
            let normalized = block.norms[0].forward(&values);
            add(
                &mut values,
                &block.self_attention.forward(
                    &normalized,
                    &block.self_attention.keys(&normalized),
                    true,
                ),
            );
            let normalized = block.norms[1].forward(&values);
            add(
                &mut values,
                &block.cross_attention.forward(&normalized, keys, false),
            );
            let mut hidden = block.hidden.forward(&block.norms[2].forward(&values));
            hidden.iter_mut().for_each(|v| *v = v.max(0.0));
            add(&mut values, &block.output.forward(&hidden));
        }
        let logits = self.output.forward(&self.norm.forward(&values));
        let mut cost = 0.0;
        for (row, target) in logits
            .as_chunks::<29>()
            .0
            .iter()
            .zip(letters.into_iter().chain(std::iter::once(28)))
        {
            let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            cost +=
                maximum + row.iter().map(|v| (v - maximum).exp()).sum::<f32>().ln() - row[target];
        }
        cost.is_finite().then_some(cost)
    }
}

fn add(target: &mut [f32], update: &[f32]) {
    debug_assert_eq!(target.len(), update.len());
    for (value, delta) in target.iter_mut().zip(update) {
        *value += delta;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(width: usize, outputs: usize) -> Linear {
        let mut weights = vec![0.0; outputs * width];
        for row in 0..outputs {
            weights[row * width + row % width] = 1.0;
        }
        Linear {
            inputs: width,
            outputs,
            weights,
            bias: vec![0.0; outputs],
        }
    }

    #[test]
    fn causal_attention_excludes_future_letters() {
        for width in [192, 384] {
            let model = Attention {
                projection: identity(width, 3 * width),
                output: identity(width, width),
            };
            let first = vec![1.0; 2 * width];
            let mut changed = first.clone();
            changed[width..].fill(2.0);
            let original = model.forward(&first, &model.keys(&first), true);
            let causal = model.forward(&changed, &model.keys(&changed), true);
            let unmasked = model.forward(&changed, &model.keys(&changed), false);
            assert_eq!(original[..width], causal[..width]);
            assert!(unmasked[0] > causal[0] + 0.5);
            assert!(causal[width] > original[width]);
        }
    }

    #[test]
    fn normalization_handles_constant_and_shifted_features() {
        let norm = Norm {
            weight: vec![1.0; WIDTH],
            bias: vec![0.0; WIDTH],
        };
        assert!(norm.forward(&vec![7.0; WIDTH]).iter().all(|&v| v == 0.0));
        let input: Vec<_> = (0..WIDTH).map(|i| i as f32 / 10.0).collect();
        let shifted: Vec<_> = input.iter().map(|v| v + 3.0).collect();
        assert!(norm
            .forward(&input)
            .iter()
            .zip(norm.forward(&shifted))
            .all(|(a, b)| (a - b).abs() < 1e-6));
    }

    #[test]
    fn reader_rejects_invalid_configuration_and_nonfinite_weights() {
        assert!(SpellingModel::from_bytes(b"SWPSP001").is_err());
        let mut bytes = vec![0; 44];
        bytes[..8].copy_from_slice(b"SWPSP001");
        assert!(SpellingModel::from_bytes(&bytes)
            .err()
            .unwrap()
            .contains("configuration"));
        bytes[32..36].copy_from_slice(&8_u32.to_le_bytes());
        assert!(SpellingModel::from_bytes(&bytes)
            .err()
            .unwrap()
            .contains("truncated"));
        bytes[..8].copy_from_slice(b"SWPSP002");
        assert!(SpellingModel::from_bytes(&bytes)
            .err()
            .unwrap()
            .contains("header"));
        bytes.extend(0_u32.to_le_bytes());
        for invalid in [0_u32, 13, u32::MAX] {
            bytes[44..48].copy_from_slice(&invalid.to_le_bytes());
            assert!(SpellingModel::from_bytes(&bytes)
                .err()
                .unwrap()
                .contains("depth"));
        }
        bytes[44..48].copy_from_slice(&9_u32.to_le_bytes());
        assert!(SpellingModel::from_bytes(&bytes)
            .err()
            .unwrap()
            .contains("truncated"));
        bytes[..8].copy_from_slice(b"SWPSP003");
        assert!(SpellingModel::from_bytes(&bytes)
            .err()
            .unwrap()
            .contains("header"));
        bytes.extend(0_u32.to_le_bytes());
        for invalid in [0_u32, 193, 768, u32::MAX] {
            bytes[48..52].copy_from_slice(&invalid.to_le_bytes());
            assert!(SpellingModel::from_bytes(&bytes)
                .err()
                .unwrap()
                .contains("width"));
        }
        bytes[48..52].copy_from_slice(&384_u32.to_le_bytes());
        assert!(SpellingModel::from_bytes(&bytes)
            .err()
            .unwrap()
            .contains("truncated"));
        let mut reader = Reader {
            bytes: &f32::NAN.to_le_bytes(),
            position: 0,
        };
        assert!(reader.values(1).unwrap_err().contains("nonfinite"));
    }
}
