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

struct Norm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl Norm {
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut result = Vec::with_capacity(input.len());
        for row in input.as_chunks::<WIDTH>().0 {
            let mean = row.iter().map(|&x| f64::from(x)).sum::<f64>() / WIDTH as f64;
            let variance = row
                .iter()
                .map(|&x| (f64::from(x) - mean).powi(2))
                .sum::<f64>()
                / WIDTH as f64;
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
        [
            self.projection.project(memory, WIDTH, WIDTH),
            self.projection.project(memory, 2 * WIDTH, WIDTH),
        ]
    }

    fn forward(&self, query: &[f32], keys: &[Vec<f32>; 2], causal: bool) -> Vec<f32> {
        let query = self.projection.project(query, 0, WIDTH);
        let count = query.len() / WIDTH;
        let key_count = keys[0].len() / WIDTH;
        let head_width = WIDTH / HEADS;
        let mut joined = vec![0.0; query.len()];
        for t in 0..count {
            for head in 0..HEADS {
                let offset = head * head_width;
                let q = &query[t * WIDTH + offset..t * WIDTH + offset + head_width];
                let end = if causal { t + 1 } else { key_count };
                let mut scores: Vec<_> = (0..end)
                    .map(|k| {
                        dot(
                            q,
                            &keys[0][k * WIDTH + offset..k * WIDTH + offset + head_width],
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
                        joined[t * WIDTH + offset + d] +=
                            score / total * keys[1][k * WIDTH + offset + d];
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

    fn norm(&mut self) -> Result<Norm, String> {
        Ok(Norm {
            weight: self.values(WIDTH)?,
            bias: self.values(WIDTH)?,
        })
    }

    fn attention(&mut self) -> Result<Attention, String> {
        Ok(Attention {
            projection: self.linear(WIDTH, 3 * WIDTH)?,
            output: self.linear(WIDTH, WIDTH)?,
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
        if bytes.len() < 44 || bytes.get(..8) != Some(b"SWPSP001") {
            return Err("unsupported spelling decoder format".into());
        }
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
            position: 44,
        };
        let memory = reader.linear(256 + channels, WIDTH)?;
        let memory_positions = reader.values(MEMORY * WIDTH)?;
        let symbols = reader.values(29 * WIDTH)?;
        let positions = reader.values(64 * WIDTH)?;
        let mut blocks = Vec::new();
        for _ in 0..3 {
            blocks.push(Block {
                norms: [reader.norm()?, reader.norm()?, reader.norm()?],
                self_attention: reader.attention()?,
                cross_attention: reader.attention()?,
                hidden: reader.linear(WIDTH, 4 * WIDTH)?,
                output: reader.linear(4 * WIDTH, WIDTH)?,
            });
        }
        let norm = reader.norm()?;
        let output = reader.linear(WIDTH, 29)?;
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
        if context.features.len() != self.channels * 64 || context.sequence.len() != 128 * MEMORY {
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
        let mut values = Vec::with_capacity(input.len() * WIDTH);
        for (t, symbol) in input.into_iter().enumerate() {
            values.extend(
                (0..WIDTH)
                    .map(|d| self.symbols[symbol * WIDTH + d] + self.positions[t * WIDTH + d]),
            );
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
