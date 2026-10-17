//! Shared native attention and projection layers for stroke decoders.
use crate::retrieval_encoder::dot;

const HEADS: usize = 6;

pub(crate) struct Linear {
    pub(crate) inputs: usize,
    pub(crate) outputs: usize,
    pub(crate) weights: Vec<f32>,
    pub(crate) bias: Vec<f32>,
}

impl Linear {
    pub(crate) fn forward(&self, input: &[f32]) -> Vec<f32> {
        self.project(input, 0, self.outputs)
    }

    pub(crate) fn project(&self, input: &[f32], start: usize, outputs: usize) -> Vec<f32> {
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

pub(crate) struct Norm {
    pub(crate) weight: Vec<f32>,
    pub(crate) bias: Vec<f32>,
}

impl Norm {
    pub(crate) fn forward(&self, input: &[f32]) -> Vec<f32> {
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

pub(crate) struct Attention {
    pub(crate) projection: Linear,
    pub(crate) output: Linear,
}

impl Attention {
    pub(crate) fn keys(&self, memory: &[f32]) -> [Vec<f32>; 2] {
        let width = self.projection.inputs;
        [
            self.projection.project(memory, width, width),
            self.projection.project(memory, 2 * width, width),
        ]
    }

    pub(crate) fn forward(&self, query: &[f32], keys: &[Vec<f32>; 2], causal: bool) -> Vec<f32> {
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

pub(crate) struct Block {
    pub(crate) norms: [Norm; 3],
    pub(crate) self_attention: Attention,
    pub(crate) cross_attention: Attention,
    pub(crate) hidden: Linear,
    pub(crate) output: Linear,
}

pub(crate) struct Reader<'a> {
    pub(crate) bytes: &'a [u8],
    pub(crate) position: usize,
}

impl Reader<'_> {
    pub(crate) fn values(&mut self, count: usize) -> Result<Vec<f32>, String> {
        let end = self.position + count * 4;
        let bytes = self
            .bytes
            .get(self.position..end)
            .ok_or("truncated transformer weights")?;
        let values: Vec<_> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect();
        if values.iter().any(|v| !v.is_finite()) {
            return Err("transformer contains nonfinite weights".into());
        }
        self.position = end;
        Ok(values)
    }

    pub(crate) fn linear(&mut self, inputs: usize, outputs: usize) -> Result<Linear, String> {
        Ok(Linear {
            inputs,
            outputs,
            weights: self.values(inputs * outputs)?,
            bias: self.values(outputs)?,
        })
    }

    pub(crate) fn norm(&mut self, width: usize) -> Result<Norm, String> {
        Ok(Norm {
            weight: self.values(width)?,
            bias: self.values(width)?,
        })
    }

    pub(crate) fn attention(&mut self, width: usize) -> Result<Attention, String> {
        Ok(Attention {
            projection: self.linear(width, 3 * width)?,
            output: self.linear(width, width)?,
        })
    }
}
