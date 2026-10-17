//! Word retrieval using a neural embedding learned from real and synthetic swipes.
use crate::character::CharacterModel;
use crate::retrieval_encoder::{dot, Encoder, DIM};
use crate::spelling::SpellingModel;
use crate::{Prediction, ScoreBreakdown};
use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::rerank::{Candidate, Geometry, Query, Ranker};

struct Word {
    text: String,
    popularity: f32,
    geometry: Geometry,
}

struct Auxiliary {
    encoder: Encoder,
    vectors: Vec<f32>,
    centers: usize,
}

#[derive(Clone, serde::Deserialize)]
struct AdditionalModel {
    weight: f32,
    encoder_tag: u64,
    bank_tag: u64,
}

/// Native neural recognizer. Load matching exported encoder and word-bank files.
///
/// The encoder maps a gesture to a unit vector. Word vectors are cached at
/// export time. Optional reranking combines neural retrieval with independent
/// shape retrieval and a learned comparison of their candidates.
pub struct RetrievalEngine {
    encoder: Encoder,
    words: Vec<Word>,
    word_ids: HashMap<String, usize>,
    vectors: Vec<f32>,
    centers: usize,
    prior_weight: f32,
    tta_degrees: f32,
    adaptive_prior: bool,
    ranker: Option<Ranker>,
    character: Option<CharacterModel>,
    spelling: Option<(Encoder, SpellingModel)>,
    spelling_required: bool,
    spelling_tag: u64,
    auxiliary: Option<Auxiliary>,
    auxiliary_weight: f32,
    auxiliary_tags: [u64; 2],
    additional_specs: Vec<AdditionalModel>,
    additional: Vec<Auxiliary>,
}

impl RetrievalEngine {
    /// Load the validated model and word vectors shipped with this crate.
    pub fn bundled() -> Result<Self, String> {
        let mut engine = Self::from_bytes(
            include_bytes!("../models/retrieval/encoder.bin"),
            include_bytes!("../models/retrieval/words.bin"),
        )?;
        engine.configure(include_str!("../models/retrieval/manifest.json"))?;
        engine.load_ranker(
            include_str!("../models/retrieval/reranker.json"),
            include_bytes!("../models/retrieval/encoder.bin"),
            include_bytes!("../models/retrieval/words.bin"),
        )?;
        Ok(engine)
    }

    pub fn from_files(directory: impl AsRef<Path>) -> Result<Self, String> {
        let encoder =
            std::fs::read(directory.as_ref().join("encoder.bin")).map_err(|e| e.to_string())?;
        let bank =
            std::fs::read(directory.as_ref().join("words.bin")).map_err(|e| e.to_string())?;
        let mut engine = Self::from_bytes(&encoder, &bank)?;
        match std::fs::read_to_string(directory.as_ref().join("manifest.json")) {
            Ok(json) => engine.configure(&json)?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.to_string()),
        }
        if engine.auxiliary_weight > 0.0 {
            if engine.prior_weight != 0.0 || engine.tta_degrees != 0.0 || engine.adaptive_prior {
                return Err("word ensembles require their exported base retrieval settings".into());
            }
            let folder = directory.as_ref().join("auxiliary");
            let other_encoder =
                std::fs::read(folder.join("encoder.bin")).map_err(|e| e.to_string())?;
            let other_bank = std::fs::read(folder.join("words.bin")).map_err(|e| e.to_string())?;
            if engine.auxiliary_tags != [fingerprint(&other_encoder), fingerprint(&other_bank)] {
                return Err("auxiliary model does not match the exported ensemble".into());
            }
            let other = Self::from_bytes(&other_encoder, &other_bank)?;
            if other.words.len() != engine.words.len()
                || other
                    .words
                    .iter()
                    .zip(&engine.words)
                    .any(|(a, b)| a.text != b.text)
            {
                return Err("ensemble word banks differ".into());
            }
            engine.auxiliary = Some(Auxiliary {
                encoder: other.encoder,
                vectors: other.vectors,
                centers: other.centers,
            });
        }
        for (index, specification) in engine.additional_specs.iter().enumerate() {
            let folder = directory.as_ref().join(format!("additional{}", index + 1));
            let encoded = std::fs::read(folder.join("encoder.bin")).map_err(|e| e.to_string())?;
            let vectors = std::fs::read(folder.join("words.bin")).map_err(|e| e.to_string())?;
            if [specification.encoder_tag, specification.bank_tag]
                != [fingerprint(&encoded), fingerprint(&vectors)]
            {
                return Err("additional model does not match the exported ensemble".into());
            }
            let other = Self::from_bytes(&encoded, &vectors)?;
            if other.words.len() != engine.words.len()
                || other
                    .words
                    .iter()
                    .zip(&engine.words)
                    .any(|(a, b)| a.text != b.text)
            {
                return Err("additional word banks differ".into());
            }
            engine.additional.push(Auxiliary {
                encoder: other.encoder,
                vectors: other.vectors,
                centers: other.centers,
            });
        }
        let ranker = directory.as_ref().join("reranker.json");
        match std::fs::read_to_string(ranker) {
            Ok(json) => engine.load_ranker(&json, &encoder, &bank)?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.to_string()),
        }
        match std::fs::read(directory.as_ref().join("character.bin")) {
            Ok(bytes) => {
                let model = CharacterModel::from_bytes(&bytes)?;
                if model.encoder_tag != fingerprint(&encoder)
                    || model.bank_tag != engine.character_binding(&bank)
                {
                    return Err(
                        "character decoder belongs to a different encoder or word bank".into(),
                    );
                }
                if engine.ranker.is_some()
                    || engine.prior_weight != 0.0
                    || engine.tta_degrees != 0.0
                {
                    return Err(
                        "character decoding requires its exported base retrieval settings".into(),
                    );
                }
                engine.character = Some(model);
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.to_string()),
        }
        match std::fs::read(directory.as_ref().join("spelling.bin")) {
            Ok(bytes) => {
                if !engine.spelling_required || engine.spelling_tag != fingerprint(&bytes) {
                    return Err("spelling weights differ from the exported manifest".into());
                }
                let model = SpellingModel::from_bytes(&bytes)?;
                let context = std::fs::read(directory.as_ref().join("spelling/encoder.bin"))
                    .map_err(|e| e.to_string())?;
                if model.context_tag != fingerprint(&context)
                    || model.encoder_tag != fingerprint(&encoder)
                    || model.bank_tag != engine.character_binding(&bank)
                {
                    return Err("spelling decoder belongs to a different retrieval context".into());
                }
                if engine.ranker.is_some()
                    || engine.character.is_some()
                    || engine.prior_weight != 0.0
                    || engine.tta_degrees != 0.0
                    || engine.adaptive_prior
                    || engine.words.iter().any(|word| word.text.len() >= 64)
                {
                    return Err("spelling decoding requires its exported retrieval settings".into());
                }
                engine.spelling = Some((Encoder::from_bytes(&context)?, model));
            }
            Err(error)
                if error.kind() == std::io::ErrorKind::NotFound && !engine.spelling_required => {}
            Err(error) => return Err(error.to_string()),
        }
        Ok(engine)
    }

    fn character_binding(&self, bank: &[u8]) -> u64 {
        if self.auxiliary.is_none() {
            return fingerprint(bank);
        }
        let mut binding = bank.to_vec();
        for tag in self.auxiliary_tags {
            binding.extend_from_slice(&tag.to_le_bytes());
        }
        binding.extend_from_slice(&self.auxiliary_weight.to_le_bytes());
        for model in &self.additional_specs {
            binding.extend_from_slice(&model.encoder_tag.to_le_bytes());
            binding.extend_from_slice(&model.bank_tag.to_le_bytes());
            binding.extend_from_slice(&model.weight.to_le_bytes());
        }
        fingerprint(&binding)
    }

    fn load_ranker(&mut self, json: &str, encoder: &[u8], bank: &[u8]) -> Result<(), String> {
        if self.auxiliary.is_some() {
            return Err("candidate ranker is not bound to this word ensemble".into());
        }
        let ranker = Ranker::parse(json)?;
        if ranker.encoder_tag != fingerprint(encoder) || ranker.bank_tag != fingerprint(bank) {
            return Err("candidate ranker belongs to a different encoder or word bank".into());
        }
        self.ranker = Some(ranker);
        Ok(())
    }

    pub fn from_bytes(model: &[u8], bank: &[u8]) -> Result<Self, String> {
        let encoder = Encoder::from_bytes(model)?;
        let multiple = bank.get(..8) == Some(b"SWPBK002");
        if (!multiple && bank.get(..8) != Some(b"SWPBK001"))
            || bank.len() < if multiple { 28 } else { 24 }
        {
            return Err("unsupported route word-bank format".into());
        }
        let tag = u64::from_le_bytes(bank[8..16].try_into().unwrap());
        if tag != fingerprint(model) {
            return Err("word bank belongs to a different route encoder".into());
        }
        let count = u32::from_le_bytes(bank[16..20].try_into().unwrap()) as usize;
        let dimension = u32::from_le_bytes(bank[20..24].try_into().unwrap()) as usize;
        if ![DIM, 512].contains(&dimension)
            || dimension != encoder.dimension()
            || count == 0
            || count > bank.len() / (2 + 4 + dimension * 4)
        {
            return Err("invalid route word-bank dimensions".into());
        }
        let centers = if multiple {
            u32::from_le_bytes(bank[24..28].try_into().unwrap()) as usize
        } else {
            1
        };
        if !(1..=8).contains(&centers) {
            return Err("invalid number of word-route centers".into());
        }
        let mut position = if multiple { 28 } else { 24 };
        if count > (bank.len() - position) / (2 + 1 + 4 + dimension * centers * 4) {
            return Err("word-bank count exceeds its payload".into());
        }
        let mut words = Vec::with_capacity(count);
        let mut vectors = Vec::with_capacity(count * centers * dimension);
        let mut seen = HashSet::new();
        for _ in 0..count {
            let length = bank
                .get(position..position + 2)
                .ok_or("truncated word-bank entry")?;
            let length = u16::from_le_bytes(length.try_into().unwrap()) as usize;
            position += 2;
            let text = bank
                .get(position..position + length)
                .ok_or("truncated word-bank word")?;
            let text = std::str::from_utf8(text)
                .map_err(|e| e.to_string())?
                .to_string();
            if text.is_empty()
                || !text.bytes().all(|b| b.is_ascii_lowercase())
                || !seen.insert(text.clone())
            {
                return Err("invalid or duplicate word-bank word".into());
            }
            position += length;
            let values = bank
                .get(position..position + (dimension * centers + 1) * 4)
                .ok_or("truncated word-bank vector")?;
            let values: Vec<_> = values
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| f32::from_le_bytes(*v))
                .collect();
            if values.iter().any(|v| !v.is_finite()) || !(0.0..=1.0).contains(&values[0]) {
                return Err("invalid word-bank values".into());
            }
            for center in values[1..].chunks_exact(dimension) {
                let norm = center.iter().map(|v| v * v).sum::<f32>();
                if !(0.99..=1.01).contains(&norm) {
                    return Err("word-bank vector is not normalized".into());
                }
                vectors.extend_from_slice(center);
            }
            if text.bytes().all(|b| b == text.as_bytes()[0]) {
                return Err("stationary word-bank route".into());
            }
            words.push(Word {
                geometry: Geometry::new(&text),
                text,
                popularity: values[0],
            });
            position += (dimension * centers + 1) * 4;
        }
        if position != bank.len() {
            return Err("word bank has trailing data".into());
        }
        Ok(Self {
            encoder,
            word_ids: words
                .iter()
                .enumerate()
                .map(|(i, w)| (w.text.clone(), i))
                .collect(),
            words,
            vectors,
            centers,
            prior_weight: 0.025,
            tta_degrees: 0.0,
            adaptive_prior: false,
            ranker: None,
            character: None,
            spelling: None,
            spelling_required: false,
            spelling_tag: 0,
            auxiliary: None,
            auxiliary_weight: 0.0,
            auxiliary_tags: [0, 0],
            additional_specs: Vec::new(),
            additional: Vec::new(),
        })
    }

    /// Apply the validation-selected prior and orientation ensemble settings.
    pub fn configure(&mut self, json: &str) -> Result<(), String> {
        #[derive(serde::Deserialize)]
        struct Settings {
            prior_weight: f32,
            #[serde(default)]
            tta_degrees: f32,
            #[serde(default)]
            adaptive_prior: bool,
            #[serde(default)]
            auxiliary_weight: f32,
            #[serde(default)]
            auxiliary_encoder_tag: u64,
            #[serde(default)]
            auxiliary_bank_tag: u64,
            #[serde(default)]
            spelling: Option<serde_json::Value>,
            #[serde(default)]
            additional_models: Vec<AdditionalModel>,
        }
        let settings: Settings = serde_json::from_str(json).map_err(|e| e.to_string())?;
        if !(0.0..=2.0).contains(&settings.prior_weight)
            || !(0.0..=24.0).contains(&settings.tta_degrees)
            || !(0.0..=1.0).contains(&settings.auxiliary_weight)
        {
            return Err("invalid route ranking settings".into());
        }
        if settings.additional_models.len() > 2
            || settings.additional_models.iter().any(|model| {
                !model.weight.is_finite()
                    || !(0.0..1.0).contains(&model.weight)
                    || model.weight == 0.0
            })
            || settings.auxiliary_weight
                + settings
                    .additional_models
                    .iter()
                    .map(|model| model.weight)
                    .sum::<f32>()
                > 1.0
            || (!settings.additional_models.is_empty() && settings.auxiliary_weight == 0.0)
        {
            return Err("invalid complete word mixture settings".into());
        }
        self.prior_weight = settings.prior_weight;
        self.tta_degrees = settings.tta_degrees;
        self.adaptive_prior = settings.adaptive_prior;
        self.auxiliary_weight = settings.auxiliary_weight;
        self.auxiliary_tags = [settings.auxiliary_encoder_tag, settings.auxiliary_bank_tag];
        self.additional_specs = settings.additional_models;
        self.spelling_required = settings.spelling.is_some();
        self.spelling_tag = match settings.spelling {
            Some(value) => value["tag"]
                .as_u64()
                .ok_or("missing spelling weight binding")?,
            None => 0,
        };
        Ok(())
    }

    pub fn word_count(&self) -> usize {
        self.words.len()
    }

    /// Restrict the frequency-ordered vocabulary, for paired benchmarks.
    pub fn truncate_vocabulary(&mut self, count: usize) {
        self.words.truncate(count);
        self.vectors
            .truncate(count * self.centers * self.encoder.dimension());
        if let Some(auxiliary) = &mut self.auxiliary {
            auxiliary
                .vectors
                .truncate(count * auxiliary.centers * auxiliary.encoder.dimension());
        }
        for model in &mut self.additional {
            model
                .vectors
                .truncate(count * model.centers * model.encoder.dimension());
        }
    }

    pub fn set_prior_weight(&mut self, weight: f32) {
        if weight.is_finite() {
            self.prior_weight = weight.clamp(0.0, 2.0);
        }
    }

    /// Expose the normalized encoder output for cross-runtime verification.
    pub fn embedding(&self, path: &[(f64, f64)]) -> Option<Vec<f32>> {
        self.encoder.embed(path)
    }

    /// Expose character log probabilities for cross-runtime verification.
    pub fn character_probabilities(&self, path: &[(f64, f64)]) -> Option<Vec<[f32; 27]>> {
        self.character
            .as_ref()?
            .log_probabilities(&self.encoder.context(path)?)
    }

    /// Expose full spelling costs for cross-runtime verification.
    pub fn spelling_costs(&self, path: &[(f64, f64)], words: &[String]) -> Option<Vec<f32>> {
        let (encoder, model) = self.spelling.as_ref()?;
        let memory = model.prepare(&encoder.context(path)?)?;
        words.iter().map(|word| model.cost(&memory, word)).collect()
    }

    /// Extract the same candidate features used during deployed reranking.
    pub fn ranking_features(&self, path: &[(f64, f64)], limit: usize) -> Vec<Candidate> {
        let predictions = self.retrieve(path, limit, true);
        let Some(query) = Query::new(path) else {
            return Vec::new();
        };
        let best = predictions.first().map_or(0.0, |p| p.rating as f32);
        predictions
            .into_iter()
            .map(|p| {
                let index = self.word_ids[&p.word];
                Candidate {
                    features: query.features(
                        &self.words[index].geometry,
                        [
                            p.raw_score as f32,
                            p.rating as f32,
                            p.popularity as f32,
                            p.rating as f32 - best,
                        ],
                    ),
                    word: p.word,
                }
            })
            .collect()
    }

    pub fn predict_from_path(&self, path: &[(f64, f64)], limit: usize) -> Vec<Prediction> {
        if self.spelling_required && self.spelling.is_none() {
            return Vec::new();
        }
        if let Some((encoder, model)) = &self.spelling {
            if limit == 0 {
                return Vec::new();
            }
            let Some(context) = encoder.context(path) else {
                return Vec::new();
            };
            let Some(memory) = model.prepare(&context) else {
                return Vec::new();
            };
            let Some(query) = self.encoder.embed(path) else {
                return Vec::new();
            };
            let mut candidates = self.rank_query(&query, path, 128, false);
            for candidate in &mut candidates {
                let Some(cost) = model.cost(&memory, &candidate.word) else {
                    return Vec::new();
                };
                candidate.rating = (cost
                    / ((candidate.word.len() + 1) as f32).powf(model.length_power)
                    - model.proxy_weight * 32.0 * (1.0 - candidate.raw_score as f32))
                    as f64;
            }
            candidates.sort_by(|a, b| a.rating.total_cmp(&b.rating).then(a.word.cmp(&b.word)));
            candidates.truncate(limit);
            return candidates;
        }
        if let Some(model) = &self.character {
            if limit == 0 {
                return Vec::new();
            }
            let Some(context) = self.encoder.context(path) else {
                return Vec::new();
            };
            let Some(probabilities) = model.log_probabilities(&context) else {
                return Vec::new();
            };
            let mut candidates = self.rank_query(&context.embedding, path, limit.max(128), false);
            for candidate in &mut candidates {
                candidate.rating = (model.cost(&probabilities, &candidate.word)
                    - model.proxy_weight * 32.0 * (1.0 - candidate.raw_score as f32))
                    as f64;
            }
            candidates.sort_by(|a, b| a.rating.total_cmp(&b.rating));
            candidates.truncate(limit);
            return candidates;
        }
        let Some(ranker) = &self.ranker else {
            return self.retrieve(path, limit, false);
        };
        if limit == 0 {
            return Vec::new();
        }
        let features = self.ranking_features(path, ranker.shortlist.max(limit));
        if features
            .first()
            .is_some_and(|c| c.features[0] < ranker.trust_distance)
        {
            if limit > ranker.shortlist / 2 {
                return self.retrieve(path, limit, false);
            }
            return features
                .into_iter()
                .take(limit)
                .map(|c| Prediction {
                    word: c.word,
                    raw_score: c.features[0] as f64,
                    popularity: c.features[2] as f64,
                    rating: c.features[1] as f64,
                    breakdown: ScoreBreakdown::default(),
                })
                .collect();
        }
        let mut candidates: Vec<_> = features
            .into_iter()
            .map(|candidate| (ranker.score(&candidate.features), candidate))
            .collect();
        candidates.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.word.cmp(&b.1.word)));
        candidates
            .into_iter()
            .take(limit)
            .map(|(rating, c)| Prediction {
                word: c.word,
                raw_score: c.features[0] as f64,
                popularity: c.features[2] as f64,
                rating: rating as f64,
                breakdown: ScoreBreakdown::default(),
            })
            .collect()
    }

    fn retrieve(&self, path: &[(f64, f64)], limit: usize, geometry_union: bool) -> Vec<Prediction> {
        if limit == 0 {
            return Vec::new();
        }
        let Some(mut query) = self.encoder.embed(path) else {
            return Vec::new();
        };
        if self.tta_degrees > 0.0 {
            for sign in [-1.0_f64, 1.0] {
                let angle = sign * (self.tta_degrees as f64).to_radians();
                let (sin, cos) = angle.sin_cos();
                let origin = path[0];
                let rotated: Vec<_> = path
                    .iter()
                    .map(|p| {
                        let x = p.0 - origin.0;
                        let y = p.1 - origin.1;
                        (cos * x - sin * y, sin * x + cos * y)
                    })
                    .collect();
                let Some(vector) = self.encoder.embed(&rotated) else {
                    return Vec::new();
                };
                for (a, b) in query.iter_mut().zip(vector) {
                    *a += b;
                }
            }
            let norm = query.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            for value in &mut query {
                *value /= norm;
            }
        }
        self.rank_query(&query, path, limit, geometry_union)
    }

    fn rank_query(
        &self,
        query: &[f32],
        path: &[(f64, f64)],
        limit: usize,
        geometry_union: bool,
    ) -> Vec<Prediction> {
        if self.auxiliary_weight > 0.0 && self.auxiliary.is_none() {
            return Vec::new();
        }
        if self.additional.len() != self.additional_specs.len() {
            return Vec::new();
        }
        let Some(additional_queries) = self
            .additional
            .iter()
            .map(|model| model.encoder.embed(path))
            .collect::<Option<Vec<_>>>()
        else {
            return Vec::new();
        };
        let additional_weight: f32 = self.additional_specs.iter().map(|model| model.weight).sum();
        let auxiliary_query = if let Some(model) = &self.auxiliary {
            let Some(embedding) = model.encoder.embed(path) else {
                return Vec::new();
            };
            Some(embedding)
        } else {
            None
        };
        let mut ranked: Vec<_> = self
            .vectors
            .chunks_exact(self.centers * query.len())
            .enumerate()
            .map(|(index, centers)| {
                let mut similarity = center_similarity(query, centers);
                if let (Some(model), Some(embedding)) = (&self.auxiliary, &auxiliary_query) {
                    let width = model.centers * embedding.len();
                    let start = index * width;
                    let other = center_similarity(embedding, &model.vectors[start..start + width]);
                    similarity = (1.0 - self.auxiliary_weight - additional_weight) * similarity
                        + self.auxiliary_weight * other;
                }
                for ((model, embedding), specification) in self
                    .additional
                    .iter()
                    .zip(&additional_queries)
                    .zip(&self.additional_specs)
                {
                    let width = model.centers * embedding.len();
                    similarity += specification.weight
                        * center_similarity(
                            embedding,
                            &model.vectors[index * width..(index + 1) * width],
                        );
                }
                let distance = (1.0 - similarity).max(0.0);
                let popularity = self.words[index].popularity;
                let rating = if self.adaptive_prior {
                    distance + self.prior_weight * (1.0 - popularity) * distance.sqrt()
                } else {
                    distance - self.prior_weight * popularity
                };
                (rating, index, distance)
            })
            .collect();
        let compare = |a: &(f32, usize, f32), b: &(f32, usize, f32)| {
            a.0.total_cmp(&b.0)
                .then(self.words[a.1].text.cmp(&self.words[b.1].text))
        };
        let count = limit.min(ranked.len());
        if count < ranked.len() {
            ranked.select_nth_unstable_by(count, compare);
        }
        if geometry_union && count >= 2 && count < ranked.len() {
            let neural_count = count / 2;
            ranked[..count].select_nth_unstable_by(neural_count, compare);
            let selected: HashSet<_> = ranked[..neural_count].iter().map(|p| p.1).collect();
            let query = Query::new(path).unwrap();
            let mut geometry: Vec<_> = self
                .words
                .iter()
                .enumerate()
                .filter(|(i, _)| !selected.contains(i))
                .map(|(i, w)| (query.coarse(&w.geometry), i))
                .collect();
            let extra = count - neural_count;
            geometry.select_nth_unstable_by(extra, |a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            let extra_ids: HashSet<_> = geometry[..extra].iter().map(|p| p.1).collect();
            ranked.retain(|p| selected.contains(&p.1) || extra_ids.contains(&p.1));
        }
        ranked.truncate(count);
        ranked.sort_by(compare);
        ranked
            .into_iter()
            .map(|(rating, index, raw)| Prediction {
                word: self.words[index].text.clone(),
                raw_score: raw as f64,
                popularity: self.words[index].popularity as f64,
                rating: rating as f64,
                breakdown: ScoreBreakdown::default(),
            })
            .collect()
    }
}

/// Soft assignment preserves distinct motor modes without adding word priors.
fn center_similarity(query: &[f32], centers: &[f32]) -> f32 {
    let count = centers.len() / query.len();
    if count == 1 {
        return dot(query, centers);
    }
    let mut scores = [0.0_f32; 8];
    for (score, center) in scores.iter_mut().zip(centers.chunks_exact(query.len())) {
        *score = dot(query, center);
    }
    let maximum = scores[..count]
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let (mut numerator, mut denominator) = (0.0, 0.0);
    for score in &scores[..count] {
        let weight = ((score - maximum) / 0.1).exp();
        numerator += weight * score;
        denominator += weight;
    }
    numerator / denominator
}

fn fingerprint(data: &[u8]) -> u64 {
    data.iter().fold(0xcbf29ce484222325, |value, byte| {
        (value ^ *byte as u64).wrapping_mul(0x100000001b3)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_bank_width_must_match_its_encoder() {
        let model = include_bytes!("../models/retrieval/encoder.bin");
        let mut bank = b"SWPBK001".to_vec();
        bank.extend(fingerprint(model).to_le_bytes());
        bank.extend(1_u32.to_le_bytes());
        bank.extend(512_u32.to_le_bytes());
        bank.extend(5_u16.to_le_bytes());
        bank.extend(b"hello");
        bank.extend(0.5_f32.to_le_bytes());
        bank.extend(1.0_f32.to_le_bytes());
        bank.extend(vec![0; 511 * 4]);
        assert!(RetrievalEngine::from_bytes(model, &bank)
            .err()
            .unwrap()
            .contains("dimensions"));
    }

    #[test]
    fn bundled_model_matches_pytorch_and_recognizes_reference_routes() {
        #[derive(serde::Deserialize)]
        struct Case {
            word: String,
            points: Vec<(f64, f64)>,
            embedding: Vec<f32>,
        }
        let engine = RetrievalEngine::bundled().unwrap();
        assert_eq!(engine.word_count(), 60000);
        let cases: Vec<Case> =
            serde_json::from_str(include_str!("../models/retrieval/verification.json")).unwrap();
        for case in cases {
            let actual = engine.embedding(&case.points).unwrap();
            let error = actual
                .iter()
                .zip(&case.embedding)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(error < 0.0001, "native embedding differs for {}", case.word);
            let predictions = engine.predict_from_path(&case.points, 5);
            assert_eq!(predictions[0].word, case.word);
            assert!(predictions.iter().all(|p| p.rating.is_finite()));
        }
        #[derive(serde::Deserialize)]
        struct RankedCase {
            points: Vec<(f64, f64)>,
            predictions: Vec<Expected>,
            reranked: bool,
        }
        #[derive(serde::Deserialize)]
        struct Expected {
            word: String,
            rating: f64,
        }
        let ranked: Vec<RankedCase> = serde_json::from_str(include_str!(
            "../models/retrieval/reranker-verification.json"
        ))
        .unwrap();
        assert!(ranked.iter().any(|c| c.reranked));
        assert!(ranked.iter().any(|c| !c.reranked));
        for case in ranked {
            let actual = engine.predict_from_path(&case.points, 5);
            assert_eq!(actual.len(), case.predictions.len());
            for (actual, expected) in actual.iter().zip(case.predictions) {
                assert_eq!(actual.word, expected.word);
                assert!((actual.rating - expected.rating).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn ensemble_combines_word_scores_and_requires_all_components() {
        let mut first = vec![0_u8; 8 + 405072 * 4];
        first[..8].copy_from_slice(b"SWPRT001");
        let bias = first.len() - DIM * 4;
        first[bias..bias + 4].copy_from_slice(&1.0_f32.to_le_bytes());
        let mut second = first.clone();
        second[bias..bias + 4].copy_from_slice(&0.0_f32.to_le_bytes());
        second[bias + 4..bias + 8].copy_from_slice(&1.0_f32.to_le_bytes());
        let mut bank = b"SWPBK001".to_vec();
        bank.extend(fingerprint(&first).to_le_bytes());
        bank.extend(3_u32.to_le_bytes());
        bank.extend((DIM as u32).to_le_bytes());
        for (name, coordinates) in [
            ("hello", [1.0_f32, 0.0]),
            ("world", [0.0, 1.0]),
            ("lunch", [0.6, 0.8]),
        ] {
            bank.extend((name.len() as u16).to_le_bytes());
            bank.extend(name.as_bytes());
            bank.extend(0.5_f32.to_le_bytes());
            let mut vector = [0.0_f32; DIM];
            vector[..2].copy_from_slice(&coordinates);
            for value in vector {
                bank.extend(value.to_le_bytes());
            }
        }
        let mut engine = RetrievalEngine::from_bytes(&first, &bank).unwrap();
        engine.prior_weight = 0.0;
        engine.auxiliary_weight = 0.75;
        let path = [(0.0, 0.0), (1.0, 2.0), (3.0, 1.0)];
        assert!(engine.predict_from_path(&path, 3).is_empty());
        let vectors = [[1.0_f32, 0.0], [0.0, 1.0], [-1.0, 0.0]]
            .into_iter()
            .flat_map(|xy| {
                let mut vector = [0.0; DIM];
                vector[..2].copy_from_slice(&xy);
                vector
            })
            .collect();
        engine.auxiliary = Some(Auxiliary {
            encoder: Encoder::from_bytes(&second).unwrap(),
            vectors,
            centers: 1,
        });
        let ranked = engine.predict_from_path(&path, 3);
        assert_eq!(
            ranked.iter().map(|p| p.word.as_str()).collect::<Vec<_>>(),
            ["world", "hello", "lunch"]
        );
        for (prediction, expected) in ranked.iter().zip([0.25, 0.75, 0.85]) {
            assert!((prediction.raw_score - expected).abs() < 1e-6);
        }
        assert!(engine.configure(r#"{"prior_weight":0,"auxiliary_weight":0.75,"additional_models":[{"weight":0.5,"encoder_tag":0,"bank_tag":0}]}"#).is_err());
        engine.auxiliary_weight = 0.25;
        engine.additional_specs.push(AdditionalModel {
            weight: 0.5,
            encoder_tag: 0,
            bank_tag: 0,
        });
        assert!(engine.predict_from_path(&path, 3).is_empty());
        let vectors = [[0.0_f32, 1.0], [-1.0, 0.0], [1.0, 0.0]]
            .into_iter()
            .flat_map(|xy| {
                let mut vector = [0.0; DIM];
                vector[..2].copy_from_slice(&xy);
                vector
            })
            .collect();
        engine.additional.push(Auxiliary {
            encoder: Encoder::from_bytes(&first).unwrap(),
            vectors,
            centers: 1,
        });
        let ranked = engine.predict_from_path(&path, 3);
        assert_eq!(
            ranked.iter().map(|p| p.word.as_str()).collect::<Vec<_>>(),
            ["lunch", "hello", "world"]
        );
        for (prediction, expected) in ranked.iter().zip([0.35, 0.75, 1.25]) {
            assert!((prediction.raw_score - expected).abs() < 1e-6);
        }
        engine.truncate_vocabulary(2);
        assert_eq!(engine.predict_from_path(&path, 3).len(), 2);
    }

    #[test]
    fn validates_bank_identity_and_values() {
        let mut model = vec![0_u8; 8 + 405072 * 4];
        model[..8].copy_from_slice(b"SWPRT001");
        let mut bank = b"SWPBK001".to_vec();
        bank.extend(fingerprint(&model).to_le_bytes());
        bank.extend(1_u32.to_le_bytes());
        bank.extend((DIM as u32).to_le_bytes());
        bank.extend(5_u16.to_le_bytes());
        bank.extend(b"hello");
        bank.extend(0.5_f32.to_le_bytes());
        bank.extend(1.0_f32.to_le_bytes());
        bank.extend(vec![0_u8; (DIM - 1) * 4]);
        let mut engine = RetrievalEngine::from_bytes(&model, &bank).unwrap();
        assert_eq!(engine.word_count(), 1);
        assert!(engine.predict_from_path(&[(1.0, 1.0); 5], 5).is_empty());
        assert!(engine
            .configure(r#"{"prior_weight":0.1,"tta_degrees":10}"#)
            .is_ok());
        assert!(engine.configure(r#"{"prior_weight":-1}"#).is_err());
        let mut mismatched = bank.clone();
        mismatched[8] ^= 1;
        assert!(RetrievalEngine::from_bytes(&model, &mismatched).is_err());
        assert!(RetrievalEngine::from_bytes(&model, &bank[..bank.len() - 1]).is_err());
        let start = bank.len() - DIM * 4;
        bank[start..start + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(RetrievalEngine::from_bytes(&model, &bank).is_err());
    }
}
