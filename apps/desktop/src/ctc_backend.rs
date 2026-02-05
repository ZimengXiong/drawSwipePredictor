use super::ui::Predictor;
use swipe_engine::{CtcPredictor, LexiconConfig, Prediction};

pub struct CtcBackend(CtcPredictor);

impl CtcBackend {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        const DICT_TEXT: &str = include_str!("../../../word_freq.txt");
        const DEFAULT_MODEL_PATH: &str = "training/checkpoints/swipe_ctc.onnx";
        const DEFAULT_CHARS_PATH: &str = "training/checkpoints/swipe_ctc_chars.json";
        const DEFAULT_N_POINTS: usize = 64;
        const DEFAULT_MAX_VOCAB: usize = 50_000;
        const DEFAULT_MIN_COUNT: u64 = 0;

        let model_path =
            std::env::var("SWIPE_CTC_MODEL").unwrap_or_else(|_| DEFAULT_MODEL_PATH.to_string());
        let chars_path =
            std::env::var("SWIPE_CTC_CHARS").unwrap_or_else(|_| DEFAULT_CHARS_PATH.to_string());
        let n_points = std::env::var("SWIPE_CTC_POINTS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_N_POINTS);
        let max_vocab = std::env::var("SWIPE_CTC_MAX_VOCAB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_MAX_VOCAB);
        let min_count = std::env::var("SWIPE_CTC_MIN_COUNT")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(DEFAULT_MIN_COUNT);
        let pop_weight = std::env::var("SWIPE_CTC_POP_WEIGHT")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.25);

        let mut predictor = CtcPredictor::new_with_lexicon(
            &model_path,
            &chars_path,
            Some(DICT_TEXT),
            n_points,
            LexiconConfig {
                max_words: max_vocab,
                min_count,
            },
        )?;
        predictor.set_pop_weight(pop_weight);

        Ok(Self(predictor))
    }
}

impl Predictor for CtcBackend {
    fn predict(&mut self, path: &[(f64, f64)], limit: usize) -> Vec<Prediction> {
        self.0.predict(path, limit)
    }
}
