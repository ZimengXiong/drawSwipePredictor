use super::ui::Predictor;
use swipe_engine::{Prediction, SwipeEngine};

pub struct DtwBackend {
    engine: SwipeEngine,
}

impl DtwBackend {
    pub fn new() -> Self {
        const DICT_TEXT: &str = include_str!("../../../word_freq.txt");
        let mut engine = SwipeEngine::new();
        engine.load_dictionary(DICT_TEXT);
        Self { engine }
    }
}

impl Predictor for DtwBackend {
    fn predict(&mut self, path: &[(f64, f64)], limit: usize) -> Vec<Prediction> {
        self.engine.predict_from_path(path, limit)
    }
}
