use super::ui::Predictor;
use swipe_engine::{MemoryEngine, Prediction};

pub struct MemoryBackend {
    engine: MemoryEngine,
}

impl Default for MemoryBackend {
    fn default() -> Self {
        let mut engine = MemoryEngine::new();
        engine.load_dictionary(include_str!("../../../word_freq.txt"));
        Self { engine }
    }
}

impl Predictor for MemoryBackend {
    fn predict(&mut self, path: &[(f64, f64)], limit: usize) -> Vec<Prediction> {
        self.engine.predict_from_path(path, limit)
    }
}
