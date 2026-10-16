use super::ui::Predictor;
use swipe_engine::{Prediction, RetrievalEngine};

pub struct RetrievalBackend {
    engine: RetrievalEngine,
}

impl RetrievalBackend {
    pub fn new() -> Result<Self, String> {
        let engine = match std::env::var_os("DRAW_SWIPE_MODEL_DIR") {
            Some(directory) => RetrievalEngine::from_files(directory)?,
            None => RetrievalEngine::bundled()?,
        };
        Ok(Self { engine })
    }
}

impl Predictor for RetrievalBackend {
    fn predict(&mut self, path: &[(f64, f64)], limit: usize) -> Vec<Prediction> {
        self.engine.predict_from_path(path, limit)
    }
}
