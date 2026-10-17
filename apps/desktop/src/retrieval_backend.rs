use super::ui::Predictor;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use swipe_engine::{Prediction, RetrievalEngine};

fn model_directory(executable: &Path, explicit: Option<OsString>) -> Option<PathBuf> {
    if let Some(directory) = explicit {
        return Some(directory.into());
    }
    let parent = executable.parent()?;
    let contents = parent.parent()?;
    if parent.file_name()? == "MacOS" && contents.file_name()? == "Contents" {
        // A packaged app requires its complete model; a missing directory must
        // fail instead of silently selecting the older compiled model.
        Some(contents.join("Resources/model"))
    } else {
        None
    }
}

pub struct RetrievalBackend {
    engine: RetrievalEngine,
}

impl RetrievalBackend {
    pub fn new() -> Result<Self, String> {
        let executable = std::env::current_exe().map_err(|e| e.to_string())?;
        let engine = match model_directory(&executable, std::env::var_os("DRAW_SWIPE_MODEL_DIR")) {
            Some(directory) => RetrievalEngine::from_files(directory)?,
            None => RetrievalEngine::bundled()?,
        };
        Ok(Self { engine })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packaged_model_is_required_and_explicit_override_has_priority() {
        let executable = Path::new("/tmp/Preview.app/Contents/MacOS/swipe-memory");
        assert_eq!(
            model_directory(executable, None),
            Some(PathBuf::from("/tmp/Preview.app/Contents/Resources/model"))
        );
        assert_eq!(
            model_directory(executable, Some("/tmp/custom".into())),
            Some(PathBuf::from("/tmp/custom"))
        );
        assert_eq!(
            model_directory(Path::new("/tmp/target/release/swipe-learned"), None),
            None
        );
    }
}

impl Predictor for RetrievalBackend {
    fn predict(&mut self, path: &[(f64, f64)], limit: usize) -> Vec<Prediction> {
        self.engine.predict_from_path(path, limit)
    }
}
