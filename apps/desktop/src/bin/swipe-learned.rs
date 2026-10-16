#[macroquad::main("Draw from Memory")]
async fn main() {
    match swipe_desktop::RetrievalBackend::new() {
        Ok(predictor) => swipe_desktop::run_app(predictor, "Learned").await,
        Err(error) => eprintln!("Could not load the recognizer: {error}"),
    }
}
