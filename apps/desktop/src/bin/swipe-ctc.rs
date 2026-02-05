#[macroquad::main("Swipe Predictor - CTC")]
async fn main() {
    let predictor = swipe_desktop::CtcBackend::new()
        .expect("Failed to load CTC model. Set SWIPE_CTC_MODEL and SWIPE_CTC_CHARS.");

    swipe_desktop::run_app(
        predictor,
        "CTC"
    ).await;
}
