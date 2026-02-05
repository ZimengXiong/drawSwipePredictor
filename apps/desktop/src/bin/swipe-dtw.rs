#[macroquad::main("Swipe Predictor - DTW")]
async fn main() {
    let predictor = swipe_desktop::DtwBackend::new();

    swipe_desktop::run_app(
        predictor,
        "DTW"
    ).await;
}
