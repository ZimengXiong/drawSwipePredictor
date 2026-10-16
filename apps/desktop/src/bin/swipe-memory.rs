#[macroquad::main("Draw from Memory")]
async fn main() {
    swipe_desktop::run_app(swipe_desktop::MemoryBackend::default(), "Memory").await;
}
