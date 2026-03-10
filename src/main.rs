mod data;
mod model;
mod train;
mod infer;
mod gui;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    match args.get(1).map(|s| s.as_str()) {
        Some("train") => train::run_training(),
        Some("test") => infer::run_evaluation(),
        Some("predict") => {
            let path = args.get(2).expect("Usage: brain_tumor_cnn predict <image_path>");
            infer::predict_single(path);
        }
        // Launch GUI when no subcommand or explicitly "gui"
        Some("gui") | None => gui::run_gui(),
        _ => {
            println!("Brain Tumor Detection CNN (Rust + Burn)");
            println!("Usage:");
            println!("  brain_tumor_cnn              - Launch GUI");
            println!("  brain_tumor_cnn gui           - Launch GUI");
            println!("  brain_tumor_cnn train         - Train the model");
            println!("  brain_tumor_cnn test          - Evaluate on test set");
            println!("  brain_tumor_cnn predict <img> - Predict single image");
        }
    }
}
