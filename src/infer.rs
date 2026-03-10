use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::data::dataloader::batcher::Batcher;

use crate::data::{
    load_image, BrainTumorBatcher, BrainTumorBatch, BrainTumorDataset,
    IMG_SIZE, NUM_CHANNELS,
};
use crate::model::BrainTumorCNN;

pub type InferBackend = burn::backend::ndarray::NdArray<f32>;
type MyBackend = InferBackend;

pub fn load_model() -> BrainTumorCNN<MyBackend> {
    let device = Default::default();
    let model: BrainTumorCNN<MyBackend> = BrainTumorCNN::new(&device);
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model
        .load_file("brain_tumor_model", &recorder, &device)
        .expect("Failed to load model. Have you trained first? Run: brain_tumor_cnn train")
}

pub fn run_evaluation() {
    let model = load_model();
    let dataset = BrainTumorDataset::from_directory("test");
    println!("Test samples: {}", dataset.len());

    let batcher = BrainTumorBatcher::<MyBackend>::new();
    let batch: BrainTumorBatch<MyBackend> = batcher.batch(dataset.items.clone());

    let logits = model.forward(batch.images);
    let predictions = logits.clone().argmax(1).squeeze::<1>(1);
    let labels = batch.labels;

    let total = dataset.len();
    let correct: usize = predictions
        .clone()
        .equal(labels.clone())
        .int()
        .sum()
        .into_scalar() as usize;

    // Compute per-class metrics
    let pred_data: Vec<i64> = (0..total)
        .map(|i| {
            predictions
                .clone()
                .slice([i..i + 1])
                .into_scalar()
        })
        .collect();
    let label_data: Vec<i64> = (0..total)
        .map(|i| {
            labels
                .clone()
                .slice([i..i + 1])
                .into_scalar()
        })
        .collect();

    let mut tp = 0; // True positives (tumor correctly detected)
    let mut fp = 0; // False positives
    let mut tn = 0; // True negatives
    let mut r#fn = 0; // False negatives

    for i in 0..total {
        match (pred_data[i], label_data[i]) {
            (1, 1) => tp += 1,
            (1, 0) => fp += 1,
            (0, 0) => tn += 1,
            (0, 1) => r#fn += 1,
            _ => {}
        }
    }

    let precision = if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        0.0
    };
    let recall = if tp + r#fn > 0 {
        tp as f64 / (tp + r#fn) as f64
    } else {
        0.0
    };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    println!("\n--- Test Results ---");
    println!("Accuracy:  {:.1}% ({}/{})", correct as f64 / total as f64 * 100.0, correct, total);
    println!("Precision: {:.3}", precision);
    println!("Recall:    {:.3}", recall);
    println!("F1 Score:  {:.3}", f1);
    println!("\nConfusion Matrix:");
    println!("              Pred No  Pred Yes");
    println!("  Actual No    {:>4}     {:>4}", tn, fp);
    println!("  Actual Yes   {:>4}     {:>4}", r#fn, tp);
}

pub fn predict_single(image_path: &str) {
    let model = load_model();
    let (label, prob_no, prob_yes) = predict_with_model(image_path, &model);
    println!("\n--- Prediction ---");
    println!("Image: {}", image_path);
    println!("Result: {}", label);
    println!("Confidence: No Tumor: {:.1}% | Tumor: {:.1}%", prob_no * 100.0, prob_yes * 100.0);
}

/// Load model without panicking — returns Err if the model file is missing.
pub fn try_load_model() -> Result<BrainTumorCNN<MyBackend>, String> {
    if !std::path::Path::new("brain_tumor_model.mpk").exists() {
        return Err("brain_tumor_model.mpk not found.\nRun:  cargo run --release -- train".to_string());
    }
    Ok(load_model())
}

/// Run inference on a single image using an already-loaded model.
/// Returns (label, prob_no_tumor, prob_tumor).
pub fn predict_with_model(image_path: &str, model: &BrainTumorCNN<MyBackend>) -> (String, f32, f32) {
    let device: <MyBackend as Backend>::Device = Default::default();
    let path = std::path::Path::new(image_path);

    let pixels = match load_image(path) {
        Some(p) => p,
        None => return ("ERROR: Could not load image".to_string(), 0.0, 0.0),
    };

    let tensor = Tensor::<MyBackend, 1>::from_floats(pixels.as_slice(), &device)
        .reshape([1, NUM_CHANNELS, IMG_SIZE, IMG_SIZE]);

    let logits = model.forward(tensor);
    let probs = burn::tensor::activation::softmax(logits, 1);
    let pred = probs.clone().argmax(1).squeeze::<1>(1).into_scalar();

    let prob_no: f32 = probs.clone().slice([0..1, 0..1]).into_scalar();
    let prob_yes: f32 = probs.slice([0..1, 1..2]).into_scalar();

    let label = if pred == 1 { "TUMOR DETECTED".to_string() } else { "NO TUMOR".to_string() };
    (label, prob_no, prob_yes)
}
