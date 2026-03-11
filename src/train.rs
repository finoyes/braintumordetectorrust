use burn::prelude::*;
use burn::backend::Autodiff;
use burn::module::AutodiffModule;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{SgdConfig, momentum::MomentumConfig, decay::WeightDecayConfig, GradientsParams, Optimizer};
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use rand::seq::SliceRandom;

use crate::data::{BrainTumorBatcher, BrainTumorDataset, BrainTumorBatch};
use crate::model::BrainTumorCNN;

use burn::data::dataloader::batcher::Batcher;

const EPOCHS: usize = 20;
const BATCH_SIZE: usize = 8;
const LEARNING_RATE: f64 = 0.001;

type InnerBackend = burn::backend::ndarray::NdArray<f32>;
type TrainBackend = Autodiff<InnerBackend>;

pub fn run_training() {
    let device = Default::default();


    let dataset = BrainTumorDataset::from_directory("train");
    println!("Training samples: {}", dataset.len());

    
    let no_count = dataset.items.iter().filter(|i| i.label == 0).count();
    let yes_count = dataset.items.iter().filter(|i| i.label == 1).count();
    let total_f = dataset.len() as f64;
    let weight_no  = total_f / (2.0 * no_count  as f64);
    let weight_yes = total_f / (2.0 * yes_count as f64);
    println!("Class counts — no: {no_count}, yes: {yes_count}  |  weights — no: {weight_no:.3}, yes: {weight_yes:.3}");

    let mut model: BrainTumorCNN<TrainBackend> = BrainTumorCNN::new(&device);

    let optimizer_config = SgdConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(1e-4)))
        .with_momentum(Some(MomentumConfig {
            momentum: 0.9,
            dampening: 0.0,
            nesterov: false,
        }));
    let mut optim = optimizer_config.init::<TrainBackend, BrainTumorCNN<TrainBackend>>();

    let batcher = BrainTumorBatcher::<TrainBackend>::new_augmented();
    let loss_fn = CrossEntropyLossConfig::new()
        .with_weights(Some(vec![weight_no as f32, weight_yes as f32]))
        .init(&device);

    println!("\n--- Starting Training ---");
    println!("Epochs: {}, Batch Size: {}, LR: {}", EPOCHS, BATCH_SIZE, LEARNING_RATE);

    for epoch in 0..EPOCHS {
        let mut epoch_loss = 0.0;
        let mut correct = 0usize;
        let mut total = 0usize;

        
        let mut items = dataset.items.clone();
        let mut rng = rand::thread_rng();
        items.shuffle(&mut rng);

        let num_batches = (items.len() + BATCH_SIZE - 1) / BATCH_SIZE;

        for batch_idx in 0..num_batches {
            let start = batch_idx * BATCH_SIZE;
            let end = (start + BATCH_SIZE).min(items.len());
            let batch_items: Vec<_> = items[start..end].to_vec();
            let batch_len = batch_items.len();

            let batch: BrainTumorBatch<TrainBackend> = batcher.batch(batch_items);

            
            let logits = model.forward(batch.images);
            let loss = loss_fn.forward(logits.clone(), batch.labels.clone());

            
            let predictions = logits.clone().argmax(1).squeeze::<1>(1);
            let correct_batch: usize = predictions
                .equal(batch.labels)
                .int()
                .sum()
                .into_scalar() as usize;
            correct += correct_batch;
            total += batch_len;

            
            let loss_val: f32 = loss.clone().into_scalar().elem();
            epoch_loss += loss_val;

            
            let grads = loss.backward();
            let grad_container = GradientsParams::from_grads(grads, &model);
            model = optim.step(LEARNING_RATE, model, grad_container);
        }

        let avg_loss = epoch_loss / num_batches as f32;
        let accuracy = correct as f64 / total as f64 * 100.0;
        println!(
            "Epoch [{:>2}/{}]  Loss: {:.4}  Accuracy: {:.1}% ({}/{})",
            epoch + 1,
            EPOCHS,
            avg_loss,
            accuracy,
            correct,
            total
        );
    }

    
    let model_valid = model.valid();
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model_valid
        .save_file("brain_tumor_model", &recorder)
        .expect("Failed to save model");
    println!("\nModel saved to brain_tumor_model.mpk");
}
