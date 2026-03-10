use burn::prelude::*;
use burn::data::dataloader::batcher::Batcher;
use image::GenericImageView;
use std::path::{Path, PathBuf};

pub const IMG_SIZE: usize = 128;
pub const NUM_CHANNELS: usize = 3;

#[derive(Clone, Debug)]
pub struct BrainTumorItem {
    pub image_path: PathBuf,
    pub label: u8, // 0 = no tumor, 1 = tumor
}

#[derive(Clone, Debug)]
pub struct BrainTumorDataset {
    pub items: Vec<BrainTumorItem>,
}

impl BrainTumorDataset {
    pub fn from_directory(base_dir: &str) -> Self {
        let mut items = Vec::new();

        // Load "no" images (label = 0)
        let no_dir = Path::new(base_dir).join("no");
        if no_dir.exists() {
            if let Ok(entries) = std::fs::read_dir(&no_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if is_image_file(&path) {
                        items.push(BrainTumorItem {
                            image_path: path,
                            label: 0,
                        });
                    }
                }
            }
        }

        // Load "yes" images (label = 1)
        let yes_dir = Path::new(base_dir).join("yes");
        if yes_dir.exists() {
            if let Ok(entries) = std::fs::read_dir(&yes_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if is_image_file(&path) {
                        items.push(BrainTumorItem {
                            image_path: path,
                            label: 1,
                        });
                    }
                }
            }
        }

        println!("Loaded {} images from {}", items.len(), base_dir);
        Self { items }
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }
}

fn is_image_file(path: &Path) -> bool {
    match path.extension().and_then(|e| e.to_str()) {
        Some(ext) => matches!(ext.to_lowercase().as_str(), "jpg" | "jpeg" | "png" | "bmp"),
        None => false,
    }
}

/// Load and preprocess a single image into a [C, H, W] float array
pub fn load_image(path: &Path) -> Option<Vec<f32>> {
    let img = image::open(path).ok()?;
    let img = img.resize_exact(
        IMG_SIZE as u32,
        IMG_SIZE as u32,
        image::imageops::FilterType::Triangle,
    );

    let mut pixels = vec![0.0f32; NUM_CHANNELS * IMG_SIZE * IMG_SIZE];

    for (x, y, pixel) in img.pixels() {
        let idx_base = (y as usize) * IMG_SIZE + (x as usize);
        // Normalize to [0, 1]
        pixels[0 * IMG_SIZE * IMG_SIZE + idx_base] = pixel[0] as f32 / 255.0; // R
        pixels[1 * IMG_SIZE * IMG_SIZE + idx_base] = pixel[1] as f32 / 255.0; // G
        pixels[2 * IMG_SIZE * IMG_SIZE + idx_base] = pixel[2] as f32 / 255.0; // B
    }

    Some(pixels)
}

/// Same as `load_image` but applies random horizontal flip and brightness jitter for training.
pub fn load_image_augmented(path: &Path) -> Option<Vec<f32>> {
    use rand::Rng;
    let img = image::open(path).ok()?;
    let img = img.resize_exact(
        IMG_SIZE as u32,
        IMG_SIZE as u32,
        image::imageops::FilterType::Triangle,
    );

    let mut rng = rand::thread_rng();

    // Random horizontal flip (50%)
    let img = if rng.gen_bool(0.5) {
        image::DynamicImage::from(image::imageops::flip_horizontal(&img))
    } else {
        img
    };

    // Slight brightness variation (±15%)
    let brightness: f32 = rng.gen_range(0.85..1.15);

    let mut pixels = vec![0.0f32; NUM_CHANNELS * IMG_SIZE * IMG_SIZE];
    for (x, y, pixel) in img.pixels() {
        let idx_base = (y as usize) * IMG_SIZE + (x as usize);
        pixels[0 * IMG_SIZE * IMG_SIZE + idx_base] = (pixel[0] as f32 / 255.0 * brightness).clamp(0.0, 1.0);
        pixels[1 * IMG_SIZE * IMG_SIZE + idx_base] = (pixel[1] as f32 / 255.0 * brightness).clamp(0.0, 1.0);
        pixels[2 * IMG_SIZE * IMG_SIZE + idx_base] = (pixel[2] as f32 / 255.0 * brightness).clamp(0.0, 1.0);
    }

    Some(pixels)
}

#[derive(Clone, Debug)]
pub struct BrainTumorBatcher<B: Backend> {
    _backend: std::marker::PhantomData<B>,
    pub augment: bool,
}

impl<B: Backend> BrainTumorBatcher<B> {
    pub fn new() -> Self {
        Self {
            _backend: std::marker::PhantomData,
            augment: false,
        }
    }

    pub fn new_augmented() -> Self {
        Self {
            _backend: std::marker::PhantomData,
            augment: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BrainTumorBatch<B: Backend> {
    pub images: Tensor<B, 4>,  // [batch, channels, height, width]
    pub labels: Tensor<B, 1, Int>, // [batch]
}

impl<B: Backend> Batcher<BrainTumorItem, BrainTumorBatch<B>> for BrainTumorBatcher<B> {
    fn batch(&self, items: Vec<BrainTumorItem>) -> BrainTumorBatch<B> {
        let batch_size = items.len();

        let mut all_images = Vec::with_capacity(batch_size);
        let mut all_labels = Vec::with_capacity(batch_size);

        for item in &items {
            let pixels = if self.augment {
                load_image_augmented(&item.image_path)
            } else {
                load_image(&item.image_path)
            };
            match pixels {
                Some(pixels) => {
                    let tensor = Tensor::<B, 1>::from_floats(
                        pixels.as_slice(),
                        &Default::default(),
                    )
                    .reshape([1, NUM_CHANNELS, IMG_SIZE, IMG_SIZE]);
                    all_images.push(tensor);
                    all_labels.push(item.label as i64);
                }
                None => {
                    eprintln!("Warning: Failed to load image {:?}", item.image_path);
                }
            }
        }

        let images = Tensor::cat(all_images, 0);
        let labels = Tensor::<B, 1, Int>::from_ints(
            all_labels.as_slice(),
            &Default::default(),
        );

        BrainTumorBatch { images, labels }
    }
}
