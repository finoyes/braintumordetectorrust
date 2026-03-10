use eframe::egui;
use crate::infer::{InferBackend, predict_with_model, try_load_model};
use crate::model::BrainTumorCNN;

pub struct BrainTumorApp {
    model: Result<BrainTumorCNN<InferBackend>, String>,
    texture: Option<egui::TextureHandle>,
    result_label: String,
    result_prob_no: f32,
    result_prob_yes: f32,
    has_result: bool,
    status_msg: String,
}

impl BrainTumorApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let model = try_load_model();
        let status_msg = match &model {
            Ok(_) => "Model loaded. Select an image to detect.".to_string(),
            Err(e) => format!("Model not loaded: {}", e),
        };
        Self {
            model,
            texture: None,
            result_label: String::new(),
            result_prob_no: 0.0,
            result_prob_yes: 0.0,
            has_result: false,
            status_msg,
        }
    }
}

impl eframe::App for BrainTumorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.heading("Brain Tumor Detector");
                ui.add_space(4.0);
            });

            ui.separator();

            // Model status banner
            match &self.model {
                Err(e) => {
                    ui.colored_label(
                        egui::Color32::from_rgb(220, 60, 60),
                        format!("⚠  {}", e),
                    );
                    return;
                }
                Ok(_) => {
                    ui.colored_label(egui::Color32::from_rgb(60, 180, 60), "● Model ready");
                }
            }

            ui.add_space(8.0);

            // Select image button
            if ui
                .add_sized(
                    [200.0, 36.0],
                    egui::Button::new("📂  Select Brain MRI Image"),
                )
                .clicked()
            {
                if let Some(path) = rfd::FileDialog::new()
                    .set_title("Select Brain MRI Image")
                    .add_filter("Images", &["jpg", "jpeg", "png", "bmp"])
                    .pick_file()
                {
                    let path_str = path.to_string_lossy().to_string();

                    // Load image for display
                    match load_display_image(&path_str) {
                        Some(color_image) => {
                            self.texture = Some(ctx.load_texture(
                                "brain_scan",
                                color_image,
                                egui::TextureOptions::LINEAR,
                            ));
                        }
                        None => {
                            self.status_msg =
                                "Failed to load image for display.".to_string();
                        }
                    }

                    // Run model prediction
                    if let Ok(model) = &self.model {
                        let (label, prob_no, prob_yes) =
                            predict_with_model(&path_str, model);
                        self.result_label = label;
                        self.result_prob_no = prob_no;
                        self.result_prob_yes = prob_yes;
                        self.has_result = true;
                        self.status_msg = format!(
                            "{}",
                            path.file_name()
                                .unwrap_or_default()
                                .to_string_lossy()
                        );
                    }
                }
            }

            ui.add_space(8.0);

            // Two-column layout: image on left, results on right
            ui.columns(2, |cols| {
                // Left: image preview
                cols[0].group(|ui| {
                    ui.set_min_size(egui::Vec2::new(300.0, 300.0));
                    ui.vertical_centered(|ui| {
                        if let Some(tex) = &self.texture {
                            let size = fit_size(tex.size_vec2(), egui::Vec2::new(290.0, 290.0));
                            ui.image((tex.id(), size));
                        } else {
                            ui.add_space(120.0);
                            ui.label("No image selected");
                        }
                    });
                });

                // Right: prediction results
                cols[1].group(|ui| {
                    ui.set_min_size(egui::Vec2::new(200.0, 300.0));
                    ui.heading("Result");
                    ui.add_space(8.0);

                    if self.has_result {
                        let (color, icon) = if self.result_label.contains("TUMOR") {
                            (egui::Color32::from_rgb(220, 60, 60), "🔴")
                        } else {
                            (egui::Color32::from_rgb(60, 180, 60), "🟢")
                        };

                        ui.add_space(8.0);
                        ui.colored_label(
                            color,
                            egui::RichText::new(format!("{} {}", icon, self.result_label))
                                .size(20.0)
                                .strong(),
                        );
                        ui.add_space(16.0);
                        ui.label("Confidence:");
                        ui.add_space(4.0);

                        // No tumor bar
                        ui.label(format!(
                            "No Tumor:  {:.1}%",
                            self.result_prob_no * 100.0
                        ));
                        ui.add(
                            egui::ProgressBar::new(self.result_prob_no)
                                .desired_width(f32::INFINITY),
                        );
                        ui.add_space(8.0);

                        // Tumor bar
                        ui.label(format!(
                            "Tumor:     {:.1}%",
                            self.result_prob_yes * 100.0
                        ));
                        ui.add(
                            egui::ProgressBar::new(self.result_prob_yes)
                                .desired_width(f32::INFINITY),
                        );
                    } else {
                        ui.add_space(60.0);
                        ui.label("Select an image to see\nthe prediction here.");
                    }
                });
            });

            ui.add_space(6.0);
            ui.separator();
            ui.small(&self.status_msg);
        });
    }
}

/// Fit `original` size into `max_size` while preserving aspect ratio.
fn fit_size(original: egui::Vec2, max_size: egui::Vec2) -> egui::Vec2 {
    let scale = (max_size.x / original.x).min(max_size.y / original.y);
    original * scale
}

/// Load an image file and convert it to an egui ColorImage for display.
fn load_display_image(path: &str) -> Option<egui::ColorImage> {
    let img = image::open(path).ok()?;
    let img = img.resize(512, 512, image::imageops::FilterType::Triangle);
    let size = [img.width() as usize, img.height() as usize];
    let rgba = img.to_rgba8();
    Some(egui::ColorImage::from_rgba_unmultiplied(
        size,
        rgba.as_flat_samples().as_slice(),
    ))
}

pub fn run_gui() {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([640.0, 480.0])
            .with_min_inner_size([480.0, 380.0])
            .with_title("Brain Tumor Detector"),
        ..Default::default()
    };

    eframe::run_native(
        "Brain Tumor Detector",
        options,
        Box::new(|cc| Ok(Box::new(BrainTumorApp::new(cc)))),
    )
    .expect("Failed to start GUI");
}
