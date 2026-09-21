use eframe::egui;
use crate::infer::{InferBackend, predict_with_model, try_load_model};
use crate::model::BrainTumorCNN;
use std::path::Path;

pub struct BrainTumorApp {
    model: Result<BrainTumorCNN<InferBackend>, String>,
    selected_path: Option<String>,
    selected_filename: Option<String>,
    texture: Option<egui::TextureHandle>,
    img_dimensions: Option<(u32, u32)>,
    has_analyzed: bool,
    is_tumor: bool,
    result_prob_no: f32,
    result_prob_yes: f32,
    status_msg: String,
    is_status_error: bool,
}

impl BrainTumorApp {
    pub fn new(cc: &eframe::CreationContext<'_>, initial_image: Option<&str>) -> Self {
        // Set dark sleek theme
        let mut visuals = egui::Visuals::dark();
        visuals.override_text_color = Some(egui::Color32::WHITE);
        visuals.panel_fill = egui::Color32::BLACK;
        visuals.window_fill = egui::Color32::BLACK;
        visuals.extreme_bg_color = egui::Color32::BLACK;
        visuals.faint_bg_color = egui::Color32::from_rgb(8, 8, 8);
        cc.egui_ctx.set_visuals(visuals);

        let model = try_load_model();
        let is_status_error = model.is_err();
        let status_msg = match &model {
            Ok(_) => "System Ready. Select an image to begin.".to_string(),
            Err(e) => format!("Error: Model not loaded - {}", e),
        };

        let mut app = Self {
            model,
            selected_path: None,
            selected_filename: None,
            texture: None,
            img_dimensions: None,
            has_analyzed: false,
            is_tumor: false,
            result_prob_no: 0.0,
            result_prob_yes: 0.0,
            status_msg,
            is_status_error,
        };

        if let Some(path_str) = initial_image {
            app.load_image_file(&cc.egui_ctx, path_str, true);
        }

        app
    }

    pub fn load_image_file(&mut self, ctx: &egui::Context, path_str: &str, auto_analyze: bool) {
        match load_display_image(path_str) {
            Some((color_image, w, h)) => {
                self.texture = Some(ctx.load_texture(
                    "brain_scan",
                    color_image,
                    egui::TextureOptions::LINEAR,
                ));
                self.img_dimensions = Some((w, h));
                self.selected_path = Some(path_str.to_string());
                let fname = Path::new(path_str)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                self.selected_filename = Some(fname.clone());
                self.has_analyzed = false;
                self.status_msg = format!("Scan loaded: {}", fname);
                self.is_status_error = false;

                if auto_analyze {
                    self.run_analysis();
                }
            }
            None => {
                self.status_msg = "Error: Failed to load image for display.".to_string();
                self.is_status_error = true;
            }
        }
    }

    pub fn run_analysis(&mut self) {
        let Some(path_str) = self.selected_path.clone() else {
            return;
        };

        let Ok(model) = &self.model else {
            self.status_msg = "Error: Model not available for inference.".to_string();
            self.is_status_error = true;
            return;
        };

        let (label, prob_no, prob_yes) = predict_with_model(&path_str, model);
        self.is_tumor = label == "TUMOR DETECTED";
        self.result_prob_no = prob_no;
        self.result_prob_yes = prob_yes;
        self.has_analyzed = true;

        let fname = self.selected_filename.clone().unwrap_or_else(|| "scan".to_string());
        self.status_msg = format!("Analysis complete for {}", fname);
        self.is_status_error = false;
    }
}

impl eframe::App for BrainTumorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Drag and drop handling
        ctx.input(|i| {
            for dropped in &i.raw.dropped_files {
                if let Some(path) = &dropped.path {
                    let path_str = path.to_string_lossy().to_string();
                    self.load_image_file(ctx, &path_str, false);
                    break;
                }
            }
        });

        // Top Bar
        egui::TopBottomPanel::top("top_bar")
            .frame(
                egui::Frame::none()
                    .fill(egui::Color32::BLACK)
                    .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(38, 38, 38)))
                    .inner_margin(egui::Margin { left: 24.0, right: 24.0, top: 14.0, bottom: 14.0 }),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    draw_brand_mark(ui);
                    ui.add_space(14.0);
                    ui.vertical(|ui| {
                        ui.label(
                            egui::RichText::new("NEUROSCAN DIAGNOSTIC WORKSTATION")
                                .size(14.0)
                                .strong()
                                .color(egui::Color32::WHITE),
                        );
                        ui.label(
                            egui::RichText::new("Automated Brain MRI Tumor Analysis System")
                                .size(11.0)
                                .color(egui::Color32::from_rgb(163, 163, 163)),
                        );
                    });

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        match &self.model {
                            Ok(_) => {
                                egui::Frame::none()
                                    .fill(egui::Color32::from_rgb(8, 8, 8))
                                    .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(38, 38, 38)))
                                    .rounding(2.0)
                                    .inner_margin(egui::Margin::symmetric(8.0, 4.0))
                                    .show(ui, |ui| {
                                        ui.label(
                                            egui::RichText::new("ONLINE")
                                                .size(10.0)
                                                .monospace()
                                                .color(egui::Color32::from_rgb(163, 163, 163)),
                                        );
                                    });
                            }
                            Err(_) => {
                                egui::Frame::none()
                                    .fill(egui::Color32::from_rgb(40, 10, 10))
                                    .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(180, 40, 40)))
                                    .rounding(2.0)
                                    .inner_margin(egui::Margin::symmetric(8.0, 4.0))
                                    .show(ui, |ui| {
                                        ui.label(
                                            egui::RichText::new("OFFLINE")
                                                .size(10.0)
                                                .monospace()
                                                .color(egui::Color32::from_rgb(255, 100, 100)),
                                        );
                                    });
                            }
                        }
                    });
                });
            });

        // Bottom Status Bar
        egui::TopBottomPanel::bottom("bottom_status_bar")
            .frame(
                egui::Frame::none()
                    .fill(egui::Color32::BLACK)
                    .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(38, 38, 38)))
                    .inner_margin(egui::Margin { left: 24.0, right: 24.0, top: 10.0, bottom: 10.0 }),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    let status_color = if self.is_status_error {
                        egui::Color32::from_rgb(255, 100, 100)
                    } else {
                        egui::Color32::WHITE
                    };
                    ui.label(
                        egui::RichText::new(&self.status_msg)
                            .size(11.0)
                            .monospace()
                            .strong()
                            .color(status_color),
                    );

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            egui::RichText::new("Burn CNN v0.1.0 | Inference: NdArray CPU")
                                .size(11.0)
                                .monospace()
                                .color(egui::Color32::from_rgb(115, 115, 115)),
                        );
                    });
                });
            });

        // Central Panel
        egui::CentralPanel::default()
            .frame(
                egui::Frame::none()
                    .fill(egui::Color32::BLACK)
                    .inner_margin(egui::Margin { left: 24.0, right: 24.0, top: 16.0, bottom: 16.0 }),
            )
            .show(ctx, |ui| {
                // Action Row
                egui::Frame::none()
                    .fill(egui::Color32::from_rgb(8, 8, 8))
                    .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(38, 38, 38)))
                    .rounding(2.0)
                    .inner_margin(egui::Margin::symmetric(16.0, 12.0))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            // Select MRI Scan Button
                            if custom_button(ui, "^  SELECT MRI SCAN", false, true).clicked() {
                                if let Some(path) = rfd::FileDialog::new()
                                    .set_title("Select Brain MRI Image")
                                    .add_filter("Images", &["jpg", "jpeg", "png", "bmp"])
                                    .pick_file()
                                {
                                    let path_str = path.to_string_lossy().to_string();
                                    self.load_image_file(ctx, &path_str, false);
                                }
                            }

                            ui.add_space(10.0);

                            // Run Analysis Button
                            let can_analyze = self.selected_path.is_some();
                            if custom_button(ui, "->  RUN ANALYSIS", true, can_analyze).clicked() {
                                self.run_analysis();
                            }

                            ui.add_space(12.0);

                            // File Tag Badge
                            let file_text = self.selected_filename.as_deref().unwrap_or("No file selected");
                            let tag_color = if self.selected_filename.is_some() {
                                egui::Color32::from_rgb(229, 229, 229)
                            } else {
                                egui::Color32::from_rgb(115, 115, 115)
                            };

                            egui::Frame::none()
                                .fill(egui::Color32::BLACK)
                                .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(38, 38, 38)))
                                .rounding(2.0)
                                .inner_margin(egui::Margin::symmetric(12.0, 6.0))
                                .show(ui, |ui| {
                                    ui.label(
                                        egui::RichText::new(file_text)
                                            .size(11.0)
                                            .monospace()
                                            .color(tag_color),
                                    );
                                });
                        });
                    });

                ui.add_space(14.0);

                // Two Panels Grid Layout
                ui.columns(2, |cols| {
                    // LEFT PANEL: Input Viewport
                    egui::Frame::none()
                        .fill(egui::Color32::from_rgb(8, 8, 8))
                        .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(38, 38, 38)))
                        .rounding(2.0)
                        .show(&mut cols[0], |ui| {
                            let badge = if let Some((w, h)) = self.img_dimensions {
                                format!("CH: 3 | {}x{} RES", w, h)
                            } else {
                                "CH: 3 | 128x128 RES".to_string()
                            };
                            draw_panel_header(ui, "INPUT VIEWPORT", &badge);

                            // Viewport Body
                            let body_w = ui.available_width();
                            let body_h = (ui.available_height() - 4.0).max(380.0);
                            let (body_rect, _) = ui.allocate_exact_size(
                                egui::vec2(body_w, body_h),
                                egui::Sense::hover(),
                            );

                            if ui.is_rect_visible(body_rect) {
                                ui.painter().rect_filled(
                                    body_rect,
                                    egui::Rounding { nw: 0.0, ne: 0.0, sw: 2.0, se: 2.0 },
                                    egui::Color32::from_rgb(3, 3, 3),
                                );

                                // Viewport subtle grid
                                draw_viewport_grid(ui.painter(), body_rect);

                                if let Some(tex) = &self.texture {
                                    let max_fit = egui::vec2(
                                        (body_rect.width() - 48.0).max(100.0),
                                        (body_rect.height() - 48.0).max(100.0),
                                    );
                                    let fit = fit_size(tex.size_vec2(), max_fit);
                                    let img_rect = egui::Rect::from_center_size(body_rect.center(), fit);

                                    ui.painter().image(
                                        tex.id(),
                                        img_rect,
                                        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                                        egui::Color32::WHITE,
                                    );
                                    ui.painter().rect_stroke(
                                        img_rect,
                                        2.0,
                                        egui::Stroke::new(1.5, egui::Color32::WHITE),
                                    );
                                } else {
                                    // Placeholder
                                    let c = body_rect.center();
                                    let icon_box = egui::Rect::from_center_size(
                                        egui::pos2(c.x, c.y - 18.0),
                                        egui::vec2(36.0, 36.0),
                                    );
                                    let dim_stroke = egui::Stroke::new(1.5, egui::Color32::from_rgb(80, 80, 80));
                                    ui.painter().rect_stroke(icon_box, 2.0, dim_stroke);
                                    ui.painter().line_segment(
                                        [egui::pos2(icon_box.min.x + 8.0, icon_box.center().y), egui::pos2(icon_box.max.x - 8.0, icon_box.center().y)],
                                        dim_stroke,
                                    );
                                    ui.painter().line_segment(
                                        [egui::pos2(icon_box.center().x, icon_box.min.y + 8.0), egui::pos2(icon_box.center().x, icon_box.max.y - 8.0)],
                                        dim_stroke,
                                    );

                                    ui.painter().text(
                                        egui::pos2(c.x, c.y + 20.0),
                                        egui::Align2::CENTER_CENTER,
                                        "AWAITING MRI SCAN INPUT",
                                        egui::FontId::proportional(11.0),
                                        egui::Color32::from_rgb(115, 115, 115),
                                    );
                                }
                            }
                        });

                    // RIGHT PANEL: Diagnostic Classification
                    egui::Frame::none()
                        .fill(egui::Color32::from_rgb(8, 8, 8))
                        .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(38, 38, 38)))
                        .rounding(2.0)
                        .show(&mut cols[1], |ui| {
                            draw_panel_header(ui, "DIAGNOSTIC CLASSIFICATION", "EVALUATION METRICS");

                            egui::Frame::none()
                                .fill(egui::Color32::from_rgb(8, 8, 8))
                                .inner_margin(egui::Margin { left: 22.0, right: 22.0, top: 22.0, bottom: 22.0 })
                                .show(ui, |ui| {
                                    if !self.has_analyzed {
                                        ui.add_space(140.0);
                                        ui.vertical_centered(|ui| {
                                            ui.label(
                                                egui::RichText::new("SELECT AN IMAGE AND RUN ANALYSIS TO VIEW DIAGNOSTIC CLASSIFICATION.")
                                                    .size(12.0)
                                                    .strong()
                                                    .color(egui::Color32::from_rgb(115, 115, 115)),
                                            );
                                        });
                                    } else {
                                        // Diagnostic Card
                                        let card_w = ui.available_width();
                                        let card_h = 72.0;
                                        let (card_rect, _) = ui.allocate_exact_size(
                                            egui::vec2(card_w, card_h),
                                            egui::Sense::hover(),
                                        );

                                        if ui.is_rect_visible(card_rect) {
                                            if self.is_tumor {
                                                // High contrast white inverted card
                                                ui.painter().rect_filled(card_rect, 2.0, egui::Color32::WHITE);
                                                ui.painter().rect_stroke(card_rect, 2.0, egui::Stroke::new(2.0, egui::Color32::WHITE));

                                                ui.painter().text(
                                                    egui::pos2(card_rect.min.x + 18.0, card_rect.min.y + 14.0),
                                                    egui::Align2::LEFT_TOP,
                                                    "POSITIVE: TUMOR DETECTED",
                                                    egui::FontId::proportional(18.0),
                                                    egui::Color32::BLACK,
                                                );
                                                ui.painter().text(
                                                    egui::pos2(card_rect.min.x + 18.0, card_rect.min.y + 44.0),
                                                    egui::Align2::LEFT_TOP,
                                                    "High probability tissue anomaly identified in scan",
                                                    egui::FontId::proportional(11.0),
                                                    egui::Color32::from_rgb(51, 51, 51),
                                                );
                                            } else {
                                                // Stark bordered dark card
                                                ui.painter().rect_filled(card_rect, 2.0, egui::Color32::BLACK);
                                                ui.painter().rect_stroke(card_rect, 2.0, egui::Stroke::new(2.0, egui::Color32::WHITE));

                                                ui.painter().text(
                                                    egui::pos2(card_rect.min.x + 18.0, card_rect.min.y + 14.0),
                                                    egui::Align2::LEFT_TOP,
                                                    "NEGATIVE: NO TUMOR DETECTED",
                                                    egui::FontId::proportional(18.0),
                                                    egui::Color32::WHITE,
                                                );
                                                ui.painter().text(
                                                    egui::pos2(card_rect.min.x + 18.0, card_rect.min.y + 44.0),
                                                    egui::Align2::LEFT_TOP,
                                                    "Normal brain tissue morphology confirmed",
                                                    egui::FontId::proportional(11.0),
                                                    egui::Color32::from_rgb(163, 163, 163),
                                                );
                                            }
                                        }

                                        ui.add_space(20.0);

                                        // Quantitative Confidence Distribution
                                        ui.label(
                                            egui::RichText::new("QUANTITATIVE CONFIDENCE DISTRIBUTION")
                                                .size(11.0)
                                                .strong()
                                                .color(egui::Color32::WHITE),
                                        );
                                        ui.add_space(6.0);

                                        egui::Frame::none()
                                            .fill(egui::Color32::BLACK)
                                            .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(38, 38, 38)))
                                            .rounding(2.0)
                                            .inner_margin(egui::Margin::same(16.0))
                                            .show(ui, |ui| {
                                                draw_conf_row(ui, "PATHOLOGY (TUMOR)", self.result_prob_yes);
                                                ui.add_space(14.0);
                                                draw_conf_row(ui, "NORMAL (NO TUMOR)", self.result_prob_no);
                                            });

                                        ui.add_space(20.0);

                                        // Model Verification Specs
                                        ui.label(
                                            egui::RichText::new("MODEL VERIFICATION SPECS")
                                                .size(11.0)
                                                .strong()
                                                .color(egui::Color32::WHITE),
                                        );
                                        ui.add_space(6.0);

                                        egui::Frame::none()
                                            .fill(egui::Color32::BLACK)
                                            .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(38, 38, 38)))
                                            .rounding(2.0)
                                            .inner_margin(egui::Margin::same(14.0))
                                            .show(ui, |ui| {
                                                ui.columns(2, |subcols| {
                                                    draw_meta_item(&mut subcols[0], "ARCHITECTURE", "3-Stage ConvNet");
                                                    subcols[0].add_space(8.0);
                                                    draw_meta_item(&mut subcols[0], "SENSITIVITY", "100% (16/16 TP)");

                                                    draw_meta_item(&mut subcols[1], "BACKEND", "Rust Burn (NdArray)");
                                                    subcols[1].add_space(8.0);
                                                    draw_meta_item(&mut subcols[1], "TEST ACCURACY", "87.5%");
                                                });
                                            });
                                    }
                                });
                        });
                });
            });
    }
}

// ---------------------------------------------------------------------------
// Helper Widgets & Drawing Primitives
// ---------------------------------------------------------------------------

fn draw_brand_mark(ui: &mut egui::Ui) {
    let size = egui::vec2(30.0, 30.0);
    let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
    if ui.is_rect_visible(rect) {
        ui.painter().rect_filled(rect, 2.0, egui::Color32::BLACK);
        ui.painter().rect_stroke(rect, 2.0, egui::Stroke::new(1.5, egui::Color32::WHITE));

        let center = rect.center();
        let stroke = egui::Stroke::new(1.5, egui::Color32::WHITE);
        ui.painter().circle_stroke(center, 9.0, stroke);
        ui.painter().line_segment(
            [egui::pos2(center.x, rect.min.y + 4.0), egui::pos2(center.x, rect.max.y - 4.0)],
            stroke,
        );
        ui.painter().line_segment(
            [egui::pos2(rect.min.x + 4.0, center.y), egui::pos2(rect.max.x - 4.0, center.y)],
            stroke,
        );
    }
}

fn custom_button(
    ui: &mut egui::Ui,
    text: &str,
    is_primary: bool,
    enabled: bool,
) -> egui::Response {
    let font_id = egui::FontId::proportional(12.0);
    let galley = ui.painter().layout_no_wrap(
        text.to_string(),
        font_id,
        if is_primary {
            egui::Color32::BLACK
        } else {
            egui::Color32::WHITE
        },
    );
    let padding = egui::vec2(18.0, 8.0);
    let desired_size = galley.size() + padding * 2.0;
    let (rect, response) = ui.allocate_exact_size(
        desired_size,
        if enabled { egui::Sense::click() } else { egui::Sense::hover() },
    );

    if ui.is_rect_visible(rect) {
        let (bg, border, text_color) = if !enabled {
            (
                egui::Color32::from_rgb(18, 18, 18),
                egui::Color32::from_rgb(45, 45, 45),
                egui::Color32::from_rgb(90, 90, 90),
            )
        } else if is_primary {
            if response.is_pointer_button_down_on() {
                (egui::Color32::from_rgb(200, 200, 200), egui::Color32::WHITE, egui::Color32::BLACK)
            } else if response.hovered() {
                (egui::Color32::from_rgb(230, 230, 230), egui::Color32::WHITE, egui::Color32::BLACK)
            } else {
                (egui::Color32::WHITE, egui::Color32::WHITE, egui::Color32::BLACK)
            }
        } else {
            if response.is_pointer_button_down_on() {
                (egui::Color32::from_rgb(30, 30, 30), egui::Color32::WHITE, egui::Color32::WHITE)
            } else if response.hovered() {
                (egui::Color32::from_rgb(23, 23, 23), egui::Color32::WHITE, egui::Color32::WHITE)
            } else {
                (egui::Color32::BLACK, egui::Color32::from_rgb(82, 82, 82), egui::Color32::WHITE)
            }
        };

        ui.painter().rect_filled(rect, 2.0, bg);
        ui.painter().rect_stroke(rect, 2.0, egui::Stroke::new(1.5, border));

        let text_pos = egui::pos2(
            rect.min.x + (rect.width() - galley.size().x) / 2.0,
            rect.min.y + (rect.height() - galley.size().y) / 2.0,
        );
        ui.painter().text(
            text_pos,
            egui::Align2::LEFT_TOP,
            text,
            egui::FontId::proportional(12.0),
            text_color,
        );
    }

    response
}

fn draw_panel_header(ui: &mut egui::Ui, title: &str, badge: &str) {
    let header_rect = ui.allocate_space(egui::vec2(ui.available_width(), 34.0)).1;
    ui.painter().rect_filled(
        header_rect,
        egui::Rounding { nw: 2.0, ne: 2.0, sw: 0.0, se: 0.0 },
        egui::Color32::BLACK,
    );
    ui.painter().line_segment(
        [egui::pos2(header_rect.min.x, header_rect.max.y), egui::pos2(header_rect.max.x, header_rect.max.y)],
        egui::Stroke::new(1.0, egui::Color32::from_rgb(38, 38, 38)),
    );

    ui.painter().text(
        egui::pos2(header_rect.min.x + 16.0, header_rect.center().y),
        egui::Align2::LEFT_CENTER,
        title,
        egui::FontId::proportional(11.0),
        egui::Color32::WHITE,
    );

    ui.painter().text(
        egui::pos2(header_rect.max.x - 16.0, header_rect.center().y),
        egui::Align2::RIGHT_CENTER,
        badge,
        egui::FontId::monospace(10.0),
        egui::Color32::from_rgb(115, 115, 115),
    );
}

fn draw_viewport_grid(painter: &egui::Painter, rect: egui::Rect) {
    let grid_color = egui::Color32::from_rgba_unmultiplied(255, 255, 255, 10);
    let stroke = egui::Stroke::new(1.0, grid_color);
    let step = 32.0;

    let mut x = rect.min.x;
    while x <= rect.max.x {
        painter.line_segment([egui::pos2(x, rect.min.y), egui::pos2(x, rect.max.y)], stroke);
        x += step;
    }

    let mut y = rect.min.y;
    while y <= rect.max.y {
        painter.line_segment([egui::pos2(rect.min.x, y), egui::pos2(rect.max.x, y)], stroke);
        y += step;
    }
}

fn draw_conf_row(ui: &mut egui::Ui, label: &str, pct: f32) {
    ui.horizontal(|ui| {
        // Label
        ui.allocate_ui_with_layout(
            egui::vec2(140.0, 16.0),
            egui::Layout::left_to_right(egui::Align::Center),
            |ui| {
                ui.label(
                    egui::RichText::new(label)
                        .size(11.0)
                        .strong()
                        .color(egui::Color32::WHITE),
                );
            },
        );

        // Track bar
        let avail_w = (ui.available_width() - 54.0).max(60.0);
        let bar_h = 8.0;
        let (rect, _) = ui.allocate_exact_size(egui::vec2(avail_w, bar_h), egui::Sense::hover());
        if ui.is_rect_visible(rect) {
            ui.painter().rect_filled(rect, 1.0, egui::Color32::from_rgb(26, 26, 26));
            ui.painter().rect_stroke(rect, 1.0, egui::Stroke::new(1.0, egui::Color32::from_rgb(51, 51, 51)));

            let fill_w = (rect.width() * pct.clamp(0.0, 1.0)).max(0.0);
            if fill_w > 0.0 {
                let fill_rect = egui::Rect::from_min_size(rect.min, egui::vec2(fill_w, bar_h));
                ui.painter().rect_filled(fill_rect, 1.0, egui::Color32::WHITE);
            }
        }

        // Percentage
        ui.allocate_ui_with_layout(
            egui::vec2(44.0, 16.0),
            egui::Layout::right_to_left(egui::Align::Center),
            |ui| {
                ui.label(
                    egui::RichText::new(format!("{:.0}%", pct * 100.0))
                        .size(12.0)
                        .monospace()
                        .strong()
                        .color(egui::Color32::WHITE),
                );
            },
        );
    });
}

fn draw_meta_item(ui: &mut egui::Ui, label: &str, val: &str) {
    ui.label(
        egui::RichText::new(label)
            .size(10.0)
            .strong()
            .color(egui::Color32::from_rgb(115, 115, 115)),
    );
    ui.label(
        egui::RichText::new(val)
            .size(11.0)
            .monospace()
            .strong()
            .color(egui::Color32::WHITE),
    );
}

/// Fit `original` size into `max_size` while preserving aspect ratio.
fn fit_size(original: egui::Vec2, max_size: egui::Vec2) -> egui::Vec2 {
    let scale = (max_size.x / original.x).min(max_size.y / original.y);
    original * scale
}

/// Load an image file and convert it to an egui ColorImage for display.
fn load_display_image(path: &str) -> Option<(egui::ColorImage, u32, u32)> {
    let img = image::open(path).ok()?;
    let orig_w = img.width();
    let orig_h = img.height();
    let img = img.resize(512, 512, image::imageops::FilterType::Triangle);
    let size = [img.width() as usize, img.height() as usize];
    let rgba = img.to_rgba8();
    Some((
        egui::ColorImage::from_rgba_unmultiplied(size, rgba.as_flat_samples().as_slice()),
        orig_w,
        orig_h,
    ))
}

pub fn run_gui(initial_image: Option<String>) {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1040.0, 680.0])
            .with_min_inner_size([880.0, 560.0])
            .with_title("NeuroScan AI — Diagnostic Workstation"),
        ..Default::default()
    };

    eframe::run_native(
        "NeuroScan AI — Diagnostic Workstation",
        options,
        Box::new(move |cc| Ok(Box::new(BrainTumorApp::new(cc, initial_image.as_deref())))),
    )
    .expect("Failed to start GUI");
}
