use axum::{
    extract::{Multipart, State},
    http::StatusCode,
    response::Html,
    routing::{get, post},
    Json, Router,
};
use serde::Serialize;
use std::sync::{Arc, Mutex};
use tower_http::cors::CorsLayer;

use crate::infer::{try_load_model, predict_with_model, InferBackend};
use crate::model::BrainTumorCNN;

struct AppState {
    model: Mutex<BrainTumorCNN<InferBackend>>,
}

#[derive(Serialize)]
struct PredictResponse {
    label: String,
    prob_no: f32,
    prob_yes: f32,
    is_tumor: bool,
}

pub fn run_server() {
    println!("Loading model...");
    let model = match try_load_model() {
        Ok(m) => {
            println!("Model loaded successfully.");
            m
        }
        Err(e) => {
            eprintln!("Error: {e}");
            eprintln!("Train the model first:  cargo run --release -- train");
            std::process::exit(1);
        }
    };

    let state = Arc::new(AppState {
        model: Mutex::new(model),
    });

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async move {
        let app = Router::new()
            .route("/", get(index_handler))
            .route("/predict", post(predict_handler))
            .layer(CorsLayer::permissive())
            .with_state(state);

        let port = std::env::var("PORT").unwrap_or_else(|_| "3000".to_string());
        let addr = format!("0.0.0.0:{port}");

        println!("Brain Tumor Detection Server");
        println!("Open: http://localhost:{port}");

        let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
        axum::serve(listener, app).await.unwrap();
    });
}

async fn index_handler() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

async fn predict_handler(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<Json<PredictResponse>, (StatusCode, String)> {
    
    let mut image_bytes = None;
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?
    {
        if field.name() == Some("image") {
            image_bytes = Some(
                field
                    .bytes()
                    .await
                    .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?,
            );
            break;
        }
    }

    let bytes = image_bytes
        .ok_or_else(|| (StatusCode::BAD_REQUEST, "No 'image' field in form data".to_string()))?;

   
    let temp_path = std::env::temp_dir().join(format!(
        "brain_mri_{}.jpg",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));

    std::fs::write(&temp_path, &bytes)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let path_str = temp_path.to_string_lossy().to_string();

    
    let (label, prob_no, prob_yes) = tokio::task::spawn_blocking(move || {
        let model = state.model.lock().unwrap();
        predict_with_model(&path_str, &model)
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let _ = std::fs::remove_file(&temp_path);

    Ok(Json(PredictResponse {
        is_tumor: label.contains("TUMOR DETECTED"),
        label,
        prob_no,
        prob_yes,
    }))
}
