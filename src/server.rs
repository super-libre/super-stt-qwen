// SPDX-License-Identifier: GPL-3.0-only
//! The `/v1` contract, served over the Unix socket the daemon hands us.
//!
//! The routes and every status code are the ones the Python backend served,
//! and its test suite is ported below alongside the new ones. Two behaviours
//! are new. `POST /v1/cancel` now stops a transcription in flight within a
//! token, where the Python backend could only acknowledge it and let the call
//! run to the end. And a streamed transcription sends `preview` frames as the
//! text is decoded, where it used to send nothing until `done`.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, PoisonError, RwLock};

use axum::Json;
use axum::body::Body;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::lang::{self, Language};
use crate::model::{self, LoadError, Outcome, QwenAsr};
use crate::model_thread::ModelThread;
use crate::progress::Report;

/// Where the model is in its lifecycle, as `GET /v1/status` reports it.
#[derive(Debug, Clone, PartialEq)]
enum LoadState {
    /// Serving, but no model has been asked for yet.
    Starting,
    /// A `POST /v1/load` is in flight, and how far it has got.
    Loading { model: String, report: Report },
    /// Ready to transcribe.
    Ready { model: String, device: &'static str },
    /// The load failed, for the contract's `reason`.
    Failed { model: String, reason: &'static str },
}

/// Everything the handlers share.
pub struct AppState {
    backend_dir: PathBuf,
    /// The model, which lives on a thread of its own because it is not `Send`.
    /// Every use of it is a job sent there, and jobs run one at a time.
    model: ModelThread,
    state: RwLock<LoadState>,
    /// A transcription is running on the model thread.
    busy: AtomicBool,
    /// Set by `POST /v1/cancel`, cleared at the start of each transcription.
    cancelled: AtomicBool,
}

impl AppState {
    /// State for a backend rooted at `backend_dir`.
    #[must_use]
    pub fn new(backend_dir: PathBuf) -> Self {
        Self {
            backend_dir,
            model: ModelThread::spawn(),
            state: RwLock::new(LoadState::Starting),
            busy: AtomicBool::new(false),
            cancelled: AtomicBool::new(false),
        }
    }

    fn set_state(&self, next: LoadState) {
        // A poisoned lock means a handler panicked mid-update. The state is a
        // plain enum with no invariant to repair, so recovering is correct.
        *self.state.write().unwrap_or_else(PoisonError::into_inner) = next;
    }

    fn state(&self) -> LoadState {
        self.state
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Record how far a load has got, unless it has already ended.
    fn set_report(&self, value: Report) {
        let mut state = self.state.write().unwrap_or_else(PoisonError::into_inner);
        if let LoadState::Loading { report, .. } = &mut *state {
            *report = value;
        }
    }
}

/// Build the router.
pub fn router(state: Arc<AppState>) -> axum::Router {
    axum::Router::new()
        .route("/v1/ping", get(ping))
        .route("/v1/status", get(status))
        .route("/v1/load", post(load))
        .route("/v1/transcribe", post(transcribe))
        .route("/v1/cancel", post(cancel))
        // Audio arrives as a JSON array of floats: a minute of it is over ten
        // megabytes of text, and the daemon caps what it records, not this.
        .layer(DefaultBodyLimit::disable())
        .with_state(state)
}

async fn ping() -> Json<Value> {
    Json(json!({ "status": "success", "message": "pong" }))
}

/// `GET /v1/status`. `status` is the envelope and is `success` whenever the
/// report itself is valid, the load having failed included; `state` is the
/// readiness.
async fn status(State(s): State<Arc<AppState>>) -> Json<Value> {
    let mut body = json!({ "status": "success", "reason": null });
    match s.state() {
        LoadState::Starting => body["state"] = json!("starting"),
        LoadState::Loading { model, report } => {
            body["state"] = json!("loading");
            body["model"] = json!({ "name": model });
            // Each only while loading, and only once the load has said: an
            // absent `progress` reads as indeterminate.
            if let Some(phase) = report.phase {
                body["phase"] = json!(phase.as_str());
            }
            if let Some(step) = report.step {
                body["step"] = json!(step.as_str());
            }
            if let Some(progress) = report.progress {
                body["progress"] = json!(progress);
            }
        }
        LoadState::Ready { model, device } => {
            body["state"] = json!("ready");
            body["model"] = json!({ "name": model });
            body["device"] = json!(device);
        }
        LoadState::Failed { model, reason } => {
            body["state"] = json!("error");
            body["model"] = json!({ "name": model });
            body["reason"] = json!(reason);
        }
    }
    Json(body)
}

/// `POST /v1/load` body. `provider` is not read: it is a compatibility echo
/// from an older identity scheme, and the contract says a new backend should
/// not validate it.
#[derive(Debug, Deserialize)]
struct LoadRequest {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    device: Option<String>,
}

/// `POST /v1/load` — answer `202` immediately and load in the background.
///
/// The daemon polls `GET /v1/status` for readiness, so returning before the
/// model is resident is the contract rather than a shortcut: mapping gigabytes
/// of weights and warming the kernels takes long enough to time out a request.
async fn load(State(s): State<Arc<AppState>>, body: Option<Json<LoadRequest>>) -> Response {
    let name = body
        .and_then(|Json(req)| req.name.map(|n| (n, req.device)))
        .filter(|(n, _)| model::MODELS.contains(&n.as_str()));
    let Some((name, device)) = name else {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_model",
            "not a model this backend serves",
        );
    };
    {
        let mut state = s.state.write().unwrap_or_else(PoisonError::into_inner);
        if matches!(*state, LoadState::Loading { .. }) {
            return json_error(
                StatusCode::CONFLICT,
                "already_loading",
                "a load is already in progress",
            );
        }
        *state = LoadState::Loading {
            model: name.clone(),
            report: Report::default(),
        };
    }

    let state = Arc::clone(&s);
    let queued = s.model.submit(move |slot| {
        let report = |r: Report| state.set_report(r);
        // A panic here — an allocation the device refused while the weights
        // were mapped, or a lost device while the old model's memory was
        // freed, say — would otherwise be caught by the model thread and
        // leave the status saying `loading` for good.
        let loaded = model::catching(|| {
            // The previous model's memory goes first: two resident at once
            // is what a GPU sized for one of them cannot hold.
            *slot = None;
            QwenAsr::load(&state.backend_dir, &name, device.as_deref(), &report)
        })
        .unwrap_or_else(|panic| Err(LoadError::Failed(anyhow::anyhow!("panicked: {panic}"))));
        match loaded {
            Ok(model) => {
                let device = model.device_name();
                *slot = Some(model);
                log::info!("{name} is ready on {device}");
                state.set_state(LoadState::Ready {
                    model: name,
                    device,
                });
            }
            Err(e) => {
                log::error!("loading {name} failed: {e}");
                state.set_state(LoadState::Failed {
                    model: name,
                    reason: e.reason(),
                });
            }
        }
    });
    if !queued {
        s.set_state(LoadState::Failed {
            model: String::new(),
            reason: "load_failed",
        });
        return json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "load_failed",
            "the model thread is gone",
        );
    }

    (
        StatusCode::ACCEPTED,
        Json(json!({ "status": "success", "message": "Loading started" })),
    )
        .into_response()
}

/// `POST /v1/cancel`: acknowledged with `200` while a transcription runs, which
/// then stops at its next token, and `409` when there is nothing to cancel.
async fn cancel(State(s): State<Arc<AppState>>) -> Response {
    if !s.busy.load(Ordering::SeqCst) {
        return json_error(
            StatusCode::CONFLICT,
            "nothing_in_progress",
            "no transcription is running",
        );
    }
    s.cancelled.store(true, Ordering::SeqCst);
    (
        StatusCode::OK,
        Json(json!({ "status": "success", "message": "Cancelled" })),
    )
        .into_response()
}

#[derive(Debug, Default, Deserialize)]
struct TranscribeOptions {
    #[serde(default)]
    stream_realtime: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct TranscribeRequest {
    #[serde(default)]
    audio_data: Option<Vec<f32>>,
    #[serde(default)]
    sample_rate: Option<u32>,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    options: Option<TranscribeOptions>,
}

/// `POST /v1/transcribe`.
///
/// Validated before the status line is sent, in the Python backend's order: a
/// model that is not ready, then missing audio, then a language the model does
/// not transcribe — so each is the `4xx` the contract names rather than an
/// error frame inside a stream that already claimed success.
async fn transcribe(
    State(s): State<Arc<AppState>>,
    body: Result<Json<TranscribeRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    if !matches!(s.state(), LoadState::Ready { .. }) {
        return json_error(StatusCode::CONFLICT, "not_ready", "no model is loaded");
    }
    let req = match body {
        Ok(Json(req)) => req,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, "invalid_audio", &e.body_text()),
    };
    let Some(audio) = req.audio_data.filter(|a| !a.is_empty()) else {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_audio",
            "audio_data is missing or empty",
        );
    };
    let sample_rate = req.sample_rate.unwrap_or(crate::qwen3::audio::SAMPLE_RATE);
    if sample_rate == 0 {
        return json_error(StatusCode::BAD_REQUEST, "invalid_audio", "sample_rate is 0");
    }
    let Some(language) = lang::resolve(req.language.as_deref()) else {
        return json_error(
            StatusCode::BAD_REQUEST,
            "unsupported_language",
            "this model does not transcribe that language",
        );
    };
    let stream = req
        .options
        .unwrap_or_default()
        .stream_realtime
        .unwrap_or(false);

    if stream {
        transcribe_streaming(&s, audio, sample_rate, language)
    } else {
        transcribe_oneshot(&s, audio, sample_rate, language).await
    }
}

/// Clears the busy flag on every way out of a transcription, a panic included.
struct Idle<'a>(&'a AtomicBool);

impl Drop for Idle<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

/// Runs on the model thread: one transcription, with the busy and cancel flags
/// kept around it.
fn run_transcription(
    slot: &mut Option<QwenAsr>,
    state: &AppState,
    audio: Vec<f32>,
    sample_rate: u32,
    language: Language,
    mut on_preview: impl FnMut(&str) -> bool,
) -> anyhow::Result<Outcome> {
    let Some(model) = slot.as_mut() else {
        anyhow::bail!("no model is loaded");
    };
    state.cancelled.store(false, Ordering::SeqCst);
    state.busy.store(true, Ordering::SeqCst);
    let _idle = Idle(&state.busy);

    let samples = crate::audio::normalize(audio, sample_rate);
    // Whether anyone still reads the previews: read by one callback and
    // written by the other.
    let listening = std::cell::Cell::new(true);
    model.transcribe(
        &samples,
        language,
        || listening.get() && !state.cancelled.load(Ordering::SeqCst),
        |text| listening.set(on_preview(text)),
    )
}

async fn transcribe_oneshot(
    s: &Arc<AppState>,
    audio: Vec<f32>,
    sample_rate: u32,
    language: Language,
) -> Response {
    let state = Arc::clone(s);
    let result = s
        .model
        .run(move |slot| run_transcription(slot, &state, audio, sample_rate, language, |_| true))
        .await;
    match result {
        // A cancelled one-shot answers with what it had transcribed, as a
        // success: the daemon's repeated passes tolerate a partial one, and the
        // Python backend answered a cancelled call with `200` too.
        Some(Ok(Outcome::Finished(text) | Outcome::Stopped(text))) => (
            StatusCode::OK,
            Json(json!({ "status": "success", "transcription": text })),
        )
            .into_response(),
        Some(Err(e)) => {
            log::error!("transcription failed: {e:#}");
            json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "inference_failed",
                &format!("{e:#}"),
            )
        }
        None => json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "inference_failed",
            "the transcription panicked or the model thread is gone",
        ),
    }
}

/// One Server-Sent Event.
fn sse(event: &str, data: &Value) -> bytes::Bytes {
    bytes::Bytes::from(format!("event: {event}\ndata: {data}\n\n"))
}

fn transcribe_streaming(
    s: &Arc<AppState>,
    audio: Vec<f32>,
    sample_rate: u32,
    language: Language,
) -> Response {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<bytes::Bytes, std::io::Error>>(16);
    let state = Arc::clone(s);
    let queued = s.model.submit(move |slot| {
        let send = |frame: bytes::Bytes| tx.blocking_send(Ok(frame)).is_ok();
        // A preview that cannot be sent means the daemon hung up, and the
        // transcription stops rather than finish for nobody.
        let result = run_transcription(slot, &state, audio, sample_rate, language, |text| {
            send(sse("preview", &json!({ "text": text })))
        });
        let _ = match result {
            Ok(Outcome::Finished(text)) => send(sse("done", &json!({ "transcription": text }))),
            // The contract ends a cancelled stream with an error frame.
            Ok(Outcome::Stopped(_)) => send(sse("error", &json!({ "message": "cancelled" }))),
            Err(e) => {
                log::error!("transcription failed: {e:#}");
                send(sse("error", &json!({ "message": "inference_failed" })))
            }
        };
    });
    if !queued {
        return json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "inference_failed",
            "the model thread is gone",
        );
    }
    // A job that panics drops the sender without a terminal frame, and the
    // stream simply ends — which the daemon reads as the failure it is.
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(
            tokio_stream::wrappers::ReceiverStream::new(rx),
        ))
        .unwrap_or_else(|e| {
            json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "inference_failed",
                &e.to_string(),
            )
        })
}

/// The JSON error envelope: `message` is the contract's code, `detail` is for
/// the person reading the log.
fn json_error(status: StatusCode, code: &str, detail: &str) -> Response {
    (
        status,
        Json(json!({ "status": "error", "message": code, "detail": detail })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::progress::{Phase, Step};
    use axum::http::Request;
    use tower::ServiceExt;

    /// A router over an empty backend directory: enough to exercise every
    /// route that does not need weights, which is every route but a successful
    /// load.
    fn app() -> (axum::Router, Arc<AppState>) {
        let state = Arc::new(AppState::new(PathBuf::from("/nonexistent")));
        (router(Arc::clone(&state)), state)
    }

    async fn call(app: axum::Router, req: Request<Body>) -> (StatusCode, Value) {
        let res = app.oneshot(req).await.expect("the router must answer");
        let status = res.status();
        let bytes = axum::body::to_bytes(res.into_body(), 1 << 20)
            .await
            .expect("a JSON body must be readable");
        let value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        (status, value)
    }

    fn get(path: &str) -> Request<Body> {
        Request::builder().uri(path).body(Body::empty()).unwrap()
    }

    fn post(path: &str, body: &Value) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri(path)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    /// Pretend a model is loaded, for the checks that run before one is used.
    fn ready(state: &AppState) {
        state.set_state(LoadState::Ready {
            model: "qwen3-asr-0.6b".to_string(),
            device: "cpu",
        });
    }

    async fn settled(app: &axum::Router) -> Value {
        for _ in 0..500 {
            let (_, body) = call(app.clone(), get("/v1/status")).await;
            if body["state"] == "ready" || body["state"] == "error" {
                return body;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        panic!("the load never settled");
    }

    #[tokio::test]
    async fn ping_answers_pong_before_any_load() {
        let (app, _) = app();
        let (status, body) = call(app, get("/v1/ping")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, json!({ "status": "success", "message": "pong" }));
    }

    #[tokio::test]
    async fn status_starts_in_starting() {
        let (app, _) = app();
        let (status, body) = call(app, get("/v1/status")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["status"], "success");
        assert_eq!(body["state"], "starting");
        assert!(body.get("model").is_none(), "no model has been asked for");
    }

    #[tokio::test]
    async fn status_reports_a_ready_model_and_its_device() {
        let (app, state) = app();
        ready(&state);
        let (_, body) = call(app, get("/v1/status")).await;
        assert_eq!(body["state"], "ready");
        assert_eq!(body["device"], "cpu");
        assert_eq!(body["model"], json!({ "name": "qwen3-asr-0.6b" }));
        assert!(body.get("progress").is_none());
    }

    #[tokio::test]
    async fn status_reports_load_progress_only_while_loading() {
        let (app, state) = app();
        state.set_state(LoadState::Loading {
            model: "qwen3-asr-1.7b".to_string(),
            report: Report::default(),
        });
        let (_, body) = call(app.clone(), get("/v1/status")).await;
        assert_eq!(body["state"], "loading");
        for field in ["phase", "step", "progress"] {
            assert!(body.get(field).is_none(), "{field} before the load said");
        }
        state.set_report(Report {
            phase: Some(Phase::InitialSetup),
            step: Some(Step::BuildingKernels),
            progress: Some(0.5),
        });
        let (_, body) = call(app.clone(), get("/v1/status")).await;
        assert_eq!(body["phase"], "initial_setup");
        assert_eq!(body["step"], "building_kernels");
        assert_eq!(body["progress"], 0.5);
        assert_eq!(body["model"]["name"], "qwen3-asr-1.7b");

        // A report that arrives after the load ended changes nothing.
        ready(&state);
        state.set_report(Report {
            phase: Some(Phase::Loading),
            ..Report::default()
        });
        let (_, body) = call(app, get("/v1/status")).await;
        assert_eq!(body["state"], "ready");
        for field in ["phase", "step", "progress"] {
            assert!(body.get(field).is_none(), "{field} once ready");
        }
    }

    #[tokio::test]
    async fn an_unknown_model_is_invalid() {
        let (app, _) = app();
        let (status, body) = call(app, post("/v1/load", &json!({ "name": "nope" }))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["message"], "invalid_model");
    }

    #[tokio::test]
    async fn a_load_without_a_name_is_invalid() {
        let (app, _) = app();
        let (status, body) = call(app, post("/v1/load", &json!({}))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["message"], "invalid_model");
    }

    /// `provider` is a compatibility echo the contract says not to validate.
    #[tokio::test]
    async fn the_provider_is_not_validated() {
        let (app, _) = app();
        let body = json!({ "name": "qwen3-asr-0.6b", "provider": "openai" });
        let (status, _) = call(app, post("/v1/load", &body)).await;
        assert_eq!(status, StatusCode::ACCEPTED);
    }

    #[tokio::test]
    async fn a_second_load_while_loading_conflicts() {
        let (app, state) = app();
        state.set_state(LoadState::Loading {
            model: "qwen3-asr-0.6b".to_string(),
            report: Report::default(),
        });
        let (status, body) =
            call(app, post("/v1/load", &json!({ "name": "qwen3-asr-0.6b" }))).await;
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(body["message"], "already_loading");
    }

    /// With no weights in the backend directory the load fails, and says so
    /// with the contract's reason rather than staying `loading` forever.
    #[tokio::test]
    async fn a_load_without_weights_ends_in_load_failed() {
        let (app, _) = app();
        let body =
            json!({ "name": "qwen3-asr-0.6b", "provider": "local_qwen3_asr", "device": "cpu" });
        let (status, _) = call(app.clone(), post("/v1/load", &body)).await;
        assert_eq!(status, StatusCode::ACCEPTED);
        let body = settled(&app).await;
        assert_eq!(body["state"], "error");
        assert_eq!(body["status"], "success");
        assert_eq!(body["reason"], "load_failed");
        assert_eq!(body["model"]["name"], "qwen3-asr-0.6b");
    }

    #[tokio::test]
    async fn transcribe_before_a_load_is_not_ready() {
        let (app, _) = app();
        let (status, body) = call(
            app,
            post("/v1/transcribe", &json!({ "audio_data": [0.1, 0.2] })),
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(body["message"], "not_ready");
    }

    #[tokio::test]
    async fn empty_or_missing_audio_is_invalid() {
        let (app, state) = app();
        ready(&state);
        for body in [
            json!({ "audio_data": [] }),
            json!({}),
            json!({ "audio_data": "x" }),
        ] {
            let (status, reply) = call(app.clone(), post("/v1/transcribe", &body)).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
            assert_eq!(reply["message"], "invalid_audio", "{body}");
        }
    }

    #[tokio::test]
    async fn an_unsupported_language_is_refused_before_any_work() {
        let (app, state) = app();
        ready(&state);
        let body = json!({ "audio_data": [0.1], "language": "xx" });
        let (status, reply) = call(app, post("/v1/transcribe", &body)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(reply["message"], "unsupported_language");
    }

    /// `auto` is not a declared code but must be accepted, never refused as
    /// unsupported. With no model actually loaded the request gets past
    /// validation and fails on the model thread — which is the assertion:
    /// it was not a `400`.
    #[tokio::test]
    async fn auto_passes_validation() {
        let (app, state) = app();
        ready(&state);
        let body = json!({ "audio_data": [0.1], "language": "auto" });
        let (status, reply) = call(app, post("/v1/transcribe", &body)).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(reply["message"], "inference_failed");
    }

    #[tokio::test]
    async fn a_streamed_failure_ends_with_an_error_frame() {
        let (app, state) = app();
        ready(&state);
        let body = json!({ "audio_data": [0.1], "options": { "stream_realtime": true } });
        let res = app.oneshot(post("/v1/transcribe", &body)).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(res.headers()[header::CONTENT_TYPE], "text/event-stream");
        let bytes = axum::body::to_bytes(res.into_body(), 1 << 20)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert_eq!(
            text,
            "event: error\ndata: {\"message\":\"inference_failed\"}\n\n"
        );
    }

    #[tokio::test]
    async fn cancel_with_nothing_running_conflicts() {
        let (app, state) = app();
        ready(&state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/cancel")
            .body(Body::empty())
            .unwrap();
        let (status, body) = call(app, req).await;
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(body["message"], "nothing_in_progress");
    }

    #[tokio::test]
    async fn cancel_while_busy_is_acknowledged_and_flagged() {
        let (app, state) = app();
        state.busy.store(true, Ordering::SeqCst);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/cancel")
            .body(Body::empty())
            .unwrap();
        let (status, body) = call(app, req).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, json!({ "status": "success", "message": "Cancelled" }));
        assert!(state.cancelled.load(Ordering::SeqCst));
    }

    #[test]
    fn sse_frames_are_event_then_data() {
        let frame = sse("done", &json!({ "transcription": "hi \"there\"" }));
        assert_eq!(
            frame,
            "event: done\ndata: {\"transcription\":\"hi \\\"there\\\"\"}\n\n"
        );
    }

    use crate::fixture::{self, Layout};

    /// A router over a backend directory holding the fixture, loaded.
    async fn loaded(backend: &fixture::Backend) -> (axum::Router, Arc<AppState>) {
        let state = Arc::new(AppState::new(backend.dir.clone()));
        let app = router(Arc::clone(&state));
        let (status, _) = call(
            app.clone(),
            post("/v1/load", &json!({ "name": fixture::MODEL })),
        )
        .await;
        assert_eq!(status, StatusCode::ACCEPTED);
        let body = settled(&app).await;
        assert_eq!(body["state"], "ready", "{body}");
        assert_eq!(body["model"]["name"], fixture::MODEL);
        (app, state)
    }

    /// `seconds` of a tone at `rate`, as a request carries it.
    fn tone(seconds: f32, rate: u32) -> Vec<f32> {
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let len = (seconds * rate as f32) as usize;
        #[allow(clippy::cast_precision_loss)]
        (0..len).map(|i| (i as f32 * 0.03).sin() * 0.3).collect()
    }

    #[tokio::test]
    async fn a_loaded_model_transcribes_whatever_the_sample_rate() {
        let backend = fixture::backend(Layout::Single);
        let (app, state) = loaded(&backend).await;
        for rate in [16_000, 48_000] {
            let body =
                json!({ "audio_data": tone(2.0, rate), "sample_rate": rate, "language": "en" });
            let (status, body) = call(app.clone(), post("/v1/transcribe", &body)).await;
            assert_eq!(status, StatusCode::OK, "{body}");
            assert_eq!(body["status"], "success");
            assert!(body["transcription"].is_string(), "{body}");
        }
        assert!(!state.busy.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn a_zero_sample_rate_is_invalid() {
        let (app, state) = app();
        ready(&state);
        let body = json!({ "audio_data": [0.1], "sample_rate": 0 });
        let (status, body) = call(app, post("/v1/transcribe", &body)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["message"], "invalid_audio");
    }

    /// The frames of a streamed transcription, read as they come; `after_first`
    /// runs once the first has arrived.
    async fn frames(
        app: axum::Router,
        body: &Value,
        after_first: impl FnOnce(),
    ) -> Vec<(String, Value)> {
        use tokio_stream::StreamExt;
        let res = app.oneshot(post("/v1/transcribe", body)).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let mut stream = res.into_body().into_data_stream();
        let mut text = String::new();
        let mut after_first = Some(after_first);
        while let Some(chunk) = stream.next().await {
            text.push_str(std::str::from_utf8(&chunk.unwrap()).unwrap());
            if let Some(f) = after_first.take() {
                f();
            }
        }
        text.split_terminator("\n\n")
            .map(|frame| {
                let (event, data) = frame.split_once('\n').unwrap();
                let event = event.strip_prefix("event: ").unwrap().to_string();
                let data = serde_json::from_str(data.strip_prefix("data: ").unwrap()).unwrap();
                (event, data)
            })
            .collect()
    }

    #[tokio::test]
    async fn a_streamed_transcription_previews_then_finishes() {
        let backend = fixture::backend(Layout::Single);
        let (app, _) = loaded(&backend).await;
        let body = json!({
            "audio_data": tone(3.0, 16_000),
            "language": "en",
            "options": { "stream_realtime": true },
        });
        let frames = frames(app, &body, || {}).await;
        let (last, previews) = frames.split_last().unwrap();
        assert_eq!(last.0, "done", "{frames:?}");
        assert!(!previews.is_empty());
        for (event, data) in previews {
            assert_eq!(event, "preview");
            assert!(data["text"].is_string());
        }
    }

    #[tokio::test]
    async fn a_cancelled_stream_ends_with_an_error_frame() {
        let backend = fixture::backend(Layout::Single);
        let (app, state) = loaded(&backend).await;
        let body = json!({
            "audio_data": tone(30.0, 16_000),
            "language": "en",
            "options": { "stream_realtime": true },
        });
        let frames = frames(app, &body, || state.cancelled.store(true, Ordering::SeqCst)).await;
        assert_eq!(
            frames.last(),
            Some(&("error".to_string(), json!({ "message": "cancelled" }))),
            "{frames:?}"
        );
    }
}
