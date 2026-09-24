// SPDX-License-Identifier: GPL-3.0-only
//! The binary as the daemon runs it: spawned with its environment, serving
//! `/v1` on the Unix socket it was handed, and gone on `SIGTERM`.
//!
//! Needs no weights. Everything checked here is answered before a model is
//! loaded — which is exactly the part of the contract the daemon relies on to
//! tell a backend that is starting from one that is broken.

use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// A running backend, killed if a test fails before stopping it.
struct Backend {
    child: Child,
    socket: PathBuf,
    _dir: TempDir,
}

impl Drop for Backend {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// A directory removed when dropped. Short, under `/tmp`: a socket path is
/// limited to 108 bytes.
struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        // Unique per test: they run in parallel in one process, and every
        // `RandomState` is seeded afresh.
        let unique = format!("qasr-{}-{:016x}", std::process::id(), rand_seed());
        let dir = Path::new("/tmp").join(unique);
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn rand_seed() -> u64 {
    use std::hash::{BuildHasher, RandomState};
    RandomState::new().hash_one(0u64)
}

fn spawn() -> Backend {
    let dir = TempDir::new();
    let socket = dir.0.join("b.sock");
    let child = Command::new(env!("CARGO_BIN_EXE_super-stt-backend-qwen"))
        .env("SUPER_STT_BACKEND_SOCKET", &socket)
        .env("SUPER_STT_BACKEND_DIR", dir.0.join("backend"))
        .env("SUPER_STT_BACKEND_CACHE_DIR", dir.0.join("cache"))
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawning the backend");
    let deadline = Instant::now() + Duration::from_secs(10);
    while !socket.exists() {
        assert!(Instant::now() < deadline, "the socket never appeared");
        std::thread::sleep(Duration::from_millis(20));
    }
    Backend {
        child,
        socket,
        _dir: dir,
    }
}

/// One HTTP/1.1 exchange over the socket: the status code and the body.
fn request(socket: &Path, method: &str, path: &str, body: Option<&str>) -> (u16, String) {
    let mut stream = UnixStream::connect(socket).expect("connecting to the socket");
    let body = body.unwrap_or("");
    write!(
        stream,
        "{method} {path} HTTP/1.1\r\nHost: backend\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).unwrap();
    let status = response
        .split(' ')
        .nth(1)
        .and_then(|s| s.parse().ok())
        .expect("a status line");
    let body = response
        .split_once("\r\n\r\n")
        .map(|(_, b)| b.to_string())
        .unwrap_or_default();
    (status, body)
}

fn json(body: &str) -> serde_json::Value {
    serde_json::from_str(body).unwrap_or_else(|e| panic!("{e}: {body:?}"))
}

#[test]
fn serves_the_contract_before_any_load() {
    let backend = spawn();
    let socket = &backend.socket;

    let (status, body) = request(socket, "GET", "/v1/ping", None);
    assert_eq!(status, 200);
    assert_eq!(json(&body)["message"], "pong");

    let (status, body) = request(socket, "GET", "/v1/status", None);
    assert_eq!(status, 200);
    assert_eq!(json(&body)["state"], "starting");

    let (status, body) = request(
        socket,
        "POST",
        "/v1/load",
        Some(r#"{"name":"whisper-tiny"}"#),
    );
    assert_eq!(status, 400);
    assert_eq!(json(&body)["message"], "invalid_model");

    let (status, body) = request(
        socket,
        "POST",
        "/v1/transcribe",
        Some(r#"{"audio_data":[0.1,0.2],"sample_rate":16000}"#),
    );
    assert_eq!(status, 409);
    assert_eq!(json(&body)["message"], "not_ready");

    let (status, body) = request(socket, "POST", "/v1/cancel", None);
    assert_eq!(status, 409);
    assert_eq!(json(&body)["message"], "nothing_in_progress");
}

/// A declared model whose files were never downloaded fails its load, and
/// says so in the contract's words rather than staying `loading`.
#[test]
fn a_load_without_weights_reports_load_failed() {
    let backend = spawn();
    let socket = &backend.socket;
    let (status, _) = request(
        socket,
        "POST",
        "/v1/load",
        Some(r#"{"name":"qwen3-asr-0.6b"}"#),
    );
    assert_eq!(status, 202);
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let (_, body) = request(socket, "GET", "/v1/status", None);
        let body = json(&body);
        if body["state"] == "error" {
            assert_eq!(body["reason"], "load_failed");
            break;
        }
        assert!(Instant::now() < deadline, "still {body} after 30 s");
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// The daemon stops a backend with `SIGTERM`: it exits promptly and takes its
/// socket with it.
#[test]
fn exits_on_sigterm_and_removes_its_socket() {
    let mut backend = spawn();
    let status = Command::new("kill")
        .args(["-TERM", &backend.child.id().to_string()])
        .status()
        .expect("running kill");
    assert!(status.success());
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(exit) = backend.child.try_wait().unwrap() {
            assert!(exit.success(), "exited with {exit}");
            break;
        }
        assert!(
            Instant::now() < deadline,
            "still running 10 s after SIGTERM"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(!backend.socket.exists(), "the socket was left behind");
}
