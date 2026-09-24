// SPDX-License-Identifier: GPL-3.0-only
//! Qwen3-ASR speech-to-text backend for Super STT.
//!
//! A subprocess backend: the daemon spawns this binary inside a hardened
//! `systemd-run --user` transient unit and drives the `/v1` contract over a
//! Unix socket. Environment variables are the whole interface —
//! `SUPER_STT_BACKEND_SOCKET` names the socket to bind, `SUPER_STT_BACKEND_DIR`
//! the directory holding `backend.toml` and the downloaded weights, and
//! `SUPER_STT_BACKEND_CACHE_DIR`, when the daemon grants one, the one writable
//! place anything may be kept between runs.
//!
//! The sandbox shapes the design more than anything else: there is no network
//! (`PrivateNetwork=yes`) and the backend directory is mounted read-only, so
//! this process never downloads or writes a model file. The daemon fetches
//! everything named in `[[models.files]]` before the first `POST /v1/load`.

mod audio;
mod lang;
#[cfg(test)]
mod manifest_probe;
mod model;
mod model_thread;
mod progress;
mod prompt;
mod qwen3;
mod server;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use hyper_util::rt::TokioIo;
use hyper_util::service::TowerToHyperService;
use tokio::net::UnixListener;

/// Names the Unix socket to bind.
const ENV_SOCKET: &str = "SUPER_STT_BACKEND_SOCKET";
/// Names the backend's own directory.
const ENV_DIR: &str = "SUPER_STT_BACKEND_DIR";
/// Names the one writable directory the sandbox grants for keeping things
/// across runs — the STT analogue of Super TTS's `SUPER_TTS_BACKEND_CACHE_DIR`,
/// spelled the same way by the Voxtral backend. The daemon does not grant it
/// yet; see [`cache_dir`] for what happens until it does.
const ENV_CACHE_DIR: &str = "SUPER_STT_BACKEND_CACHE_DIR";

/// Where the compiled kernels go.
///
/// The granted directory when there is one. Otherwise a directory under the
/// unit's private `/tmp`, which is writable but dies with the process, so the
/// kernels are compiled and tuned again on every load, and every load reports
/// itself as an initial setup.
fn cache_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os(ENV_CACHE_DIR) {
        log::info!("keeping compiled kernels in {}", Path::new(&dir).display());
        return PathBuf::from(dir);
    }
    let dir = std::env::temp_dir().join("qwen3-asr-cache");
    log::warn!(
        "{ENV_CACHE_DIR} is not set, so compiled GPU kernels live in {} and are rebuilt on \
         every start; a daemon that grants a cache directory makes them persist",
        dir.display()
    );
    dir
}

fn main() -> Result<()> {
    // CubeCL's ROCm compiler goes through pliron, which logs its whole IR
    // after every pass at `info`: a cold ROCm load under the daemon's
    // `RUST_LOG=info` wrote 10 GB of log. `RUST_LOG` is parsed after this
    // default, so `pliron=info` there still brings it back.
    env_logger::Builder::new()
        .filter_module("pliron", log::LevelFilter::Warn)
        .parse_env(env_logger::Env::default().default_filter_or("info"))
        .init();

    // Before anything touches a device: the kernel cache's location is frozen
    // the first time `CubeCL` reads its configuration, and the driver's own
    // cache the first time CUDA initializes.
    let cache = cache_dir();
    if let Err(e) = std::fs::create_dir_all(&cache) {
        log::warn!("creating {}: {e}", cache.display());
    }
    // The NVIDIA driver keeps its PTX-to-SASS translations under
    // `$HOME/.nv`, read-only in the sandbox. Only when nothing chose a place:
    // the daemon's own choice wins.
    if std::env::var_os("CUDA_CACHE_PATH").is_none() {
        // Safety: no other thread exists yet — the runtime is built below, and
        // nothing before this line spawns one.
        unsafe { std::env::set_var("CUDA_CACHE_PATH", cache.join("nv")) };
    }
    model::configure_kernel_cache(&cache);

    let backend_dir =
        PathBuf::from(std::env::var(ENV_DIR).with_context(|| format!("{ENV_DIR} is not set"))?);

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("building the async runtime")?
        .block_on(serve(backend_dir))
}

async fn serve(backend_dir: PathBuf) -> Result<()> {
    let socket_path = PathBuf::from(
        std::env::var(ENV_SOCKET).with_context(|| format!("{ENV_SOCKET} is not set"))?,
    );
    // Before the socket exists: the daemon may send SIGTERM as soon as it
    // sees it, and one that lands before the handler kills the process
    // outright, leaving the socket behind.
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .context("installing the SIGTERM handler")?;
    // The socket directory is the one path the sandbox leaves writable, and a
    // stale socket from a killed unit would make `bind` fail with EADDRINUSE.
    if let Some(parent) = socket_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let _ = std::fs::remove_file(&socket_path);
    let listener = UnixListener::bind(&socket_path)
        .with_context(|| format!("binding {}", socket_path.display()))?;
    log::info!(
        "qwen3-asr backend listening on {} (dir {})",
        socket_path.display(),
        backend_dir.display()
    );

    let app = server::router(Arc::new(server::AppState::new(backend_dir)));

    loop {
        let (stream, _) = tokio::select! {
            accepted = listener.accept() => match accepted {
                Ok(pair) => pair,
                Err(e) => {
                    log::error!("accept failed: {e}");
                    continue;
                }
            },
            // The daemon stops a backend with SIGTERM: close the socket and go.
            _ = terminate.recv() => {
                log::info!("SIGTERM; shutting down");
                let _ = std::fs::remove_file(&socket_path);
                return Ok(());
            }
        };
        let service = TowerToHyperService::new(app.clone());
        tokio::spawn(async move {
            if let Err(e) = hyper::server::conn::http1::Builder::new()
                .serve_connection(TokioIo::new(stream), service)
                .await
            {
                log::debug!("connection ended: {e}");
            }
        });
    }
}
