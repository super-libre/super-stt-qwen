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

/// The argument that runs the kernel exporter instead of the server.
const EXPORT_KERNELS: &str = "export-kernels";

/// Where the compiled kernels go.
///
/// The granted directory when there is one. Otherwise a directory under the
/// unit's private `/tmp`, which is writable but dies with the process — so the
/// kernels are compiled again on every load, but the autotune bundle shipped
/// in `kernels/` still has somewhere to be imported to, and that is the slow
/// half. The Python backend kept its caches in the same place, for the same
/// reason.
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

/// Produce the kernel bundle a release ships, then exit.
///
/// The daemon never passes arguments, so this mode is unreachable from a
/// running install: it is a developer tool, kept in this binary rather than a
/// second one because a bundle is only valid for the exact `CubeCL` the
/// consuming binary links.
///
/// ```text
/// export-kernels [--warm <model>] [--everything] <out.bundle> ["Bundle name"]
/// ```
///
/// `--warm` loads the model first, which fills the cache by running the same
/// ladder a real load does. Point `SUPER_STT_BACKEND_CACHE_DIR` at an empty
/// directory so what comes out is this model's cold load and nothing else.
fn export_kernels(backend_dir: &Path, args: &[String]) -> Result<()> {
    const USAGE: &str = "usage: export-kernels [--warm <model>] [--everything] <out.bundle> [name]";

    let everything = args.iter().any(|a| a == "--everything");
    let args: Vec<String> = args
        .iter()
        .filter(|a| *a != "--everything")
        .cloned()
        .collect();
    let (warm, rest) = match args.first().map(String::as_str) {
        Some("--warm") => (Some(args.get(1).context(USAGE)?.clone()), &args[2..]),
        _ => (None, args.as_slice()),
    };
    let out = PathBuf::from(rest.first().context(USAGE)?);
    let name = rest
        .get(1)
        .cloned()
        .or_else(|| warm.clone())
        .unwrap_or_else(|| "super-stt-qwen".to_string());

    if let Some(model_name) = &warm {
        log::info!(
            "warming {model_name} to export its kernels; against an empty cache this takes minutes"
        );
        let model = model::QwenAsr::load(backend_dir, model_name, None, &|_| {})
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let device = model.device_name();
        // Dropped before the export so nothing is still writing to the cache.
        drop(model);
        log::info!("exporting the kernels {model_name} compiled on {device}");
    }
    model::export_kernel_bundle(&out, &name, everything)
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

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
    // Before any device exists, and once per process: nothing in the cache is
    // keyed by model.
    model::import_kernel_bundle(&backend_dir);

    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|a| a == EXPORT_KERNELS) {
        return export_kernels(&backend_dir, &args[1..]);
    }

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
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .context("installing the SIGTERM handler")?;

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
