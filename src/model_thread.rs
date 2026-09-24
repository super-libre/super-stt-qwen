// SPDX-License-Identifier: GPL-3.0-only
//! The one thread the model lives on.
//!
//! [`QwenAsr`] is not `Send`: its captured decode steps keep their state in
//! `Rc<RefCell<..>>`, deliberately, since that state is
//! threaded through a single sequential decode loop and an atomic refcount
//! would be paid on every step. So the model cannot sit behind a mutex in the
//! shared state the handlers hold, and it cannot be moved onto a
//! `spawn_blocking` thread either.
//!
//! It gets a thread of its own instead, and every use of it is a job sent
//! there. That also gives the serialization a mutex would: jobs run one at a
//! time, in the order they were submitted, which is what a single GPU and a
//! model with one key/value cache want anyway.
//!
//! Adapted from the Qwen TTS backend's thread of the same name.

use tokio::sync::mpsc::{UnboundedSender, unbounded_channel};

use crate::model::QwenAsr;

/// Work to run on the model thread. `None` before the first successful load,
/// which is why a job takes the slot rather than the model.
type Job = Box<dyn FnOnce(&mut Option<QwenAsr>) + Send>;

/// A handle to the model thread. `Send + Sync`, and the only way to reach the
/// model.
#[derive(Debug)]
pub struct ModelThread {
    jobs: UnboundedSender<Job>,
}

impl ModelThread {
    /// Start the thread. It owns the model slot and lives until the sender is
    /// dropped, which happens when the process is shutting down.
    ///
    /// # Panics
    /// If the operating system refuses a thread. A backend that cannot start
    /// its model thread can serve `/v1/ping` and nothing else, which is a worse
    /// failure than not starting.
    #[must_use]
    pub fn spawn() -> Self {
        let (jobs, mut rx) = unbounded_channel::<Job>();
        std::thread::Builder::new()
            .name("qwen-asr-model".to_string())
            .spawn(move || {
                let mut model: Option<QwenAsr> = None;
                // `blocking_recv` is correct here and only here: this is a
                // plain thread, not one of the runtime's, so blocking it parks
                // nothing the runtime needed.
                while let Some(job) = rx.blocking_recv() {
                    // A panic here would otherwise end the thread and leave a
                    // process that still answers `/v1/ping` while every real
                    // request fails — the one failure the daemon cannot see,
                    // because a backend that exits is restarted and one that
                    // lies is not.
                    //
                    // The model is dropped rather than kept: a decode that
                    // unwound half way through leaves the key/value cache and
                    // the captured step describing a pass that never finished,
                    // and answering the next request from that is worse than
                    // refusing it. `POST /v1/load` builds a fresh one.
                    let run = std::panic::AssertUnwindSafe(|| job(&mut model));
                    if std::panic::catch_unwind(run).is_err() {
                        log::error!("a model job panicked; dropping the model");
                        model = None;
                    }
                }
                log::debug!("the model thread is done");
            })
            .expect("spawning the model thread");
        Self { jobs }
    }

    /// Queue `job` and return without waiting for it.
    ///
    /// Returns `false` if the model thread is gone. Used for the jobs whose
    /// results reach the caller some other way: a load, which reports through
    /// `GET /v1/status`, and a streamed transcription, which writes into a
    /// response that has already been sent.
    pub fn submit(&self, job: impl FnOnce(&mut Option<QwenAsr>) + Send + 'static) -> bool {
        self.jobs.send(Box::new(job)).is_ok()
    }

    /// Queue `job` and wait for what it returns.
    ///
    /// `None` if the model thread is gone or the job panicked — neither is
    /// recoverable by the caller, and both are reported as a failure.
    pub async fn run<T: Send + 'static>(
        &self,
        job: impl FnOnce(&mut Option<QwenAsr>) -> T + Send + 'static,
    ) -> Option<T> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.jobs
            .send(Box::new(move |model| {
                // The receiver is gone when the request was abandoned while
                // queued. The job still ran; there is just nobody to tell.
                let _ = tx.send(job(model));
            }))
            .ok()?;
        rx.await.ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A panicking job must not take the thread with it. Losing the thread
    /// would leave a process that still answers `/v1/ping` while every real
    /// request fails.
    ///
    /// The panic prints its message to stderr while this runs. That is the
    /// panic hook doing its job, not a failure.
    #[tokio::test]
    async fn a_panicking_job_does_not_take_the_thread_with_it() {
        let thread = ModelThread::spawn();
        assert!(thread.submit(|_| panic!("a job went wrong")));
        // Answering at all is the assertion: the thread had to survive the
        // panic to run this.
        assert_eq!(thread.run(|slot| slot.is_none()).await, Some(true));
    }

    /// Jobs run in the order they were submitted — the serialization the model
    /// needs, since one key/value cache cannot serve two transcriptions at
    /// once.
    #[tokio::test]
    async fn jobs_run_in_order() {
        use std::sync::{Arc, Mutex};

        let thread = ModelThread::spawn();
        let seen = Arc::new(Mutex::new(Vec::new()));
        for i in 0..8 {
            let seen = Arc::clone(&seen);
            assert!(thread.submit(move |_| seen.lock().unwrap().push(i)));
        }
        thread.run(|_| ()).await.expect("the thread answers");
        assert_eq!(*seen.lock().unwrap(), (0..8).collect::<Vec<i32>>());
    }
}
