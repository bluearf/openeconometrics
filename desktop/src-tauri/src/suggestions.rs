//! Optional, local-only code completion. The native broker owns the model cache
//! and worker; the webview cannot choose a URL, executable, model, or credential.
use cap_fs_ext::{FollowSymlinks, OpenOptionsFollowExt};
use cap_std::fs::{Dir, OpenOptions};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::VecDeque,
    io::{Read, Write},
    net::TcpListener,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicU64, Ordering},
        Mutex,
    },
    time::{Duration, Instant},
};
use tokio::io::AsyncWriteExt;
use tokio_util::sync::CancellationToken;

pub const MODEL_ID: &str = "qwen2.5-coder-0.5b-q8_0";
pub const MODEL_BYTES: u64 = 531_068_128;
pub const MODEL_SHA256: &str = "d0f8cd6c49bab52a0abdbe47948518b1f4d9b1a8a2a6825099cea31cea10ac56";
pub const MODEL_URL: &str = "https://huggingface.co/ggml-org/Qwen2.5-Coder-0.5B-Q8_0-GGUF/resolve/5788aee90e725673490be18f553d234feaa8303d/qwen2.5-coder-0.5b-q8_0.gguf";
const MODEL_FILE: &str = "qwen2.5-coder-0.5b-q8_0.gguf";
const CACHE_DIR: &str = ".local-suggestions";
const SETTINGS_FILE: &str = ".local-suggestions.json";
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
const ENGINE_MANIFEST: &str = include_str!("../../suggestions/manifest.json");
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
const ENGINE_MANIFEST: &str = "{}";
const MAX_PREFIX: usize = 8192;
const MAX_SUFFIX: usize = 2048;
const MAX_RESPONSE: usize = 64 * 1024;
const MAX_OUTPUT: usize = 2048;
const REQUEST_TIMEOUT: Duration = Duration::from_millis(2500);

#[derive(Clone, Copy, Serialize, PartialEq, Eq, Debug)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Off,
    NotInstalled,
    Installing,
    Starting,
    Ready,
    Error,
}

#[derive(Serialize, Clone, Debug)]
pub struct Status {
    pub enabled: bool,
    pub installed: bool,
    pub state: Phase,
    pub model: &'static str,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_code: Option<&'static str>,
}

#[derive(Deserialize, Serialize, Default)]
#[serde(deny_unknown_fields)]
struct Settings {
    enabled: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompletionRequest {
    pub request_id: String,
    pub prefix: String,
    pub suffix: String,
    pub language: String,
}

#[derive(Serialize)]
pub struct CompletionResponse {
    pub request_id: String,
    pub text: String,
}

struct Worker {
    child: Child,
    origin: String,
    token: String,
}

#[derive(Clone, Serialize)]
struct StartTimings {
    model_hash_seconds: f64,
    engine_hash_seconds: f64,
    spawn_seconds: f64,
    health_seconds: f64,
}
impl Worker {
    fn stop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        self.stop();
    }
}

struct Inner {
    enabled: bool,
    installed: bool,
    checked: bool,
    generation: u64,
    phase: Phase,
    error: Option<&'static str>,
    worker: Option<Worker>,
    active: Option<(String, CancellationToken)>,
    cancelled_ids: VecDeque<String>,
    download: Option<CancellationToken>,
    startup: Option<CancellationToken>,
    last_start_timings: Option<StartTimings>,
}

pub struct Manager {
    root: Dir,
    data_root: PathBuf,
    resources: PathBuf,
    inner: Mutex<Inner>,
    lifecycle: tokio::sync::Mutex<()>,
    inference: tokio::sync::Mutex<()>,
    downloaded: AtomicU64,
    local_client: reqwest::Client,
    download_client: reqwest::Client,
}

fn error(code: &'static str) -> String {
    code.into()
}
fn safe_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 96
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}
fn validate_request(request: &CompletionRequest) -> Result<(), String> {
    if !safe_id(&request.request_id)
        || request.language != "python"
        || request.prefix.is_empty()
        || request.prefix.len() > MAX_PREFIX
        || request.suffix.len() > MAX_SUFFIX
        || request.prefix.contains('\0')
        || request.suffix.contains('\0')
    {
        return Err(error("AI_INVALID_REQUEST"));
    }
    Ok(())
}
fn safe_redirect(url: &url::Url) -> bool {
    url.scheme() == "https"
        && url.username().is_empty()
        && url.password().is_none()
        && url.port_or_known_default() == Some(443)
        && url.host_str().is_some_and(|host| {
            host == "huggingface.co"
                || host.ends_with(".huggingface.co")
                || host.ends_with(".hf.co")
        })
}
fn bounded_text(text: &str) -> String {
    // A completion is plain source text, never HTML/Markdown or an execution.
    if text.contains('\0') || text.trim_start().starts_with("```") {
        return String::new();
    }
    let mut end = text.len().min(MAX_OUTPUT);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    let text = &text[..end];
    let mut lines = 0;
    let mut end = text.len();
    for (index, character) in text.char_indices() {
        if character == '\n' {
            lines += 1;
            if lines == 3 {
                end = index;
                break;
            }
        }
    }
    text[..end].to_string()
}

fn read_settings(root: &Dir) -> Result<Settings, String> {
    let mut options = OpenOptions::new();
    options.read(true).follow(FollowSymlinks::No);
    let mut file = match root.open_with(SETTINGS_FILE, &options) {
        Ok(file) => file,
        Err(failure) if failure.kind() == std::io::ErrorKind::NotFound => {
            return Ok(Settings::default())
        }
        Err(_) => return Err(error("AI_CONFIG_INVALID")),
    };
    let info = file.metadata().map_err(|_| error("AI_CONFIG_INVALID"))?;
    if !info.is_file() || info.len() > 256 {
        return Err(error("AI_CONFIG_INVALID"));
    }
    let mut bytes = Vec::new();
    Read::by_ref(&mut file)
        .take(257)
        .read_to_end(&mut bytes)
        .map_err(|_| error("AI_CONFIG_INVALID"))?;
    serde_json::from_slice(&bytes).map_err(|_| error("AI_CONFIG_INVALID"))
}

impl Manager {
    pub fn new(resources: &Path, data_root: &Path) -> Result<Self, String> {
        let root = Dir::open_ambient_dir(data_root, cap_std::ambient_authority())
            .map_err(|_| error("AI_CONFIG_INVALID"))?;
        let settings = read_settings(&root);
        let config_error = settings.as_ref().err().map(|_| "AI_CONFIG_INVALID");
        let enabled = settings.map(|settings| settings.enabled).unwrap_or(false);
        let local_client = reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_millis(500))
            .timeout(REQUEST_TIMEOUT)
            .build()
            .map_err(|_| error("AI_ENGINE_UNAVAILABLE"))?;
        let download_client = reqwest::Client::builder()
            .https_only(true)
            .no_proxy()
            .redirect(reqwest::redirect::Policy::custom(|attempt| {
                if attempt.previous().len() > 5 || !safe_redirect(attempt.url()) {
                    attempt.stop()
                } else {
                    attempt.follow()
                }
            }))
            .connect_timeout(Duration::from_secs(15))
            .timeout(Duration::from_secs(15 * 60))
            .build()
            .map_err(|_| error("AI_ENGINE_UNAVAILABLE"))?;
        Ok(Self {
            root,
            data_root: data_root.to_owned(),
            resources: resources.to_owned(),
            inner: Mutex::new(Inner {
                enabled,
                installed: false,
                checked: false,
                generation: 0,
                phase: if config_error.is_some() {
                    Phase::Error
                } else {
                    Phase::Off
                },
                error: config_error,
                worker: None,
                active: None,
                cancelled_ids: VecDeque::new(),
                download: None,
                startup: None,
                last_start_timings: None,
            }),
            lifecycle: tokio::sync::Mutex::new(()),
            inference: tokio::sync::Mutex::new(()),
            downloaded: AtomicU64::new(0),
            local_client,
            download_client,
        })
    }

    fn snapshot(&self) -> Result<Status, String> {
        let mut inner = self.inner.lock().map_err(|_| error("AI_UNAVAILABLE"))?;
        if let Some(worker) = inner.worker.as_mut() {
            if worker
                .child
                .try_wait()
                .map_err(|_| error("AI_ENGINE_UNAVAILABLE"))?
                .is_some()
            {
                inner.worker.take();
                inner.phase = Phase::Error;
                inner.error = Some("AI_ENGINE_UNAVAILABLE");
            }
        }
        Ok(Status {
            enabled: inner.enabled,
            installed: inner.installed,
            state: inner.phase,
            model: MODEL_ID,
            downloaded_bytes: if inner.installed {
                MODEL_BYTES
            } else {
                self.downloaded.load(Ordering::Relaxed)
            },
            total_bytes: MODEL_BYTES,
            error_code: inner.error,
        })
    }

    fn cache(&self, create: bool) -> Result<Dir, String> {
        match self.root.symlink_metadata(CACHE_DIR) {
            Ok(info) if info.is_dir() && !info.file_type().is_symlink() => (),
            Err(failure) if failure.kind() == std::io::ErrorKind::NotFound && create => {
                self.root
                    .create_dir(CACHE_DIR)
                    .map_err(|_| error("AI_CACHE_INVALID"))?;
            }
            _ => return Err(error("AI_CACHE_INVALID")),
        }
        let cache = self
            .root
            .open_dir(CACHE_DIR)
            .map_err(|_| error("AI_CACHE_INVALID"))?;
        #[cfg(unix)]
        {
            use cap_std::fs::PermissionsExt;
            cache
                .set_permissions(".", cap_std::fs::Permissions::from_mode(0o700))
                .map_err(|_| error("AI_CACHE_INVALID"))?;
        }
        Ok(cache)
    }

    fn verify_model(&self) -> Result<bool, String> {
        if self.root.symlink_metadata(CACHE_DIR).is_err() {
            return Ok(false);
        }
        let cache = self.cache(false)?;
        let mut options = OpenOptions::new();
        options.read(true).follow(FollowSymlinks::No);
        let mut file = match cache.open_with(MODEL_FILE, &options) {
            Ok(file) => file,
            Err(failure) if failure.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(_) => return Err(error("AI_MODEL_INVALID")),
        };
        let info = file.metadata().map_err(|_| error("AI_MODEL_INVALID"))?;
        if !info.is_file() || info.len() != MODEL_BYTES {
            return Err(error("AI_MODEL_INVALID"));
        }
        let mut digest = Sha256::new();
        let mut buffer = vec![0; 256 * 1024];
        loop {
            let count = file
                .read(&mut buffer)
                .map_err(|_| error("AI_MODEL_INVALID"))?;
            if count == 0 {
                break;
            }
            digest.update(&buffer[..count]);
        }
        if format!("{:x}", digest.finalize()) != MODEL_SHA256 {
            return Err(error("AI_MODEL_INVALID"));
        }
        Ok(true)
    }

    fn check_model(&self) -> Result<(), String> {
        if self
            .inner
            .lock()
            .map_err(|_| error("AI_UNAVAILABLE"))?
            .checked
        {
            return Ok(());
        }
        let result = self.verify_model();
        let mut inner = self.inner.lock().map_err(|_| error("AI_UNAVAILABLE"))?;
        inner.checked = true;
        match result {
            Ok(installed) => {
                inner.installed = installed;
                if inner.error.is_none() {
                    inner.phase = if inner.enabled && !installed {
                        Phase::NotInstalled
                    } else {
                        Phase::Off
                    };
                }
            }
            Err(_) => {
                inner.installed = false;
                inner.phase = Phase::Error;
                inner.error = Some("AI_MODEL_INVALID");
            }
        }
        Ok(())
    }

    fn save_enabled(&self, enabled: bool) -> Result<(), String> {
        if let Ok(info) = self.root.symlink_metadata(SETTINGS_FILE) {
            if !info.is_file() || info.file_type().is_symlink() {
                return Err(error("AI_CONFIG_INVALID"));
            }
        }
        let temporary = format!(".local-suggestions-{}.json", uuid::Uuid::new_v4());
        let mut options = OpenOptions::new();
        options
            .write(true)
            .create_new(true)
            .follow(FollowSymlinks::No);
        let result = (|| {
            let mut file = self
                .root
                .open_with(&temporary, &options)
                .map_err(|_| error("AI_CONFIG_INVALID"))?;
            #[cfg(unix)]
            {
                use cap_std::fs::PermissionsExt;
                file.set_permissions(cap_std::fs::Permissions::from_mode(0o600))
                    .map_err(|_| error("AI_CONFIG_INVALID"))?;
            }
            file.write_all(
                &serde_json::to_vec(&Settings { enabled })
                    .map_err(|_| error("AI_CONFIG_INVALID"))?,
            )
            .map_err(|_| error("AI_CONFIG_INVALID"))?;
            file.sync_all().map_err(|_| error("AI_CONFIG_INVALID"))?;
            drop(file);
            self.root
                .rename(&temporary, &self.root, SETTINGS_FILE)
                .map_err(|_| error("AI_CONFIG_INVALID"))
        })();
        if result.is_err() {
            let _ = self.root.remove_file(&temporary);
        }
        result
    }

    pub async fn status(&self) -> Result<Status, String> {
        // Status never downloads. A persisted ON setting may lazily restart only
        // its already-installed, verified worker when the editor opens.
        if let Ok(_guard) = self.lifecycle.try_lock() {
            self.check_model()?;
            let start = {
                let inner = self.inner.lock().map_err(|_| error("AI_UNAVAILABLE"))?;
                inner.enabled && inner.installed && inner.worker.is_none() && inner.error.is_none()
            };
            if start {
                let _ = self.start().await;
            }
        }
        self.snapshot()
    }

    pub async fn configure(&self, enabled: bool) -> Result<Status, String> {
        let generation = {
            let mut inner = self.inner.lock().map_err(|_| error("AI_UNAVAILABLE"))?;
            inner.enabled = enabled;
            inner.generation += 1;
            if !enabled {
                if let Some((_, token)) = inner.active.take() {
                    token.cancel();
                }
                if let Some(token) = inner.startup.take() {
                    token.cancel();
                }
                if let Some(token) = inner.download.take() {
                    token.cancel();
                }
                inner.worker.take();
                inner.phase = Phase::Off;
            }
            // Serialize the tiny private settings write with admission so an
            // earlier ON call cannot overwrite a later OFF while waiting to load.
            self.save_enabled(enabled)?;
            inner.error = None;
            inner.generation
        };
        let _guard = self.lifecycle.lock().await;
        if self
            .inner
            .lock()
            .map_err(|_| error("AI_UNAVAILABLE"))?
            .generation
            != generation
        {
            return self.snapshot();
        }
        self.check_model()?;
        if self
            .inner
            .lock()
            .map_err(|_| error("AI_UNAVAILABLE"))?
            .enabled
        {
            if !self
                .inner
                .lock()
                .map_err(|_| error("AI_UNAVAILABLE"))?
                .installed
            {
                self.inner
                    .lock()
                    .map_err(|_| error("AI_UNAVAILABLE"))?
                    .phase = Phase::NotInstalled;
            } else {
                self.start().await?;
            }
        }
        self.snapshot()
    }

    pub fn stop(&self) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.enabled = false;
            inner.generation += 1;
            if let Some((_, token)) = inner.active.take() {
                token.cancel();
            }
            if let Some(token) = inner.startup.take() {
                token.cancel();
            }
            if let Some(token) = inner.download.take() {
                token.cancel();
            }
            inner.worker.take();
            inner.phase = Phase::Off;
        }
    }

    pub fn cancel(&self, request_id: &str) -> Result<(), String> {
        if !safe_id(request_id) {
            return Err(error("AI_INVALID_REQUEST"));
        }
        let mut inner = self.inner.lock().map_err(|_| error("AI_UNAVAILABLE"))?;
        if !inner.cancelled_ids.iter().any(|id| id == request_id) {
            if inner.cancelled_ids.len() == 64 {
                inner.cancelled_ids.pop_front();
            }
            inner.cancelled_ids.push_back(request_id.into());
        }
        if let Some((id, token)) = &inner.active {
            if id == request_id {
                token.cancel();
            }
        }
        Ok(())
    }

    pub async fn install(&self) -> Result<Status, String> {
        let generation = self
            .inner
            .lock()
            .map_err(|_| error("AI_UNAVAILABLE"))?
            .generation;
        let _guard = self.lifecycle.lock().await;
        if self
            .inner
            .lock()
            .map_err(|_| error("AI_UNAVAILABLE"))?
            .generation
            != generation
        {
            return Err(error("AI_CANCELLED"));
        }
        self.check_model()?;
        if self
            .inner
            .lock()
            .map_err(|_| error("AI_UNAVAILABLE"))?
            .installed
        {
            return self.snapshot();
        }
        // Do not spend bandwidth on platforms without this packaged engine or
        // when its signed resource manifest no longer matches its actual bytes.
        self.engine_path()?;
        let cache = self.cache(true)?;
        let temporary = format!(".download-{}", uuid::Uuid::new_v4());
        let token = CancellationToken::new();
        {
            let mut inner = self.inner.lock().map_err(|_| error("AI_UNAVAILABLE"))?;
            if inner.generation != generation {
                return Err(error("AI_CANCELLED"));
            }
            inner.download = Some(token.clone());
            inner.phase = Phase::Installing;
            inner.error = None;
        }
        self.downloaded.store(0, Ordering::Relaxed);
        let result = token
            .run_until_cancelled(self.download_model(&cache, &temporary))
            .await
            .unwrap_or_else(|| Err(error("AI_CANCELLED")));
        if result.is_err() {
            let _ = cache.remove_file(&temporary);
        }
        {
            let mut inner = self.inner.lock().map_err(|_| error("AI_UNAVAILABLE"))?;
            inner.download = None;
            if result.is_ok() {
                inner.installed = true;
                inner.checked = true;
                inner.error = None;
                inner.phase = Phase::Off;
            } else if token.is_cancelled() {
                inner.phase = Phase::Off;
                self.downloaded.store(0, Ordering::Relaxed);
            } else {
                inner.phase = Phase::Error;
                inner.error = Some(
                    if result
                        .as_ref()
                        .err()
                        .is_some_and(|e| e == "AI_MODEL_INVALID")
                    {
                        "AI_MODEL_INVALID"
                    } else {
                        "AI_NETWORK"
                    },
                );
            }
        }
        result?;
        if self
            .inner
            .lock()
            .map_err(|_| error("AI_UNAVAILABLE"))?
            .enabled
        {
            self.start().await?;
        }
        self.snapshot()
    }

    async fn download_model(&self, cache: &Dir, temporary: &str) -> Result<(), String> {
        let mut response = self
            .download_client
            .get(MODEL_URL)
            .send()
            .await
            .map_err(|_| error("AI_NETWORK"))?;
        if !response.status().is_success()
            || response
                .content_length()
                .is_some_and(|length| length != MODEL_BYTES)
        {
            return Err(error("AI_NETWORK"));
        }
        let mut options = OpenOptions::new();
        options
            .write(true)
            .create_new(true)
            .follow(FollowSymlinks::No);
        let file = cache
            .open_with(temporary, &options)
            .map_err(|_| error("AI_CACHE_INVALID"))?;
        #[cfg(unix)]
        {
            use cap_std::fs::PermissionsExt;
            file.set_permissions(cap_std::fs::Permissions::from_mode(0o600))
                .map_err(|_| error("AI_CACHE_INVALID"))?;
        }
        let mut file = tokio::fs::File::from_std(file.into_std());
        let mut digest = Sha256::new();
        let mut size = 0u64;
        while let Some(chunk) = response.chunk().await.map_err(|_| error("AI_NETWORK"))? {
            size = size
                .checked_add(chunk.len() as u64)
                .ok_or_else(|| error("AI_MODEL_INVALID"))?;
            if size > MODEL_BYTES {
                return Err(error("AI_MODEL_INVALID"));
            }
            digest.update(&chunk);
            file.write_all(&chunk)
                .await
                .map_err(|_| error("AI_CACHE_INVALID"))?;
            self.downloaded.store(size, Ordering::Relaxed);
        }
        if size != MODEL_BYTES || format!("{:x}", digest.finalize()) != MODEL_SHA256 {
            return Err(error("AI_MODEL_INVALID"));
        }
        file.sync_all()
            .await
            .map_err(|_| error("AI_CACHE_INVALID"))?;
        drop(file);
        if let Ok(info) = cache.symlink_metadata(MODEL_FILE) {
            if !info.is_file() || info.file_type().is_symlink() {
                return Err(error("AI_MODEL_INVALID"));
            }
        }
        cache
            .rename(temporary, cache, MODEL_FILE)
            .map_err(|_| error("AI_CACHE_INVALID"))
    }

    fn engine_path(&self) -> Result<PathBuf, String> {
        if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            return Err(error("AI_ENGINE_UNAVAILABLE"));
        }
        let directory = self.resources.join("suggestions");
        if std::fs::symlink_metadata(&directory)
            .map_err(|_| error("AI_ENGINE_UNAVAILABLE"))?
            .file_type()
            .is_symlink()
        {
            return Err(error("AI_ENGINE_UNAVAILABLE"));
        }
        let manifest: Value =
            serde_json::from_str(ENGINE_MANIFEST).map_err(|_| error("AI_ENGINE_UNAVAILABLE"))?;
        if manifest["version"] != "b11146" || manifest["model"]["sha256"] != MODEL_SHA256 {
            return Err(error("AI_ENGINE_UNAVAILABLE"));
        }
        let files = manifest["files"]
            .as_array()
            .ok_or_else(|| error("AI_ENGINE_UNAVAILABLE"))?;
        for entry in files {
            let name = entry["name"]
                .as_str()
                .ok_or_else(|| error("AI_ENGINE_UNAVAILABLE"))?;
            if name.is_empty()
                || name.contains('/')
                || name.contains('\\')
                || matches!(name, "." | "..")
            {
                return Err(error("AI_ENGINE_UNAVAILABLE"));
            }
            let path = directory.join(name);
            let info =
                std::fs::symlink_metadata(&path).map_err(|_| error("AI_ENGINE_UNAVAILABLE"))?;
            if !info.is_file()
                || info.file_type().is_symlink()
                || Some(info.len()) != entry["size_bytes"].as_u64()
            {
                return Err(error("AI_ENGINE_UNAVAILABLE"));
            }
            let bytes = std::fs::read(path).map_err(|_| error("AI_ENGINE_UNAVAILABLE"))?;
            if Some(format!("{:x}", Sha256::digest(bytes)).as_str()) != entry["sha256"].as_str() {
                return Err(error("AI_ENGINE_UNAVAILABLE"));
            }
        }
        if !files.iter().any(|file| file["name"] == "llama-server") {
            return Err(error("AI_ENGINE_UNAVAILABLE"));
        }
        Ok(directory.join("llama-server"))
    }

    async fn start(&self) -> Result<(), String> {
        let (generation, token) = {
            let mut inner = self.inner.lock().map_err(|_| error("AI_UNAVAILABLE"))?;
            if !inner.enabled || inner.worker.is_some() {
                return Ok(());
            }
            inner.phase = Phase::Starting;
            inner.error = None;
            let token = CancellationToken::new();
            inner.startup = Some(token.clone());
            (inner.generation, token)
        };
        let result = token
            .run_until_cancelled(self.spawn_worker(generation))
            .await
            .unwrap_or_else(|| Err(error("AI_CANCELLED")));
        let mut inner = self.inner.lock().map_err(|_| error("AI_UNAVAILABLE"))?;
        inner.startup = None;
        if !inner.enabled || inner.generation != generation {
            inner.worker.take();
            inner.phase = Phase::Off;
            return Ok(());
        }
        match result {
            Ok(()) => {
                inner.phase = Phase::Ready;
                inner.error = None;
                Ok(())
            }
            Err(failure) => {
                inner.worker.take();
                inner.phase = Phase::Error;
                let code = if failure == "AI_MODEL_INVALID" {
                    inner.installed = false;
                    inner.checked = false;
                    "AI_MODEL_INVALID"
                } else {
                    "AI_ENGINE_UNAVAILABLE"
                };
                inner.error = Some(code);
                Err(error(code))
            }
        }
    }

    async fn spawn_worker(&self, generation: u64) -> Result<(), String> {
        let started = Instant::now();
        if !self.verify_model()? {
            return Err(error("AI_MODEL_INVALID"));
        }
        let model_hash_seconds = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let engine = self.engine_path()?;
        let engine_hash_seconds = started.elapsed().as_secs_f64();
        let model = self.data_root.join(CACHE_DIR).join(MODEL_FILE);
        let listener =
            TcpListener::bind(("127.0.0.1", 0)).map_err(|_| error("AI_ENGINE_UNAVAILABLE"))?;
        let port = listener
            .local_addr()
            .map_err(|_| error("AI_ENGINE_UNAVAILABLE"))?
            .port();
        drop(listener);
        let token = format!(
            "{}{}",
            uuid::Uuid::new_v4().simple(),
            uuid::Uuid::new_v4().simple()
        );
        let origin = format!("http://127.0.0.1:{port}");
        let mut command = Command::new(engine);
        command
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("LLAMA_API_KEY", &token);
        if let Some(tmp) = std::env::var_os("TMPDIR") {
            command.env("TMPDIR", tmp);
        }
        command
            .args(["-m"])
            .arg(model)
            .args([
                "--host",
                "127.0.0.1",
                "--port",
                &port.to_string(),
                "--offline",
                "--no-webui",
                "--no-agent",
                "--no-slots",
                "-c",
                "2048",
                "-np",
                "1",
                "-t",
                "4",
                "--poll",
                "0",
                "--poll-batch",
                "0",
                "--prio",
                "-1",
                "--temp",
                "0",
                "--log-verbosity",
                "0",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        let started = Instant::now();
        let child = command
            .spawn()
            .map_err(|_| error("AI_ENGINE_UNAVAILABLE"))?;
        let spawn_seconds = started.elapsed().as_secs_f64();
        let mut worker = Worker {
            child,
            origin: origin.clone(),
            token: token.clone(),
        };
        {
            let mut inner = self.inner.lock().map_err(|_| error("AI_UNAVAILABLE"))?;
            if !inner.enabled || inner.generation != generation {
                worker.stop();
                return Err(error("AI_CANCELLED"));
            }
            inner.worker = Some(worker);
        }
        let started = Instant::now();
        while started.elapsed() < Duration::from_secs(25) {
            let alive = {
                let mut inner = self.inner.lock().map_err(|_| error("AI_UNAVAILABLE"))?;
                if !inner.enabled || inner.generation != generation {
                    return Err(error("AI_CANCELLED"));
                }
                inner.worker.as_mut().is_some_and(|worker| {
                    worker.child.try_wait().is_ok_and(|result| result.is_none())
                })
            };
            if !alive {
                return Err(error("AI_ENGINE_UNAVAILABLE"));
            }
            if let Ok(response) = self
                .local_client
                .get(format!("{origin}/health"))
                .bearer_auth(&token)
                .send()
                .await
            {
                if response.status().is_success() {
                    self.inner
                        .lock()
                        .map_err(|_| error("AI_UNAVAILABLE"))?
                        .last_start_timings = Some(StartTimings {
                        model_hash_seconds,
                        engine_hash_seconds,
                        spawn_seconds,
                        health_seconds: started.elapsed().as_secs_f64(),
                    });
                    return Ok(());
                }
            }
            tokio::time::sleep(Duration::from_millis(75)).await;
        }
        Err(error("AI_ENGINE_UNAVAILABLE"))
    }

    pub async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, String> {
        validate_request(&request)?;
        let token = CancellationToken::new();
        let (origin, credential, generation) = {
            let mut inner = self.inner.lock().map_err(|_| error("AI_UNAVAILABLE"))?;
            if inner
                .cancelled_ids
                .iter()
                .any(|id| id == &request.request_id)
            {
                return Err(error("AI_CANCELLED"));
            }
            if !inner.enabled || inner.phase != Phase::Ready {
                return Err(error("AI_NOT_READY"));
            }
            let worker = inner.worker.as_ref().ok_or_else(|| error("AI_NOT_READY"))?;
            let origin = worker.origin.clone();
            let credential = worker.token.clone();
            if let Some((_, active)) = inner.active.take() {
                active.cancel();
            }
            inner.active = Some((request.request_id.clone(), token.clone()));
            (origin, credential, inner.generation)
        };
        let result = tokio::time::timeout(REQUEST_TIMEOUT, token.run_until_cancelled(async {
            let _guard = self.inference.lock().await;
            let response = self.local_client.post(format!("{origin}/infill")).bearer_auth(credential)
                .json(&json!({"input_prefix": request.prefix, "input_suffix": request.suffix,
                    "n_predict": 64, "temperature": 0, "cache_prompt": true, "stream": false,
                    "stop": ["<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>", "<|file_sep|>"]}))
                .send().await.map_err(|_| error("AI_NETWORK"))?;
            read_completion(response).await
        })).await;
        let result = match result {
            Ok(Some(result)) => result,
            Ok(None) => Err(error("AI_CANCELLED")),
            Err(_) => Err(error("AI_TIMEOUT")),
        };
        let mut inner = self.inner.lock().map_err(|_| error("AI_UNAVAILABLE"))?;
        if inner
            .active
            .as_ref()
            .is_some_and(|(id, _)| id == &request.request_id)
        {
            inner.active = None;
        }
        if token.is_cancelled() || !inner.enabled || inner.generation != generation {
            return Err(error("AI_CANCELLED"));
        }
        result.map(|text| CompletionResponse {
            request_id: request.request_id,
            text,
        })
    }
}

impl Drop for Manager {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Exercise the same native manager against an explicitly supplied, checksum-
/// verified test model. This owns a fresh cache and never opens human projects.
pub fn smoke_test(resources: &Path, model_source: &Path) -> Result<Value, String> {
    let data_root = std::env::temp_dir().join(format!(
        "openecon-suggestions-smoke-{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir(&data_root).map_err(|_| error("AI_CACHE_INVALID"))?;
    let result = (|| -> Result<Value, String> {
        let manager = std::sync::Arc::new(Manager::new(resources, &data_root)?);
        tauri::async_runtime::block_on(async {
            let initial = manager.status().await?;
            if initial.enabled
                || initial.installed
                || manager
                    .inner
                    .lock()
                    .map_err(|_| error("AI_UNAVAILABLE"))?
                    .worker
                    .is_some()
            {
                return Err(error("AI_SMOKE_FAILED"));
            }
            let cache = manager.cache(true)?;
            let destination = data_root.join(CACHE_DIR).join(MODEL_FILE);
            let info =
                std::fs::symlink_metadata(model_source).map_err(|_| error("AI_MODEL_INVALID"))?;
            if !info.is_file() || info.file_type().is_symlink() || info.len() != MODEL_BYTES {
                return Err(error("AI_MODEL_INVALID"));
            }
            std::fs::copy(model_source, &destination).map_err(|_| error("AI_CACHE_INVALID"))?;
            #[cfg(unix)]
            {
                use cap_std::fs::PermissionsExt;
                cache
                    .set_permissions(MODEL_FILE, cap_std::fs::Permissions::from_mode(0o600))
                    .map_err(|_| error("AI_CACHE_INVALID"))?;
            }
            manager
                .inner
                .lock()
                .map_err(|_| error("AI_UNAVAILABLE"))?
                .checked = false;
            let installed = manager.status().await?;
            if !installed.installed || installed.enabled {
                return Err(error("AI_SMOKE_FAILED"));
            }
            let started = Instant::now();
            let ready = manager.configure(true).await?;
            let ready_seconds = started.elapsed().as_secs_f64();
            let first_start_timings = manager
                .inner
                .lock()
                .map_err(|_| error("AI_UNAVAILABLE"))?
                .last_start_timings
                .clone();
            if ready.state != Phase::Ready {
                return Err(error("AI_SMOKE_FAILED"));
            }
            let (origin, first_pid) = {
                let inner = manager.inner.lock().map_err(|_| error("AI_UNAVAILABLE"))?;
                let worker = inner
                    .worker
                    .as_ref()
                    .ok_or_else(|| error("AI_SMOKE_FAILED"))?;
                (worker.origin.clone(), worker.child.id())
            };
            let unauthenticated = manager
                .local_client
                .post(format!("{origin}/infill"))
                .json(&json!({"input_prefix":"x = ","input_suffix":"","n_predict":1}))
                .send()
                .await
                .map_err(|_| error("AI_SMOKE_FAILED"))?
                .status()
                .as_u16();
            if unauthenticated != 401 {
                return Err(error("AI_SMOKE_FAILED"));
            }
            let started = Instant::now();
            let completion = manager
                .complete(CompletionRequest {
                    request_id: "smoke-square".into(),
                    prefix: "def square(x):\n    return ".into(),
                    suffix: "\n".into(),
                    language: "python".into(),
                })
                .await?;
            let completion_seconds = started.elapsed().as_secs_f64();
            if completion.text.is_empty() || completion.text.len() > MAX_OUTPUT {
                return Err(error("AI_SMOKE_FAILED"));
            }
            manager
                .cancel("smoke-early")
                .map_err(|_| error("AI_SMOKE_FAILED"))?;
            let early = manager
                .complete(CompletionRequest {
                    request_id: "smoke-early".into(),
                    prefix: "x = ".into(),
                    suffix: String::new(),
                    language: "python".into(),
                })
                .await;
            if !early.is_err_and(|failure| failure == "AI_CANCELLED") {
                return Err(error("AI_SMOKE_FAILED"));
            }
            let active_manager = manager.clone();
            let active = tauri::async_runtime::spawn(async move {
                active_manager
                    .complete(CompletionRequest {
                        request_id: "smoke-active".into(),
                        prefix: "# List 100 Python functions\nfunctions = [".into(),
                        suffix: String::new(),
                        language: "python".into(),
                    })
                    .await
            });
            tokio::time::sleep(Duration::from_millis(10)).await;
            manager.cancel("smoke-active")?;
            if !active
                .await
                .map_err(|_| error("AI_SMOKE_FAILED"))?
                .is_err_and(|failure| failure == "AI_CANCELLED")
            {
                return Err(error("AI_SMOKE_FAILED"));
            }
            let off = manager.configure(false).await?;
            let off_stopped = manager
                .inner
                .lock()
                .map_err(|_| error("AI_UNAVAILABLE"))?
                .worker
                .is_none()
                && manager
                    .local_client
                    .get(format!("{origin}/health"))
                    .send()
                    .await
                    .is_err();
            if off.enabled || off.state != Phase::Off || !off_stopped {
                return Err(error("AI_SMOKE_FAILED"));
            }
            let restarted = Manager::new(resources, &data_root)?;
            if restarted.snapshot()?.enabled {
                return Err(error("AI_SMOKE_FAILED"));
            }
            drop(restarted);
            let started = Instant::now();
            manager.configure(true).await?;
            let warm_ready_seconds = started.elapsed().as_secs_f64();
            let warm_start_timings = manager
                .inner
                .lock()
                .map_err(|_| error("AI_UNAVAILABLE"))?
                .last_start_timings
                .clone();
            let (exit_origin, exit_pid) = {
                let inner = manager.inner.lock().map_err(|_| error("AI_UNAVAILABLE"))?;
                let worker = inner
                    .worker
                    .as_ref()
                    .ok_or_else(|| error("AI_SMOKE_FAILED"))?;
                (worker.origin.clone(), worker.child.id())
            };
            let client = manager.local_client.clone();
            drop(manager);
            let exit_stopped = client
                .get(format!("{exit_origin}/health"))
                .send()
                .await
                .is_err();
            if !exit_stopped {
                return Err(error("AI_SMOKE_FAILED"));
            }
            Ok(
                json!({"status":"ok","model":MODEL_ID,"model_sha256":MODEL_SHA256,
                "default_off":true,"installed_model_verified":true,"ready_seconds":ready_seconds,
                "first_start_timings":first_start_timings,"warm_ready_seconds":warm_ready_seconds,"warm_start_timings":warm_start_timings,
                "completion_seconds":completion_seconds,"completion":completion.text,
                "unauthenticated_inference":unauthenticated,"early_cancel":true,"active_cancel":true,
                "off_stops_worker":off_stopped,"off_persists_after_restart":true,"exit_stops_worker":exit_stopped,
                "owned_test_worker_pids":[first_pid,exit_pid],"no_human_projects":true}),
            )
        })
    })();
    let _ = std::fs::remove_dir_all(&data_root);
    result
}

async fn read_completion(mut response: reqwest::Response) -> Result<String, String> {
    if !response.status().is_success() {
        return Err(error("AI_ENGINE_UNAVAILABLE"));
    }
    if response
        .content_length()
        .is_some_and(|length| length > MAX_RESPONSE as u64)
    {
        return Err(error("AI_INVALID_RESPONSE"));
    }
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(|_| error("AI_NETWORK"))? {
        if bytes.len() + chunk.len() > MAX_RESPONSE {
            return Err(error("AI_INVALID_RESPONSE"));
        }
        bytes.extend_from_slice(&chunk);
    }
    parse_completion(&bytes)
}

fn parse_completion(bytes: &[u8]) -> Result<String, String> {
    if bytes.len() > MAX_RESPONSE {
        return Err(error("AI_INVALID_RESPONSE"));
    }
    let value: Value = serde_json::from_slice(bytes).map_err(|_| error("AI_INVALID_RESPONSE"))?;
    let text = value["content"]
        .as_str()
        .ok_or_else(|| error("AI_INVALID_RESPONSE"))?;
    Ok(bounded_text(text))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn temporary() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "openecon-suggestions-test-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir(&path).unwrap();
        path
    }
    #[test]
    fn completion_request_is_bounded_and_cannot_choose_provider() {
        let request = CompletionRequest {
            request_id: "request-1".into(),
            prefix: "return ".into(),
            suffix: String::new(),
            language: "python".into(),
        };
        assert!(validate_request(&request).is_ok());
        for id in ["", "../worker", "a/b", "a?token=x", "a\n"] {
            assert!(!safe_id(id));
        }
        let mut request = request;
        request.prefix = "x".repeat(MAX_PREFIX + 1);
        assert!(validate_request(&request).is_err());
        request.prefix = "return ".into();
        request.suffix = "x".repeat(MAX_SUFFIX + 1);
        assert!(validate_request(&request).is_err());
        assert!(serde_json::from_value::<CompletionRequest>(json!({"request_id":"x","prefix":"x","suffix":"","language":"python","url":"https://evil.example"})).is_err());
    }
    #[test]
    fn engine_response_is_plain_source_and_schema_bounded() {
        assert_eq!(
            parse_completion(br#"{"content":"x * x"}"#).unwrap(),
            "x * x"
        );
        assert!(parse_completion(br#"{"error":"provider token secret"}"#).is_err());
        assert!(parse_completion(&vec![b'x'; MAX_RESPONSE + 1]).is_err());
        assert!(parse_completion(b"<script>fake</script>").is_err());
    }
    #[test]
    fn model_redirects_are_only_trusted_https_without_credentials() {
        for value in [MODEL_URL, "https://cas-bridge.xethub.hf.co/model"] {
            assert!(safe_redirect(&value.parse().unwrap()));
        }
        for value in [
            "http://huggingface.co/model",
            "https://evil.example/model",
            "https://huggingface.co.evil.example/model",
            "https://user@huggingface.co/model",
            "https://huggingface.co:8443/model",
            "http://127.0.0.1/model",
        ] {
            assert!(!safe_redirect(&value.parse().unwrap()));
        }
    }
    #[test]
    fn suggestions_preserve_source_whitespace_and_bound_utf8_lines() {
        assert_eq!(bounded_text("x * x\n"), "x * x\n");
        assert_eq!(bounded_text("one\ntwo\nthree\nfour"), "one\ntwo\nthree");
        assert_eq!(bounded_text("```python\nx\n```"), "");
        assert!(bounded_text(&"ü".repeat(3000)).len() <= MAX_OUTPUT);
        assert_eq!(bounded_text("a\0b"), "");
    }
    #[test]
    fn default_off_never_spawns_or_downloads_and_settings_are_private() {
        let directory = temporary();
        let manager = Manager::new(Path::new("/missing-engine"), &directory).unwrap();
        let status = manager.snapshot().unwrap();
        assert!(!status.enabled);
        assert_eq!(status.state, Phase::Off);
        assert!(manager.inner.lock().unwrap().worker.is_none());
        assert!(!directory.join(CACHE_DIR).exists());
        manager.save_enabled(true).unwrap();
        assert!(read_settings(&manager.root).unwrap().enabled);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(directory.join(SETTINGS_FILE))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
        }
        drop(manager);
        std::fs::remove_dir_all(directory).unwrap();
    }
    #[test]
    fn invalid_settings_disable_only_suggestions() {
        let directory = temporary();
        std::fs::write(
            directory.join(SETTINGS_FILE),
            b"{\"enabled\":true,\"url\":\"evil\"}",
        )
        .unwrap();
        let manager = Manager::new(Path::new("/missing-engine"), &directory).unwrap();
        let status = manager.snapshot().unwrap();
        assert!(!status.enabled);
        assert_eq!(status.error_code, Some("AI_CONFIG_INVALID"));
        drop(manager);
        std::fs::remove_dir_all(directory).unwrap();
    }
    #[test]
    fn cancellation_before_admission_and_previous_ids_are_bounded() {
        let directory = temporary();
        let manager = Manager::new(Path::new("/missing-engine"), &directory).unwrap();
        for index in 0..100 {
            manager.cancel(&format!("request-{index}")).unwrap();
        }
        let token = CancellationToken::new();
        manager.inner.lock().unwrap().active = Some(("new".into(), token.clone()));
        manager.cancel("old").unwrap();
        assert!(!token.is_cancelled());
        manager.cancel("new").unwrap();
        assert!(token.is_cancelled());
        assert_eq!(manager.inner.lock().unwrap().cancelled_ids.len(), 64);
        drop(manager);
        std::fs::remove_dir_all(directory).unwrap();
    }
    #[test]
    fn later_off_wins_over_on_waiting_for_lifecycle() {
        let directory = temporary();
        let manager =
            std::sync::Arc::new(Manager::new(Path::new("/missing-engine"), &directory).unwrap());
        tauri::async_runtime::block_on(async {
            let guard = manager.lifecycle.lock().await;
            let enabled = manager.clone();
            let on = tauri::async_runtime::spawn(async move { enabled.configure(true).await });
            tokio::time::timeout(Duration::from_secs(2), async {
                while !manager.inner.lock().unwrap().enabled {
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
            })
            .await
            .unwrap();
            let disabled = manager.clone();
            let off = tauri::async_runtime::spawn(async move { disabled.configure(false).await });
            tokio::time::timeout(Duration::from_secs(2), async {
                while manager.inner.lock().unwrap().enabled {
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
            })
            .await
            .unwrap();
            drop(guard);
            assert!(!on.await.unwrap().unwrap().enabled);
            assert!(!off.await.unwrap().unwrap().enabled);
            assert!(!read_settings(&manager.root).unwrap().enabled);
            assert!(manager.inner.lock().unwrap().worker.is_none());
        });
        drop(manager);
        std::fs::remove_dir_all(directory).unwrap();
    }
    #[test]
    fn later_off_cancels_install_admission_before_any_download() {
        use std::{
            future::Future,
            task::{Context, Poll, Waker},
        };
        let directory = temporary();
        let manager = Manager::new(Path::new("/missing-engine"), &directory).unwrap();
        tauri::async_runtime::block_on(async {
            let guard = manager.lifecycle.lock().await;
            let mut install = Box::pin(manager.install());
            let mut context = Context::from_waker(Waker::noop());
            assert!(matches!(install.as_mut().poll(&mut context), Poll::Pending));
            let mut off = Box::pin(manager.configure(false));
            assert!(matches!(off.as_mut().poll(&mut context), Poll::Pending));
            drop(guard);
            assert!(install
                .await
                .is_err_and(|failure| failure == "AI_CANCELLED"));
            assert!(!off.await.unwrap().enabled);
            assert!(manager.inner.lock().unwrap().download.is_none());
            assert!(!directory.join(CACHE_DIR).exists());
            assert_eq!(manager.downloaded.load(Ordering::Relaxed), 0);
        });
        drop(manager);
        std::fs::remove_dir_all(directory).unwrap();
    }
    #[test]
    fn cancel_before_completion_admission_never_needs_a_worker() {
        let directory = temporary();
        let manager = Manager::new(Path::new("/missing-engine"), &directory).unwrap();
        manager.cancel("cancelled-before-start").unwrap();
        let result = tauri::async_runtime::block_on(manager.complete(CompletionRequest {
            request_id: "cancelled-before-start".into(),
            prefix: "return ".into(),
            suffix: String::new(),
            language: "python".into(),
        }));
        assert!(result.is_err_and(|failure| failure == "AI_CANCELLED"));
        assert!(manager.inner.lock().unwrap().worker.is_none());
        drop(manager);
        std::fs::remove_dir_all(directory).unwrap();
    }
    #[cfg(unix)]
    #[test]
    fn cache_and_config_symlinks_cannot_escape() {
        use std::os::unix::fs::symlink;
        let directory = temporary();
        let outside = temporary();
        symlink(&outside, directory.join(CACHE_DIR)).unwrap();
        let manager = Manager::new(Path::new("/missing-engine"), &directory).unwrap();
        assert!(manager.cache(true).is_err());
        std::fs::write(outside.join("settings"), b"{}").unwrap();
        symlink(outside.join("settings"), directory.join(SETTINGS_FILE)).unwrap();
        assert!(manager.save_enabled(true).is_err());
        assert_eq!(std::fs::read(outside.join("settings")).unwrap(), b"{}");
        drop(manager);
        std::fs::remove_dir_all(directory).unwrap();
        std::fs::remove_dir_all(outside).unwrap();
    }
}
