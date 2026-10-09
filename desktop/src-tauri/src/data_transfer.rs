//! Bounded, membership-scoped data transfers. Cloud credentials exist only in
//! request headers; durable journals contain local content pins, never tokens.
use crate::broker::{self, CachedFile, CloudResponse};
use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
use cap_std::fs::{Dir, OpenOptions};
use fs2::FileExt;
use reqwest::{Client, Method};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

pub const PART_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_FILE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
pub const MAX_PROJECT_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const MAX_JOURNAL_BYTES: u64 = 256 * 1024;
const SCHEMA: &str = "chunked-v1";
const INTEGRITY: &str = "OPENECON_DATA_INTEGRITY";

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PartPin {
    pub index: usize,
    pub sha256: String,
    pub size_bytes: u64,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct UploadJournal {
    schema: String,
    request_id: String,
    project_id: String,
    local_path: String,
    name: String,
    size_bytes: u64,
    sha256: String,
    parts: Vec<PartPin>,
    transfer_id: Option<String>,
}

#[derive(Serialize, Clone)]
pub struct TransferView {
    pub request_id: String,
    pub name: String,
    pub size_bytes: u64,
    pub sha256: String,
    pub state: String,
    pub acknowledged_bytes: u64,
}

pub enum UploadOutcome {
    Complete(Value),
    Pending(TransferView),
    Unsupported,
}

#[derive(Deserialize)]
struct RemoteStatus {
    schema: String,
    id: String,
    state: String,
    name: String,
    size_bytes: u64,
    sha256: String,
    part_count: usize,
    part_bytes: usize,
    file_id: String,
    #[serde(default)]
    parts: Vec<PartPin>,
}

#[derive(Deserialize, Serialize, Clone)]
struct Manifest {
    schema: String,
    file: Value,
    size_bytes: u64,
    sha256: String,
    part_bytes: usize,
    parts: Vec<PartPin>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DownloadJournal {
    schema: String,
    project_id: String,
    file_id: String,
    size_bytes: u64,
    sha256: String,
    completed: Vec<PartPin>,
}

fn lower_hex(value: &str, size: usize) -> bool {
    value.len() == size
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn part_size(size: u64, index: usize) -> Result<u64, String> {
    if !(1..=MAX_FILE_BYTES).contains(&size) || index as u64 >= size.div_ceil(PART_BYTES as u64) {
        return Err(INTEGRITY.into());
    }
    Ok((size - index as u64 * PART_BYTES as u64).min(PART_BYTES as u64))
}

fn validate_parts(size: u64, parts: &[PartPin], complete: bool) -> Result<(), String> {
    if !(1..=MAX_FILE_BYTES).contains(&size)
        || parts.len() > size.div_ceil(PART_BYTES as u64) as usize
    {
        return Err(INTEGRITY.into());
    }
    if complete && parts.len() != size.div_ceil(PART_BYTES as u64) as usize {
        return Err(INTEGRITY.into());
    }
    let mut seen = std::collections::HashSet::new();
    for (position, part) in parts.iter().enumerate() {
        if !seen.insert(part.index)
            || (complete && part.index != position)
            || part.size_bytes != part_size(size, part.index)?
            || !lower_hex(&part.sha256, 64)
        {
            return Err(INTEGRITY.into());
        }
    }
    Ok(())
}

fn directory(root: &Dir, project: &str) -> Result<Dir, String> {
    if !lower_hex(project, 32) {
        return Err("Invalid project identifier.".into());
    }
    let project = broker::project_dir(root, project)?;
    match project.create_dir(".transfers") {
        Ok(()) => (),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => (),
        Err(_) => return Err("The transfer journal directory could not be created.".into()),
    }
    project
        .open_dir_nofollow(".transfers")
        .map_err(|_| "The transfer directory is unsafe.".into())
}

fn load<T: serde::de::DeserializeOwned>(dir: &Dir, name: &str) -> Result<Option<T>, String> {
    let mut options = OpenOptions::new();
    options.read(true).follow(FollowSymlinks::No);
    let file = match dir.open_with(name, &options) {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(_) => return Err("The transfer journal is unsafe.".into()),
    };
    let meta = file
        .metadata()
        .map_err(|_| "The transfer journal cannot be read.")?;
    if !meta.is_file() || meta.len() > MAX_JOURNAL_BYTES {
        return Err(INTEGRITY.into());
    }
    let mut bytes = Vec::new();
    file.take(MAX_JOURNAL_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| "The transfer journal cannot be read.")?;
    if bytes.len() as u64 > MAX_JOURNAL_BYTES {
        return Err(INTEGRITY.into());
    }
    serde_json::from_slice(&bytes)
        .map(Some)
        .map_err(|_| INTEGRITY.into())
}

fn save<T: Serialize>(dir: &Dir, name: &str, record: &T) -> Result<(), String> {
    let bytes = serde_json::to_vec(record).map_err(|_| INTEGRITY)?;
    if bytes.len() as u64 > MAX_JOURNAL_BYTES {
        return Err(INTEGRITY.into());
    }
    if let Ok(meta) = dir.symlink_metadata(name) {
        if !meta.is_file() || meta.file_type().is_symlink() {
            return Err("The transfer journal is unsafe.".into());
        }
    }
    let temp = format!(".write-{}", uuid::Uuid::new_v4().simple());
    let mut options = OpenOptions::new();
    options
        .write(true)
        .create_new(true)
        .follow(FollowSymlinks::No);
    let mut file = dir
        .open_with(&temp, &options)
        .map_err(|_| "The transfer journal cannot be saved.")?;
    let result: Result<(), String> = (|| {
        file.write_all(&bytes)
            .map_err(|_| "The transfer journal cannot be saved.")?;
        file.sync_all()
            .map_err(|_| "The transfer journal cannot be saved.")?;
        dir.rename(&temp, dir, name)
            .map_err(|_| "The transfer journal cannot be installed.")?;
        #[cfg(unix)]
        dir.try_clone()
            .map_err(|_| "The transfer directory cannot be synced.")?
            .into_std_file()
            .sync_all()
            .map_err(|_| "The transfer directory cannot be synced.")?;
        Ok(())
    })();
    if result.is_err() {
        let _ = dir.remove_file(&temp);
    }
    result
}

fn open_source(path: &Path) -> Result<std::fs::File, String> {
    let parent = path.parent().ok_or("The selected path is invalid.")?;
    let name = path.file_name().ok_or("The selected path is invalid.")?;
    let dir = Dir::open_ambient_dir(parent, cap_std::ambient_authority())
        .map_err(|_| "The selected file cannot be opened.")?;
    let mut options = OpenOptions::new();
    options.read(true).follow(FollowSymlinks::No);
    let file = dir
        .open_with(name, &options)
        .map_err(|_| "Choose a safe regular data file.")?;
    if !file
        .metadata()
        .map_err(|_| "The selected file cannot be inspected.")?
        .is_file()
    {
        return Err("Choose a regular data file.".into());
    }
    Ok(file.into_std())
}

fn pin_source(path: &Path) -> Result<(u64, String, Vec<PartPin>), String> {
    let mut file = open_source(path)?;
    let size = file
        .metadata()
        .map_err(|_| "The selected file cannot be inspected.")?
        .len();
    if !(1..=MAX_FILE_BYTES).contains(&size) {
        return Err("Shared CSV/Parquet files must be between 1 byte and 2 GiB.".into());
    }
    let mut buffer = vec![0; PART_BYTES];
    let mut whole = Sha256::new();
    let mut parts = Vec::with_capacity(size.div_ceil(PART_BYTES as u64) as usize);
    for index in 0..size.div_ceil(PART_BYTES as u64) as usize {
        let count = part_size(size, index)? as usize;
        file.read_exact(&mut buffer[..count])
            .map_err(|_| "The selected file changed while being read.")?;
        whole.update(&buffer[..count]);
        parts.push(PartPin {
            index,
            sha256: hash(&buffer[..count]),
            size_bytes: count as u64,
        });
    }
    if file
        .read(&mut buffer[..1])
        .map_err(|_| "The selected file cannot be read.")?
        != 0
        || file
            .metadata()
            .map_err(|_| "The selected file cannot be inspected.")?
            .len()
            != size
    {
        return Err("The selected file changed while being read.".into());
    }
    Ok((size, format!("{:x}", whole.finalize()), parts))
}

async fn pin_async(path: &Path) -> Result<(u64, String, Vec<PartPin>), String> {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || pin_source(&path))
        .await
        .map_err(|_| "The selected file could not be inspected.")?
}

fn validate_journal(j: &UploadJournal, project: &str, request: &str) -> Result<(), String> {
    if j.schema != SCHEMA
        || j.project_id != project
        || j.request_id != request
        || !lower_hex(request, 32)
        || !lower_hex(&j.sha256, 64)
        || !broker::safe_filename(&j.name)
        || !matches!(
            Path::new(&j.name)
                .extension()
                .and_then(|x| x.to_str())
                .map(str::to_ascii_lowercase)
                .as_deref(),
            Some("csv" | "parquet")
        )
        || !Path::new(&j.local_path).is_absolute()
        || j.local_path.len() > 4096
        || j.transfer_id.as_ref().is_some_and(|id| !lower_hex(id, 32))
    {
        return Err(INTEGRITY.into());
    }
    validate_parts(j.size_bytes, &j.parts, true)
}

fn view(j: &UploadJournal, state: &str, acknowledged_bytes: u64) -> TransferView {
    TransferView {
        request_id: j.request_id.clone(),
        name: j.name.clone(),
        size_bytes: j.size_bytes,
        sha256: j.sha256.clone(),
        state: state.into(),
        acknowledged_bytes,
    }
}

fn lock(dir: &Dir, request: &str) -> Result<std::fs::File, String> {
    let mut options = OpenOptions::new();
    options
        .read(true)
        .write(true)
        .create(true)
        .follow(FollowSymlinks::No);
    let file = dir
        .open_with(format!("{request}.lock"), &options)
        .map_err(|_| "The transfer lock is unsafe.")?
        .into_std();
    if !file
        .metadata()
        .map_err(|_| "The transfer lock is unsafe.")?
        .is_file()
    {
        return Err(INTEGRITY.into());
    }
    file.try_lock_exclusive()
        .map_err(|_| "This transfer is already running.")?;
    Ok(file)
}

struct Remote<'a> {
    client: &'a Client,
    token: &'a str,
    origin: &'a str,
    project: &'a str,
}

impl Remote<'_> {
    fn builder(&self, method: Method, suffix: &str) -> Result<reqwest::RequestBuilder, String> {
        if self.token.is_empty()
            || self.token.len() > 8192
            || self.token.chars().any(char::is_control)
        {
            return Err("The sign-in credential is invalid.".into());
        }
        Ok(self
            .client
            .request(
                method,
                format!(
                    "{}/api/projects/{}/workspace{suffix}",
                    self.origin, self.project
                ),
            )
            .bearer_auth(self.token))
    }
    async fn json(
        &self,
        method: Method,
        suffix: &str,
        body: Option<Value>,
    ) -> Result<CloudResponse, String> {
        let mut builder = self.builder(method, suffix)?;
        if let Some(body) = body {
            builder = builder.json(&body);
        }
        let response = builder.send().await.map_err(|_| "OPENECON_NETWORK")?;
        let status = response.status().as_u16();
        if !(200..300).contains(&status) {
            // Never turn a known authorization denial into a network fallback.
            let body = if status == 409 {
                read_json(response).await.unwrap_or(Value::Null)
            } else {
                Value::Null
            };
            return Ok(CloudResponse { status, body });
        }
        Ok(CloudResponse {
            status,
            body: read_json(response).await?,
        })
    }
}

async fn read_json(mut response: reqwest::Response) -> Result<Value, String> {
    if response
        .content_length()
        .is_some_and(|n| n > MAX_JOURNAL_BYTES)
    {
        return Err(INTEGRITY.into());
    }
    let mut data = Vec::new();
    while let Some(part) = response.chunk().await.map_err(|_| "OPENECON_NETWORK")? {
        if data.len() + part.len() > MAX_JOURNAL_BYTES as usize {
            return Err(INTEGRITY.into());
        }
        data.extend_from_slice(&part);
    }
    serde_json::from_slice(&data).map_err(|_| INTEGRITY.into())
}

fn success(response: CloudResponse) -> Result<Value, String> {
    if !(200..300).contains(&response.status) {
        return Err(format!("OPENECON_HTTP:{}", response.status));
    }
    Ok(response.body)
}

fn error_code(body: &Value) -> Option<&str> {
    body.get("code")
        .and_then(Value::as_str)
        .or_else(|| {
            body.get("detail")
                .and_then(|e| e.get("code"))
                .and_then(Value::as_str)
        })
        .or_else(|| {
            body.get("error")
                .and_then(|e| e.get("code"))
                .and_then(Value::as_str)
        })
}

fn validate_status(body: Value, j: &UploadJournal) -> Result<RemoteStatus, String> {
    let status: RemoteStatus = serde_json::from_value(body).map_err(|_| INTEGRITY)?;
    if status.schema != SCHEMA
        || !lower_hex(&status.id, 32)
        || !lower_hex(&status.file_id, 32)
        || status.name != j.name
        || status.size_bytes != j.size_bytes
        || status.sha256 != j.sha256
        || status.part_bytes != PART_BYTES
        || status.part_count != j.parts.len()
        || !matches!(status.state.as_str(), "uploading" | "ready")
        || j.transfer_id.as_ref().is_some_and(|id| *id != status.id)
    {
        return Err(INTEGRITY.into());
    }
    validate_parts(j.size_bytes, &status.parts, false)?;
    for part in &status.parts {
        if *part != j.parts[part.index] {
            return Err(INTEGRITY.into());
        }
    }
    Ok(status)
}

async fn resume_inner(
    remote: &Remote<'_>,
    dir: &Dir,
    j: UploadJournal,
) -> Result<UploadOutcome, String> {
    resume_with_initial_pin(remote, dir, j, None).await
}

// This pin is ephemeral and never restored from a journal or accepted over IPC.
// Only a newly planned upload can avoid repeating its initial whole-file read.
struct InitialSourcePin {
    size: u64,
    digest: String,
    parts: Vec<PartPin>,
}

async fn resume_with_initial_pin(
    remote: &Remote<'_>,
    dir: &Dir,
    mut j: UploadJournal,
    initial: Option<InitialSourcePin>,
) -> Result<UploadOutcome, String> {
    let name = format!("{}.json", j.request_id);
    let _lock = lock(dir, &j.request_id)?;
    let cancel = format!("{}.cancel", j.request_id);
    if dir.symlink_metadata(&cancel).is_ok() {
        return Err("This transfer is waiting to be cancelled.".into());
    }
    let (size, digest, parts) = match initial {
        Some(pin) => (pin.size, pin.digest, pin.parts),
        None => pin_async(Path::new(&j.local_path)).await?,
    };
    if size != j.size_bytes || digest != j.sha256 || parts != j.parts {
        return Err("The local file changed. Cancel this transfer and select the new file.".into());
    }
    let begin = remote
        .json(
            Method::POST,
            "/transfers",
            Some(json!({"request_id": j.request_id,
        "name":j.name, "size_bytes":j.size_bytes, "sha256":j.sha256})),
        )
        .await;
    let begin = match begin {
        Err(e) if e == "OPENECON_NETWORK" => {
            return Ok(UploadOutcome::Pending(view(&j, "waiting", 0)))
        }
        Err(e) => return Err(e),
        Ok(v) => v,
    };
    if j.transfer_id.is_none() && matches!(begin.status, 404 | 405 | 501) {
        return Ok(UploadOutcome::Unsupported);
    }
    let initial = validate_status(success(begin)?, &j)?;
    j.transfer_id = Some(initial.id.clone());
    if dir.symlink_metadata(&cancel).is_ok() {
        return Err("The transfer was stopped for cancellation.".into());
    }
    save(dir, &name, &j)?;
    let status = match remote
        .json(Method::GET, &format!("/transfers/{}", initial.id), None)
        .await
    {
        Err(e) if e == "OPENECON_NETWORK" => {
            return Ok(UploadOutcome::Pending(view(&j, "waiting", 0)))
        }
        Err(e) => return Err(e),
        Ok(v) => validate_status(success(v)?, &j)?,
    };
    let mut acknowledged: u64 = status.parts.iter().map(|p| p.size_bytes).sum();
    if status.state == "ready" {
        let manifest = remote
            .json(
                Method::GET,
                &format!("/files/{}/manifest", status.file_id),
                None,
            )
            .await?;
        let manifest = validate_manifest(success(manifest)?, &status.file_id)?;
        if manifest.size_bytes != j.size_bytes
            || manifest.sha256 != j.sha256
            || manifest.file["name"] != j.name
        {
            return Err(INTEGRITY.into());
        }
        if dir.symlink_metadata(&cancel).is_ok() {
            return Err("The transfer was stopped for cancellation.".into());
        }
        let _ = dir.remove_file(&name);
        return Ok(UploadOutcome::Complete(manifest.file));
    }
    let mut file = open_source(Path::new(&j.local_path))?;
    let remote_parts: std::collections::HashSet<_> = status.parts.iter().map(|p| p.index).collect();
    for pin in &j.parts {
        if dir.symlink_metadata(&cancel).is_ok() {
            return Err("The transfer was stopped for cancellation.".into());
        }
        if remote_parts.contains(&pin.index) {
            continue;
        }
        file.seek(SeekFrom::Start(pin.index as u64 * PART_BYTES as u64))
            .map_err(|_| "The selected file cannot be read.")?;
        let mut bytes = vec![0; pin.size_bytes as usize];
        file.read_exact(&mut bytes)
            .map_err(|_| "The selected file changed while uploading.")?;
        if hash(&bytes) != pin.sha256 {
            return Err("The selected file changed while uploading.".into());
        }
        let send = remote
            .builder(
                Method::PUT,
                &format!("/transfers/{}/parts/{}", status.id, pin.index),
            )?
            .header("Content-Type", "application/octet-stream")
            .header("X-OpenEcon-SHA256", &pin.sha256)
            .body(bytes)
            .send()
            .await;
        let response = match send {
            Err(_) => return Ok(UploadOutcome::Pending(view(&j, "waiting", acknowledged))),
            Ok(v) => v,
        };
        if !response.status().is_success() {
            let status_code = response.status().as_u16();
            let body = if status_code == 409 {
                read_json(response).await.unwrap_or(Value::Null)
            } else {
                Value::Null
            };
            if status_code == 409 && error_code(&body) == Some("PART_BUSY") {
                return Ok(UploadOutcome::Pending(view(&j, "waiting", acknowledged)));
            }
            return Err(format!("OPENECON_HTTP:{status_code}"));
        }
        let body = match read_json(response).await {
            Err(e) if e == "OPENECON_NETWORK" => {
                return Ok(UploadOutcome::Pending(view(&j, "waiting", acknowledged)))
            }
            Err(e) => return Err(e),
            Ok(v) => v,
        };
        if body["index"].as_u64() != Some(pin.index as u64)
            || body["size_bytes"].as_u64() != Some(pin.size_bytes)
            || body["sha256"].as_str() != Some(pin.sha256.as_str())
        {
            return Err(INTEGRITY.into());
        }
        acknowledged += pin.size_bytes;
    }
    let (size, digest, parts) = pin_async(Path::new(&j.local_path)).await?;
    if size != j.size_bytes || digest != j.sha256 || parts != j.parts {
        return Err("The selected file changed before completion.".into());
    }
    if dir.symlink_metadata(&cancel).is_ok() {
        return Err("The transfer was stopped for cancellation.".into());
    }
    let response = match remote
        .json(
            Method::POST,
            &format!("/transfers/{}/complete", status.id),
            None,
        )
        .await
    {
        Err(e) if e == "OPENECON_NETWORK" => {
            return Ok(UploadOutcome::Pending(view(&j, "waiting", acknowledged)))
        }
        Err(e) => return Err(e),
        Ok(v) => v,
    };
    if response.status == 409 && error_code(&response.body) == Some("TRANSFER_BUSY") {
        return Ok(UploadOutcome::Pending(view(&j, "waiting", acknowledged)));
    }
    let body = success(response)?;
    if body["id"].as_str() != Some(status.file_id.as_str())
        || body["name"] != j.name
        || body["size_bytes"].as_u64() != Some(j.size_bytes)
        || body["data_hash"].as_str() != Some(j.sha256.as_str())
        || body["transfer"].as_str() != Some(SCHEMA)
    {
        return Err(INTEGRITY.into());
    }
    if dir.symlink_metadata(&cancel).is_ok() {
        return Err("The transfer was stopped for cancellation.".into());
    }
    let _ = dir.remove_file(&name);
    Ok(UploadOutcome::Complete(body))
}

pub async fn upload_selected(
    client: &Client,
    root: &Dir,
    project: &str,
    path: &Path,
    token: &str,
) -> Result<UploadOutcome, String> {
    let dir = directory(root, project)?;
    // Serialize journal reservations, rather than allowing concurrent picker
    // commands to race past the four-upload budget. Release before network IO.
    let reservation = lock(&dir, "upload-budget")?;
    let name = path
        .file_name()
        .and_then(|x| x.to_str())
        .ok_or("The selected filename is invalid.")?;
    if !broker::safe_filename(name)
        || !matches!(
            path.extension()
                .and_then(|x| x.to_str())
                .map(str::to_ascii_lowercase)
                .as_deref(),
            Some("csv" | "parquet")
        )
    {
        return Err("Share large data as CSV or Parquet.".into());
    }
    let journals = journals(&dir, project)?;
    let local_path = path
        .to_str()
        .ok_or("The selected path cannot be saved for retry.")?;
    if let Some(prior) = journals
        .iter()
        .find(|j| j.local_path == local_path && j.name == name)
    {
        drop(reservation);
        return resume_inner(
            &Remote {
                client,
                token,
                origin: broker::CLOUD_ORIGIN,
                project,
            },
            &dir,
            prior.clone(),
        )
        .await;
    }
    if journals.len() >= 4 {
        return Err("Finish or cancel an existing transfer first (maximum 4).".into());
    }
    let (size_bytes, sha256, parts) = pin_async(path).await?;
    let initial = InitialSourcePin {
        size: size_bytes,
        digest: sha256.clone(),
        parts: parts.clone(),
    };
    let j = UploadJournal {
        schema: SCHEMA.into(),
        request_id: uuid::Uuid::new_v4().simple().to_string(),
        project_id: project.into(),
        local_path: local_path.into(),
        name: name.into(),
        size_bytes,
        sha256,
        parts,
        transfer_id: None,
    };
    validate_journal(&j, project, &j.request_id)?;
    save(&dir, &format!("{}.json", j.request_id), &j)?;
    drop(reservation);
    let request = j.request_id.clone();
    let result = resume_with_initial_pin(
        &Remote {
            client,
            token,
            origin: broker::CLOUD_ORIGIN,
            project,
        },
        &dir,
        j,
        Some(initial),
    )
    .await;
    if matches!(result, Ok(UploadOutcome::Unsupported)) {
        let _ = dir.remove_file(format!("{request}.json"));
    }
    if matches!(
        result,
        Ok(UploadOutcome::Complete(_)) | Ok(UploadOutcome::Unsupported)
    ) {
        let _ = dir.remove_file(format!("{request}.lock"));
    }
    result
}

/// Install the already selected upload source without downloading it again.
/// Content is rechecked against the completed cloud snapshot; originals survive.
pub async fn cache_uploaded(
    root: &Dir,
    project: &str,
    path: &Path,
    metadata: &Value,
) -> Result<CachedFile, String> {
    let root = root
        .try_clone()
        .map_err(|_| "The local cache is unavailable.")?;
    let project = project.to_string();
    let path = path.to_path_buf();
    let metadata = metadata.clone();
    tokio::task::spawn_blocking(move || {
        let id = metadata["id"]
            .as_str()
            .filter(|id| lower_hex(id, 32))
            .ok_or(INTEGRITY)?;
        let name = metadata["name"]
            .as_str()
            .filter(|name| broker::safe_filename(name))
            .ok_or(INTEGRITY)?;
        let size = metadata["size_bytes"]
            .as_u64()
            .filter(|n| (1..=MAX_FILE_BYTES).contains(n))
            .ok_or(INTEGRITY)?;
        let digest = metadata["data_hash"]
            .as_str()
            .filter(|h| lower_hex(h, 64))
            .ok_or(INTEGRITY)?;
        if metadata["transfer"].as_str() != Some(SCHEMA) {
            return Err(INTEGRITY.into());
        }
        let dir = broker::project_dir(&root, &project)?;
        let mut cached = CachedFile {
            cloud_id: id.into(),
            name: name.into(),
            python_path: name.into(),
            data_hash: digest.into(),
            sha256: digest.into(),
            size_bytes: size,
            reused: false,
        };
        if broker::cache_matches(&dir, name, size, digest)? {
            cached.reused = true;
            return Ok(cached);
        }
        if dir.symlink_metadata(name).is_ok() {
            return Err(
                "The local filename contains different data; existing data was preserved.".into(),
            );
        }
        let temporary = format!(".uploaded-{}", uuid::Uuid::new_v4().simple());
        let mut options = OpenOptions::new();
        options
            .write(true)
            .create_new(true)
            .follow(FollowSymlinks::No);
        let mut target = dir
            .open_with(&temporary, &options)
            .map_err(|_| "The upload cache could not be created.")?;
        let result: Result<(), String> = (|| {
            let mut source = open_source(&path)?;
            let mut buffer = [0u8; 65536];
            let mut whole = Sha256::new();
            let mut total = 0u64;
            loop {
                let count = source
                    .read(&mut buffer)
                    .map_err(|_| "The uploaded source cannot be read.")?;
                if count == 0 {
                    break;
                }
                total += count as u64;
                if total > size {
                    return Err(INTEGRITY.into());
                }
                whole.update(&buffer[..count]);
                target
                    .write_all(&buffer[..count])
                    .map_err(|_| "The upload cache could not be saved.")?;
            }
            if total != size || format!("{:x}", whole.finalize()) != digest {
                return Err(INTEGRITY.into());
            }
            target
                .sync_all()
                .map_err(|_| "The upload cache could not be saved.")?;
            dir.hard_link(&temporary, &dir, name).map_err(|_| {
                "The destination appeared during upload; existing data was preserved."
            })?;
            Ok(())
        })();
        drop(target);
        let _ = dir.remove_file(&temporary);
        result?;
        Ok(cached)
    })
    .await
    .map_err(|_| "The uploaded source cache could not be installed.")?
}

fn journals(dir: &Dir, project: &str) -> Result<Vec<UploadJournal>, String> {
    let mut result = Vec::new();
    for (count, entry) in dir
        .entries()
        .map_err(|_| "Transfer journals cannot be listed.")?
        .enumerate()
    {
        if count >= 128 {
            return Err("The local transfer directory exceeds its entry budget.".into());
        }
        let entry = entry.map_err(|_| "Transfer journals cannot be listed.")?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(id) = name.strip_suffix(".json") else {
            continue;
        };
        if !lower_hex(id, 32) {
            continue;
        }
        if result.len() >= 4 {
            return Err("The local transfer journal count exceeds its budget.".into());
        }
        let j: UploadJournal = load(dir, &name)?.ok_or(INTEGRITY)?;
        validate_journal(&j, project, id)?;
        result.push(j);
    }
    Ok(result)
}

fn outcome_response(outcome: UploadOutcome) -> Result<CloudResponse, String> {
    Ok(match outcome {
        UploadOutcome::Complete(file) => CloudResponse {
            status: 200,
            body: json!({"state":"ready", "file":file}),
        },
        UploadOutcome::Pending(v) => CloudResponse {
            status: 202,
            body: serde_json::to_value(v).map_err(|_| INTEGRITY)?,
        },
        UploadOutcome::Unsupported => return Err("OPENECON_TRANSFER_UNSUPPORTED".into()),
    })
}

async fn cancel_inner(
    remote: &Remote<'_>,
    dir: &Dir,
    mut j: UploadJournal,
) -> Result<CloudResponse, String> {
    let request = j.request_id.clone();
    let name = format!("{request}.json");
    let marker = format!("{request}.cancel");
    save(dir, &marker, &json!({"cancel":true}))?;
    if j.transfer_id.is_none() {
        let initial = remote
            .json(
                Method::POST,
                "/transfers",
                Some(json!({"request_id":j.request_id,
            "name":j.name,"size_bytes":j.size_bytes,"sha256":j.sha256})),
            )
            .await?;
        let status = validate_status(success(initial)?, &j)?;
        j.transfer_id = Some(status.id);
        save(dir, &name, &j)?;
    }
    let response = remote
        .json(
            Method::DELETE,
            &format!("/transfers/{}", j.transfer_id.as_ref().unwrap()),
            None,
        )
        .await?;
    if response.status == 409 && error_code(&response.body) == Some("CLEANUP_PENDING") {
        return Ok(CloudResponse {
            status: 202,
            body: serde_json::to_value(view(&j, "cancel_pending", 0)).map_err(|_| INTEGRITY)?,
        });
    }
    if response.status == 409 && error_code(&response.body) == Some("TRANSFER_COMPLETE") {
        // A completed catalogue entry cannot be undone by transfer cancellation.
        // Keep the journal so a later explicit resume can recover its ready cache.
        dir.remove_file(&marker)
            .map_err(|_| "The completed transfer state could not be reconciled.")?;
        return Err("OPENECON_TRANSFER_COMPLETE".into());
    }
    success(response)?;
    // Active resume may still hold a cloned journal. Keep the tombstone until
    // it has stopped and cannot resurrect the cancelled operation.
    let guard = match lock(dir, &request) {
        Ok(guard) => guard,
        Err(e) if e == "This transfer is already running." => {
            return Ok(CloudResponse {
                status: 202,
                body: serde_json::to_value(view(&j, "cancel_pending", 0)).map_err(|_| INTEGRITY)?,
            })
        }
        Err(e) => return Err(e),
    };
    dir.remove_file(&name)
        .map_err(|_| "The cancelled transfer journal could not be removed.")?;
    dir.remove_file(&marker)
        .map_err(|_| "The cancellation marker could not be removed.")?;
    drop(guard);
    let _ = dir.remove_file(format!("{request}.lock"));
    Ok(CloudResponse {
        status: 200,
        body: json!({"cancelled":true,"request_id":request}),
    })
}

pub async fn transfer_action(
    client: &Client,
    root: &Dir,
    project: &str,
    request: Option<&str>,
    token: &str,
    action: &str,
) -> Result<CloudResponse, String> {
    let dir = directory(root, project)?;
    if action == "list" {
        let values: Vec<_> = journals(&dir, project)?
            .iter()
            .map(|j| {
                view(
                    j,
                    if dir
                        .symlink_metadata(format!("{}.cancel", j.request_id))
                        .is_ok()
                    {
                        "cancel_pending"
                    } else {
                        "waiting"
                    },
                    0,
                )
            })
            .collect();
        return Ok(CloudResponse {
            status: 200,
            body: json!({"transfers":values}),
        });
    }
    let request = request
        .filter(|id| lower_hex(id, 32))
        .ok_or("Invalid transfer request identifier.")?;
    let remote = Remote {
        client,
        token,
        origin: broker::CLOUD_ORIGIN,
        project,
    };
    if action == "cancel_download" {
        return cancel_download_inner(&remote, root, request).await;
    }
    let name = format!("{request}.json");
    let j: UploadJournal = load(&dir, &name)?.ok_or("This local transfer is no longer pending.")?;
    validate_journal(&j, project, request)?;
    match action {
        "resume" => {
            let path = j.local_path.clone();
            let outcome = resume_inner(&remote, &dir, j).await?;
            if let UploadOutcome::Complete(file) = &outcome {
                let _ = dir.remove_file(format!("{request}.lock"));
                cache_uploaded(root, project, Path::new(&path), file).await?;
            }
            outcome_response(outcome)
        }
        "status" => {
            if dir.symlink_metadata(format!("{request}.cancel")).is_ok() {
                return Ok(CloudResponse {
                    status: 202,
                    body: serde_json::to_value(view(&j, "cancel_pending", 0))
                        .map_err(|_| INTEGRITY)?,
                });
            }
            if let Some(id) = &j.transfer_id {
                let response = remote
                    .json(Method::GET, &format!("/transfers/{id}"), None)
                    .await?;
                let status = validate_status(success(response)?, &j)?;
                Ok(CloudResponse {
                    status: 200,
                    body: serde_json::to_value(view(
                        &j,
                        &status.state,
                        status.parts.iter().map(|p| p.size_bytes).sum(),
                    ))
                    .map_err(|_| INTEGRITY)?,
                })
            } else {
                Ok(CloudResponse {
                    status: 200,
                    body: serde_json::to_value(view(&j, "waiting", 0)).map_err(|_| INTEGRITY)?,
                })
            }
        }
        "cancel" => cancel_inner(&remote, &dir, j).await,
        _ => Err("Unknown transfer action.".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{TcpListener, TcpStream};
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    };
    use std::time::Duration;

    struct Sandbox {
        path: std::path::PathBuf,
        root: Dir,
    }
    impl Sandbox {
        fn new() -> Self {
            let path =
                std::env::temp_dir().join(format!("openecon-transfer-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir(&path).unwrap();
            let root = Dir::open_ambient_dir(&path, cap_std::ambient_authority()).unwrap();
            Self { path, root }
        }
        fn source(&self) -> std::path::PathBuf {
            self.source_with_size(PART_BYTES + 9)
        }
        fn source_with_size(&self, size: usize) -> std::path::PathBuf {
            let path = self.path.join("large.csv");
            let mut file = std::fs::File::create(&path).unwrap();
            let mut remaining = size;
            let block = [b'7'; 65536];
            while remaining > 0 {
                let count = remaining.min(block.len());
                file.write_all(&block[..count]).unwrap();
                remaining -= count;
            }
            path
        }
    }
    impl Drop for Sandbox {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    struct MockState {
        journal: UploadJournal,
        stored: Vec<PartPin>,
        puts: Vec<usize>,
        ready: bool,
        lose_first_ack: bool,
        denied: bool,
        downloads: usize,
        corrupt_download: bool,
        lose_second_download: bool,
        download_counts: Vec<usize>,
        busy_part: bool,
        busy_complete: bool,
        cleanup_pending: bool,
        cancel_conflict: bool,
        cancelled: bool,
        cancels: usize,
        pause_second_download: bool,
        second_requested: Arc<AtomicBool>,
        release_second: Arc<AtomicBool>,
        edit_source_after_upload: bool,
    }
    struct Mock {
        origin: String,
        state: Arc<Mutex<MockState>>,
        stop: Arc<AtomicBool>,
        join: Option<std::thread::JoinHandle<()>>,
    }
    impl Mock {
        fn new(journal: UploadJournal, lose_first_ack: bool) -> Self {
            let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
            listener.set_nonblocking(true).unwrap();
            let origin = format!("http://{}", listener.local_addr().unwrap());
            let part_count = journal.parts.len();
            let state = Arc::new(Mutex::new(MockState {
                journal,
                stored: Vec::new(),
                puts: vec![0; part_count],
                ready: false,
                lose_first_ack,
                denied: false,
                downloads: 0,
                corrupt_download: false,
                lose_second_download: false,
                download_counts: vec![0; part_count],
                busy_part: false,
                busy_complete: false,
                cleanup_pending: false,
                cancel_conflict: false,
                cancelled: false,
                cancels: 0,
                pause_second_download: false,
                second_requested: Arc::new(AtomicBool::new(false)),
                release_second: Arc::new(AtomicBool::new(false)),
                edit_source_after_upload: false,
            }));
            let stop = Arc::new(AtomicBool::new(false));
            let shared = state.clone();
            let stopping = stop.clone();
            let join = std::thread::spawn(move || {
                let mut handlers = Vec::new();
                while !stopping.load(Ordering::Relaxed) {
                    match listener.accept() {
                        Ok((socket, _)) => {
                            let state = shared.clone();
                            handlers.push(std::thread::spawn(move || Self::handle(socket, &state)));
                        }
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            std::thread::sleep(Duration::from_millis(5))
                        }
                        Err(e) => panic!("mock listener failed: {e}"),
                    }
                }
                for handler in handlers {
                    handler.join().unwrap();
                }
            });
            Self {
                origin,
                state,
                stop,
                join: Some(join),
            }
        }
        fn handle(mut socket: TcpStream, state: &Arc<Mutex<MockState>>) {
            socket.set_nonblocking(false).unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(10)))
                .unwrap();
            let mut header = Vec::new();
            let mut block = [0; 8192];
            let split = loop {
                let count = socket.read(&mut block).unwrap();
                assert!(count > 0);
                header.extend_from_slice(&block[..count]);
                if let Some(i) = header.windows(4).position(|v| v == b"\r\n\r\n") {
                    break i + 4;
                }
                assert!(header.len() < 16384);
            };
            let text = std::str::from_utf8(&header[..split])
                .unwrap()
                .to_ascii_lowercase();
            assert!(text.contains("authorization: bearer synthetic-test-token"));
            let line: Vec<_> = text.lines().next().unwrap().split_whitespace().collect();
            let method = line[0];
            let path = line[1];
            let length: usize = text
                .lines()
                .find_map(|v| {
                    v.strip_prefix("content-length:")
                        .map(|n| n.trim().parse().unwrap())
                })
                .unwrap_or(0);
            assert!(length <= PART_BYTES);
            let mut body = header[split..].to_vec();
            while body.len() < length {
                let n = socket
                    .read(&mut block[..(length - body.len()).min(8192)])
                    .unwrap();
                assert!(n > 0);
                body.extend_from_slice(&block[..n]);
            }
            let mut s = state.lock().unwrap();
            if s.denied {
                write!(
                    socket,
                    "HTTP/1.1 403 Forbidden\r\nContent-Length: 999\r\nConnection: close\r\n\r\n"
                )
                .unwrap();
                return;
            }
            let id = "a".repeat(32);
            let file_id = "b".repeat(32);
            let mut status = 200;
            let mut extra = String::new();
            let response = if method == "delete" && path.contains("/transfers/") {
                s.cancels += 1;
                if s.ready {
                    status = 409;
                    json!({"detail":{"code":"TRANSFER_COMPLETE"}})
                        .to_string()
                        .into_bytes()
                } else if s.cancel_conflict {
                    status = 409;
                    json!({"detail":{"code":"DATA_INTEGRITY"}})
                        .to_string()
                        .into_bytes()
                } else {
                    s.cancelled = true;
                    if s.cleanup_pending {
                        s.cleanup_pending = false;
                        status = 409;
                        json!({"detail":{"code":"CLEANUP_PENDING"}})
                            .to_string()
                            .into_bytes()
                    } else {
                        s.stored.clear();
                        json!({"cancelled":true}).to_string().into_bytes()
                    }
                }
            } else if s.cancelled {
                status = 410;
                json!({"detail":{"code":"TRANSFER_CANCELLED"}})
                    .to_string()
                    .into_bytes()
            } else if method == "post" && path.ends_with("/transfers") {
                let begin: Value = serde_json::from_slice(&body).unwrap();
                assert_eq!(begin["request_id"], s.journal.request_id);
                assert_eq!(begin["sha256"], s.journal.sha256);
                status = 201;
                json!({"schema":SCHEMA,"id":id,"state":if s.ready {"ready"} else {"uploading"},"name":s.journal.name,
                    "size_bytes":s.journal.size_bytes,"sha256":s.journal.sha256,"part_bytes":PART_BYTES,
                    "part_count":s.journal.parts.len(),"file_id":file_id}).to_string().into_bytes()
            } else if method == "get" && path.contains("/transfers/") {
                json!({"schema":SCHEMA,"id":id,"state":if s.ready {"ready"} else {"uploading"},"name":s.journal.name,
                    "size_bytes":s.journal.size_bytes,"sha256":s.journal.sha256,"part_bytes":PART_BYTES,
                    "part_count":s.journal.parts.len(),"file_id":file_id,"parts":s.stored}).to_string().into_bytes()
            } else if method == "put" {
                let index: usize = path.rsplit('/').next().unwrap().parse().unwrap();
                let pin = s.journal.parts[index].clone();
                assert_eq!(hash(&body), pin.sha256);
                assert_eq!(body.len() as u64, pin.size_bytes);
                assert!(text.contains(&format!("x-openecon-sha256: {}", pin.sha256)));
                if s.busy_part {
                    s.busy_part = false;
                    let response = json!({"detail":{"code":"PART_BUSY"}}).to_string();
                    write!(socket,"HTTP/1.1 409 Conflict\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response}",response.len()).unwrap();
                    return;
                }
                s.puts[index] += 1;
                if !s.stored.iter().any(|p| p.index == index) {
                    s.stored.push(pin.clone());
                }
                if s.edit_source_after_upload && s.stored.len() == s.journal.parts.len() {
                    s.edit_source_after_upload = false;
                    let mut source = std::fs::OpenOptions::new()
                        .write(true)
                        .open(&s.journal.local_path)
                        .unwrap();
                    source.write_all(b"X").unwrap();
                }
                if s.lose_first_ack {
                    s.lose_first_ack = false;
                    write!(
                        socket,
                        "HTTP/1.1 200 OK\r\nContent-Length: 100\r\nConnection: close\r\n\r\n"
                    )
                    .unwrap();
                    return;
                }
                json!({"index":index,"sha256":pin.sha256,"size_bytes":pin.size_bytes,"reused":false}).to_string().into_bytes()
            } else if method == "post" && path.ends_with("/complete") {
                if s.busy_complete {
                    s.busy_complete = false;
                    let response = json!({"detail":{"code":"TRANSFER_BUSY"}}).to_string();
                    write!(socket,"HTTP/1.1 409 Conflict\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response}",response.len()).unwrap();
                    return;
                }
                assert_eq!(s.stored.len(), s.journal.parts.len());
                s.ready = true;
                json!({"id":file_id,"name":s.journal.name,"size_bytes":s.journal.size_bytes,"data_hash":s.journal.sha256,"transfer":SCHEMA}).to_string().into_bytes()
            } else if method == "get" && path.ends_with("/manifest") {
                json!({"schema":SCHEMA,"file":{"id":file_id,"name":s.journal.name,"size_bytes":s.journal.size_bytes,"data_hash":s.journal.sha256,"transfer":SCHEMA},
                    "size_bytes":s.journal.size_bytes,"sha256":s.journal.sha256,"part_bytes":PART_BYTES,"parts":s.journal.parts}).to_string().into_bytes()
            } else if method == "get" && path.contains("/parts/") {
                let index: usize = path.rsplit('/').next().unwrap().parse().unwrap();
                s.download_counts[index] += 1;
                if index == 1 && s.pause_second_download {
                    s.pause_second_download = false;
                    let requested = s.second_requested.clone();
                    let release = s.release_second.clone();
                    drop(s);
                    requested.store(true, Ordering::Release);
                    let deadline = std::time::Instant::now() + Duration::from_secs(10);
                    while !release.load(Ordering::Acquire) && std::time::Instant::now() < deadline {
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    s = state.lock().unwrap();
                }
                if index == 1 && s.lose_second_download {
                    s.lose_second_download = false;
                    write!(socket,"HTTP/1.1 200 OK\r\nX-OpenEcon-SHA256: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",s.journal.parts[index].sha256,s.journal.parts[index].size_bytes).unwrap();
                    return;
                }
                let pin = &s.journal.parts[index];
                let mut source = open_source(Path::new(&s.journal.local_path)).unwrap();
                source
                    .seek(SeekFrom::Start(index as u64 * PART_BYTES as u64))
                    .unwrap();
                let mut data = vec![0; pin.size_bytes as usize];
                source.read_exact(&mut data).unwrap();
                extra = format!("X-OpenEcon-SHA256: {}\r\n", pin.sha256);
                s.downloads += 1;
                if s.corrupt_download {
                    data[0] ^= 1;
                }
                data
            } else {
                status = 404;
                b"null".to_vec()
            };
            write!(
                socket,
                "HTTP/1.1 {status} OK\r\n{extra}Content-Length: {}\r\nConnection: close\r\n\r\n",
                response.len()
            )
            .unwrap();
            drop(s);
            socket.write_all(&response).unwrap();
        }
    }
    impl Drop for Mock {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
            if let Some(j) = self.join.take() {
                let _ = j.join();
            }
        }
    }

    fn journal(path: &Path) -> UploadJournal {
        let (size_bytes, sha256, parts) = pin_source(path).unwrap();
        UploadJournal {
            schema: SCHEMA.into(),
            project_id: "f".repeat(32),
            request_id: "c".repeat(32),
            local_path: path.to_str().unwrap().into(),
            name: "large.csv".into(),
            size_bytes,
            sha256,
            parts,
            transfer_id: None,
        }
    }
    fn client() -> Client {
        Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap()
    }

    #[test]
    fn chunk_transfer_unknown_ack_resumes_only_missing_parts_without_tokens_in_journal() {
        // Exceeds the broker's 24 MiB inline-upload threshold: seven physical
        // parts travel over local HTTP with one intentionally lost ACK.
        let sandbox = Sandbox::new();
        let path = sandbox.source_with_size(25 * 1024 * 1024 + 17);
        let j = journal(&path);
        let mock = Mock::new(j.clone(), true);
        assert_eq!(j.size_bytes, 25 * 1024 * 1024 + 17);
        assert_eq!(j.parts.len(), 7);
        let dir = directory(&sandbox.root, &j.project_id).unwrap();
        save(&dir, &format!("{}.json", j.request_id), &j).unwrap();
        let client = client();
        tauri::async_runtime::block_on(async {
            let remote = Remote {
                client: &client,
                token: "synthetic-test-token",
                origin: &mock.origin,
                project: &j.project_id,
            };
            let first = resume_inner(&remote, &dir, j.clone()).await.unwrap();
            assert!(matches!(first, UploadOutcome::Pending(_)));
            assert!(!mock.state.lock().unwrap().ready);
            let saved: UploadJournal = load(&dir, &format!("{}.json", j.request_id))
                .unwrap()
                .unwrap();
            let serialized = serde_json::to_string(&saved).unwrap();
            assert!(!serialized.contains("token"));
            assert!(!serialized.contains("synthetic-test-token"));
            let complete = resume_inner(&remote, &dir, saved).await.unwrap();
            let UploadOutcome::Complete(file) = complete else {
                panic!("expected complete");
            };
            assert_eq!(mock.state.lock().unwrap().puts, vec![1; 7]);
            assert!(
                load::<UploadJournal>(&dir, &format!("{}.json", j.request_id))
                    .unwrap()
                    .is_none()
            );
            let cached = cache_uploaded(&sandbox.root, &j.project_id, &path, &file)
                .await
                .unwrap();
            assert!(!cached.reused);
            assert_eq!(cached.sha256, j.sha256);
            assert_eq!(mock.state.lock().unwrap().downloads, 0);
        });
    }

    #[test]
    fn chunk_transfer_checks_local_changes_remote_ack_geometry_and_current_membership() {
        let sandbox = Sandbox::new();
        let path = sandbox.source();
        let j = journal(&path);
        let mock = Mock::new(j.clone(), false);
        let dir = directory(&sandbox.root, &j.project_id).unwrap();
        let client = client();
        tauri::async_runtime::block_on(async {
            let remote = Remote {
                client: &client,
                token: "synthetic-test-token",
                origin: &mock.origin,
                project: &j.project_id,
            };
            mock.state.lock().unwrap().denied = true;
            assert_eq!(
                resume_inner(&remote, &dir, j.clone()).await.err().unwrap(),
                "OPENECON_HTTP:403"
            );
            mock.state.lock().unwrap().denied = false;
            let mut invalid = j.parts[0].clone();
            invalid.sha256 = "0".repeat(64);
            mock.state.lock().unwrap().stored = vec![invalid];
            assert_eq!(
                resume_inner(&remote, &dir, j.clone()).await.err().unwrap(),
                INTEGRITY
            );
            mock.state.lock().unwrap().stored.clear();
            let mut edit = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
            edit.write_all(b"X").unwrap();
            assert!(resume_inner(&remote, &dir, j.clone())
                .await
                .err()
                .unwrap()
                .contains("local file changed"));
            assert_eq!(mock.state.lock().unwrap().puts, vec![0, 0]);
        });
    }

    #[test]
    fn chunk_fresh_initial_pin_keeps_part_and_final_source_change_guards() {
        let sandbox = Sandbox::new();
        let path = sandbox.source();
        let j = journal(&path);
        let mock = Mock::new(j.clone(), false);
        let dir = directory(&sandbox.root, &j.project_id).unwrap();
        let client = client();
        let initial = || InitialSourcePin {
            size: j.size_bytes,
            digest: j.sha256.clone(),
            parts: j.parts.clone(),
        };
        tauri::async_runtime::block_on(async {
            let remote = Remote {
                client: &client,
                token: "synthetic-test-token",
                origin: &mock.origin,
                project: &j.project_id,
            };
            let mut source = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
            source.write_all(b"X").unwrap();
            assert!(
                resume_with_initial_pin(&remote, &dir, j.clone(), Some(initial()))
                    .await
                    .err()
                    .unwrap()
                    .contains("changed while uploading")
            );
            assert_eq!(mock.state.lock().unwrap().puts, vec![0, 0]);
            source.seek(SeekFrom::Start(0)).unwrap();
            source.write_all(b"7").unwrap();
            mock.state.lock().unwrap().edit_source_after_upload = true;
            assert!(
                resume_with_initial_pin(&remote, &dir, j.clone(), Some(initial()))
                    .await
                    .err()
                    .unwrap()
                    .contains("changed before completion")
            );
            assert_eq!(mock.state.lock().unwrap().puts, vec![1, 1]);
            assert!(!mock.state.lock().unwrap().ready);
            assert!(path.exists());
        });
    }

    #[test]
    fn chunk_download_listing_pin_mismatches_reject_before_staging_any_data() {
        let sandbox = Sandbox::new();
        let path = sandbox.source();
        let j = journal(&path);
        let mock = Mock::new(j.clone(), false);
        let client = client();
        mock.state.lock().unwrap().ready = true;
        let expected = CachedFile {
            cloud_id: "b".repeat(32),
            name: j.name.clone(),
            python_path: j.name.clone(),
            sha256: j.sha256.clone(),
            data_hash: j.sha256.clone(),
            size_bytes: j.size_bytes,
            reused: false,
        };
        tauri::async_runtime::block_on(async {
            let remote = Remote {
                client: &client,
                token: "synthetic-test-token",
                origin: &mock.origin,
                project: &j.project_id,
            };
            let id = expected.cloud_id.clone();
            let mut mismatches = Vec::new();
            let mut altered = expected.clone();
            altered.cloud_id = "d".repeat(32);
            mismatches.push(altered);
            let mut altered = expected.clone();
            altered.name = "other.csv".into();
            mismatches.push(altered);
            let mut altered = expected.clone();
            altered.size_bytes += 1;
            mismatches.push(altered);
            let mut altered = expected.clone();
            altered.sha256 = "0".repeat(64);
            mismatches.push(altered);
            let mut altered = expected.clone();
            altered.data_hash = "0".repeat(64);
            mismatches.push(altered);
            for altered in mismatches {
                assert_eq!(
                    download_pinned_inner(&remote, &sandbox.root, &id, Some(&altered))
                        .await
                        .err()
                        .unwrap(),
                    INTEGRITY
                );
            }
            let dir = broker::project_dir(&sandbox.root, &j.project_id).unwrap();
            assert!(dir.symlink_metadata("large.csv").is_err());
            assert!(dir
                .symlink_metadata(format!(".download-chunks-{id}"))
                .is_err());
            assert_eq!(mock.state.lock().unwrap().downloads, 0);
            let cached = download_pinned_inner(&remote, &sandbox.root, &id, Some(&expected))
                .await
                .unwrap();
            assert_eq!(cached.sha256, expected.sha256);
            assert_eq!(cached.size_bytes, expected.size_bytes);
            assert!(broker::cache_matches(
                &dir,
                &expected.name,
                expected.size_bytes,
                &expected.sha256
            )
            .unwrap());
        });
    }

    #[test]
    fn chunk_download_is_verified_atomic_reusable_and_never_masks_denial_or_corruption() {
        let sandbox = Sandbox::new();
        let path = sandbox.source();
        let j = journal(&path);
        let mock = Mock::new(j.clone(), false);
        let client = client();
        tauri::async_runtime::block_on(async {
            let remote = Remote {
                client: &client,
                token: "synthetic-test-token",
                origin: &mock.origin,
                project: &j.project_id,
            };
            let file = "b".repeat(32);
            mock.state.lock().unwrap().corrupt_download = true;
            assert_eq!(
                download_inner(&remote, &sandbox.root, &file)
                    .await
                    .err()
                    .unwrap(),
                INTEGRITY
            );
            let dir = broker::project_dir(&sandbox.root, &j.project_id).unwrap();
            assert!(dir.symlink_metadata("large.csv").is_err());
            mock.state.lock().unwrap().corrupt_download = false;
            let complete = download_inner(&remote, &sandbox.root, &file).await.unwrap();
            assert!(!complete.reused);
            let count = mock.state.lock().unwrap().downloads;
            let cached = download_inner(&remote, &sandbox.root, &file).await.unwrap();
            assert!(cached.reused);
            assert_eq!(mock.state.lock().unwrap().downloads, count);
            mock.state.lock().unwrap().denied = true;
            assert_eq!(
                download_inner(&remote, &sandbox.root, &file)
                    .await
                    .err()
                    .unwrap(),
                "OPENECON_HTTP:403"
            );
        });
    }

    #[test]
    fn chunk_transfer_manifest_and_durable_journal_geometry_are_strict() {
        let sandbox = Sandbox::new();
        let path = sandbox.source();
        let j = journal(&path);
        validate_journal(&j, &j.project_id, &j.request_id).unwrap();
        let mut changed = j.clone();
        changed.size_bytes = MAX_FILE_BYTES + 1;
        assert!(validate_journal(&changed, &j.project_id, &j.request_id).is_err());
        changed = j.clone();
        changed.parts[1].index = 0;
        assert!(validate_journal(&changed, &j.project_id, &j.request_id).is_err());
        let mut json = serde_json::to_value(&j).unwrap();
        json["token"] = json!("forbidden");
        assert!(serde_json::from_value::<UploadJournal>(json).is_err());
        let dir = directory(&sandbox.root, &j.project_id).unwrap();
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        let oversized = dir.open_with("bad.json", &options).unwrap();
        oversized.set_len(MAX_JOURNAL_BYTES + 1).unwrap();
        assert!(load::<UploadJournal>(&dir, "bad.json").is_err());
        for size in [0, MAX_FILE_BYTES + 1] {
            assert!(part_size(size, 0).is_err());
        }
        assert_eq!(part_size(MAX_FILE_BYTES, 511).unwrap(), PART_BYTES as u64);
        assert!(part_size(MAX_FILE_BYTES, 512).is_err());
    }

    #[test]
    fn chunk_download_reconnect_reuses_only_locally_verified_acknowledged_parts() {
        let sandbox = Sandbox::new();
        let path = sandbox.source();
        let j = journal(&path);
        let mock = Mock::new(j.clone(), false);
        let client = client();
        mock.state.lock().unwrap().lose_second_download = true;
        tauri::async_runtime::block_on(async {
            let remote = Remote {
                client: &client,
                token: "synthetic-test-token",
                origin: &mock.origin,
                project: &j.project_id,
            };
            let id = "b".repeat(32);
            assert_eq!(
                download_inner(&remote, &sandbox.root, &id)
                    .await
                    .err()
                    .unwrap(),
                "OPENECON_NETWORK"
            );
            let dir = broker::project_dir(&sandbox.root, &j.project_id).unwrap();
            assert!(dir.symlink_metadata("large.csv").is_err());
            let journal_dir = directory(&sandbox.root, &j.project_id).unwrap();
            let partial: DownloadJournal = load(&journal_dir, &format!("download-{id}.json"))
                .unwrap()
                .unwrap();
            assert_eq!(partial.completed.len(), 1);
            let ready = download_inner(&remote, &sandbox.root, &id).await.unwrap();
            assert!(!ready.reused);
            assert_eq!(mock.state.lock().unwrap().download_counts, vec![1, 2]);
            assert!(broker::cache_matches(&dir, "large.csv", j.size_bytes, &j.sha256).unwrap());
        });
    }

    #[test]
    fn chunk_transfer_busy_claims_stay_retryable_without_duplicate_parts_or_false_completion() {
        let sandbox = Sandbox::new();
        let path = sandbox.source();
        let j = journal(&path);
        let mock = Mock::new(j.clone(), false);
        let dir = directory(&sandbox.root, &j.project_id).unwrap();
        let client = client();
        mock.state.lock().unwrap().busy_part = true;
        mock.state.lock().unwrap().busy_complete = true;
        tauri::async_runtime::block_on(async {
            let remote = Remote {
                client: &client,
                token: "synthetic-test-token",
                origin: &mock.origin,
                project: &j.project_id,
            };
            assert!(matches!(
                resume_inner(&remote, &dir, j.clone()).await.unwrap(),
                UploadOutcome::Pending(_)
            ));
            let saved: UploadJournal = load(&dir, &format!("{}.json", j.request_id))
                .unwrap()
                .unwrap();
            assert!(matches!(
                resume_inner(&remote, &dir, saved).await.unwrap(),
                UploadOutcome::Pending(_)
            ));
            assert!(!mock.state.lock().unwrap().ready);
            let saved: UploadJournal = load(&dir, &format!("{}.json", j.request_id))
                .unwrap()
                .unwrap();
            assert!(matches!(
                resume_inner(&remote, &dir, saved).await.unwrap(),
                UploadOutcome::Complete(_)
            ));
            assert_eq!(mock.state.lock().unwrap().puts, vec![1, 1]);
        });
    }

    #[test]
    fn chunk_transfer_cancel_cleanup_and_active_resume_keep_durable_intent_until_retry() {
        let sandbox = Sandbox::new();
        let path = sandbox.source();
        let j = journal(&path);
        let mock = Mock::new(j.clone(), false);
        let dir = directory(&sandbox.root, &j.project_id).unwrap();
        let journal_name = format!("{}.json", j.request_id);
        let marker = format!("{}.cancel", j.request_id);
        save(&dir, &journal_name, &j).unwrap();
        let client = client();
        mock.state.lock().unwrap().cleanup_pending = true;
        tauri::async_runtime::block_on(async {
            let remote = Remote {
                client: &client,
                token: "synthetic-test-token",
                origin: &mock.origin,
                project: &j.project_id,
            };
            let first = cancel_inner(&remote, &dir, j.clone()).await.unwrap();
            assert_eq!(first.status, 202);
            assert_eq!(first.body["state"], "cancel_pending");
            assert!(dir.symlink_metadata(&marker).is_ok());
            let saved: UploadJournal = load(&dir, &journal_name).unwrap().unwrap();
            assert_eq!(saved.transfer_id, Some("a".repeat(32)));
            // An in-flight resume owns this lock and may hold its own journal
            // clone; acknowledged remote cleanup must retain the tombstone.
            let active = lock(&dir, &j.request_id).unwrap();
            let pending = cancel_inner(&remote, &dir, saved.clone()).await.unwrap();
            assert_eq!(pending.status, 202);
            assert!(load::<UploadJournal>(&dir, &journal_name)
                .unwrap()
                .is_some());
            drop(active);
            assert!(resume_inner(&remote, &dir, saved.clone())
                .await
                .err()
                .unwrap()
                .contains("waiting to be cancelled"));
            let done = cancel_inner(&remote, &dir, saved).await.unwrap();
            assert_eq!(done.status, 200);
            assert_eq!(done.body["cancelled"], true);
            assert!(load::<UploadJournal>(&dir, &journal_name)
                .unwrap()
                .is_none());
            assert!(dir.symlink_metadata(&marker).is_err());
            assert_eq!(mock.state.lock().unwrap().puts, vec![0, 0]);
            assert!(path.exists());
        });
    }

    #[test]
    fn chunk_transfer_cancel_refusals_preserve_source_and_reconcile_only_exact_ready_state() {
        let sandbox = Sandbox::new();
        let path = sandbox.source();
        let mut j = journal(&path);
        j.transfer_id = Some("a".repeat(32));
        let mock = Mock::new(j.clone(), false);
        let dir = directory(&sandbox.root, &j.project_id).unwrap();
        let journal_name = format!("{}.json", j.request_id);
        let marker = format!("{}.cancel", j.request_id);
        save(&dir, &journal_name, &j).unwrap();
        let client = client();
        tauri::async_runtime::block_on(async {
            let remote = Remote {
                client: &client,
                token: "synthetic-test-token",
                origin: &mock.origin,
                project: &j.project_id,
            };
            mock.state.lock().unwrap().denied = true;
            assert_eq!(
                cancel_inner(&remote, &dir, j.clone()).await.err().unwrap(),
                "OPENECON_HTTP:403"
            );
            assert!(dir.symlink_metadata(&marker).is_ok());
            mock.state.lock().unwrap().denied = false;
            mock.state.lock().unwrap().cancel_conflict = true;
            assert_eq!(
                cancel_inner(&remote, &dir, j.clone()).await.err().unwrap(),
                "OPENECON_HTTP:409"
            );
            assert!(dir.symlink_metadata(&marker).is_ok());
            mock.state.lock().unwrap().cancel_conflict = false;
            mock.state.lock().unwrap().ready = true;
            assert_eq!(
                cancel_inner(&remote, &dir, j.clone()).await.err().unwrap(),
                "OPENECON_TRANSFER_COMPLETE"
            );
            assert!(dir.symlink_metadata(&marker).is_err());
            let saved: UploadJournal = load(&dir, &journal_name).unwrap().unwrap();
            assert!(matches!(
                resume_inner(&remote, &dir, saved).await.unwrap(),
                UploadOutcome::Complete(_)
            ));
            assert!(path.exists());
            assert_eq!(mock.state.lock().unwrap().puts, vec![0, 0]);
        });
    }

    #[test]
    fn chunk_download_cancel_between_verified_parts_requires_membership_and_explicit_resume() {
        let sandbox = Sandbox::new();
        let path = sandbox.source();
        let j = journal(&path);
        let mock = Mock::new(j.clone(), false);
        let client = client();
        let (requested, release) = {
            let mut state = mock.state.lock().unwrap();
            state.pause_second_download = true;
            (state.second_requested.clone(), state.release_second.clone())
        };
        tauri::async_runtime::block_on(async {
            let owned_root = sandbox.root.try_clone().unwrap();
            let owned_client = client.clone();
            let origin = mock.origin.clone();
            let project = j.project_id.clone();
            let id = "b".repeat(32);
            let download = tauri::async_runtime::spawn(async move {
                let remote = Remote {
                    client: &owned_client,
                    token: "synthetic-test-token",
                    origin: &origin,
                    project: &project,
                };
                download_inner(&remote, &owned_root, &"b".repeat(32)).await
            });
            let deadline = std::time::Instant::now() + Duration::from_secs(10);
            while !requested.load(Ordering::Acquire) && std::time::Instant::now() < deadline {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            assert!(requested.load(Ordering::Acquire));
            let remote = Remote {
                client: &client,
                token: "synthetic-test-token",
                origin: &mock.origin,
                project: &j.project_id,
            };
            let intent = cancel_download_inner(&remote, &sandbox.root, &id)
                .await
                .unwrap();
            assert_eq!(intent.status, 200);
            assert_eq!(intent.body["cancel_requested"], true);
            release.store(true, Ordering::Release);
            assert_eq!(
                download.await.unwrap().err().unwrap(),
                "OPENECON_DOWNLOAD_CANCELLED"
            );
            let dir = broker::project_dir(&sandbox.root, &j.project_id).unwrap();
            assert!(dir.symlink_metadata("large.csv").is_err());
            let journal_dir = directory(&sandbox.root, &j.project_id).unwrap();
            let partial: DownloadJournal = load(&journal_dir, &format!("download-{id}.json"))
                .unwrap()
                .unwrap();
            assert_eq!(partial.completed.len(), 1);
            assert!(journal_dir
                .symlink_metadata(format!("download-{id}.cancel"))
                .is_ok());
            mock.state.lock().unwrap().denied = true;
            assert_eq!(
                cancel_download_inner(&remote, &sandbox.root, &id)
                    .await
                    .err()
                    .unwrap(),
                "OPENECON_HTTP:403"
            );
            assert_eq!(
                download_inner(&remote, &sandbox.root, &id)
                    .await
                    .err()
                    .unwrap(),
                "OPENECON_HTTP:403"
            );
            assert!(journal_dir
                .symlink_metadata(format!("download-{id}.cancel"))
                .is_ok());
            mock.state.lock().unwrap().denied = false;
            let ready = download_inner(&remote, &sandbox.root, &id).await.unwrap();
            assert!(!ready.reused);
            assert_eq!(mock.state.lock().unwrap().download_counts, vec![1, 2]);
            assert!(broker::cache_matches(&dir, "large.csv", j.size_bytes, &j.sha256).unwrap());
            assert!(journal_dir
                .symlink_metadata(format!("download-{id}.cancel"))
                .is_err());
        });
    }
}

fn validate_manifest(body: Value, file: &str) -> Result<Manifest, String> {
    let manifest: Manifest = serde_json::from_value(body).map_err(|_| INTEGRITY)?;
    if manifest.schema != SCHEMA
        || manifest.part_bytes != PART_BYTES
        || !lower_hex(&manifest.sha256, 64)
        || manifest.file["id"].as_str() != Some(file)
        || manifest.file["size_bytes"].as_u64() != Some(manifest.size_bytes)
        || manifest.file["data_hash"].as_str() != Some(manifest.sha256.as_str())
        || manifest.file["transfer"].as_str() != Some(SCHEMA)
        || !manifest.file["name"]
            .as_str()
            .is_some_and(broker::safe_filename)
        || !matches!(
            manifest.file["name"]
                .as_str()
                .and_then(|name| Path::new(name).extension().and_then(|x| x.to_str()))
                .map(str::to_ascii_lowercase)
                .as_deref(),
            Some("csv" | "parquet")
        )
        || manifest
            .file
            .get("python_path")
            .is_some_and(|p| p != &manifest.file["name"])
        || manifest.file.get("blob").is_some()
    {
        return Err(INTEGRITY.into());
    }
    validate_parts(manifest.size_bytes, &manifest.parts, true)?;
    Ok(manifest)
}

pub async fn download_chunked(
    client: &Client,
    root: &Dir,
    project: &str,
    file: &str,
    token: &str,
    expected: &CachedFile,
) -> Result<CachedFile, String> {
    if !lower_hex(project, 32) || !lower_hex(file, 32) {
        return Err("Invalid cloud file identifier.".into());
    }
    download_pinned_inner(
        &Remote {
            client,
            token,
            origin: broker::CLOUD_ORIGIN,
            project,
        },
        root,
        file,
        Some(expected),
    )
    .await
}

async fn cancel_download_inner(
    remote: &Remote<'_>,
    root: &Dir,
    file: &str,
) -> Result<CloudResponse, String> {
    // Cancelling local staging requires current read membership, rather than an
    // uploader/editor role. It never deletes the shared catalogue entry.
    validate_manifest(
        success(
            remote
                .json(Method::GET, &format!("/files/{file}/manifest"), None)
                .await?,
        )?,
        file,
    )?;
    let dir = directory(root, remote.project)?;
    let _intent = lock(&dir, &format!("download-intent-{file}"))?;
    save(
        &dir,
        &format!("download-{file}.cancel"),
        &json!({"cancel":true,"generation":uuid::Uuid::new_v4().simple().to_string()}),
    )?;
    Ok(CloudResponse {
        status: 200,
        body: json!({"cancel_requested":true,"file_id":file}),
    })
}

fn check_download_cancel(dir: &Dir, marker: &str) -> Result<(), String> {
    if dir.symlink_metadata(marker).is_ok() {
        return Err("OPENECON_DOWNLOAD_CANCELLED".into());
    }
    Ok(())
}

async fn cache_matches_async(
    dir: &Dir,
    name: &str,
    size: u64,
    digest: &str,
) -> Result<bool, String> {
    let dir = dir
        .try_clone()
        .map_err(|_| "The local cache could not be inspected.")?;
    let name = name.to_string();
    let digest = digest.to_string();
    tokio::task::spawn_blocking(move || broker::cache_matches(&dir, &name, size, &digest))
        .await
        .map_err(|_| "The local cache could not be inspected.")?
}

fn validate_expected_manifest(manifest: &Manifest, expected: &CachedFile) -> Result<(), String> {
    if manifest.file["id"].as_str() != Some(expected.cloud_id.as_str())
        || manifest.file["name"].as_str() != Some(expected.name.as_str())
        || manifest.size_bytes != expected.size_bytes
        || manifest.sha256 != expected.sha256
        || manifest.sha256 != expected.data_hash
    {
        return Err(INTEGRITY.into());
    }
    Ok(())
}

#[cfg(test)]
async fn download_inner(
    remote: &Remote<'_>,
    root: &Dir,
    file_id: &str,
) -> Result<CachedFile, String> {
    download_pinned_inner(remote, root, file_id, None).await
}

async fn download_pinned_inner(
    remote: &Remote<'_>,
    root: &Dir,
    file_id: &str,
    expected: Option<&CachedFile>,
) -> Result<CachedFile, String> {
    let journal_dir = directory(root, remote.project)?;
    let _lock = lock(&journal_dir, &format!("download-{file_id}"))?;
    let marker = format!("download-{file_id}.cancel");
    // A new explicit download may resume an earlier cancelled staging file.
    // Capture its marker before awaiting membership; a cancellation arriving
    // during that request has a new generation and must stop this invocation.
    let previous_cancel: Option<Value> = {
        let _intent = lock(&journal_dir, &format!("download-intent-{file_id}"))?;
        load(&journal_dir, &marker)?
    };
    // This fresh authenticated manifest is required even for a matching cache.
    let response = remote
        .json(Method::GET, &format!("/files/{file_id}/manifest"), None)
        .await?;
    let manifest = validate_manifest(success(response)?, file_id)?;
    if let Some(expected) = expected {
        validate_expected_manifest(&manifest, expected)?;
    }
    {
        let _intent = lock(&journal_dir, &format!("download-intent-{file_id}"))?;
        let current_cancel: Option<Value> = load(&journal_dir, &marker)?;
        if current_cancel != previous_cancel {
            return Err("OPENECON_DOWNLOAD_CANCELLED".into());
        }
        if previous_cancel.is_some() {
            journal_dir
                .remove_file(&marker)
                .map_err(|_| "The prior download cancellation could not be cleared.")?;
        }
    }
    let name = manifest.file["name"].as_str().unwrap();
    let dir = broker::project_dir(root, remote.project)?;
    let mut cached = CachedFile {
        cloud_id: file_id.into(),
        name: name.into(),
        python_path: name.into(),
        data_hash: manifest.sha256.clone(),
        sha256: manifest.sha256.clone(),
        size_bytes: manifest.size_bytes,
        reused: false,
    };
    if cache_matches_async(&dir, name, manifest.size_bytes, &manifest.sha256).await? {
        check_download_cancel(&journal_dir, &marker)?;
        cached.reused = true;
        return Ok(cached);
    }
    if dir.symlink_metadata(name).is_ok() {
        return Err(
            "The local filename contains different data. Preserve it before downloading again."
                .into(),
        );
    }
    let journal_name = format!("download-{file_id}.json");
    let temporary = format!(".download-chunks-{file_id}");
    let mut j: DownloadJournal = match load(&journal_dir, &journal_name)? {
        Some(j) => j,
        None => DownloadJournal {
            schema: SCHEMA.into(),
            project_id: remote.project.into(),
            file_id: file_id.into(),
            size_bytes: manifest.size_bytes,
            sha256: manifest.sha256.clone(),
            completed: Vec::new(),
        },
    };
    if j.schema != SCHEMA
        || j.project_id != remote.project
        || j.file_id != file_id
        || j.size_bytes != manifest.size_bytes
        || j.sha256 != manifest.sha256
    {
        return Err(INTEGRITY.into());
    }
    validate_parts(j.size_bytes, &j.completed, false)?;
    let mut options = OpenOptions::new();
    options
        .read(true)
        .write(true)
        .create(true)
        .follow(FollowSymlinks::No);
    let mut file = dir
        .open_with(&temporary, &options)
        .map_err(|_| "A safe partial download could not be opened.")?;
    if !file.metadata().map_err(|_| INTEGRITY)?.is_file() {
        return Err(INTEGRITY.into());
    }
    if !j.completed.is_empty() && file.metadata().map_err(|_| INTEGRITY)?.len() != j.size_bytes {
        return Err(INTEGRITY.into());
    }
    file.set_len(j.size_bytes)
        .map_err(|_| "The partial download could not be sized.")?;
    for part in &j.completed {
        check_download_cancel(&journal_dir, &marker)?;
        if *part != manifest.parts[part.index] {
            return Err(INTEGRITY.into());
        }
        file.seek(SeekFrom::Start(part.index as u64 * PART_BYTES as u64))
            .map_err(|_| INTEGRITY)?;
        let mut bytes = vec![0; part.size_bytes as usize];
        file.read_exact(&mut bytes).map_err(|_| INTEGRITY)?;
        if hash(&bytes) != part.sha256 {
            return Err(INTEGRITY.into());
        }
        tokio::task::yield_now().await;
    }
    save(&journal_dir, &journal_name, &j)?;
    for part in &manifest.parts {
        check_download_cancel(&journal_dir, &marker)?;
        if j.completed.iter().any(|p| p.index == part.index) {
            continue;
        }
        let mut response = remote
            .builder(
                Method::GET,
                &format!("/files/{file_id}/parts/{}", part.index),
            )?
            .send()
            .await
            .map_err(|_| "OPENECON_NETWORK")?;
        if !response.status().is_success() {
            return Err(format!("OPENECON_HTTP:{}", response.status().as_u16()));
        }
        if response
            .content_length()
            .is_some_and(|n| n != part.size_bytes)
            || response
                .headers()
                .get("X-OpenEcon-SHA256")
                .and_then(|h| h.to_str().ok())
                != Some(part.sha256.as_str())
        {
            return Err(INTEGRITY.into());
        }
        let mut bytes = Vec::with_capacity(part.size_bytes as usize);
        while let Some(chunk) = response.chunk().await.map_err(|_| "OPENECON_NETWORK")? {
            check_download_cancel(&journal_dir, &marker)?;
            if bytes.len() + chunk.len() > part.size_bytes as usize {
                return Err(INTEGRITY.into());
            }
            bytes.extend_from_slice(&chunk);
        }
        check_download_cancel(&journal_dir, &marker)?;
        if bytes.len() as u64 != part.size_bytes || hash(&bytes) != part.sha256 {
            return Err(INTEGRITY.into());
        }
        file.seek(SeekFrom::Start(part.index as u64 * PART_BYTES as u64))
            .map_err(|_| INTEGRITY)?;
        file.write_all(&bytes)
            .map_err(|_| "The data part could not be saved.")?;
        file.sync_all()
            .map_err(|_| "The data part could not be saved.")?;
        j.completed.push(part.clone());
        save(&journal_dir, &journal_name, &j)?;
    }
    check_download_cancel(&journal_dir, &marker)?;
    if !cache_matches_async(&dir, &temporary, manifest.size_bytes, &manifest.sha256).await? {
        return Err(INTEGRITY.into());
    }
    // Refresh membership after downloading and before installing the local cache.
    let final_manifest = validate_manifest(
        success(
            remote
                .json(Method::GET, &format!("/files/{file_id}/manifest"), None)
                .await?,
        )?,
        file_id,
    )?;
    if let Some(expected) = expected {
        validate_expected_manifest(&final_manifest, expected)?;
    }
    if final_manifest.sha256 != manifest.sha256
        || final_manifest.parts != manifest.parts
        || final_manifest.file["name"] != name
    {
        return Err(INTEGRITY.into());
    }
    drop(file);
    // Create the final directory entry without clobbering any existing user file.
    // Cancellation and final installation share this short local lock, so an
    // acknowledged cancellation cannot race between its last check and install.
    let _intent = lock(&journal_dir, &format!("download-intent-{file_id}"))?;
    check_download_cancel(&journal_dir, &marker)?;
    dir.hard_link(&temporary, &dir, name)
        .map_err(|_| "The destination appeared during download; existing data was preserved.")?;
    let _ = dir.remove_file(&temporary);
    let _ = journal_dir.remove_file(&journal_name);
    Ok(cached)
}
