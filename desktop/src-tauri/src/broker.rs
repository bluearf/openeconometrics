//! The webview can request a small fixed API surface. URLs and credentials never
//! pass to the Python runtime. Redirects and environment proxy settings are disabled.
use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
use cap_std::fs::{Dir, OpenOptions};
use reqwest::{Client, Method};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    io::{Read, Write},
    path::Path,
    time::Duration,
};

pub const CLOUD_ORIGIN: &str = "https://openecon-291739190496.us-central1.run.app";
pub const MAX_FILE_BYTES: u64 = 24 * 1024 * 1024;
const MAX_REQUEST_BYTES: usize = 128 * 1024;
const MAX_JSON_BYTES: usize = 10 * 1024 * 1024;

#[derive(Serialize)]
pub struct CloudResponse {
    pub status: u16,
    pub body: Value,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CachedFile {
    pub cloud_id: String,
    pub name: String,
    pub python_path: String,
    pub data_hash: String,
    pub sha256: String,
    pub size_bytes: u64,
    pub reused: bool,
}

pub fn client() -> Result<Client, String> {
    Client::builder()
        .https_only(true)
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(Duration::from_secs(15))
        .timeout(Duration::from_secs(60))
        .build()
        .map_err(|_| "The secure cloud connection could not start.".into())
}

pub fn identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-')
}

pub fn hex_id(value: &str, length: usize) -> bool {
    value.len() == length && value.bytes().all(|c| c.is_ascii_hexdigit())
}

fn account_grant_id(value: &str) -> bool {
    hex_id(value, 32) && !value.bytes().any(|c| c.is_ascii_uppercase())
}

/// Only the history read endpoint accepts a bounded, named query string.
/// Path encodings, ambiguous normalization and compute endpoints stay rejected.
pub fn allowed_route(method: &str, path: &str) -> bool {
    if let Some((route, query)) = path.split_once('?') {
        let parts: Vec<&str> = route.split('/').collect();
        if method != "GET" || path.len() > 8192 || path.contains('#') || path.contains('\\')
            || !matches!(parts.as_slice(), ["", "api", "projects", project, "workspace", "console", "history"] if identifier(project))
            || query.is_empty() || route.contains('%') {
            return false;
        }
        let mut keys = std::collections::HashSet::new();
        return url::form_urlencoded::parse(query.as_bytes()).all(|(key, value)| {
            if !keys.insert(key.to_string()) || value.chars().any(char::is_control) { return false; }
            match key.as_ref() {
                "query" => value.chars().count() <= 256,
                "cursor" => value.len() <= 2048 && !value.is_empty()
                    && value.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-'),
                "since" | "until" => value.is_empty() || (value.len() == 10
                    && value.bytes().enumerate().all(|(i, c)| if i == 4 || i == 7 { c == b'-' } else { c.is_ascii_digit() })),
                "limit" => value.parse::<u32>().is_ok_and(|v| (1..=50).contains(&v)),
                "max_bytes" => value.parse::<u32>().is_ok_and(|v| (4096..=262144).contains(&v)),
                _ => false,
            }
        });
    }
    if path.len() > 1024
        || path.contains("console/execute")
        || path.contains('%')
        || path.contains('?')
        || path.contains('#')
        || path.contains('\\')
    {
        return false;
    }
    let p: Vec<&str> = path.split('/').collect();
    if p.first() != Some(&"") || p.get(1) != Some(&"api") {
        return false;
    }
    if p.iter().skip(2).any(|part| !identifier(part)) {
        return false;
    }
    match p.as_slice() {
        ["", "api", "auth", "config"] | ["", "api", "me"] => method == "GET",
        ["", "api", "desktop", "login"] => method == "POST",
        ["", "api", "desktop", "login", request] => {
            method == "GET" && account_grant_id(request)
        }
        ["", "api", "desktop", "login", request, "exchange"] => {
            method == "POST" && hex_id(request, 32)
        }
        ["", "api", "desktop", "account-link"] => method == "POST",
        ["", "api", "desktop", "account-link", request] => {
            method == "DELETE" && account_grant_id(request)
        }
        ["", "api", "desktop", "account-link", request, "exchange"] => {
            method == "POST" && account_grant_id(request)
        }
        ["", "api", "projects"] => matches!(method, "GET" | "POST"),
        ["", "api", "projects", _] => method == "PATCH",
        ["", "api", "invitations", _, "accept"] => method == "POST",
        ["", "api", "projects", _, "members"] => method == "GET",
        ["", "api", "projects", _, "members", _] => matches!(method, "PATCH" | "DELETE"),
        ["", "api", "projects", _, "invitations"] => method == "POST",
        ["", "api", "projects", _, "invitations", _] => method == "DELETE",
        ["", "api", "projects", _, "workspace", "session" | "bootstrap" | "config" | "datasets" | "console"] => {
            method == "GET"
        }
        ["", "api", "projects", _, "workspace", "console", "script"] => {
            matches!(method, "GET" | "PUT")
        }
        ["", "api", "projects", _, "workspace", "console", "history"] => method == "GET",
        ["", "api", "projects", _, "workspace", "runs", _, "record"] => method == "GET",
        ["", "api", "projects", _, "workspace", "console", "scripts"] => {
            matches!(method, "GET" | "POST")
        }
        ["", "api", "projects", _, "workspace", "console", "scripts", id] => {
            matches!(method, "GET" | "PUT")
                && (*id == "analysis"
                    || (id.len() == 32
                        && id.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))))
        }
        ["", "api", "projects", _, "workspace", "environment"] => {
            matches!(method, "GET" | "PUT")
        }
        ["", "api", "projects", _, "workspace", "files", "layout"] => {
            matches!(method, "GET" | "PUT")
        }
        ["", "api", "projects", _, "workspace", "datasets", "example"] => method == "POST",
        ["", "api", "projects", _, "workspace", "desktop", "results"] => method == "POST",
        _ => false,
    }
}

fn authorize(
    builder: reqwest::RequestBuilder,
    token: Option<&str>,
    public: bool,
) -> Result<reqwest::RequestBuilder, String> {
    if public && token.is_none_or(str::is_empty) {
        return Ok(builder);
    }
    let token = token.ok_or("Sign in to your OpenEconometrics account.")?;
    if token.is_empty() || token.len() > 8192 || token.chars().any(char::is_control) {
        return Err("The sign-in credential is invalid.".into());
    }
    Ok(builder.bearer_auth(token))
}

async fn json_response(mut response: reqwest::Response) -> Result<CloudResponse, String> {
    let status = response.status().as_u16();
    if response
        .content_length()
        .is_some_and(|size| size > MAX_JSON_BYTES as u64)
    {
        return Err("The cloud response exceeds the desktop limit.".into());
    }
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(|_| {
        if (200..300).contains(&status) {
            "OPENECON_NETWORK".to_string()
        } else {
            format!("OPENECON_HTTP:{status}")
        }
    })? {
        if bytes.len() + chunk.len() > MAX_JSON_BYTES {
            return Err("The cloud response exceeds the desktop limit.".into());
        }
        bytes.extend_from_slice(&chunk);
    }
    // Preserve non-success HTTP status even if a gateway returns plain text.
    // Do not expose that response text or mistake an authorization failure for
    // a lost connection when deciding whether cached files may be opened.
    let body = match serde_json::from_slice(&bytes) {
        Ok(body) => body,
        Err(_) if !(200..300).contains(&status) => Value::Null,
        Err(_) => return Err("The cloud returned an invalid JSON response.".into()),
    };
    Ok(CloudResponse { status, body })
}

/// Authenticated account linking is distinct from public sign-in exchanges.
/// The native link bridge only accepts a target/challenge or a verifier; it
/// cannot carry a password, Google credential, caller-selected UID or token.
fn auth_request_policy(method: &str, path: &str, body: Option<&Value>) -> Result<bool, String> {
    let parts: Vec<&str> = path.split('/').collect();
    let (key, public, target) = match (method, parts.as_slice()) {
        ("POST", ["", "api", "desktop", "login"]) => ("challenge", true, false),
        ("POST", ["", "api", "desktop", "login", _, "exchange"]) => ("verifier", true, false),
        ("POST", ["", "api", "desktop", "account-link"]) => ("challenge", false, true),
        ("POST", ["", "api", "desktop", "account-link", _, "exchange"]) => ("verifier", false, false),
        ("GET", ["", "api", "desktop", "login", _])
        | ("DELETE", ["", "api", "desktop", "account-link", _]) => {
            if body.is_some() { return Err("This account operation does not accept a body.".into()); }
            return Ok(false);
        }
        _ => return Ok(false),
    };
    let object = body.and_then(Value::as_object).ok_or("Invalid account request.")?;
    let valid_proof = object.get(key).and_then(Value::as_str).is_some_and(|value| {
        value.len() == 64 && value.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    });
    if object.len() != (if target { 2 } else { 1 }) || !valid_proof
        || (target && !object.get("target").and_then(Value::as_str).is_some_and(|value| matches!(value, "google.com" | "password"))) {
        return Err("Invalid account request.".into());
    }
    Ok(public)
}

pub async fn request(
    client: &Client,
    method: &str,
    path: &str,
    token: Option<&str>,
    body: Option<Value>,
) -> Result<CloudResponse, String> {
    if !allowed_route(method, path) {
        return Err("This cloud operation is not available in the desktop app.".into());
    }
    if matches!(method, "GET" | "DELETE") && body.is_some() {
        return Err("This cloud operation does not accept a body.".into());
    }
    let public_login = auth_request_policy(method, path, body.as_ref())?;
    let method = Method::from_bytes(method.as_bytes()).map_err(|_| "Invalid cloud method.")?;
    let mut builder = authorize(
        client.request(method, format!("{CLOUD_ORIGIN}{path}")),
        token,
        path == "/api/auth/config" || public_login,
    )?;
    if let Some(value) = body {
        let bytes = serde_json::to_vec(&value).map_err(|_| "Invalid cloud request body.")?;
        let maximum = if path.ends_with("/workspace/desktop/results") {
            3 * 1024 * 1024
        } else if path.ends_with("/workspace/files/layout") {
            1024 * 1024
        } else {
            MAX_REQUEST_BYTES
        };
        if bytes.len() > maximum {
            return Err("The cloud request exceeds the desktop limit.".into());
        }
        builder = builder
            .header("Content-Type", "application/json")
            .body(bytes);
    }
    json_response(builder.send().await.map_err(|_| "OPENECON_NETWORK")?).await
}

pub fn safe_filename(name: &str) -> bool {
    let extension = Path::new(name)
        .extension()
        .and_then(|v| v.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if !matches!(extension.as_str(), "csv" | "parquet" | "xlsx" | "dta")
        || name.is_empty()
        || name.len() > 180
        || name.starts_with('.')
        || name.ends_with(' ')
        || name.chars().any(|c| {
            c.is_control() || matches!(c, '/' | '\\' | ':' | '<' | '>' | '"' | '|' | '?' | '*')
        })
    {
        return false;
    }
    // Windows device names are reserved even with an extension.
    let stem = name.split('.').next().unwrap_or("").to_ascii_uppercase();
    !matches!(
        stem.as_str(),
        "CON"
            | "PRN"
            | "AUX"
            | "NUL"
            | "COM1"
            | "COM2"
            | "COM3"
            | "COM4"
            | "COM5"
            | "COM6"
            | "COM7"
            | "COM8"
            | "COM9"
            | "LPT1"
            | "LPT2"
            | "LPT3"
            | "LPT4"
            | "LPT5"
            | "LPT6"
            | "LPT7"
            | "LPT8"
            | "LPT9"
    )
}

fn ensure_dir(parent: &Dir, name: &str) -> Result<Dir, String> {
    match parent.create_dir(name) {
        Ok(()) => (),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => (),
        Err(_) => return Err("The local project folder could not be created.".into()),
    }
    parent
        .open_dir_nofollow(name)
        .map_err(|_| "The project folder is not a safe directory.".into())
}

pub fn project_dir(root: &Dir, project: &str) -> Result<Dir, String> {
    if !identifier(project) {
        return Err("Invalid project identifier.".into());
    }
    let projects = ensure_dir(root, "projects")?;
    ensure_dir(&projects, project)
}

pub fn cache_matches(dir: &Dir, name: &str, size: u64, hash: &str) -> Result<bool, String> {
    let mut options = OpenOptions::new();
    options.read(true).follow(FollowSymlinks::No);
    let mut file = match dir.open_with(name, &options) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(_) => return Err("The cached file is not a safe regular file.".into()),
    };
    let metadata = file
        .metadata()
        .map_err(|_| "The cached file could not be inspected.")?;
    if !metadata.is_file() {
        return Err("The cached file is not a regular file.".into());
    }
    if metadata.len() != size {
        return Ok(false);
    }
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 65536];
    let mut total = 0u64;
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|_| "The cached file could not be read.")?;
        if count == 0 {
            break;
        }
        total += count as u64;
        if total > size {
            return Ok(false);
        }
        digest.update(&buffer[..count]);
    }
    Ok(total == size && format!("{:x}", digest.finalize()) == hash)
}

fn pinned_metadata(body: &Value, file_id: &str) -> Result<CachedFile, String> {
    let item = body
        .get("datasets")
        .and_then(Value::as_array)
        .and_then(|items| {
            items
                .iter()
                .find(|item| item.get("id").and_then(Value::as_str) == Some(file_id))
        })
        .ok_or("This cloud file is no longer available.")?;
    let name = item
        .get("name")
        .and_then(Value::as_str)
        .ok_or("Missing cloud filename.")?;
    let hash = item
        .get("data_hash")
        .and_then(Value::as_str)
        .ok_or("Missing cloud SHA-256.")?;
    let size = item
        .get("size_bytes")
        .and_then(Value::as_u64)
        .ok_or("Missing cloud file size.")?;
    if !safe_filename(name)
        || hash.len() != 64
        || !hash.bytes().all(|c| c.is_ascii_hexdigit())
        || size > if item.get("transfer").and_then(Value::as_str) == Some("chunked-v1") {
            crate::data_transfer::MAX_FILE_BYTES
        } else { MAX_FILE_BYTES }
        || (item.get("transfer").and_then(Value::as_str) == Some("chunked-v1")
            && (!matches!(Path::new(name).extension().and_then(|x| x.to_str()).map(str::to_ascii_lowercase).as_deref(), Some("csv" | "parquet")) || size == 0))
    {
        return Err("The cloud file metadata is invalid.".into());
    }
    Ok(CachedFile {
        cloud_id: file_id.into(),
        name: name.into(),
        python_path: name.into(),
        data_hash: hash.to_ascii_lowercase(),
        sha256: hash.to_ascii_lowercase(),
        size_bytes: size,
        reused: false,
    })
}

pub async fn download(
    client: &Client,
    root: &Dir,
    project: &str,
    file_id: &str,
    token: &str,
) -> Result<CachedFile, String> {
    if !identifier(project) || !identifier(file_id) {
        return Err("Invalid cloud file identifier.".into());
    }
    // Metadata is fetched from the authenticated project listing, never trusted
    // from JavaScript. The immutable object's content must match this pinned hash.
    let listing = file_listing(client, project, token).await?;
    // Sequential UI downloads retain the same whole-project geometry check as
    // the batch command; an individual call cannot bypass its resource budget.
    let metadata = selected_project_metadata(&listing, file_id)?;
    if chunked_file(&listing, file_id) {
        return crate::data_transfer::download_chunked(client, root, project, file_id, token, &metadata).await;
    }
    download_metadata(client, root, project, metadata, token).await
}

pub async fn download_all(
    client: &Client,
    root: &Dir,
    project: &str,
    token: &str,
) -> Result<Vec<CachedFile>, String> {
    if !identifier(project) {
        return Err("Invalid project identifier.".into());
    }
    let listing = file_listing(client, project, token).await?;
    let items = project_metadata(&listing)?;
    let mut files = Vec::with_capacity(items.len());
    for metadata in items {
        if chunked_file(&listing, &metadata.cloud_id) {
            files.push(crate::data_transfer::download_chunked(client, root, project, &metadata.cloud_id, token, &metadata).await?);
        } else {
            files.push(download_metadata(client, root, project, metadata, token).await?);
        }
    }
    Ok(files)
}

async fn file_listing(client: &Client, project: &str, token: &str) -> Result<Value, String> {
    let response = authorize(
        client.get(format!(
            "{CLOUD_ORIGIN}/api/projects/{project}/workspace/datasets"
        )),
        Some(token),
        false,
    )?
    .send()
    .await
    .map_err(|_| "OPENECON_NETWORK")?;
    file_listing_response(response).await
}

async fn file_listing_response(response: reqwest::Response) -> Result<Value, String> {
    // Check authorization before reading a body, including truncated gateway
    // responses. A known denied status must never become an offline fallback.
    if response.status().as_u16() != 200 {
        return Err(format!("OPENECON_HTTP:{}", response.status().as_u16()));
    }
    Ok(json_response(response).await?.body)
}

fn chunked_file(body: &Value, id: &str) -> bool {
    body.get("datasets").and_then(Value::as_array).is_some_and(|items| items.iter().any(|item|
        item.get("id").and_then(Value::as_str) == Some(id)
        && item.get("transfer").and_then(Value::as_str) == Some("chunked-v1")))
}

fn project_metadata(body: &Value) -> Result<Vec<CachedFile>, String> {
    let items = body
        .get("datasets")
        .and_then(Value::as_array)
        .ok_or("The cloud file listing is invalid.")?;
    if items.len() > 20 {
        return Err("The project exceeds the desktop limit of 20 input files.".into());
    }
    let mut files = Vec::with_capacity(items.len());
    let mut size = 0u64;
    let mut legacy_size = 0u64;
    let mut names = std::collections::HashSet::new();
    let mut ids = std::collections::HashSet::new();
    for item in items {
        let id = item
            .get("id")
            .and_then(Value::as_str)
            .ok_or("The cloud file listing is invalid.")?;
        if !hex_id(id, 32) || !ids.insert(id.to_string()) {
            return Err("The cloud file identifier is invalid.".into());
        }
        let metadata = pinned_metadata(body, id)?;
        size = size
            .checked_add(metadata.size_bytes)
            .ok_or("The project file size is invalid.")?;
        if !chunked_file(body, id) {
            legacy_size = legacy_size.checked_add(metadata.size_bytes).ok_or("The project file size is invalid.")?;
        }
        if size > crate::data_transfer::MAX_PROJECT_BYTES || legacy_size > 64 * 1024 * 1024 {
            return Err("The project exceeds 8 GiB of shared data or 64 MiB of legacy input data.".into());
        }
        if !names.insert(metadata.name.to_lowercase()) {
            return Err("The project contains conflicting input filenames.".into());
        }
        files.push(metadata);
    }
    Ok(files)
}

fn selected_project_metadata(body: &Value, id: &str) -> Result<CachedFile, String> {
    project_metadata(body)?.into_iter().find(|file| file.cloud_id == id)
        .ok_or_else(|| "This cloud file is no longer available.".into())
}

async fn download_metadata(
    client: &Client,
    root: &Dir,
    project: &str,
    mut metadata: CachedFile,
    token: &str,
) -> Result<CachedFile, String> {
    let file_id = &metadata.cloud_id;
    let dir = project_dir(root, project)?;
    if cache_matches(
        &dir,
        &metadata.name,
        metadata.size_bytes,
        &metadata.data_hash,
    )? {
        metadata.reused = true;
        return Ok(metadata);
    }
    let builder = authorize(
        client.get(format!(
            "{CLOUD_ORIGIN}/api/projects/{project}/workspace/files/{file_id}/download"
        )),
        Some(token),
        false,
    )?;
    let mut response = builder.send().await.map_err(|_| "OPENECON_NETWORK")?;
    if !response.status().is_success() {
        return Err(format!("OPENECON_HTTP:{}", response.status().as_u16()));
    }
    if response
        .content_length()
        .is_some_and(|size| size != metadata.size_bytes || size > MAX_FILE_BYTES)
    {
        return Err("The downloaded file size does not match its metadata.".into());
    }
    let temporary = format!(".download-{}", uuid::Uuid::new_v4());
    let mut options = OpenOptions::new();
    options
        .write(true)
        .create_new(true)
        .follow(FollowSymlinks::No);
    let mut file = dir
        .open_with(&temporary, &options)
        .map_err(|_| "A safe download file could not be created.")?;
    let result: Result<(), String> = async {
        let mut digest = Sha256::new();
        let mut total = 0u64;
        while let Some(chunk) = response.chunk().await.map_err(|_| "OPENECON_NETWORK")? {
            total += chunk.len() as u64;
            if total > metadata.size_bytes || total > MAX_FILE_BYTES {
                return Err("The downloaded file exceeds its pinned size.".into());
            }
            digest.update(&chunk);
            file.write_all(&chunk)
                .map_err(|_| "The downloaded file could not be saved.")?;
        }
        if total != metadata.size_bytes || format!("{:x}", digest.finalize()) != metadata.data_hash
        {
            return Err("The downloaded file failed SHA-256 verification.".into());
        }
        file.sync_all()
            .map_err(|_| "The downloaded file could not be saved.")?;
        drop(file);
        if let Ok(existing) = dir.symlink_metadata(&metadata.name) {
            if !existing.is_file() || existing.file_type().is_symlink() {
                return Err("The destination is not a safe regular file.".into());
            }
        }
        dir.rename(&temporary, &dir, &metadata.name)
            .map_err(|_| "The verified download could not be installed.")?;
        Ok(())
    }
    .await;
    if result.is_err() {
        let _ = dir.remove_file(&temporary);
    }
    result?;
    Ok(metadata)
}

fn local_workspace(origin: &str, project: &str) -> Result<String, String> {
    let url = url::Url::parse(origin).map_err(|_| "The local workspace origin is invalid.")?;
    if url.scheme() != "http"
        || !matches!(url.host_str(), Some("127.0.0.1" | "localhost" | "[::1]" | "::1"))
        || url.port().is_none_or(|port| port < 1024)
        || !url.username().is_empty() || url.password().is_some()
        || url.query().is_some() || url.fragment().is_some() || url.path() != "/"
        || !hex_id(project, 32)
    {
        return Err("The local workspace origin or project is invalid.".into());
    }
    Ok(format!("{}/api/desktop/projects/{project}/workspace", url.origin().ascii_serialization()))
}

fn editable_session(body: &Value) -> bool {
    body.get("read_only").and_then(Value::as_bool) == Some(false)
}

fn editable_local_state(body: &Value) -> bool {
    matches!(body.get("role").and_then(Value::as_str), Some("owner" | "editor"))
        && matches!(body.get("access_denied"), None | Some(Value::Bool(false)))
}

fn local_upload(size: u64, name: &str) -> Result<bool, String> {
    if size <= MAX_FILE_BYTES {
        return Ok(false);
    }
    let extension = Path::new(name).extension().and_then(|suffix| suffix.to_str()).unwrap_or("").to_ascii_lowercase();
    if !matches!(extension.as_str(), "csv" | "parquet") {
        return Err("Convert large XLSX/DTA files to CSV or Parquet before importing locally.".into());
    }
    Ok(true)
}

async fn local_import(project: &str, origin: &str, file: tokio::fs::File,
                      size: u64, name: &str) -> Result<CloudResponse, String> {
    let workspace = local_workspace(origin, project)?;
    let client = Client::builder().no_proxy().redirect(reqwest::redirect::Policy::none())
        .connect_timeout(Duration::from_secs(15)).build()
        .map_err(|_| "The local connection could not start.")?;
    let session = json_response(client.get(format!("{workspace}/session")).send().await
        .map_err(|_| "The local session could not be opened.")?).await?;
    if session.status != 200 {
        return Err("The local project session is unavailable.".into());
    }
    let token = session.body.get("token").and_then(Value::as_str)
        .filter(|value| !value.is_empty() && value.len() <= 8192 && !value.chars().any(char::is_control))
        .ok_or("The local project token is invalid.")?;
    let saved = json_response(client.get(format!("{workspace}/desktop-sync-state"))
        .header("X-OpenEcon-Token", token).send().await
        .map_err(|_| "The local project state is unavailable.")?).await?;
    if saved.status != 200 || !editable_local_state(&saved.body) {
        return Err("You do not have permission to edit this local project.".into());
    }
    let stream = tokio_util::io::ReaderStream::with_capacity(tokio::io::AsyncReadExt::take(file, size), 65536);
    let part = reqwest::multipart::Part::stream_with_length(reqwest::Body::wrap_stream(stream), size)
        .file_name(name.to_string());
    // Only the fixed loopback endpoint receives the per-project local token.
    // A cloud bearer is never passed to the Python runtime.
    let response = json_response(client.post(format!("{workspace}/datasets/upload"))
        .header("Origin", origin).header("X-OpenEcon-Token", token)
        .multipart(reqwest::multipart::Form::new().part("file", part))
        .send().await.map_err(|_| "The local file could not be imported.")?).await?;
    if (200..300).contains(&response.status) &&
       (response.body.get("local_only").and_then(Value::as_bool) != Some(true)
        || response.body.get("id").and_then(Value::as_str).is_none_or(|id| uuid::Uuid::parse_str(id).is_err())) {
        return Err("The local-only dataset record could not be verified.".into());
    }
    Ok(response)
}

pub async fn upload(client: &Client, root: &Dir, project: &str, token: &str, local_origin: &str) -> Result<CloudResponse, String> {
    if !identifier(project) {
        return Err("Invalid project identifier.".into());
    }
    let choice = rfd::AsyncFileDialog::new()
        .set_title("Import project data")
        .add_filter("Data files", &["csv", "parquet", "xlsx", "dta"])
        .pick_file()
        .await;
    let Some(choice) = choice else {
        return Ok(CloudResponse {
            status: 204,
            body: serde_json::json!({"cancelled": true}),
        });
    };
    let name = choice.file_name();
    if !safe_filename(&name) {
        return Err("Choose a CSV, Parquet, XLSX, or DTA file with a safe name.".into());
    }
    let file = tokio::fs::File::open(choice.path())
        .await
        .map_err(|_| "The selected file could not be read.")?;
    let info = file
        .metadata()
        .await
        .map_err(|_| "The selected file could not be inspected.")?;
    if !info.is_file() {
        return Err("Choose a regular data file.".into());
    }
    if local_upload(info.len(), &name)? {
        // Fresh authorization checks distinguish a denied project from network
        // loss. Importing never falls back after an explicit cloud denial.
        let permission = request(client, "GET", &format!("/api/projects/{project}/workspace/session"), Some(token), None).await?;
        if permission.status != 200 {
            return Ok(permission);
        }
        if !editable_session(&permission.body) {
            return Err("You do not have permission to edit this project.".into());
        }
        if info.len() > crate::data_transfer::MAX_FILE_BYTES {
            // Larger physical files remain usable locally under the existing
            // local importer; the 2 GiB cap applies only to shared transfers.
            return local_import(project, local_origin, file, info.len(), &name).await;
        }
        match crate::data_transfer::upload_selected(client, root, project, choice.path(), token).await? {
            crate::data_transfer::UploadOutcome::Complete(body) => {
                // Copy the selected source into a verified managed cache without a second
                // network download. Cloud credentials never enter local Python imports.
                crate::data_transfer::cache_uploaded(root, project, choice.path(), &body).await?;
                return Ok(CloudResponse { status: 201, body });
            },
            crate::data_transfer::UploadOutcome::Pending(transfer) => {
                let mut imported = local_import(project, local_origin, file, info.len(), &name).await?;
                if let Some(object) = imported.body.as_object_mut() {
                    object.insert("sharing_pending_transfer".into(), serde_json::to_value(transfer).map_err(|_| "The transfer status could not be saved.")?);
                }
                return Ok(imported);
            },
            crate::data_transfer::UploadOutcome::Unsupported => {
                return local_import(project, local_origin, file, info.len(), &name).await;
            },
        }
    }
    let stream = tokio_util::io::ReaderStream::new(tokio::io::AsyncReadExt::take(file, info.len()));
    let part = reqwest::multipart::Part::stream_with_length(
        reqwest::Body::wrap_stream(stream),
        info.len(),
    )
    .file_name(name);
    let builder = authorize(
        client.post(format!(
            "{CLOUD_ORIGIN}/api/projects/{project}/workspace/datasets/upload"
        )),
        Some(token),
        false,
    )?;
    json_response(
        builder
            .multipart(reqwest::multipart::Form::new().part("file", part))
            .send()
            .await
            .map_err(|_| "OPENECON_NETWORK")?,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_model_json_preserves_float_bits_across_request_and_response() {
        // Values observed in an actual saved HC3 model's interval, predictions
        // and p-value. The native JSON broker must not round these a second time.
        let raw = "[1.8183335056294878,44.726301815618164,-1.5466964797787845,1.4552615370818851e-180]";
        let expected: [f64; 4] = [1.8183335056294878, 44.726301815618164,
                                 -1.5466964797787845, 1.4552615370818851e-180];
        let mut value: Value = serde_json::from_str(raw).unwrap();
        for _ in 0..3 {
            for (index, original) in expected.iter().enumerate() {
                assert_eq!(value[index].as_f64().unwrap().to_bits(), original.to_bits());
            }
            value = serde_json::from_slice(&serde_json::to_vec(&value).unwrap()).unwrap();
        }
    }

    #[test]
    fn account_link_routes_are_canonical_and_do_not_allow_native_completion() {
        let id = "a".repeat(32);
        for (method, path) in [
            ("POST", "/api/desktop/account-link".to_string()),
            ("GET", format!("/api/desktop/login/{id}")),
            ("DELETE", format!("/api/desktop/account-link/{id}")),
            ("POST", format!("/api/desktop/account-link/{id}/exchange")),
        ] {
            assert!(allowed_route(method, &path));
            for wrong in ["GET", "POST", "PUT", "PATCH", "DELETE"] {
                if wrong != method { assert!(!allowed_route(wrong, &path)); }
            }
        }
        for path in [
            format!("/api/desktop/account-link/{id}/complete"),
            format!("/api/desktop/account-link/{id}/authorize"),
            format!("/api/desktop/account-link/{id}/exchange?host=evil"),
            format!("/api/desktop/account-link/{}/exchange", "A".repeat(32)),
            "/api/desktop/account-link/../exchange".to_string(),
            "/api/desktop/account-link/short/exchange".to_string(),
        ] {
            for method in ["GET", "POST", "DELETE"] { assert!(!allowed_route(method, &path)); }
        }
    }

    #[test]
    fn account_link_policy_requires_auth_and_never_accepts_credentials_or_uid() {
        let proof = "b".repeat(64);
        let id = "a".repeat(32);
        for target in ["google.com", "password"] {
            let body = serde_json::json!({"challenge": proof, "target": target});
            assert_eq!(auth_request_policy("POST", "/api/desktop/account-link", Some(&body)), Ok(false));
        }
        let exchange = serde_json::json!({"verifier": proof});
        assert_eq!(auth_request_policy("POST", &format!("/api/desktop/account-link/{id}/exchange"), Some(&exchange)), Ok(false));
        assert_eq!(auth_request_policy("GET", &format!("/api/desktop/login/{id}"), None), Ok(false));
        assert_eq!(auth_request_policy("DELETE", &format!("/api/desktop/account-link/{id}"), None), Ok(false));
        let login = serde_json::json!({"challenge": proof});
        assert_eq!(auth_request_policy("POST", "/api/desktop/login", Some(&login)), Ok(true));
        assert_eq!(auth_request_policy("POST", &format!("/api/desktop/login/{id}/exchange"), Some(&exchange)), Ok(true));
        for body in [
            serde_json::json!({"challenge": proof, "target": "google.com", "uid": "other"}),
            serde_json::json!({"challenge": proof, "target": "password", "password": "not-for-transport"}),
            serde_json::json!({"challenge": proof, "target": "github.com"}),
            serde_json::json!({"challenge": "B".repeat(64), "target": "password"}),
            serde_json::json!({"target": "password"}),
            serde_json::json!([]),
        ] {
            assert_eq!(auth_request_policy("POST", "/api/desktop/account-link", Some(&body)), Err("Invalid account request.".to_string()));
        }
        let client = Client::new();
        let public = auth_request_policy("POST", "/api/desktop/account-link", Some(&serde_json::json!({"challenge": proof, "target": "password"}))).unwrap();
        assert!(authorize(client.post(CLOUD_ORIGIN), None, public).is_err());
        assert!(auth_request_policy("GET", &format!("/api/desktop/login/{id}"), Some(&exchange)).is_err());
    }
    #[test]
    fn large_local_upload_has_explicit_format_origin_and_role_boundaries() {
        assert!(!local_upload(MAX_FILE_BYTES, "wages.csv").unwrap());
        for name in ["wages.csv", "wide.PARQUET"] {
            assert!(local_upload(MAX_FILE_BYTES + 1, name).unwrap());
        }
        for name in ["large.xlsx", "large.dta", "large.py", "large"] {
            assert!(local_upload(MAX_FILE_BYTES + 1, name).is_err());
        }
        let project = "f".repeat(32);
        for origin in ["http://127.0.0.1:8765", "http://localhost:8765", "http://[::1]:8765"] {
            assert!(local_workspace(origin, &project).unwrap().ends_with(&format!("/api/desktop/projects/{project}/workspace")));
        }
        for origin in ["https://127.0.0.1:8765", "http://127.0.0.1", "http://127.0.0.1:80", "http://example.com:8765", "http://127.0.0.1:8765/other", "http://user@127.0.0.1:8765", "http://127.0.0.1:8765?redirect=x", "http://127.0.0.1:8765#x"] {
            assert!(local_workspace(origin, &project).is_err());
        }
        assert!(local_workspace("http://127.0.0.1:8765", "../project").is_err());
        assert!(editable_session(&serde_json::json!({"read_only":false})));
        for value in [serde_json::json!({}), serde_json::json!({"read_only":true}), serde_json::json!({"read_only":"false"})] {
            assert!(!editable_session(&value));
        }
        for role in ["owner", "editor"] {
            assert!(editable_local_state(&serde_json::json!({"role":role})));
            for denied in [serde_json::json!(true), serde_json::json!("false"), serde_json::json!(null)] {
                assert!(!editable_local_state(&serde_json::json!({"role":role,"access_denied":denied})));
            }
        }
        assert!(!editable_local_state(&serde_json::json!({"role":"viewer"})));
    }

    #[test]
    fn native_local_import_streams_fixed_loopback_with_local_token_only() {
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let address = listener.local_addr().unwrap();
        let project = "f".repeat(32);
        let expected_path = format!("/api/desktop/projects/{project}/workspace");
        let file_path = std::env::temp_dir().join(format!("openecon-local-upload-{}.csv", uuid::Uuid::new_v4()));
        let bytes = 1_048_576;
        {
            let mut file = std::fs::File::create(&file_path).unwrap();
            for _ in 0..bytes / 1024 { file.write_all(&[b'7'; 1024]).unwrap(); }
        }
        let server = std::thread::spawn(move || {
            for suffix in ["/session", "/desktop-sync-state", "/datasets/upload"] {
                let (mut socket, _) = listener.accept().unwrap();
                socket.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
                let mut buffer = [0u8; 65536];
                let mut header = Vec::new();
                let split = loop {
                    let count = socket.read(&mut buffer).unwrap();
                    assert!(count > 0);
                    header.extend_from_slice(&buffer[..count]);
                    if let Some(index) = header.windows(4).position(|value| value == b"\r\n\r\n") { break index + 4; }
                    assert!(header.len() < 16384);
                };
                let text = std::str::from_utf8(&header[..split]).unwrap().to_ascii_lowercase();
                assert!(text.lines().next().unwrap().contains(&format!("{expected_path}{suffix}")));
                assert!(!text.contains("authorization:"));
                if suffix != "/session" { assert!(text.contains("x-openecon-token: local-test-token")); }
                let response = if suffix == "/session" {
                    serde_json::json!({"token":"local-test-token"})
                } else if suffix == "/desktop-sync-state" {
                    serde_json::json!({"role":"editor","access_denied":false})
                } else {
                    assert!(text.contains(&format!("origin: http://{address}")));
                    assert!(text.contains("multipart/form-data; boundary="));
                    let size: usize = text.lines().find_map(|line| line.strip_prefix("content-length:").map(|value| value.trim().parse().unwrap())).unwrap();
                    let mut received = header.len() - split;
                    while received < size {
                        let count = socket.read(&mut buffer[..(size - received).min(65536)]).unwrap();
                        assert!(count > 0); received += count;
                    }
                    assert!(received > bytes && received < bytes + 1024);
                    serde_json::json!({"id":"11111111-1111-4111-8111-111111111111","local_only":true})
                }.to_string();
                write!(socket, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response}", response.len()).unwrap();
            }
        });
        let result = tauri::async_runtime::block_on(async {
            let file = tokio::fs::File::open(&file_path).await.unwrap();
            local_import(&project, &format!("http://{address}"), file, bytes as u64, "large.csv").await
        });
        std::fs::remove_file(file_path).unwrap();
        server.join().unwrap();
        let result = result.unwrap();
        assert_eq!(result.status, 200);
        assert_eq!(result.body["local_only"], true);
    }
    #[test]
    fn denied_http_is_preserved_even_if_body_is_truncated() {
        for listing in [true, false] {
            for (status, size, body, expected) in [
                (403, 50, "", "OPENECON_HTTP:403"),
                (200, 50, "", "OPENECON_NETWORK"),
                (
                    200,
                    5,
                    "wrong",
                    "The cloud returned an invalid JSON response.",
                ),
            ] {
                let server = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
                let address = server.local_addr().unwrap();
                let thread = std::thread::spawn(move || {
                    let (mut socket, _) = server.accept().unwrap();
                    let mut request = [0u8; 4096];
                    socket.read(&mut request).unwrap();
                    write!(socket, "HTTP/1.1 {status} Test\r\nContent-Length: {size}\r\nConnection: close\r\n\r\n{body}").unwrap();
                });
                let result = tauri::async_runtime::block_on(async {
                    let response = Client::builder()
                        .no_proxy()
                        .build()
                        .unwrap()
                        .get(format!("http://{address}/"))
                        .send()
                        .await
                        .unwrap();
                    if listing {
                        file_listing_response(response).await
                    } else {
                        json_response(response).await.map(|value| value.body)
                    }
                });
                thread.join().unwrap();
                assert_eq!(result.unwrap_err(), expected);
            }
        }
    }

    #[test]
    fn history_reads_accept_only_bounded_named_queries_and_never_compute() {
        let path = "/api/projects/p/workspace/console/history";
        for suffix in ["", "?limit=20", "?query=%F0%9F%8C%8D&since=2026-10-01&until=&limit=50&max_bytes=4096", "?cursor=abc_123-xyz"] {
            assert!(allowed_route("GET", &format!("{path}{suffix}")));
            for method in ["POST", "PUT", "PATCH", "DELETE"] {
                assert!(!allowed_route(method, &format!("{path}{suffix}")));
            }
        }
        for suffix in ["?", "?host=evil", "?limit=51", "?limit=0", "?max_bytes=262145", "?max_bytes=4095", "?query=a&query=b", "?cursor=../escape", "?query=%00", "?limit=20#fragment", "?since=wrong"] {
            assert!(!allowed_route("GET", &format!("{path}{suffix}")), "{suffix}");
        }
        assert!(!allowed_route("GET", &format!("{path}?query={}", "a".repeat(257))));
        assert!(!allowed_route("GET", &format!("{path}?cursor={}", "a".repeat(2049))));
        assert!(!allowed_route("GET", "/api/projects/%70/workspace/console/history?limit=20"));
        assert!(!allowed_route("GET", "/api/projects/p/workspace/console/execute?limit=20"));
        let record = "/api/projects/p/workspace/runs/run_123/record";
        assert!(allowed_route("GET", record));
        for method in ["POST", "PUT", "DELETE"] { assert!(!allowed_route(method, record)); }
        assert!(!allowed_route("GET", &format!("{record}?host=evil")));
        assert!(!allowed_route("GET", "/api/projects/p/workspace/runs/../record"));
    }

    #[test]
    fn cloud_route_is_narrow_and_never_executes() {
        assert!(allowed_route(
            "GET",
            "/api/projects/project-1/workspace/datasets"
        ));
        assert!(allowed_route(
            "PUT",
            "/api/projects/project-1/workspace/console/script"
        ));
        assert!(allowed_route("GET", "/api/projects/p/workspace/environment"));
        assert!(allowed_route("PUT", "/api/projects/p/workspace/environment"));
        for path in [
            "/api/projects/p/workspace/console/execute",
            "/api/console/execute",
            "https://evil.example/api/me",
            "//evil.example/api/me",
            "/api/me?host=evil",
            "/api/projects/%2e%2e/members",
            "/api/projects/../members",
            "/api/projects/p/workspace/console/%65xecute",
            "/api/projects/p/workspace/files/f/download",
            "/api/projects/p/workspace/environment/install",
            "/api/projects/p/workspace/environment/restore",
        ] {
            for method in ["GET", "POST", "PUT", "DELETE"] {
                assert!(!allowed_route(method, path), "{method} {path}");
            }
        }
        assert!(!allowed_route(
            "POST",
            "/api/projects/p/workspace/console/reset"
        ));
    }
    #[test]
    fn project_rename_allows_only_exact_patch_and_valid_project_identifiers() {
        for id in [
            "p",
            "project-1",
            "Project_2",
            "0123456789abcdef0123456789abcdef",
        ] {
            let path = format!("/api/projects/{id}");
            assert!(allowed_route("PATCH", &path));
            for method in [
                "GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "patch",
            ] {
                assert!(!allowed_route(method, &path), "{method} {path}");
            }
        }
        assert!(allowed_route(
            "PATCH",
            &format!("/api/projects/{}", "p".repeat(128))
        ));
        assert!(!allowed_route(
            "PATCH",
            &format!("/api/projects/{}", "p".repeat(129))
        ));
        for path in [
            "/api/projects/",
            "/api/projects/p/",
            "/api/projects//p",
            "/api/projects/p/name",
            "/api/projects/p/workspace",
            "/api/projects/p/workspace/console/execute",
            "/api/projects/p?name=other",
            "/api/projects/p#fragment",
            "/api/projects/%70",
            "/api/projects/p%2fname",
            "/api/projects/..",
            "/api/projects/.",
            "/api/projects/p\\name",
            "/api/projects/türkçe",
            "/api/projects/p name",
            "//api/projects/p",
            "https://evil.example/api/projects/p",
        ] {
            assert!(!allowed_route("PATCH", path), "PATCH {path}");
        }
    }
    #[test]
    fn named_scripts_allow_only_catalog_and_canonical_document_routes() {
        let catalog = "/api/projects/p/workspace/console/scripts";
        for method in ["GET", "POST"] {
            assert!(allowed_route(method, catalog));
        }
        for method in ["PUT", "PATCH", "DELETE"] {
            assert!(!allowed_route(method, catalog));
        }
        for id in ["analysis", "0123456789abcdef0123456789abcdef"] {
            let path = format!("{catalog}/{id}");
            for method in ["GET", "PUT"] {
                assert!(allowed_route(method, &path));
            }
            for method in ["POST", "PATCH", "DELETE"] {
                assert!(!allowed_route(method, &path));
            }
        }
        for id in ["ANALYSIS", "Analysis", "analysis.py", "abcdef", "0123456789ABCDEF0123456789ABCDEF",
                   "0123456789abcdef0123456789abcdef/execute", "analysis/", "..", "%61nalysis"] {
            let path = format!("{catalog}/{id}");
            for method in ["GET", "POST", "PUT", "DELETE"] {
                assert!(!allowed_route(method, &path), "{method} {path}");
            }
        }
    }
    #[test]
    fn file_layout_allows_only_canonical_metadata_routes() {
        let path = "/api/projects/p/workspace/files/layout";
        for method in ["GET", "PUT"] {
            assert!(allowed_route(method, path));
        }
        for method in ["POST", "PATCH", "DELETE"] {
            assert!(!allowed_route(method, path));
        }
        for path in [
            "/api/projects/p/workspace/files/layout/",
            "/api/projects/p/workspace/files/layout/execute",
            "/api/projects/p/workspace/files/%6cayout",
            "/api/projects/p/workspace/files/layout?host=evil",
            "/api/projects/p/workspace/files/layout#fragment",
            "/api/projects/p/workspace/files/../layout",
        ] {
            for method in ["GET", "PUT"] {
                assert!(!allowed_route(method, path), "{method} {path}");
            }
        }
    }
    #[test]
    fn filenames_are_portable_and_cannot_escape() {
        for name in [
            "wages.csv",
            "türkçe veri.xlsx",
            "data.parquet",
            "survey.dta",
        ] {
            assert!(safe_filename(name));
        }
        for name in [
            "../wages.csv",
            "a/b.csv",
            "a\\b.csv",
            "CON.csv",
            "COM1.xlsx",
            ".hidden.csv",
            "a.csv ",
            "a:stream.csv",
            "run.py",
            "a\n.csv",
        ] {
            assert!(!safe_filename(name), "{name}");
        }
    }

    #[test]
    fn empty_credential_is_allowed_only_for_public_requests() {
        let client = client().unwrap();
        assert!(authorize(client.get(CLOUD_ORIGIN), Some(""), true).is_ok());
        assert!(authorize(client.get(CLOUD_ORIGIN), None, true).is_ok());
        assert!(authorize(client.get(CLOUD_ORIGIN), Some(""), false).is_err());
        assert!(authorize(client.get(CLOUD_ORIGIN), Some("bad\nheader"), false).is_err());
    }
    #[test]
    fn metadata_must_include_valid_hash_and_bounded_size() {
        let item = serde_json::json!({"datasets":[{"id":"file1","name":"wages.csv","data_hash":"f".repeat(64),"size_bytes":12}]});
        assert_eq!(
            pinned_metadata(&item, "file1").unwrap().python_path,
            "wages.csv"
        );
        assert!(pinned_metadata(&item, "file2").is_err());
        let bad = serde_json::json!({"datasets":[{"id":"file1","name":"../wages.csv","data_hash":"f".repeat(64),"size_bytes":12}]});
        assert!(pinned_metadata(&bad, "file1").is_err());
    }

    #[test]
    fn individual_download_metadata_cannot_bypass_project_geometry() {
        let make = |count: usize, size: u64, chunked: bool| serde_json::json!({"datasets":(0..count).map(|index| {
            let mut file = serde_json::json!({"id":format!("{index:032x}"),"name":format!("data{index}.csv"),"data_hash":"a".repeat(64),"size_bytes":size});
            if chunked { file["transfer"] = serde_json::json!("chunked-v1"); }
            file
        }).collect::<Vec<_>>()});
        let selected = "0".repeat(32);
        assert!(selected_project_metadata(&make(21,1,true),&selected).is_err());
        assert!(selected_project_metadata(&make(5,crate::data_transfer::MAX_FILE_BYTES,true),&selected).is_err());
        assert!(selected_project_metadata(&make(3,MAX_FILE_BYTES,false),&selected).is_err());
        let exact = make(4,crate::data_transfer::MAX_FILE_BYTES,true);
        assert_eq!(selected_project_metadata(&exact,&selected).unwrap().size_bytes,crate::data_transfer::MAX_FILE_BYTES);
        assert!(selected_project_metadata(&exact,&"f".repeat(32)).is_err());
    }

    #[test]
    fn batch_metadata_rejects_duplicate_names_and_excessive_input() {
        let item = serde_json::json!({"id":"a".repeat(32),"name":"wages.csv","data_hash":"f".repeat(64),"size_bytes":12});
        let valid = serde_json::json!({"datasets":[item.clone()]});
        assert_eq!(project_metadata(&valid).unwrap().len(), 1);
        let mut collision = item.clone();
        collision["id"] = Value::String("b".repeat(32));
        collision["name"] = Value::String("WAGES.csv".into());
        assert!(
            project_metadata(&serde_json::json!({"datasets":[item.clone(),collision]})).is_err()
        );
        assert!(project_metadata(&serde_json::json!({"datasets":vec![item.clone();21]})).is_err());
        let big: Vec<Value> = (0..3).map(|index| serde_json::json!({"id":format!("{index:032x}"),"name":format!("file{index}.csv"),"data_hash":"f".repeat(64),"size_bytes":MAX_FILE_BYTES})).collect();
        assert!(project_metadata(&serde_json::json!({"datasets":big})).is_err());
    }
    #[test]
    fn cache_hash_detects_local_edits_and_directories() {
        let path =
            std::env::temp_dir().join(format!("openecon-cache-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&path).unwrap();
        let root = Dir::open_ambient_dir(&path, cap_std::ambient_authority()).unwrap();
        let dir = project_dir(&root, "project1").unwrap();
        dir.write("wages.csv", b"hello").unwrap();
        let hash = format!("{:x}", Sha256::digest(b"hello"));
        assert!(cache_matches(&dir, "wages.csv", 5, &hash).unwrap());
        dir.write("wages.csv", b"world").unwrap();
        assert!(!cache_matches(&dir, "wages.csv", 5, &hash).unwrap());
        dir.create_dir("bad.csv").unwrap();
        assert!(cache_matches(&dir, "bad.csv", 0, &hash).is_err());
        // Windows retains directory handles until they are explicitly closed.
        drop(dir);
        drop(root);
        std::fs::remove_dir_all(path).unwrap();
    }
    #[cfg(unix)]
    #[test]
    fn symlinked_project_and_file_are_rejected() {
        let path =
            std::env::temp_dir().join(format!("openecon-link-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(path.join("projects")).unwrap();
        std::os::unix::fs::symlink(std::env::temp_dir(), path.join("projects/escape")).unwrap();
        let root = Dir::open_ambient_dir(&path, cap_std::ambient_authority()).unwrap();
        assert!(project_dir(&root, "escape").is_err());
        let dir = project_dir(&root, "project1").unwrap();
        std::os::unix::fs::symlink("/etc/hosts", path.join("projects/project1/wages.csv")).unwrap();
        assert!(cache_matches(&dir, "wages.csv", 0, "").is_err());
        drop(dir);
        drop(root);
        std::fs::remove_dir_all(path).unwrap();
    }
}
