#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
mod broker;
mod data_transfer;
mod export;
mod runtime;
mod suggestions;
#[cfg(target_os = "macos")]
mod macos_layout;

use cap_std::fs::Dir;
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex,
    },
};
use tauri::{Manager, State, WebviewUrl, WebviewWindow, WebviewWindowBuilder};

struct DesktopState {
    runtime: Mutex<Option<runtime::LocalRuntime>>,
    local_origin: Mutex<Option<String>>,
    exiting: AtomicBool,
    export_busy: AtomicBool,
    data_root: PathBuf,
    root: Dir,
    client: reqwest::Client,
}

#[derive(Serialize)]
struct DesktopInfo {
    local_origin: String,
    data_root: String,
    cloud_origin: &'static str,
    version: &'static str,
}

fn trusted_window(window: &WebviewWindow, state: &DesktopState) -> Result<(), String> {
    let url = window
        .url()
        .map_err(|_| "The local window is unavailable.")?;
    let expected = state
        .local_origin
        .lock()
        .map_err(|_| "The local workspace is unavailable.")?;
    if window.label() != "main"
        || expected.as_deref() != Some(url.origin().ascii_serialization().as_str())
    {
        return Err("Only the local OpenEconometrics window can use this operation.".into());
    }
    Ok(())
}

fn diagnostic_native_title(enabled: bool, title: &str) -> Option<&'static str> {
    if !enabled {
        return None;
    }
    match title {
        "OpenEconometrics diagnostics: IPC pending" => Some("OpenEconometrics diagnostics: IPC pending"),
        "OpenEconometrics diagnostics: IPC unavailable" => Some("OpenEconometrics diagnostics: IPC unavailable"),
        "OpenEconometrics diagnostics: IPC failed" => Some("OpenEconometrics diagnostics: IPC failed"),
        "OpenEconometrics diagnostics: IPC verified" => Some("OpenEconometrics diagnostics: IPC verified"),
        _ => None,
    }
}

#[tauri::command]
async fn save_text_export(
    window: WebviewWindow,
    state: State<'_, DesktopState>,
    filename: String,
    content: String,
) -> Result<export::ExportReceipt, String> {
    trusted_window(&window, &state)?;
    export::validate(&filename, &content)?;
    if state.export_busy.swap(true, Ordering::AcqRel) {
        return Err("Finish or cancel the open export dialog first.".into());
    }
    struct BusyGuard<'a>(&'a AtomicBool);
    impl Drop for BusyGuard<'_> {
        fn drop(&mut self) { self.0.store(false, Ordering::Release); }
    }
    let _guard = BusyGuard(&state.export_busy);
    let selected = rfd::AsyncFileDialog::new().set_parent(&window)
        .set_title("Save export").set_file_name(&filename).save_file().await;
    trusted_window(&window, &state)?;
    let Some(selected) = selected else { return Ok(export::cancelled()); };
    let path = selected.path().to_path_buf();
    tauri::async_runtime::spawn_blocking(move || export::write_selected(&path, &content))
        .await.map_err(|_| "The export operation could not finish.")?
}

#[tauri::command]
fn desktop_info(
    window: WebviewWindow,
    state: State<'_, DesktopState>,
) -> Result<DesktopInfo, String> {
    trusted_window(&window, &state)?;
    #[cfg(target_os = "macos")]
    let _ = macos_layout::reconcile(&window, "workspace-ready");
    Ok(DesktopInfo {
        local_origin: state
            .local_origin
            .lock()
            .map_err(|_| "The local workspace is unavailable.")?
            .clone()
            .ok_or("The local workspace is starting.")?,
        data_root: state.data_root.to_string_lossy().into(),
        cloud_origin: broker::CLOUD_ORIGIN,
        version: env!("CARGO_PKG_VERSION"),
    })
}

#[tauri::command]
async fn cloud_request(
    window: WebviewWindow,
    state: State<'_, DesktopState>,
    method: String,
    path: String,
    token: Option<String>,
    body: Option<Value>,
) -> Result<broker::CloudResponse, String> {
    trusted_window(&window, &state)?;
    broker::request(&state.client, &method, &path, token.as_deref(), body).await
}

#[tauri::command]
async fn download_project_file(
    window: WebviewWindow,
    state: State<'_, DesktopState>,
    project_id: String,
    file_id: String,
    token: String,
) -> Result<broker::CachedFile, String> {
    trusted_window(&window, &state)?;
    broker::download(&state.client, &state.root, &project_id, &file_id, &token).await
}

#[tauri::command]
async fn download_project_files(
    window: WebviewWindow,
    state: State<'_, DesktopState>,
    project_id: String,
    token: String,
) -> Result<Vec<broker::CachedFile>, String> {
    trusted_window(&window, &state)?;
    broker::download_all(&state.client, &state.root, &project_id, &token).await
}

#[tauri::command]
async fn upload_project_file(
    window: WebviewWindow,
    state: State<'_, DesktopState>,
    project_id: String,
    token: String,
) -> Result<broker::CloudResponse, String> {
    trusted_window(&window, &state)?;
    let origin = state.local_origin.lock()
        .map_err(|_| "The local workspace is unavailable.")?
        .clone().ok_or("The local workspace is starting.")?;
    broker::upload(&state.client, &state.root, &project_id, &token, &origin).await
}

#[tauri::command]
async fn project_transfer_action(
    window: WebviewWindow,
    state: State<'_, DesktopState>,
    project_id: String,
    request_id: Option<String>,
    token: String,
    action: String,
) -> Result<broker::CloudResponse, String> {
    trusted_window(&window, &state)?;
    data_transfer::transfer_action(&state.client, &state.root, &project_id,
                                   request_id.as_deref(), &token, &action).await
}

#[tauri::command]
fn open_desktop_login(
    window: WebviewWindow,
    state: State<'_, DesktopState>,
    request_id: String,
) -> Result<(), String> {
    trusted_window(&window, &state)?;
    if !broker::hex_id(&request_id, 32) {
        return Err("Invalid sign-in request identifier.".into());
    }
    open::that(format!(
        "{}/?desktop_login={request_id}",
        broker::CLOUD_ORIGIN
    ))
    .map_err(|_| "The system browser could not be opened.".into())
}

#[tauri::command]
async fn suggestions_status(
    window: WebviewWindow,
    desktop: State<'_, DesktopState>,
    suggestions: State<'_, suggestions::Manager>,
) -> Result<suggestions::Status, String> {
    trusted_window(&window, &desktop)?;
    suggestions.status().await
}

#[tauri::command]
async fn suggestions_install(
    window: WebviewWindow,
    desktop: State<'_, DesktopState>,
    suggestions: State<'_, suggestions::Manager>,
) -> Result<suggestions::Status, String> {
    trusted_window(&window, &desktop)?;
    suggestions.install().await
}

#[tauri::command]
async fn suggestions_configure(
    window: WebviewWindow,
    desktop: State<'_, DesktopState>,
    suggestions: State<'_, suggestions::Manager>,
    enabled: bool,
) -> Result<suggestions::Status, String> {
    trusted_window(&window, &desktop)?;
    suggestions.configure(enabled).await
}

#[tauri::command]
async fn suggestions_complete(
    window: WebviewWindow,
    desktop: State<'_, DesktopState>,
    suggestions: State<'_, suggestions::Manager>,
    request: suggestions::CompletionRequest,
) -> Result<suggestions::CompletionResponse, String> {
    trusted_window(&window, &desktop)?;
    suggestions.complete(request).await
}

#[tauri::command]
fn suggestions_cancel(
    window: WebviewWindow,
    desktop: State<'_, DesktopState>,
    suggestions: State<'_, suggestions::Manager>,
    request_id: String,
) -> Result<(), String> {
    trusted_window(&window, &desktop)?;
    suggestions.cancel(&request_id)
}

fn smoke_test() -> Result<(), String> {
    let resources = runtime::resource_dir_for_executable()?;
    let data_root =
        std::env::temp_dir().join(format!("openecon-packaged-smoke-{}", uuid::Uuid::new_v4()));
    let started = std::time::Instant::now();
    let mut runtime = runtime::LocalRuntime::start(&resources, &data_root)?;
    let ready_seconds = started.elapsed().as_secs_f64();
    let first_port = runtime.ready.port;
    let origin = runtime.ready.url.clone();
    let token = runtime.ready.token.clone();
    let result = tauri::async_runtime::block_on(async {
        let client = reqwest::Client::builder()
            .no_proxy()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .map_err(|_| "Local client failed.")?;
        let project = "0123456789abcdef0123456789abcdef";
        let workspace = format!("{origin}/api/desktop/projects/{project}/workspace");
        let session: Value = client
            .get(format!("{workspace}/session"))
            .send()
            .await
            .map_err(|_| "Local session unavailable.")?
            .json()
            .await
            .map_err(|_| "Invalid local session.")?;
        let token = session
            .get("token")
            .and_then(Value::as_str)
            .unwrap_or(&token);
        let result = client.post(format!("{workspace}/console/execute")).header("X-OpenEcon-Token", token)
            .json(&json!({"code": "import openecon as oe\nimport torch\nprint('bundled-runtime-ok', torch.__version__)\ndf = oe.example()\nprint(df.shape)\nresult = oe.ols(data=df, y='wage', x=['education', 'experience'])\nprint(result.latex)\n", "timeout_seconds": 60}))
            .send().await.map_err(|_| "Bundled Python execution failed.")?;
        let status = result.status();
        let body: Value = result
            .json()
            .await
            .map_err(|_| "Invalid Python execution response.")?;
        if !status.is_success() || body.get("status").and_then(Value::as_str) != Some("ok") {
            return Err(format!("Bundled execution failed: {body}"));
        }
        Ok(body)
    });
    runtime.stop();
    drop(runtime);
    let execution = result?;
    let started = std::time::Instant::now();
    let mut second = runtime::LocalRuntime::start(&resources, &data_root)?;
    let warm_ready_seconds = started.elapsed().as_secs_f64();
    let same_port = second.ready.port == first_port;
    second.stop();
    drop(second);
    if !same_port {
        return Err("The desktop origin changed after restart.".into());
    }
    println!(
        "{}",
        json!({"status":"ok","bundled":true,"stable_origin":same_port,"ready_seconds":ready_seconds,"warm_ready_seconds":warm_ready_seconds,"local_origin":origin,"execution":execution})
    );
    let _ = std::fs::remove_dir_all(data_root);
    Ok(())
}

fn main() {
    if std::env::args().any(|arg| arg == "--suggestions-smoke-test") {
        let arguments: Vec<String> = std::env::args().collect();
        let value_after = |flag: &str| arguments.iter().position(|arg| arg == flag)
            .and_then(|position| arguments.get(position + 1)).map(PathBuf::from);
        let result = (|| {
            let resources = value_after("--suggestions-resources-path")
                .map(Ok).unwrap_or_else(runtime::resource_dir_for_executable)?;
            let model = value_after("--suggestions-model-path").ok_or("AI_SMOKE_MODEL_REQUIRED")?;
            suggestions::smoke_test(&resources, &model)
        })();
        match result { Ok(report) => println!("{report}"), Err(failure) => { eprintln!("{failure}"); std::process::exit(1); } }
        return;
    }
    if std::env::args().any(|arg| arg == "--smoke-test") {
        if let Err(error) = smoke_test() {
            eprintln!("{error}");
            std::process::exit(1);
        }
        return;
    }
    let diagnostic_window = std::env::args().any(|arg| arg == "--diagnostic-window");
    let app = tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![desktop_info, save_text_export, cloud_request, download_project_file, download_project_files, upload_project_file, project_transfer_action, open_desktop_login, suggestions_status, suggestions_install, suggestions_configure, suggestions_complete, suggestions_cancel])
        .setup(move |app| {
            let resources = app.path().resource_dir()?;
            let data_root = app.path().app_data_dir()?;
            std::fs::create_dir_all(&data_root)?;
            if std::fs::symlink_metadata(&data_root)?.file_type().is_symlink() {
                return Err(std::io::Error::other("The OpenEconometrics data folder cannot be a symbolic link.").into());
            }
            let root = Dir::open_ambient_dir(&data_root, cap_std::ambient_authority())?;
            app.manage(suggestions::Manager::new(&resources, &data_root).map_err(std::io::Error::other)?);
            app.manage(DesktopState { runtime: Mutex::new(None), local_origin: Mutex::new(None), exiting: AtomicBool::new(false), export_busy: AtomicBool::new(false), data_root: data_root.clone(), root, client: broker::client().map_err(std::io::Error::other)? });
            let navigation_app = app.handle().clone();
            let window = WebviewWindowBuilder::new(app, "main", WebviewUrl::App("index.html".into()))
                .title("OpenEconometrics").inner_size(1280.0, 820.0).min_inner_size(800.0, 600.0)
                // The sidebar handles internal file moves through HTML drag events.
                .disable_drag_drop_handler()
                .devtools(cfg!(debug_assertions))
                .on_navigation(move |url| {
                    let state = navigation_app.state::<DesktopState>();
                    let origin = state.local_origin.lock().ok().and_then(|value| value.clone());
                    if let Some(expected) = origin { return url.origin().ascii_serialization() == expected; }
                    // The initial, bundled splash has no application capability.
                    (url.scheme() == "tauri" && url.host_str() == Some("localhost"))
                        || (matches!(url.scheme(), "http" | "https") && url.host_str() == Some("tauri.localhost"))
                })
                .on_new_window(|_, _| tauri::webview::NewWindowResponse::Deny)
                .on_document_title_changed(move |window, title| {
                    if let Some(title) = diagnostic_native_title(diagnostic_window, &title) {
                        let state = window.state::<DesktopState>();
                        if trusted_window(&window, &state).is_ok() {
                            let _ = window.set_title(title);
                        }
                    }
                })
                .on_page_load(move |window, payload| {
                    if payload.event() == tauri::webview::PageLoadEvent::Finished {
                        let state = window.state::<DesktopState>();
                        let origin = state.local_origin.lock().ok().and_then(|value| value.clone());
                        if origin.as_deref() == Some(payload.url().origin().ascii_serialization().as_str()) && trusted_window(&window, &state).is_ok() {
                            #[cfg(target_os = "macos")]
                            let _ = macos_layout::reconcile(&window, "document-finished");
                            if diagnostic_window {
                                let _ = window.eval("document.title='OpenEconometrics diagnostics: IPC pending'; if(typeof window.__TAURI__?.core?.invoke==='function'){window.__TAURI__.core.invoke('desktop_info').then(()=>document.title='OpenEconometrics diagnostics: IPC verified').catch(()=>document.title='OpenEconometrics diagnostics: IPC failed')}else{document.title='OpenEconometrics diagnostics: IPC unavailable'}");
                            }
                        }
                    }
                })
                .build()?;
            let handle = app.handle().clone();
            // Show the bundled splash immediately; importing the Python engine
            // never blocks creation of the native application window.
            std::thread::spawn(move || {
                let start = || -> Result<(), String> {
                    let mut local = runtime::LocalRuntime::start(&resources, &data_root)?;
                    let state = handle.state::<DesktopState>();
                    if state.exiting.load(Ordering::Acquire) { local.stop(); return Ok(()); }
                    let origin = local.ready.url.trim_end_matches('/').to_string();
                    let capability = json!({"identifier":"local-workspace","windows":["main"],"local":false,"remote":{"urls":[format!("{origin}/*")]},"permissions":["allow-desktop-info","allow-save-text-export","allow-cloud-request","allow-download-project-file","allow-download-project-files","allow-upload-project-file","allow-project-transfer-action","allow-open-desktop-login","allow-suggestions-status","allow-suggestions-install","allow-suggestions-configure","allow-suggestions-complete","allow-suggestions-cancel"]});
                    handle.add_capability(capability.to_string()).map_err(|_| "The local workspace permission could not start.")?;
                    *state.local_origin.lock().map_err(|_| "The local workspace is unavailable.")? = Some(origin.clone());
                    *state.runtime.lock().map_err(|_| "The local workspace is unavailable.")? = Some(local);
                    if state.exiting.load(Ordering::Acquire) {
                        if let Ok(mut runtime) = state.runtime.lock() { if let Some(local) = runtime.as_mut() { local.stop(); } }
                        return Ok(());
                    }
                    window.navigate(origin.parse().map_err(|_| "The local workspace URL is invalid.")?).map_err(|_| "The local workspace window could not open.")?;
                    Ok(())
                };
                if let Err(error) = start() {
                    let message = serde_json::to_string(&error).unwrap_or_else(|_| "\"OpenEconometrics could not start.\"".into());
                    let _ = window.eval(&format!("document.getElementById('status').textContent={message};document.getElementById('loader').hidden=true"));
                }
            });
            Ok(())
        })
        .on_window_event(|window, event| {
            #[cfg(target_os = "macos")]
            if matches!(event, tauri::WindowEvent::Resized(_) | tauri::WindowEvent::Focused(true)) {
                if let Some(webview) = window.app_handle().get_webview_window(window.label()) {
                    let state = webview.state::<DesktopState>();
                    if trusted_window(&webview, &state).is_ok() {
                        let _ = macos_layout::reconcile(&webview, "native-visible-or-resized");
                    }
                }
            }
            if matches!(event, tauri::WindowEvent::Destroyed) { window.app_handle().exit(0); }
        })
        .build(tauri::generate_context!()).expect("OpenEconometrics desktop application could not start");
    app.run(|app, event| {
        if matches!(
            event,
            tauri::RunEvent::Exit | tauri::RunEvent::ExitRequested { .. }
        ) {
            if let Some(suggestions) = app.try_state::<suggestions::Manager>() { suggestions.stop(); }
            if let Some(state) = app.try_state::<DesktopState>() {
                state.exiting.store(true, Ordering::Release);
                if let Ok(mut runtime) = state.runtime.lock() {
                    if let Some(local) = runtime.as_mut() {
                        local.stop();
                    }
                }
            }
        }
    });
}

#[cfg(test)]
mod diagnostic_tests {
    use super::diagnostic_native_title;

    #[test]
    fn normal_windows_do_not_adopt_diagnostic_document_titles() {
        assert_eq!(diagnostic_native_title(false, "OpenEconometrics diagnostics: IPC verified"), None);
    }

    #[test]
    fn diagnostic_titles_require_exact_fixed_statuses() {
        assert_eq!(diagnostic_native_title(true, "OpenEconometrics diagnostics: IPC verified"), Some("OpenEconometrics diagnostics: IPC verified"));
        for title in ["OpenEconometrics", "OpenEconometrics diagnostics: arbitrary error", "OpenEconometrics diagnostics: IPC verified\n", "OpenEconometrics diagnostics: IPC verified - forged"] {
            assert_eq!(diagnostic_native_title(true, title), None);
        }
    }
}
