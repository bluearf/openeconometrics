fn main() {
    tauri_build::try_build(tauri_build::Attributes::new().app_manifest(
        tauri_build::AppManifest::new().commands(&[
            "desktop_info",
            "save_text_export",
            "cloud_request",
            "download_project_file",
            "download_project_files",
            "upload_project_file",
            "project_transfer_action",
            "open_desktop_login",
            "suggestions_status",
            "suggestions_install",
            "suggestions_configure",
            "suggestions_complete",
            "suggestions_cancel",
        ]),
    ))
    .expect("desktop build metadata");
}
