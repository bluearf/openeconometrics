//! Reconcile the native content view after WKWebView replaces its document.
//! This runs on the UI thread without changing the window's size or focus.

use tauri::{Manager, Webview, WebviewWindow};

fn trace(window: &WebviewWindow, value: serde_json::Value) {
    // Opt-in local diagnosis contains geometry only. Never follow a trace
    // alias or let repeated native events grow an unbounded file.
    let Ok(root) = window.app_handle().path().app_data_dir() else { return };
    let marker = root.join(".native-layout-diagnostics");
    if !std::fs::symlink_metadata(marker).is_ok_and(|m| m.is_file()) { return; }
    let path = root.join("native-layout-diagnostics.jsonl");
    if let Ok(metadata) = std::fs::symlink_metadata(&path) {
        if !metadata.is_file() || metadata.len() > 64 * 1024 { return; }
    }
    if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        use std::io::Write;
        let _ = writeln!(file, "{value}");
    }
}

pub fn reconcile(window: &WebviewWindow, reason: &'static str) -> tauri::Result<()> {
    let webview: &Webview = window.as_ref();
    let window = window.clone();
    webview.with_webview(move |platform| {
        // Tauri owns this WKWebView (an NSView subclass), and invokes this
        // callback on the main thread while the platform view remains alive.
        let view = unsafe { &*platform.inner().cast::<objc2_app_kit::NSView>() };
        let before = view.frame();
        // A completed web document can precede AppKit's host layout. Resolve
        // the ancestor constraints before reading the bounds for its child.
        let attached = view.window();
        let content = attached.as_ref().and_then(|native| native.contentView());
        if let Some(content) = content.as_ref() {
            content.setNeedsLayout(true);
            content.layoutSubtreeIfNeeded();
        }
        let mut admitted = false;
        let mut parent_bounds = None;
        if let Some(parent) = unsafe { view.superview() } {
            parent.setNeedsLayout(true);
            parent.layoutSubtreeIfNeeded();
            let bounds = parent.bounds();
            parent_bounds = Some(format!("{bounds:?}"));
            if bounds.size.width > 0.0 && bounds.size.height > 0.0 {
                admitted = true;
                view.setFrame(bounds);
                view.setNeedsLayout(true);
                view.layoutSubtreeIfNeeded();
                view.setNeedsDisplay(true);
                parent.setNeedsDisplay(true);
            }
        }
        if let Some(content) = content.as_ref() { content.setNeedsDisplay(true); }
        trace(&window, serde_json::json!({"reason":reason,"attached":attached.is_some(),
            "resolved":admitted,"view_before":format!("{before:?}"),
            "parent_bounds":parent_bounds,"view_after":format!("{:?}",view.frame()),
            "content_bounds":content.as_ref().map(|v|format!("{:?}",v.bounds()))}));
    })
}
