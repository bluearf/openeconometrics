//! User-selected, bounded text exports. No renderer-supplied destination path.
use cap_std::{
    ambient_authority,
    fs::{Dir, OpenOptions},
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    io::{Read, Write},
    path::Path,
};

pub const MAX_EXPORT_BYTES: usize = 32 * 1024 * 1024;

#[derive(Serialize)]
pub struct ExportReceipt {
    pub status: &'static str,
    pub path: Option<String>,
    pub bytes: usize,
    pub sha256: Option<String>,
}

pub fn cancelled() -> ExportReceipt {
    ExportReceipt {
        status: "cancelled",
        path: None,
        bytes: 0,
        sha256: None,
    }
}

pub fn validate(filename: &str, content: &str) -> Result<(), String> {
    if filename.is_empty()
        || filename.len() > 240
        || filename.trim() != filename
        || matches!(filename, "." | "..")
        || filename.ends_with('.')
        || filename
            .chars()
            .any(|c| c.is_control() || matches!(c, '/' | '\\' | ':'))
    {
        return Err("Choose a valid export filename without a directory path.".into());
    }
    if content.len() > MAX_EXPORT_BYTES {
        return Err("This text export exceeds the 32 MiB desktop export limit.".into());
    }
    Ok(())
}

/// Called only after the native Save panel selects and confirms a destination.
pub fn write_selected(path: &Path, content: &str) -> Result<ExportReceipt, String> {
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or("Invalid export filename.")?;
    validate(name, content)?;
    let parent = path.parent().ok_or("Invalid export destination.")?;
    let dir = Dir::open_ambient_dir(parent, ambient_authority())
        .map_err(|_| "The selected export folder could not be opened.")?;
    match dir.symlink_metadata(name) {
        Ok(m) if !m.is_file() || m.file_type().is_symlink() => {
            return Err("Choose a regular file destination, not a link or directory.".into())
        }
        Ok(_) => (),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
        Err(_) => return Err("The selected export destination could not be checked.".into()),
    }
    let temporary = format!(".openecon-export-{}", uuid::Uuid::new_v4());
    let result = (|| {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        let mut file = dir
            .open_with(&temporary, &options)
            .map_err(|_| "The export file could not be created.")?;
        file.write_all(content.as_bytes())
            .and_then(|_| file.sync_all())
            .map_err(|_| "The export file could not be saved.")?;
        drop(file);
        dir.rename(&temporary, &dir, name)
            .map_err(|_| "The saved export could not replace the selected file.")?;
        let mut saved = Vec::new();
        dir.open(name)
            .map_err(|_| "The saved export could not be verified.")?
            .take((MAX_EXPORT_BYTES + 1) as u64)
            .read_to_end(&mut saved)
            .map_err(|_| "The saved export could not be verified.")?;
        if saved != content.as_bytes() {
            return Err("The saved export changed before it could be verified.".into());
        }
        Ok(ExportReceipt {
            status: "saved",
            path: Some(path.to_string_lossy().into()),
            bytes: saved.len(),
            sha256: Some(format!("{:x}", Sha256::digest(&saved))),
        })
    })();
    let _ = dir.remove_file(&temporary);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    fn folder() -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("openecon-export-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&p).unwrap();
        p
    }
    #[test]
    fn unicode_empty_content_and_replacing_regular_file_are_exact() {
        let p = folder();
        let file = p.join("sonuç.tex");
        std::fs::write(&file, "old bytes to replace").unwrap();
        let s = "\\begin{tabular}{l}\nİstanbul α 𝛽\n\\end{tabular}\n";
        let r = write_selected(&file, s).unwrap();
        assert_eq!(r.status, "saved");
        assert_eq!(r.bytes, s.len());
        assert_eq!(
            r.sha256.unwrap(),
            format!("{:x}", Sha256::digest(s.as_bytes()))
        );
        assert_eq!(std::fs::read_to_string(&file).unwrap(), s);
        write_selected(&file, "").unwrap();
        assert_eq!(std::fs::metadata(&file).unwrap().len(), 0);
        assert_eq!(std::fs::read_dir(&p).unwrap().count(), 1);
        std::fs::remove_dir_all(p).unwrap();
    }
    #[test]
    fn rejects_paths_controls_and_oversized_utf8() {
        for name in [
            "", ".", "..", " /x", "../x", "C:\\x", "a/b", "a:b", "a\0b", "a\nb", "x.", "x ",
        ] {
            assert!(validate(name, "").is_err(), "{name:?}");
        }
        assert!(validate(&"x".repeat(241), "").is_err());
        assert!(validate("x.tex", &"α".repeat(MAX_EXPORT_BYTES / 2 + 1)).is_err());
    }
    #[test]
    fn cancellation_has_no_saved_file_claim() {
        let r = cancelled();
        assert_eq!(r.status, "cancelled");
        assert!(r.path.is_none() && r.sha256.is_none());
        assert_eq!(r.bytes, 0);
    }
    #[test]
    fn rejects_directory_without_modifying_existing_files() {
        let p = folder();
        let existing = p.join("kept");
        std::fs::create_dir(&existing).unwrap();
        assert!(write_selected(&existing, "new").is_err());
        assert!(existing.is_dir());
        assert_eq!(std::fs::read_dir(&p).unwrap().count(), 1);
        std::fs::remove_dir_all(p).unwrap();
    }
    #[cfg(unix)]
    #[test]
    fn refuses_symlink_and_does_not_follow_it() {
        let p = folder();
        let actual = p.join("actual");
        std::fs::write(&actual, "keep").unwrap();
        let link = p.join("export.tex");
        std::os::unix::fs::symlink(&actual, &link).unwrap();
        assert!(write_selected(&link, "replace").is_err());
        assert_eq!(std::fs::read_to_string(actual).unwrap(), "keep");
        std::fs::remove_dir_all(p).unwrap();
    }
}
