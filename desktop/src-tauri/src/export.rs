//! User-selected, bounded exports. No renderer-supplied destination path.
use cap_fs_ext::{FollowSymlinks, OpenOptionsFollowExt};
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

/// Stream a pinned input from the managed cache; large files never enter the webview.
pub fn write_cached_selected(
    path: &Path, source_dir: &Dir, source_name: &str, size: u64, hash: &str,
) -> Result<ExportReceipt, String> {
    validate(source_name, "")?;
    let name = path.file_name().and_then(|s| s.to_str()).ok_or("Invalid export filename.")?;
    validate(name, "")?;
    if !(1..=crate::data_transfer::MAX_FILE_BYTES).contains(&size) || !crate::broker::hex_id(hash, 64) {
        return Err("Invalid shared file size or checksum.".into());
    }
    let mut read_options = OpenOptions::new();
    read_options.read(true).follow(FollowSymlinks::No);
    let mut source = source_dir.open_with(source_name, &read_options)
        .map_err(|_| "The cached input is not a safe regular file.")?;
    let metadata = source.metadata().map_err(|_| "The cached input could not be checked.")?;
    if !metadata.is_file() || metadata.len() != size {
        return Err("The cached input size does not match its metadata.".into());
    }
    let parent = path.parent().ok_or("Invalid export destination.")?;
    let dir = Dir::open_ambient_dir(parent, ambient_authority())
        .map_err(|_| "The selected export folder could not be opened.")?;
    match dir.symlink_metadata(name) {
        Ok(m) if !m.is_file() || m.file_type().is_symlink() => {
            return Err("Choose a regular file destination, not a link or directory.".into());
        }
        Ok(_) => (),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
        Err(_) => return Err("The selected export destination could not be checked.".into()),
    }
    let temporary = format!(".openecon-export-{}", uuid::Uuid::new_v4());
    let result = (|| {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true).follow(FollowSymlinks::No);
        let mut file = dir.open_with(&temporary, &options)
            .map_err(|_| "The export file could not be created.")?;
        let mut digest = Sha256::new();
        let mut buffer = [0u8; 64 * 1024];
        let mut bytes = 0u64;
        loop {
            let count = source.read(&mut buffer).map_err(|_| "The cached input could not be read.")?;
            if count == 0 { break; }
            bytes = bytes.checked_add(count as u64).ok_or("The cached input size is invalid.")?;
            if bytes > size { return Err("The cached input exceeds its pinned size.".into()); }
            digest.update(&buffer[..count]);
            file.write_all(&buffer[..count]).map_err(|_| "The data export could not be saved.")?;
        }
        if bytes != size || format!("{:x}", digest.finalize()) != hash {
            return Err("The cached input does not match its shared checksum.".into());
        }
        file.sync_all().map_err(|_| "The data export could not be saved.")?;
        drop(file);
        dir.rename(&temporary, &dir, name).map_err(|_| "The data export could not replace the selected file.")?;
        if !crate::broker::cache_matches(&dir, name, size, hash)? {
            return Err("The saved data changed before it could be verified.".into());
        }
        Ok(ExportReceipt {
            status: "saved", path: Some(path.to_string_lossy().into()),
            bytes: size as usize, sha256: Some(hash.to_owned()),
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
    #[test]
    fn streams_binary_input_beyond_text_limit_and_replaces_exactly() {
        let p = folder();
        let dir = Dir::open_ambient_dir(&p, ambient_authority()).unwrap();
        let source = p.join("source.parquet");
        let block: Vec<u8> = (0..65536).map(|n| (n % 251) as u8).collect();
        let mut file = std::fs::File::create(&source).unwrap();
        let mut digest = Sha256::new();
        for _ in 0..513 { file.write_all(&block).unwrap(); digest.update(&block); }
        drop(file);
        let size = 513 * block.len() as u64;
        assert!(size > MAX_EXPORT_BYTES as u64);
        let hash = format!("{:x}", digest.finalize());
        let target = p.join("sonuç.parquet");
        std::fs::write(&target, "replace me").unwrap();
        let r = write_cached_selected(&target, &dir, "source.parquet", size, &hash).unwrap();
        assert_eq!(r.status, "saved"); assert_eq!(r.bytes as u64, size);
        assert_eq!(r.sha256.as_deref(), Some(hash.as_str()));
        assert!(crate::broker::cache_matches(&dir, "sonuç.parquet", size, &hash).unwrap());
        assert_eq!(std::fs::read_dir(&p).unwrap().count(), 2);
        drop(dir); // Windows holds the directory open until the capability is released.
        std::fs::remove_dir_all(p).unwrap();
    }
    #[test]
    fn corrupt_or_invalid_input_preserves_destination_and_removes_temporary_files() {
        let p = folder();
        let dir = Dir::open_ambient_dir(&p, ambient_authority()).unwrap();
        std::fs::write(p.join("source.csv"), b"a,b\n1,2\n").unwrap();
        let target = p.join("target.csv");
        std::fs::write(&target, "keep").unwrap();
        for (name, size, hash) in [
            ("source.csv", 8, "0".repeat(64)), ("source.csv", 7, "0".repeat(64)),
            ("source.csv", 0, "0".repeat(64)),
            ("source.csv", crate::data_transfer::MAX_FILE_BYTES + 1, "0".repeat(64)),
            ("../source.csv", 8, "0".repeat(64)), ("source.csv", 8, "invalid".to_owned()),
        ] {
            assert!(write_cached_selected(&target, &dir, name, size, &hash).is_err());
            assert_eq!(std::fs::read_to_string(&target).unwrap(), "keep");
            assert_eq!(std::fs::read_dir(&p).unwrap().count(), 2);
        }
        drop(dir); // Windows holds the directory open until the capability is released.
        std::fs::remove_dir_all(p).unwrap();
    }
    #[cfg(unix)]
    #[test]
    fn binary_export_rejects_source_and_destination_links() {
        let p = folder();
        let dir = Dir::open_ambient_dir(&p, ambient_authority()).unwrap();
        std::fs::write(p.join("actual.csv"), "keep").unwrap();
        std::os::unix::fs::symlink(p.join("actual.csv"), p.join("link.csv")).unwrap();
        let hash = format!("{:x}", Sha256::digest(b"keep"));
        assert!(write_cached_selected(&p.join("target.csv"), &dir, "link.csv", 4, &hash).is_err());
        assert!(write_cached_selected(&p.join("link.csv"), &dir, "actual.csv", 4, &hash).is_err());
        assert_eq!(std::fs::read_to_string(p.join("actual.csv")).unwrap(), "keep");
        assert_eq!(std::fs::read_dir(&p).unwrap().count(), 2);
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
