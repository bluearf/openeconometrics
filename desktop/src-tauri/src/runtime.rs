use cap_fs_ext::{FollowSymlinks, OpenOptionsFollowExt};
use cap_std::fs::{Dir, OpenOptions};
use serde::Deserialize;
use std::{
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::mpsc,
    time::{Duration, Instant},
};

#[derive(Deserialize)]
pub struct Ready {
    #[serde(rename = "type")]
    kind: String,
    pub port: u16,
    pub url: String,
    #[serde(default)]
    pub token: String,
}

pub struct LocalRuntime {
    pub child: Child,
    pub ready: Ready,
    _instance_lock: std::fs::File,
}

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct PortPreference {
    port: u16,
}

fn instance_and_port(data_root: &Path) -> Result<(std::fs::File, u16, Dir), String> {
    let dir = Dir::open_ambient_dir(data_root, cap_std::ambient_authority())
        .map_err(|_| "The local data folder is unavailable.")?;
    let mut options = OpenOptions::new();
    options
        .read(true)
        .write(true)
        .create(true)
        .follow(FollowSymlinks::No);
    let file = dir
        .open_with(".desktop.lock", &options)
        .map_err(|_| "The desktop instance lock is unsafe.")?
        .into_std();
    if !file
        .metadata()
        .map_err(|_| "The desktop instance lock is unavailable.")?
        .is_file()
    {
        return Err("The desktop instance lock is unsafe.".into());
    }
    fs2::FileExt::try_lock_exclusive(&file)
        .map_err(|_| "OpenEconometrics is already running. Open its existing window.")?;
    let mut options = OpenOptions::new();
    options.read(true).follow(FollowSymlinks::No);
    let port = match dir.open_with(".runtime-port.json", &options) {
        Ok(mut preference) => {
            let info = preference
                .metadata()
                .map_err(|_| "The saved desktop port is unavailable.")?;
            if !info.is_file() || info.len() > 256 {
                return Err("The saved desktop port is unsafe.".into());
            }
            let mut bytes = Vec::new();
            std::io::Read::by_ref(&mut preference)
                .take(257)
                .read_to_end(&mut bytes)
                .map_err(|_| "The saved desktop port is unavailable.")?;
            let preference: PortPreference =
                serde_json::from_slice(&bytes).map_err(|_| "The saved desktop port is invalid.")?;
            if preference.port < 1024 {
                return Err("The saved desktop port is invalid.".into());
            }
            preference.port
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
        Err(_) => return Err("The saved desktop port is unsafe.".into()),
    };
    if port != 0 {
        // Only a child we started may announce readiness. Never attach to a
        // process that already occupies the remembered loopback port.
        let probe = socket2::Socket::new(
            socket2::Domain::IPV4,
            socket2::Type::STREAM,
            Some(socket2::Protocol::TCP),
        )
        .map_err(|_| "The local desktop port could not be inspected.")?;
        #[cfg(unix)]
        probe
            .set_reuse_address(true)
            .map_err(|_| "The local desktop port could not be inspected.")?;
        let address = std::net::SocketAddr::from(([127, 0, 0, 1], port));
        probe.bind(&address.into()).map_err(|_| "OpenEconometrics's saved local port is already in use. Close the other application and reopen OpenEconometrics.")?;
        drop(probe);
    }
    Ok((file, port, dir))
}

fn remember_port(dir: &Dir, port: u16) -> Result<(), String> {
    if port < 1024 {
        return Err("The local runtime returned an invalid desktop port.".into());
    }
    let temporary = format!(".runtime-port-{}.json", uuid::Uuid::new_v4());
    let mut options = OpenOptions::new();
    options
        .write(true)
        .create_new(true)
        .follow(FollowSymlinks::No);
    let result = (|| -> Result<(), String> {
        let mut file = dir
            .open_with(&temporary, &options)
            .map_err(|_| "The desktop port could not be saved.")?;
        file.write_all(
            &serde_json::to_vec(&PortPreference { port })
                .map_err(|_| "The desktop port could not be saved.")?,
        )
        .map_err(|_| "The desktop port could not be saved.")?;
        file.sync_all()
            .map_err(|_| "The desktop port could not be saved.")?;
        drop(file);
        if dir.symlink_metadata(".runtime-port.json").is_ok() {
            return Err("The desktop port was changed during startup.".into());
        }
        dir.rename(&temporary, dir, ".runtime-port.json")
            .map_err(|_| "The desktop port could not be saved.")?;
        Ok(())
    })();
    if result.is_err() {
        let _ = dir.remove_file(&temporary);
    }
    result
}

fn executable_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "openecon-runtime.exe"
    } else {
        "openecon-runtime"
    }
}

pub fn bundled_executable(resources: &Path) -> PathBuf {
    resources
        .join("runtime")
        .join("openecon-runtime")
        .join(executable_name())
}

pub fn resource_dir_for_executable() -> Result<PathBuf, String> {
    let executable = std::env::current_exe().map_err(|_| "Application location is unavailable.")?;
    let parent = executable
        .parent()
        .ok_or("Application location is unavailable.")?;
    if cfg!(target_os = "macos") && parent.file_name().is_some_and(|name| name == "MacOS") {
        return Ok(parent
            .parent()
            .ok_or("Application resources are missing.")?
            .join("Resources"));
    }
    Ok(parent.to_path_buf())
}

fn command(resources: &Path) -> Result<Command, String> {
    let frozen = bundled_executable(resources);
    if frozen.is_file() {
        return Ok(Command::new(frozen));
    }
    let project = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("Development checkout is unavailable.")?;
    // This fallback is compiled out of production. Installed applications always
    // run their own frozen runtime and never discover a user's Python or uv.
    if cfg!(debug_assertions) {
        let python = if cfg!(windows) {
            project.join(".venv/Scripts/python.exe")
        } else {
            project.join(".venv/bin/python")
        };
        if python.is_file() {
            let mut command = Command::new(python);
            command.arg("-m").arg("openecon.desktop_entry");
            command.env("PYTHONPATH", project.join("src"));
            return Ok(command);
        }
    }
    Err("The bundled OpenEconometrics runtime is missing. Reinstall the desktop application.".into())
}

impl LocalRuntime {
    pub fn start(resources: &Path, data_root: &Path) -> Result<Self, String> {
        std::fs::create_dir_all(data_root)
            .map_err(|_| "The OpenEconometrics data folder could not be created.")?;
        if std::fs::symlink_metadata(data_root)
            .map_err(|_| "The OpenEconometrics data folder is unavailable.")?
            .file_type()
            .is_symlink()
        {
            return Err("The OpenEconometrics data folder cannot be a symbolic link.".into());
        }
        let (instance_lock, requested_port, directory) = instance_and_port(data_root)?;
        let mut command = command(resources)?;
        // Preserve OS essentials only. Sign-in tokens and developer cloud
        // credentials are never inherited by the local Python process.
        let python_path = command
            .get_envs()
            .find(|(key, _)| *key == "PYTHONPATH")
            .and_then(|(_, value)| value)
            .map(|value| value.to_os_string());
        command.env_clear();
        for key in [
            "HOME",
            "USERPROFILE",
            "APPDATA",
            "LOCALAPPDATA",
            "SYSTEMROOT",
            "WINDIR",
            "COMSPEC",
            "HOMEDRIVE",
            "HOMEPATH",
            "TMPDIR",
            "TEMP",
            "TMP",
            "LANG",
        ] {
            if let Some(value) = std::env::var_os(key) {
                command.env(key, value);
            }
        }
        if let Some(path) = python_path {
            command.env("PYTHONPATH", path);
        }
        command
            .env("PYTHONNOUSERSITE", "1")
            .env("PYTHONUNBUFFERED", "1")
            .env("OPENECON_DESKTOP", "1")
            .env("OMP_NUM_THREADS", "1")
            .env("MKL_NUM_THREADS", "1")
            .env("OPENBLAS_NUM_THREADS", "1")
            .arg("--port")
            .arg(requested_port.to_string())
            .arg("--data-root")
            .arg(data_root)
            .current_dir(data_root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        #[cfg(feature = "qa-metrics")]
        command.arg("--qa-metrics");
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x08000000); // CREATE_NO_WINDOW
        }
        let mut child = command
            .spawn()
            .map_err(|_| "The local OpenEconometrics runtime could not start.")?;
        let stdout = child
            .stdout
            .take()
            .ok_or("The local runtime has no readiness channel.")?;
        let (sender, receiver) = mpsc::sync_channel(1);
        std::thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            let mut line = String::new();
            // The single ready record is deliberately small and bounded.
            let result = std::io::Read::by_ref(&mut reader)
                .take(16384)
                .read_line(&mut line)
                .map_err(|_| "The local runtime readiness channel failed.".to_string())
                .and_then(|_| {
                    serde_json::from_str::<Ready>(&line).map_err(|_| {
                        "The local runtime did not return a valid readiness record.".into()
                    })
                });
            let _ = sender.send(result);
            // Drain stdout without sending it to the webview or logging tokens.
            let _ = std::io::copy(&mut reader, &mut std::io::sink());
        });
        let result = receiver
            .recv_timeout(Duration::from_secs(90))
            .map_err(|_| "The local OpenEconometrics runtime startup timed out.".to_string())
            .and_then(|ready| ready)
            .and_then(|ready| {
                let expected = format!("http://127.0.0.1:{}", ready.port);
                if ready.kind != "ready"
                    || ready.port == 0
                    || (requested_port != 0 && ready.port != requested_port)
                    || ready.url.trim_end_matches('/') != expected
                {
                    return Err("The local runtime returned an untrusted origin.".into());
                }
                if requested_port == 0 {
                    remember_port(&directory, ready.port)?;
                }
                Ok(ready)
            });
        match result {
            Ok(ready) => Ok(Self {
                child,
                ready,
                _instance_lock: instance_lock,
            }),
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                Err(error)
            }
        }
    }

    pub fn stop(&mut self) {
        if let Some(mut input) = self.child.stdin.take() {
            let _ = input.write_all(b"{\"type\":\"shutdown\"}\n");
        }
        let deadline = Instant::now() + Duration::from_secs(3);
        while Instant::now() < deadline {
            if self.child.try_wait().ok().flatten().is_some() {
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for LocalRuntime {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn remembered_port_and_instance_lock_are_persistent() {
        let root =
            std::env::temp_dir().join(format!("openecon-port-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&root).unwrap();
        let (lock, port, dir) = instance_and_port(&root).unwrap();
        assert_eq!(port, 0);
        assert!(instance_and_port(&root).is_err());
        let probe = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let chosen = probe.local_addr().unwrap().port();
        drop(probe);
        remember_port(&dir, chosen).unwrap();
        drop(dir);
        drop(lock);
        let (lock, remembered, _) = instance_and_port(&root).unwrap();
        assert_eq!(remembered, chosen);
        drop(lock);
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn invalid_port_and_occupied_port_are_rejected() {
        let root =
            std::env::temp_dir().join(format!("openecon-port-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&root).unwrap();
        std::fs::write(root.join(".runtime-port.json"), "{\"port\":80}").unwrap();
        assert!(instance_and_port(&root).is_err());
        let probe = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        std::fs::write(
            root.join(".runtime-port.json"),
            format!("{{\"port\":{}}}", probe.local_addr().unwrap().port()),
        )
        .unwrap();
        assert!(instance_and_port(&root).is_err());
        drop(probe);
        std::fs::remove_dir_all(root).unwrap();
    }
    #[cfg(unix)]
    #[test]
    fn symlinked_port_preference_is_rejected() {
        let root =
            std::env::temp_dir().join(format!("openecon-port-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&root).unwrap();
        std::os::unix::fs::symlink("/etc/hosts", root.join(".runtime-port.json")).unwrap();
        assert!(instance_and_port(&root).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}
