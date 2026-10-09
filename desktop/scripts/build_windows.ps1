param([switch]$SkipRuntime)
$ErrorActionPreference = 'Stop'
$DesktopRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ProjectRoot = Split-Path -Parent $DesktopRoot
Set-Location $DesktopRoot
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) { throw 'Rust with the x86_64-pc-windows-msvc target and Visual Studio C++ Build Tools is required to build.' }
if ([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture -ne 'X64') { throw 'Run the Windows build on a Windows x64 host.' }
python -m venv "$DesktopRoot\.build-venv"
if ($LASTEXITCODE -ne 0) { throw 'Build environment creation failed.' }
$BuildPython = "$DesktopRoot\.build-venv\Scripts\python.exe"
& $BuildPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Build environment initialization failed.' }
# Build tools are isolated and pinned; application dependencies use uv.lock.
& $BuildPython -m pip install 'uv==0.9.26' 'pyinstaller==6.22.3'
if ($LASTEXITCODE -ne 0) { throw 'Build tool installation failed.' }
$RuntimeLock = & $BuildPython "$DesktopRoot\scripts\export_locked_runtime.py" | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw 'Locked runtime dependency export failed.' }
# CPU index + exact locked version avoids CUDA and a fresh dependency resolution.
# A +cpu local wheel version satisfies the lock's exact public version.
& $BuildPython -m pip install "torch==$($RuntimeLock.torch_version)" --no-deps --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'Locked CPU PyTorch installation failed.' }
& $BuildPython -m pip install --no-deps -r $RuntimeLock.requirements
if ($LASTEXITCODE -ne 0) { throw 'Locked runtime dependency installation failed.' }
& $BuildPython -m pip install --no-deps "$ProjectRoot\packages\openecon-charts" "$ProjectRoot"
if ($LASTEXITCODE -ne 0) { throw 'openecon wheel installation failed.' }
Set-Location "$ProjectRoot\web"
npm ci
if ($LASTEXITCODE -ne 0) { throw 'Web dependency installation failed.' }
Set-Location $DesktopRoot
npm ci
if ($LASTEXITCODE -ne 0) { throw 'Desktop dependency installation failed.' }
if ($SkipRuntime) { & $BuildPython scripts/build_desktop.py --bundles nsis --skip-runtime }
else { & $BuildPython scripts/build_desktop.py --bundles nsis }
if ($LASTEXITCODE -ne 0) { throw 'Windows desktop build failed.' }
Write-Host 'Windows x64 installer: desktop/src-tauri/target/release/bundle/nsis/'
