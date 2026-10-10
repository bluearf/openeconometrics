param(
    [Parameter(Mandatory = $true)][string]$InstallerPath,
    [Parameter(Mandatory = $true)][string]$OutputDirectory,
    [Parameter(Mandatory = $true)][string]$PreviousInstallerPath,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-f]{40}$')][string]$SourceCommit,
    [string]$ControllerPython = '',
    [string]$DiagnosticProducerReceipt = ''
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if (-not $IsWindows) { throw 'NSIS installed acceptance requires Windows.' }
if ([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture -ne 'X64') { throw 'This installed acceptance is specifically Windows x64.' }
if ($env:GITHUB_ACTIONS -ne 'true' -or -not $env:RUNNER_TEMP -or -not $env:GITHUB_WORKSPACE) { throw 'Use only a disposable GitHub Actions Windows runner for this installed acceptance.' }
if (-not $env:ImageOS -or $env:ImageOS -notmatch '^win' -or -not $env:ImageVersion -or $env:RUNNER_ENVIRONMENT -ne 'github-hosted') { throw 'Machine-level dependency/firewall tests require an ephemeral GitHub-hosted Windows image.' }
if (-not ([Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw 'Hosted dependency/firewall acceptance requires the disposable runner administrator.' }
$DesktopRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ProjectRoot = Split-Path -Parent $DesktopRoot
. (Join-Path $DesktopRoot 'scripts/windows_webview_attribution.ps1')
if (-not (Test-Path -LiteralPath $env:RUNNER_TEMP -PathType Container) -or -not (Test-Path -LiteralPath $env:GITHUB_WORKSPACE -PathType Container) -or -not $ProjectRoot.Equals((Resolve-Path -LiteralPath $env:GITHUB_WORKSPACE).Path, [StringComparison]::OrdinalIgnoreCase)) { throw 'GitHub runner workspace identity is incorrect.' }
$Installer = (Resolve-Path -LiteralPath $InstallerPath).Path
$PreviousInstaller = (Resolve-Path -LiteralPath $PreviousInstallerPath).Path
$PreviousInstallerSha = '6a6587e4b570f57ae783ce37cfbdc383d6862eadae0b74f648b08cd77a403938'
$PreviousSource = '8d844ef70a1f6c4ec78d3d3da6790c44591a36e9'
if ((Get-FileHash -LiteralPath $PreviousInstaller -Algorithm SHA256).Hash.ToLowerInvariant() -ne $PreviousInstallerSha) { throw 'The actual previous public Windows0.3.44 installer differs from its reviewed SHA-256.' }
$Output = [System.IO.Path]::GetFullPath($OutputDirectory)
[System.IO.Directory]::CreateDirectory($Output) | Out-Null
if (-not $ControllerPython) { $ControllerPython = (Get-Command python -ErrorAction Stop).Source }
$ControllerPython = (Resolve-Path -LiteralPath $ControllerPython).Path
$ControllerNode = (Get-Command node -ErrorAction Stop).Source
# Git attributes can retain native Windows CRLF; source identity must parse both endings.
$ExpectedSdk = [regex]::Match((Get-Content -LiteralPath (Join-Path $ProjectRoot 'pyproject.toml') -Raw), '(?m)^version = "([^"\r\n]+)"\r?$').Groups[1].Value
$ExpectedCharts = [regex]::Match((Get-Content -LiteralPath (Join-Path $ProjectRoot 'packages/openecon-charts/pyproject.toml') -Raw), '(?m)^version = "([^"\r\n]+)"\r?$').Groups[1].Value
$ExpectedDesktop = (Get-Content -LiteralPath (Join-Path $DesktopRoot 'src-tauri/tauri.conf.json') -Raw | ConvertFrom-Json).version
if (-not $ExpectedSdk -or -not $ExpectedCharts) { throw 'Source package version identity is unavailable.' }
if ([version]$ExpectedDesktop -le [version]'0.3.44') { throw 'Previous-version upgrade acceptance requires a source-versioned candidate newer than0.3.44.' }
$KnownFolderDataRoot = Join-Path ([Environment]::GetFolderPath([Environment+SpecialFolder]::ApplicationData)) 'org.openecon.desktop'
$KnownFolderLocalRoot = Join-Path ([Environment]::GetFolderPath([Environment+SpecialFolder]::LocalApplicationData)) 'org.openecon.desktop'
if ((Test-Path -LiteralPath $KnownFolderDataRoot) -or (Test-Path -LiteralPath $KnownFolderLocalRoot)) { throw 'An existing native application profile must not be touched by installed acceptance.' }
$OwnedRoot = Join-Path $env:RUNNER_TEMP ('OpenEconometrics ölçüm alanı ' + [guid]::NewGuid().ToString('N'))
$InstallDirectory = Join-Path $OwnedRoot 'Kurulu uygulama Türkçe'
$NativeProfile = Join-Path $OwnedRoot 'Yerel kullanıcı profili'
[System.IO.Directory]::CreateDirectory($OwnedRoot) | Out-Null
foreach ($Directory in @($NativeProfile, (Join-Path $NativeProfile 'Roaming'), (Join-Path $NativeProfile 'Local'), (Join-Path $OwnedRoot 'Geçici dosyalar'))) {
    [System.IO.Directory]::CreateDirectory($Directory) | Out-Null
}
# The exact test-only HKLM profile policy avoids guessing KnownFolder expansion.
# Its browser storage is kept intact across every cold restart/migration phase.
$OwnedBrowserRoot = Join-Path $NativeProfile 'Owned WebView2'
$BrowserProfileCandidates = @($OwnedBrowserRoot, (Join-Path $OwnedBrowserRoot 'EBWebView'))
foreach ($Candidate in $BrowserProfileCandidates) {
    if (Test-Path -LiteralPath $Candidate) { throw 'A WebView2 acceptance profile already exists.' }
}
$DiagnosticProducer = $null
if ($DiagnosticProducerReceipt) {
    if ($env:GITHUB_REPOSITORY -cne 'bluearf/openecon' -or $env:GITHUB_WORKFLOW -cne 'OpenEconometrics Windows UI diagnostic replay' -or $env:GITHUB_EVENT_NAME -cnotin @('workflow_dispatch', 'push') -or $env:GITHUB_REF -cne 'refs/heads/codex/windows-hosted-acceptance' -or $SourceCommit -cne 'e281bcf7b98c8b3cbb673a9fd749a054af495703') { throw 'Diagnostic reuse is restricted to the fixed trusted Windows producer and separate manual workflow.' }
    $ProducerPath = (Resolve-Path -LiteralPath $DiagnosticProducerReceipt).Path
    if ((Get-FileHash -LiteralPath $ProducerPath -Algorithm SHA256).Hash.ToLowerInvariant() -cne 'a99c8ab3f6868017cc38a809ee1d52d4e2d0e0170eb2ef2c3e9ed2a806896a56') { throw 'Diagnostic producer receipt bytes differ from the reviewed failed-run artifact.' }
    $DiagnosticProducer = Get-Content -LiteralPath $ProducerPath -Raw | ConvertFrom-Json -AsHashtable
    if ($DiagnosticProducer.status -cne 'error' -or $DiagnosticProducer.source_sha -cne $SourceCommit -or $DiagnosticProducer.installer_sha256 -cne 'df7b47988f548b018695bc4d05f0819e9be9b734787a058bb4d744c8e93c1a05' -or (Get-FileHash -LiteralPath $Installer -Algorithm SHA256).Hash.ToLowerInvariant() -cne $DiagnosticProducer.installer_sha256 -or $DiagnosticProducer.desktop_version -cne $ExpectedDesktop -or $DiagnosticProducer.sdk_version -cne $ExpectedSdk -or $DiagnosticProducer.charts_version -cne $ExpectedCharts) { throw 'Diagnostic reuse did not retain its exact old installer and version identities.' }
}
$Receipt = [ordered]@{
    status = 'running'; source_sha = $SourceCommit; controller_source_sha = $env:GITHUB_SHA; desktop_version = $ExpectedDesktop
    acceptance_kind = if ($null -ne $DiagnosticProducer) { 'diagnostic replay of an unverified fixed older producer; no release/current-source acceptance' } else { 'fresh exact-source installed release acceptance' }
    diagnostic_producer_run_id = if ($null -ne $DiagnosticProducer) { '37868521174' } else { $null }
    sdk_version = $ExpectedSdk; charts_version = $ExpectedCharts
    installer = $Installer; installer_sha256 = (Get-FileHash -LiteralPath $Installer -Algorithm SHA256).Hash.ToLowerInvariant()
    installed_path = $InstallDirectory; owned_profile = $NativeProfile
    native_profile_scope = 'fresh disposable runner KnownFolder, child APPDATA, or child USERPROFILE/AppData/Roaming; attributed by native runtime lock and persisted port after launch'
    native_knownfolder_profile_before_launch = $KnownFolderDataRoot; native_knownfolder_preexisted = $false
    native_knownfolder_webview_profile_before_launch = $KnownFolderLocalRoot
    exact_webview_profile_candidates_absent_before_launch = $BrowserProfileCandidates
    os_description = [System.Runtime.InteropServices.RuntimeInformation]::OSDescription
    os_architecture = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString()
    runner_image = $env:ImageOS; runner_image_version = $env:ImageVersion
    controller_python = $ControllerPython; controller_scope = 'stdlib HTTP orchestration only; application uses installed frozen Python'
    human_data_access = $false; checks = [ordered]@{}
    previous_installer = @{ version = '0.3.44'; source_sha = $PreviousSource; sha256 = $PreviousInstallerSha; public_url = 'https://github.com/bluearf/openeconometrics/releases/download/v0.3.44-windows-alpha.1/OpenEconometrics_0.3.44_x64-setup.exe' }
    boundaries = @('Unsigned / Authenticode and SmartScreen remain separate', 'Actual GitHub-hosted Windows x64; consumer machines and Windows ARM64 remain separate', 'Live cloud/team authentication remains separate', 'Embedded WebView2 bootstrapper requires Microsoft network access')
}
$Native = $null
$NativeLaunchAttempted = $false
$DefaultOwnedBrowserProcesses = @()
$ObservedNativeDataRoot = $null
$FirewallBefore = @()
$FirewallRuleNames = [System.Collections.Generic.List[string]]::new()
$FirewallRuleRecords = [System.Collections.Generic.List[object]]::new()
$UiProjectName = 'Windows UI ölçüm'
$OwnedDebugPolicy = $null

function New-OwnedProcess([string]$Executable, [string[]]$Arguments, [string]$RawArguments = '', [int]$BrowserDebugPort = 0, [switch]$UiController) {
    $Info = [System.Diagnostics.ProcessStartInfo]::new()
    $Info.FileName = $Executable
    $Info.UseShellExecute = $false
    $Info.WorkingDirectory = $OwnedRoot
    $Info.RedirectStandardOutput = $true
    $Info.RedirectStandardError = $true
    if ($RawArguments) { $Info.Arguments = $RawArguments }
    else { foreach ($Argument in $Arguments) { $Info.ArgumentList.Add($Argument) } }
    $Info.Environment.Clear()
    foreach ($Name in @('SYSTEMROOT', 'WINDIR', 'COMSPEC', 'HOMEDRIVE', 'LANG')) {
        $Value = [Environment]::GetEnvironmentVariable($Name)
        if ($null -ne $Value) { $Info.Environment[$Name] = $Value }
    }
    $Info.Environment['PATH'] = Join-Path $env:SYSTEMROOT 'System32'
    $Info.Environment['USERPROFILE'] = $NativeProfile
    $Info.Environment['APPDATA'] = Join-Path $NativeProfile 'Roaming'
    $Info.Environment['LOCALAPPDATA'] = Join-Path $NativeProfile 'Local'
    $Info.Environment['TEMP'] = Join-Path $OwnedRoot 'Geçici dosyalar'
    $Info.Environment['TMP'] = $Info.Environment['TEMP']
    $Info.Environment['PYTHONNOUSERSITE'] = '1'
    if ($UiController) {
        if ($env:GITHUB_ACTIONS -ne 'true' -or -not $Executable.Equals($ControllerNode, [StringComparison]::OrdinalIgnoreCase) -or $Arguments[0] -ne (Join-Path $DesktopRoot 'scripts/verify_windows_ui.mjs')) { throw 'Only the exact hosted Node UI controller may retain its validated Actions guard.' }
        $Info.Environment['GITHUB_ACTIONS'] = 'true'
    }
    if ($BrowserDebugPort) {
        if ($BrowserDebugPort -lt 1024 -or $BrowserDebugPort -gt 65535) { throw 'Owned debugger port is outside its loopback range.' }
        $Info.Environment['WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS'] = "--remote-debugging-address=127.0.0.1 --remote-debugging-port=$BrowserDebugPort"
    }
    $Process = [System.Diagnostics.Process]::new()
    $Process.StartInfo = $Info
    try {
        if (-not $Process.Start()) { throw 'The owned process could not start.' }
        return @{ process = $Process; stdout = $Process.StandardOutput.ReadToEndAsync(); stderr = $Process.StandardError.ReadToEndAsync(); disposed = $false }
    }
    catch { $Process.Dispose(); throw }
}

function Wait-OwnedProcess($Owned, [int]$TimeoutSeconds, [string]$Name) {
    try {
        if (-not $Owned.process.WaitForExit($TimeoutSeconds * 1000)) {
            $Owned.process.Kill($true)
            $Owned.process.WaitForExit(10000) | Out-Null
            throw "$Name exceeded its bounded timeout."
        }
        $Text = $Owned.stdout.GetAwaiter().GetResult()
        $Errors = $Owned.stderr.GetAwaiter().GetResult()
        [System.IO.File]::WriteAllText((Join-Path $Output "$Name.stderr"), $Errors, [System.Text.UTF8Encoding]::new($false))
        if ($Owned.process.ExitCode -ne 0) { throw "$Name failed with exit code $($Owned.process.ExitCode). See retained stderr." }
        # Native JSON may grow additional readiness fields later. Persist only its explicit safe summary.
        if ($Name -eq 'installed-native-smoke') {
            $NativeResult = $Text | ConvertFrom-Json
            $SafeText = @{ status = $NativeResult.status; bundled = $NativeResult.bundled; stable_origin = $NativeResult.stable_origin; ready_seconds = $NativeResult.ready_seconds; warm_ready_seconds = $NativeResult.warm_ready_seconds; execution_status = $NativeResult.execution.status } | ConvertTo-Json
        }
        else { $SafeText = $Text }
        [System.IO.File]::WriteAllText((Join-Path $Output "$Name.stdout"), $SafeText, [System.Text.UTF8Encoding]::new($false))
        return $Text
    }
    finally {
        $Owned.process.Dispose()
        $Owned.disposed = $true
    }
}

function Install-Owned([string]$Package = $Installer, [string]$Name = 'nsis-install') {
    # NSIS /D must be last and unquoted, including when the directory contains spaces.
    $Process = New-OwnedProcess -Executable $Package -RawArguments ('/S /D=' + $InstallDirectory)
    Wait-OwnedProcess -Owned $Process -TimeoutSeconds 600 -Name $Name | Out-Null
    $Executables = @(Get-ChildItem -LiteralPath $InstallDirectory -Filter 'openecon-desktop.exe' -File)
    if ($Executables.Count -ne 1) { throw 'NSIS did not install exactly one native application executable in the owned directory.' }
    return $Executables[0].FullName
}

function Get-WebViewState {
    $Client = 'SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}'
    $ClientState = $Client.Replace('\Clients\', '\ClientState\')
    $Rows = @()
    $Commands = @()
    foreach ($Hive in @([Microsoft.Win32.RegistryHive]::LocalMachine, [Microsoft.Win32.RegistryHive]::CurrentUser)) {
        foreach ($View in @([Microsoft.Win32.RegistryView]::Registry32, [Microsoft.Win32.RegistryView]::Registry64)) {
            $Base = [Microsoft.Win32.RegistryKey]::OpenBaseKey($Hive, $View)
            try {
                $Key = $Base.OpenSubKey($Client)
                try { if ($null -ne $Key) { $Rows += @{ hive = $Hive.ToString(); view = $View.ToString(); pv = [string]$Key.GetValue('pv', '') } } }
                finally { if ($null -ne $Key) { $Key.Dispose() } }
                foreach ($Path in @($ClientState, 'SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Microsoft EdgeWebView')) {
                    $Key = $Base.OpenSubKey($Path)
                    try {
                        if ($null -ne $Key) {
                            $Command = [string]$Key.GetValue('UninstallString', '')
                            if ($Command) { $Commands += @{ command = $Command; arguments = [string]$Key.GetValue('UninstallArguments', '') } }
                        }
                    }
                    finally { if ($null -ne $Key) { $Key.Dispose() } }
                }
            }
            finally { $Base.Dispose() }
        }
    }
    $Roots = @([Environment]::GetFolderPath([Environment+SpecialFolder]::ProgramFiles), [Environment]::GetFolderPath([Environment+SpecialFolder]::ProgramFilesX86), [Environment]::GetFolderPath([Environment+SpecialFolder]::LocalApplicationData)) | Select-Object -Unique
    $Files = @()
    $Preview = @()
    $Updater = @()
    foreach ($Root in $Roots) {
        $Application = Join-Path $Root 'Microsoft/EdgeWebView/Application'
        if (Test-Path -LiteralPath $Application -PathType Container) {
            foreach ($Directory in @(Get-ChildItem -LiteralPath $Application -Directory)) {
                if (($Directory.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw 'A WebView dependency directory must not be a reparse point.' }
                $File = Join-Path $Directory.FullName 'msedgewebview2.exe'
                if (Test-Path -LiteralPath $File -PathType Leaf) { $Files += $File }
            }
        }
        foreach ($Channel in @('Edge Beta', 'Edge Dev', 'Edge SxS')) {
            if (Test-Path -LiteralPath (Join-Path $Root "Microsoft/$Channel/Application") -PathType Container) { $Preview += $Channel }
        }
        foreach ($Name in @('MicrosoftEdgeUpdate.exe', 'Disabled_MicrosoftEdgeUpdate.exe')) {
            $File = Join-Path $Root "Microsoft/EdgeUpdate/$Name"
            if (Test-Path -LiteralPath $File -PathType Leaf) { $Updater += @{ file = $File; sha256 = (Get-FileHash -LiteralPath $File -Algorithm SHA256).Hash.ToLowerInvariant() } }
        }
    }
    $RegistrationRoots = @{
        LocalMachine = @([Environment]::GetFolderPath([Environment+SpecialFolder]::ProgramFiles), [Environment]::GetFolderPath([Environment+SpecialFolder]::ProgramFilesX86))
        CurrentUser = @([Environment]::GetFolderPath([Environment+SpecialFolder]::LocalApplicationData))
    }
    $RegisteredFiles = @(Get-RegisteredWebViewExecutables -Registrations $Rows -InstallationRoots $RegistrationRoots -DiscoveredFiles $Files)
    return @{ registrations = $Rows; runtime_executables = @($Files | Select-Object -Unique); registered_runtime_executables = @($RegisteredFiles | Select-Object -Unique); registered_uninstallers = $Commands; preview_channels = @($Preview | Select-Object -Unique); updater = $Updater }
}

function Remove-HostedWebView {
    # This modifies only the expressly guarded ephemeral GitHub machine, never a user computer.
    $Before = Get-WebViewState
    $Receipt.webview_bootstrap = @{ before = $Before; absence_verified = $false; network_required = $true }
    if ($Before.preview_channels.Count) { throw 'An installed Edge preview channel would invalidate genuine WebView2 absence.' }
    if (-not $Before.runtime_executables.Count) { throw 'The hosted runtime fixture must begin with an actual WebView2 executable to remove.' }
    $Uninstaller = $null
    foreach ($Registered in $Before.registered_uninstallers) {
        $Match = [regex]::Match($Registered.command, '^"([^\"]+setup\.exe)"\s*(.*)$', 'IgnoreCase')
        if (-not $Match.Success) { $Match = [regex]::Match($Registered.command, '^(.+?setup\.exe)(?:\s+(.*))?$', 'IgnoreCase') }
        if (-not $Match.Success) { continue }
        $File = $Match.Groups[1].Value
        $Allowed = @($Before.runtime_executables | ForEach-Object { Join-Path (Split-Path -Parent $_) 'Installer/setup.exe' })
        $Args = ($Match.Groups[2].Value + ' ' + $Registered.arguments).Trim()
        if ($File -in $Allowed -and $Args -match '--uninstall\b' -and $Args -match '--msedgewebview\b') { $Uninstaller = @{ file = $File; arguments = $Args }; break }
    }
    if ($null -eq $Uninstaller) { throw 'A genuine registered Microsoft WebView2 uninstaller is unavailable.' }
    $Signature = Get-AuthenticodeSignature -LiteralPath $Uninstaller.file
    if ($Signature.Status -ne 'Valid' -or $Signature.SignerCertificate.Subject -notmatch 'O=Microsoft Corporation') { throw 'The registered WebView2 setup must have a valid Microsoft signature.' }
    $SetupHash = (Get-FileHash -LiteralPath $Uninstaller.file -Algorithm SHA256).Hash.ToLowerInvariant()
    foreach ($Process in @(Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -in $Before.runtime_executables })) { Stop-Process -Id $Process.ProcessId -Force -ErrorAction Stop }
    $Info = [Diagnostics.ProcessStartInfo]::new($Uninstaller.file, ($Uninstaller.arguments + ' --force-uninstall'))
    $Info.UseShellExecute = $false
    $Info.RedirectStandardOutput = $true
    $Info.RedirectStandardError = $true
    $Process = [Diagnostics.Process]::new()
    $Process.StartInfo = $Info
    try {
        if (-not $Process.Start()) { throw 'The registered WebView2 uninstaller did not start.' }
        $Out = $Process.StandardOutput.ReadToEndAsync()
        $Err = $Process.StandardError.ReadToEndAsync()
        if (-not $Process.WaitForExit(180000)) { $Process.Kill($true); $Process.WaitForExit(10000) | Out-Null; throw 'WebView2 uninstall exceeded its bounded CI timeout.' }
        $Receipt.webview_bootstrap.uninstall = @{ file = $Uninstaller.file; setup_sha256 = $SetupHash; exit_code = $Process.ExitCode; registered_arguments = $Uninstaller.arguments; force_uninstall_requested = $true }
        [IO.File]::WriteAllText((Join-Path $Output 'webview-uninstall.stdout'), $Out.GetAwaiter().GetResult())
        [IO.File]::WriteAllText((Join-Path $Output 'webview-uninstall.stderr'), $Err.GetAwaiter().GetResult())
    }
    finally { $Process.Dispose() }
    $After = Get-WebViewState
    $Receipt.webview_bootstrap.after_uninstall = $After
    # Tauri2.12.1 skips its bootstrapper for ANY nonempty pv, including0.0.0.0.
    # Do not edit registry/files/environment to pretend the dependency is absent.
    if ($After.runtime_executables.Count -or @($After.registrations | Where-Object { $_.pv }).Count) { throw 'The real WebView2 dependency was not absent after registered uninstall.' }
    $Receipt.webview_bootstrap.absence_verified = $true
}

function Enable-OwnedFirewall([string[]]$Programs) {
    Assert-OwnedFirewallHost
    $ExpectedPrograms = @($Executable, $Runtime)
    if ($FirewallBefore.Count -or $FirewallRuleRecords.Count -or $Programs.Count -ne 2 -or $Programs[0] -cne $ExpectedPrograms[0] -or $Programs[1] -cne $ExpectedPrograms[1]) { throw 'Only the two exact fresh installed executable paths may receive owned firewall rules.' }
    $script:FirewallBefore = @(Get-NetFirewallProfile | Select-Object Name, Enabled)
    if ($FirewallBefore.Count -ne 3) { throw 'Expected the three actual Windows firewall profiles.' }
    foreach ($Profile in $FirewallBefore) { Set-NetFirewallProfile -Name $Profile.Name -Enabled True }
    if (@(Get-NetFirewallProfile | Where-Object { $_.Enabled -ne 'True' }).Count) { throw 'Actual Windows firewall profiles did not enable.' }
    foreach ($Program in $Programs) {
        $Name = 'OpenEcon-owned-' + [guid]::NewGuid().ToString('N')
        if (Get-NetFirewallRule -Name $Name -ErrorAction SilentlyContinue) { throw 'A generated owned firewall rule name already exists.' }
        $FirewallRuleNames.Add($Name)
        $Record = @{ name = $Name; program = $Program; enabled = $true; readback_verified = $false }
        $FirewallRuleRecords.Add($Record)
        New-NetFirewallRule -Name $Name -DisplayName $Name -Direction Outbound -Action Block -Program $Program -RemoteAddress Internet -Profile Any -Enabled True | Out-Null
        $Rule = Get-NetFirewallRule -Name $Name
        $Address = $Rule | Get-NetFirewallAddressFilter
        $Application = $Rule | Get-NetFirewallApplicationFilter
        Assert-OwnedFirewallRule $Record $Rule $Address $Application 'True'
        $Record.readback_verified = $true
    }
    $Receipt.firewall = @{ actual_profiles_enabled = $true; owned_outbound_internet_block_rules = @($FirewallRuleNames); exact_rule_identities = @($FirewallRuleRecords.ToArray()); programs = $Programs; loopback_excluded_from_remote_internet_scope = $true; phases = [System.Collections.Generic.List[object]]::new(); scope = 'Actual offline native/numerical computation; owned application Internet blocks disabled only for real PyPI package installation, then re-enabled for migration and native cold reopen' }
    $Receipt.firewall.phases.Add(@{ phase = 'offline_native_create'; owned_rules_enabled = $true; exact_readback_verified = $true; three_profiles_enabled = $true })
}

function Invoke-Migration([string]$Phase, [string]$Name, [string]$Sdk, [string]$Charts, [string]$Sha) {
    $Result = Join-Path $Output "$Name.json"
    & $ControllerPython -I (Join-Path $DesktopRoot 'scripts/verify_windows_runtime.py') --runtime (Join-Path $InstallDirectory 'runtime/openecon-runtime/openecon-runtime.exe') --profile (Join-Path $OwnedRoot 'Yükseltme analiz projesi') --output $Result --sdk-version $Sdk --charts-version $Charts --source-sha $Sha --migration-phase $Phase --snapshot (Join-Path $OwnedRoot 'migration-snapshot.json')
    if ($LASTEXITCODE -ne 0) { throw "Actual installed migration phase failed: $Name" }
    $Record = Get-Content -LiteralPath $Result -Raw | ConvertFrom-Json
    if ($Record.status -ne 'passed' -or $Record.migration_phase -ne $Phase -or -not $Record.owned_runtime_stopped -or $Record.source_sha -ne $Sha) { throw "Migration receipt did not pass its exact phase identity: $Name" }
    return @{ file = $Result; sha256 = (Get-FileHash -LiteralPath $Result -Algorithm SHA256).Hash.ToLowerInvariant(); source_sha = $Sha; sdk_version = $Sdk; charts_version = $Charts; runtime_sha256 = (Get-FileHash -LiteralPath (Join-Path $InstallDirectory 'runtime/openecon-runtime/openecon-runtime.exe') -Algorithm SHA256).Hash.ToLowerInvariant() }
}

function Invoke-DefaultWebViewStartup {
    if ($null -ne $OwnedDebugPolicy) { throw 'Default WebView2 startup must precede owned browser overrides.' }
    $Base = [Microsoft.Win32.RegistryKey]::OpenBaseKey([Microsoft.Win32.RegistryHive]::LocalMachine, [Microsoft.Win32.RegistryView]::Registry64)
    try {
        foreach ($Path in @('SOFTWARE\Policies\Microsoft\Edge\WebView2\AdditionalBrowserArguments', 'SOFTWARE\Policies\Microsoft\Edge\WebView2\UserDataFolder')) {
            $Key = $Base.OpenSubKey($Path, $false)
            try {
                if ($null -ne $Key -and @($Key.GetValueNames() | Where-Object { $_ -in @('openecon-desktop.exe', 'org.openecon.desktop', '*') }).Count) { throw 'Default native startup must not inherit an existing app or global browser override policy.' }
            }
            finally { if ($null -ne $Key) { $Key.Dispose() } }
        }
    }
    finally { $Base.Dispose() }
    $script:NativeLaunchAttempted = $true
    # This existing diagnostic flag changes only the title's trusted-IPC proof.
    # The production window, local origin and default WebView2 storage are used.
    $script:Native = New-OwnedProcess -Executable $Executable -Arguments @('--diagnostic-window')
    $Started = $Native.process.StartTime.ToUniversalTime()
    $Deadline = [DateTime]::UtcNow.AddSeconds(120)
    while ([DateTime]::UtcNow -lt $Deadline) {
        if ($Native.process.HasExited) { throw 'Default native WebView2 startup exited before trusted IPC readiness.' }
        $Native.process.Refresh()
        if ($Native.process.MainWindowTitle -ceq 'OpenEconometrics diagnostics: IPC verified') { break }
        Start-Sleep -Milliseconds 200
    }
    if ($Native.process.MainWindowTitle -cne 'OpenEconometrics diagnostics: IPC verified') { throw 'The actual default WebView2 did not render and verify trusted native IPC.' }
    $Processes = @{}
    foreach ($Process in @(Get-CimInstance Win32_Process)) { $Processes[[uint32]$Process.ProcessId] = $Process }
    $script:DefaultOwnedBrowserProcesses = @()
    foreach ($Process in $Processes.Values) {
        if ($Process.Name -ne 'msedgewebview2.exe') { continue }
        $Identity = Get-WebViewLineage -Processes $Processes -BrowserPid $Process.ProcessId -NativePid $Native.process.Id -NativeStarted $Started -NativeExecutable $Executable
        if ($Identity.owned_native_descendant) {
            if (-not $Process.ExecutablePath -or @($RestoredWebView.registered_runtime_executables | Where-Object { $_.Equals($Process.ExecutablePath, [StringComparison]::OrdinalIgnoreCase) }).Count -ne 1) { throw 'Default native WebView2 did not use its exact registered Microsoft runtime.' }
            $script:DefaultOwnedBrowserProcesses += @{ pid = $Process.ProcessId; creation_date = ([DateTime]$Process.CreationDate).ToUniversalTime().ToString('o'); exact_registered_runtime = $true }
        }
    }
    if ($DefaultOwnedBrowserProcesses.Count -lt 1 -or $DefaultOwnedBrowserProcesses.Count -gt 64) { throw 'Default native WebView2 processes cannot be bounded and attributed.' }
    $Screenshot = Capture-NativeWindow -Process $Native.process -Name 'installed-native-default-webview.png'
    if (-not $Native.process.CloseMainWindow()) { throw 'Default native application did not receive its close request.' }
    Wait-OwnedProcess -Owned $Native -TimeoutSeconds 45 -Name 'installed-native-default-webview' | Out-Null
    $script:Native = $null
    $Deadline = [DateTime]::UtcNow.AddSeconds(30)
    do {
        $Remaining = @($DefaultOwnedBrowserProcesses | Where-Object {
            $Process = Get-CimInstance Win32_Process -Filter "ProcessId=$($_.pid)"
            $null -ne $Process -and ([DateTime]$Process.CreationDate).ToUniversalTime().ToString('o') -ceq $_.creation_date
        })
        if ($Remaining.Count -eq 0) { break }
        Start-Sleep -Milliseconds 200
    } while ([DateTime]::UtcNow -lt $Deadline)
    if ($Remaining.Count) { throw 'Attributed default WebView2 children survived native shutdown.' }
    return @{ actual_default_webview_storage = $true; user_data_folder_override = $false; debugger_override = $false; trusted_native_ipc_verified = $true; title_only_diagnostic_flag = $true; screenshot = $Screenshot; attributed_browser_processes = $DefaultOwnedBrowserProcesses; actual_owned_browser_children_stopped = $true }
}

function Invoke-NormalUi([string]$Mode) {
    $Listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, 0)
    $Listener.ExclusiveAddressUse = $true
    $Listener.Start()
    $Port = $Listener.LocalEndpoint.Port
    $Listener.Stop()
    Set-OwnedDebugPolicy -Port $Port -OwnedRoot $OwnedRoot -NativeExecutable $Executable -BrowserProfile $OwnedBrowserRoot
    try { return Invoke-NormalUiCore -Mode $Mode -Port $Port }
    finally { Remove-OwnedDebugPolicy }
}

function Invoke-NormalUiCore([string]$Mode, [int]$Port) {
    $script:NativeLaunchAttempted = $true
    $PolicyEvidence = @($OwnedDebugPolicy.entries | ForEach-Object { @{ registry_hive = 'HKLM'; registry_view = 'Registry64'; path = $_.path; exact_executable = $_.name; value = $_.value; readback_verified = $_.readback_verified; preexisting_key = $_.key_existed } })
    $script:Native = New-OwnedProcess -Executable $Executable -Arguments @() -BrowserDebugPort $Port
    $Started = $Native.process.StartTime.ToUniversalTime()
    $Deadline = [DateTime]::UtcNow.AddSeconds(120)
    $OwnedDebugger = $null
    $DebuggerEvidence = $null
    $LastListeners = @()
    $LastSummaries = @()
    while ([DateTime]::UtcNow -lt $Deadline) {
        if ($Native.process.HasExited) { throw 'The real normal native app exited before its owned WebView2 debugger.' }
        $Processes = @{}
        foreach ($Process in @(Get-CimInstance Win32_Process)) { $Processes[[uint32]$Process.ProcessId] = $Process }
        $LastListeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object LocalAddress, LocalPort, OwningProcess)
        $LastSummaries = @()
        foreach ($Connection in $LastListeners) {
            $Evidence = Get-WebViewAttribution -Processes $Processes -BrowserPid $Connection.OwningProcess -NativePid $Native.process.Id -NativeStarted $Started -NativeExecutable $Executable -FreshProfileCandidates $BrowserProfileCandidates -RuntimeExecutables $RestoredWebView.registered_runtime_executables -Port $Port
            $LastSummaries += $Evidence
            if ($Evidence.owned -and $Connection.LocalAddress -in @('127.0.0.1', '::1')) {
                Assert-OwnedWebViewPolicyProfile -OwnedRoot $OwnedRoot -NativeExecutable $Executable -BrowserProfile $OwnedBrowserRoot | Out-Null
                $OwnedDebugger = $Connection.OwningProcess; $DebuggerEvidence = $Evidence; break
            }
        }
        if ($null -ne $OwnedDebugger) { break }
        Start-Sleep -Milliseconds 200
    }
    if ($null -eq $OwnedDebugger) {
        $Receipt.debugger_failure = @{ mode = $Mode; port = $Port; policy_channel = 'HKLM Registry64, exact executable name'; policy_readback = $PolicyEvidence; native_pid = $Native.process.Id; native_creation_date = $Started.ToString('o'); listeners = $LastListeners; listener_process_summaries = $LastSummaries }
        throw 'The normal app debugger could not be attributed to a fresh exact native-descendant loopback WebView2 process.'
    }
    $Result = Join-Path $Output "installed-native-ui-$Mode.json"
    $UiArguments = @((Join-Path $DesktopRoot 'scripts/verify_windows_ui.mjs'), '--port', "$Port", '--output', $Result, '--mode', $Mode, '--project-name', $UiProjectName)
    if ($null -eq $DiagnosticProducer) { $UiArguments += @('--regularized-all-family', 'true', '--saved-binomial', 'true', '--source-sha', $SourceCommit) }
    if ($null -eq $DiagnosticProducer -and $Mode -eq 'reopen') {
        $UiArguments += @('--original-receipt', (Join-Path $Output 'installed-native-ui-create.json'))
    }
    $Controller = New-OwnedProcess -Executable $ControllerNode -UiController -Arguments $UiArguments
    try { Wait-OwnedProcess -Owned $Controller -TimeoutSeconds 600 -Name "installed-native-ui-$Mode-controller" | Out-Null }
    catch {
        try { $Receipt.native_ui_failure_screenshot = Capture-NativeWindow -Process $Native.process -Name "installed-native-$Mode-failure.png" }
        catch { $Receipt.native_ui_failure_screenshot_error = $_.Exception.Message }
        throw
    }
    $Record = Get-Content -LiteralPath $Result -Raw | ConvertFrom-Json -AsHashtable
    $Gate = if ($Mode -eq 'create') { 'normal_ui_run_ols_logit_poisson_oracles_and_saved_state' } else { 'cold_reopen_visible_history_without_refit' }
    if ($Record.status -ne 'passed' -or $Record.mode -ne $Mode -or $Record.desktop_ipc_verified -ne $true -or $Record.project_name -ne $UiProjectName -or $Record.checks[$Gate] -ne $true) { throw "The actual normal native UI did not complete its $Mode gate." }
    if ($null -eq $DiagnosticProducer) {
        $RegularizedGate = if ($Mode -eq 'create') { 'normal_ui_run_four_regularized_methods_complete_state_and_tables' } else { 'cold_reopen_four_regularized_tables_without_refit' }
        if ($Record.regularized_all_family_enabled -ne $true -or $Record.checks.four_regularized_tables_rendered -ne $true -or $Record.checks[$RegularizedGate] -ne $true -or $Record.original_user_command_count -ne 1 -or $Record.regularized.saved_files.Count -ne 10 -or $Record.regularized.tables.Count -ne 4 -or $Record.regularized.rows_per_table -ne 4) { throw 'The actual native UI did not complete its four-family model/persistence gate.' }
        $SavedBinomialGate = if ($Mode -eq 'create') { 'normal_ui_run_saved_binomial_full_state_and_tables' } else { 'cold_reopen_saved_binomial_tables_without_refit' }
        if ($Record.saved_binomial_enabled -ne $true -or $Record.checks.eight_saved_binomial_tables_rendered -ne $true -or $Record.checks[$SavedBinomialGate] -ne $true -or $Record.saved_binomial.source_sha -cne $SourceCommit -or $Record.saved_binomial.saved_files.Count -ne 10 -or $Record.saved_binomial.tables.Count -ne 8 -or $Record.saved_binomial.rows_per_table -ne 7) { throw 'The actual native UI did not complete its eight saved-binomial full-state/table gate.' }
    }
    $Screenshot = Capture-NativeWindow -Process $Native.process -Name "installed-native-$Mode.png"
    if (-not $Native.process.CloseMainWindow()) { throw 'The normal native application did not receive its close request.' }
    Wait-OwnedProcess -Owned $Native -TimeoutSeconds 45 -Name "installed-native-window-$Mode" | Out-Null
    $script:Native = $null
    return @{ receipt = $Result; sha256 = (Get-FileHash -LiteralPath $Result -Algorithm SHA256).Hash.ToLowerInvariant(); screenshot = $Screenshot; debugger_port = $Port; debugger_owned_pid = $OwnedDebugger; debugger_attribution = $DebuggerEvidence; debugger_policy_scope = 'Temporary HKLM Registry64 exact executable debugger and owned browser-storage values; removed after this phase without resetting storage'; policy_readback = $PolicyEvidence; owned_browser_profile = $OwnedBrowserRoot; child_environment_scope = 'Only this owned native process; never global' }
}

function Get-NativeProjectSnapshot([string]$DataRoot) {
    $CatalogFile = Join-Path $DataRoot 'local-projects.json'
    $CatalogItem = Get-Item -LiteralPath $CatalogFile -Force
    if ($CatalogItem.Length -gt 512KB -or ($CatalogItem.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'The real native project catalog is unsafe.' }
    $Catalog = Get-Content -LiteralPath $CatalogFile -Raw | ConvertFrom-Json -AsHashtable
    $Projects = @($Catalog.projects | Where-Object { $_.name -eq $UiProjectName })
    if ($Projects.Count -ne 1 -or $Projects[0].id -notmatch '^[0-9a-f]{32}$') { throw 'Exactly one actual normal-UI project is required for uninstall preservation.' }
    $ProjectDirectory = Join-Path $DataRoot ('projects/' + $Projects[0].id)
    if (((Get-Item -LiteralPath $ProjectDirectory -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'The real native project must not be a reparse point.' }
    $Items = @(Get-ChildItem -LiteralPath $ProjectDirectory -Recurse -Force)
    if ($Items.Count -gt 2048 -or @($Items | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) { throw 'The owned project snapshot exceeds its bounds or contains a reparse point.' }
    $Rows = [ordered]@{}
    $Total = 0L
    foreach ($Item in @($Items | Where-Object { -not $_.PSIsContainer } | Sort-Object FullName)) {
        $Total += $Item.Length
        if ($Total -gt 128MB) { throw 'The owned project snapshot exceeds its byte bound.' }
        $Relative = [IO.Path]::GetRelativePath($ProjectDirectory, $Item.FullName).Replace('\', '/')
        $Rows[$Relative] = @{ sha256 = (Get-FileHash -LiteralPath $Item.FullName -Algorithm SHA256).Hash.ToLowerInvariant(); bytes = $Item.Length }
    }
    $RequiredFiles = @('windows-ui.csv', 'windows-ui-ols.json', 'windows-ui-logit.json', 'windows-ui-poisson.json', 'console/history.json')
    if ($null -eq $DiagnosticProducer) {
        $RequiredFiles += @('regularized-all-family-models.json', 'regularized-all-family-data.csv')
        foreach ($Method in @('ridge', 'lasso', 'elasticnet', 'pls')) { $RequiredFiles += @("regularized-all-family-$Method.json", "regularized-all-family-$Method.tex") }
        $RequiredFiles += @('saved-binomial-suest-models.json', 'saved-binomial-suest-data.parquet')
        foreach ($Case in @('cloglog_offset', 'cloglog_frequency_cluster', 'fractional_logit_corners', 'fractional_probit_corners', 'fractional_logit_frequency', 'fractional_probit_probability_cluster', 'fractional_logit_analytic', 'fractional_probit_analytic_cluster')) { $RequiredFiles += "saved-binomial-suest-$Case.tex" }
    }
    foreach ($Required in $RequiredFiles) {
        if (-not $Rows.Contains($Required)) { throw "The actual UI project's required saved data/result/history file is missing: $Required" }
    }
    return @{ project_id = $Projects[0].id; project_name = $UiProjectName; catalog_sha256 = (Get-FileHash -LiteralPath $CatalogFile -Algorithm SHA256).Hash.ToLowerInvariant(); files = $Rows; file_count = $Rows.Count; total_bytes = $Total }
}

function Assert-RegularizedUiFiles([string]$UiReceiptFile, $Snapshot, $Previous = $null) {
    $Ui = Get-Content -LiteralPath $UiReceiptFile -Raw | ConvertFrom-Json -AsHashtable
    $Saved = $Ui.regularized.saved_files
    if ($Saved.Count -ne 10) { throw 'The UI receipt lacks its exact ten saved regularized files.' }
    foreach ($Name in $Saved.Keys) {
        if (-not $Snapshot.files.Contains($Name) -or $Saved[$Name].sha256 -cne $Snapshot.files[$Name].sha256 -or $Saved[$Name].bytes -ne $Snapshot.files[$Name].bytes) { throw 'A UI regularized saved marker differs from the actual native project file bytes.' }
        if ($null -ne $Previous -and (-not $Previous.ContainsKey($Name) -or $Previous[$Name].sha256 -cne $Snapshot.files[$Name].sha256 -or $Previous[$Name].bytes -ne $Snapshot.files[$Name].bytes)) { throw 'A saved regularized artifact changed across cold reopening or reinstall.' }
    }
    return $Saved
}

function Assert-SavedBinomialUiFiles([string]$UiReceiptFile, $Snapshot, $Previous = $null) {
    $Ui = Get-Content -LiteralPath $UiReceiptFile -Raw | ConvertFrom-Json -AsHashtable
    $Saved = $Ui.saved_binomial.saved_files
    if ($Saved.Count -ne 10) { throw 'The UI receipt lacks its exact ten saved-binomial files.' }
    foreach ($Name in $Saved.Keys) {
        if (-not $Snapshot.files.Contains($Name) -or $Saved[$Name].sha256 -cne $Snapshot.files[$Name].sha256 -or $Saved[$Name].bytes -ne $Snapshot.files[$Name].bytes) { throw 'A saved-binomial UI marker differs from the actual native project file bytes.' }
        if ($null -ne $Previous -and (-not $Previous.ContainsKey($Name) -or $Previous[$Name].sha256 -cne $Snapshot.files[$Name].sha256 -or $Previous[$Name].bytes -ne $Snapshot.files[$Name].bytes)) { throw 'A saved-binomial artifact changed across cold reopening or reinstall.' }
    }
    return $Saved
}

function Get-NativeProfileEvidence([string[]]$Candidates) {
    foreach ($Candidate in $Candidates) {
        $Exists = Test-Path -LiteralPath $Candidate -PathType Container
        $Lock = Join-Path $Candidate '.desktop.lock'
        $Port = Join-Path $Candidate '.runtime-port.json'
        $HasLock = Test-Path -LiteralPath $Lock -PathType Leaf
        $HasPort = Test-Path -LiteralPath $Port -PathType Leaf
        $PortHash = $null
        if ($Exists) {
            if (((Get-Item -LiteralPath $Candidate -Force).Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) { throw 'An observed native profile must not be a reparse point.' }
            if ($HasLock -and $HasPort) {
                $LockItem = Get-Item -LiteralPath $Lock -Force
                $PortItem = Get-Item -LiteralPath $Port -Force
                if ((($LockItem.Attributes -bor $PortItem.Attributes) -band [System.IO.FileAttributes]::ReparsePoint) -ne 0 -or $PortItem.Length -gt 256) { throw 'Native runtime profile evidence is unsafe.' }
                $Preference = Get-Content -LiteralPath $Port -Raw | ConvertFrom-Json -AsHashtable
                if ($Preference.Count -ne 1 -or -not $Preference.ContainsKey('port') -or $Preference.port -isnot [long] -or $Preference.port -lt 1024 -or $Preference.port -gt 65535) { throw 'Native runtime profile port evidence is invalid.' }
                $PortHash = (Get-FileHash -LiteralPath $Port -Algorithm SHA256).Hash.ToLowerInvariant()
            }
        }
        [pscustomobject]@{ path = $Candidate; exists = $Exists; runtime_lock_exists = $HasLock; runtime_port_exists = $HasPort; runtime_port_sha256 = $PortHash; native_runtime_evidence = ($null -ne $PortHash) }
    }
}

function Get-NsisNativeIdentity([string]$BuildFile) {
    # Locked Tauri CLI 2.12.1 bundle.rs patches this one token for NSIS,
    # then restores the unpatched source executable after producing the installer.
    $SourceToken = '__TAURI_BUNDLE_TYPE_VAR_UNK'
    $NsisToken = '__TAURI_BUNDLE_TYPE_VAR_NSS'
    $Bytes = [System.IO.File]::ReadAllBytes($BuildFile)
    # Latin1 maps every byte to one character, preserving exact byte offsets.
    $ByteText = [System.Text.Encoding]::Latin1.GetString($Bytes)
    $Count = 0
    $Offset = -1
    $Next = 0
    while ($Next -lt $ByteText.Length) {
        $Found = $ByteText.IndexOf($SourceToken, $Next, [StringComparison]::Ordinal)
        if ($Found -lt 0) { break }
        $Count++
        $Offset = $Found
        $Next = $Found + $SourceToken.Length
    }
    if ($Count -ne 1) { throw "The source native executable must contain exactly one Tauri bundle token; observed $Count." }
    $OriginalHash = [Convert]::ToHexString([System.Security.Cryptography.SHA256]::HashData($Bytes)).ToLowerInvariant()
    $ExistingNsisCount = ($ByteText.Length - $ByteText.Replace($NsisToken, '').Length) / $NsisToken.Length
    [Array]::Copy([System.Text.Encoding]::ASCII.GetBytes($NsisToken), 0, $Bytes, $Offset, $SourceToken.Length)
    return @{
        original_sha256 = $OriginalHash
        expected_sha256 = [Convert]::ToHexString([System.Security.Cryptography.SHA256]::HashData($Bytes)).ToLowerInvariant()
        patch = @{ tauri_cli_version = '2.12.1'; source = 'https://github.com/tauri-apps/tauri/blob/tauri-cli-v2.12.1/crates/tauri-bundler/src/bundle.rs'; source_token = $SourceToken; installed_token = $NsisToken; marker_count = $Count; byte_offset = $Offset; original_nsis_token_count = $ExistingNsisCount; expected_nsis_token_count = $ExistingNsisCount + 1 }
    }
}

function Capture-NativeWindow([System.Diagnostics.Process]$Process, [string]$Name = 'installed-native-webview2.png') {
    Add-Type -AssemblyName System.Drawing
    if (-not ('OpenEconAcceptanceWindow' -as [type])) {
        Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class OpenEconAcceptanceWindow {
  [StructLayout(LayoutKind.Sequential)] public struct Rect { public int Left,Top,Right,Bottom; }
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr hwnd,out Rect rect);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hwnd);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hwnd,int command);
  [DllImport("user32.dll")] public static extern bool MoveWindow(IntPtr hwnd,int x,int y,int width,int height,bool repaint);
}
'@
    }
    $Handle = $Process.MainWindowHandle
    if ($Handle -eq [IntPtr]::Zero) { throw 'The real installed WebView2 application has no native window.' }
    # Preserve the actual normal window geometry; resizing can repair a blank surface.
    [OpenEconAcceptanceWindow]::SetForegroundWindow($Handle) | Out-Null
    Start-Sleep -Seconds 2
    $Rect = [OpenEconAcceptanceWindow+Rect]::new()
    if (-not [OpenEconAcceptanceWindow]::GetWindowRect($Handle, [ref]$Rect)) { throw 'The native window rectangle is unavailable.' }
    $Width = $Rect.Right - $Rect.Left
    $Height = $Rect.Bottom - $Rect.Top
    if ($Width -lt 800 -or $Height -lt 500) { throw 'The native application window did not reach a useful rendered size.' }
    $Bitmap = [System.Drawing.Bitmap]::new($Width, $Height)
    $Graphics = [System.Drawing.Graphics]::FromImage($Bitmap)
    try {
        $Graphics.CopyFromScreen($Rect.Left, $Rect.Top, 0, 0, [System.Drawing.Size]::new($Width, $Height))
        $Colours = [System.Collections.Generic.HashSet[int]]::new()
        for ($X = 5; $X -lt $Width; $X += 17) {
            for ($Y = 5; $Y -lt $Height; $Y += 17) { $Colours.Add($Bitmap.GetPixel($X, $Y).ToArgb()) | Out-Null }
        }
        if ($Colours.Count -lt 12) { throw 'The captured real native window is blank or did not render.' }
        $Screenshot = Join-Path $Output $Name
        $Bitmap.Save($Screenshot, [System.Drawing.Imaging.ImageFormat]::Png)
        return @{ file = $Screenshot; sha256 = (Get-FileHash -LiteralPath $Screenshot -Algorithm SHA256).Hash.ToLowerInvariant(); width = $Width; height = $Height; sampled_colours = $Colours.Count }
    }
    finally { $Graphics.Dispose(); $Bitmap.Dispose() }
}

try {
    $LockedCli = (Get-Content -LiteralPath (Join-Path $DesktopRoot 'package-lock.json') -Raw | ConvertFrom-Json -AsHashtable).packages.'node_modules/@tauri-apps/cli'.version
    if ($LockedCli -ne '2.12.1') { throw 'Review the exact native NSIS byte transform when changing the locked Tauri CLI version.' }
    $NativeReferencePath = Join-Path $Output 'native-build-reference.exe'
    $NativeReferenceMetadataPath = Join-Path $Output 'native-build-reference.json'
    if (-not (Test-Path -LiteralPath $NativeReferencePath -PathType Leaf) -or -not (Test-Path -LiteralPath $NativeReferenceMetadataPath -PathType Leaf)) { throw 'The immutable pre-test native build reference is absent.' }
    $NativeReference = Get-Content -LiteralPath $NativeReferenceMetadataPath -Raw | ConvertFrom-Json
    if ($NativeReference.source_sha -ne $SourceCommit -or $NativeReference.tauri_cli_version -ne $LockedCli -or $NativeReference.source_sha256 -ne (Get-FileHash -LiteralPath $NativeReferencePath -Algorithm SHA256).Hash.ToLowerInvariant() -or $NativeReference.bytes -ne (Get-Item -LiteralPath $NativeReferencePath).Length) { throw 'The immutable native build reference does not match its exact source identity.' }
    $Receipt.native_build_reference = $NativeReference
    $Config = Get-Content -LiteralPath (Join-Path $DesktopRoot 'src-tauri/tauri.conf.json') -Raw | ConvertFrom-Json
    if ($Config.bundle.windows.webviewInstallMode.type -ne 'embedBootstrapper' -or $Config.bundle.windows.webviewInstallMode.silent -ne $true) { throw 'Fresh dependency acceptance requires the actual source-configured embedded silent WebView2 bootstrapper.' }
    Remove-HostedWebView
    $Executable = Install-Owned
    $RestoredWebView = Get-WebViewState
    $ValidRegistrations = @($RestoredWebView.registrations | Where-Object { $_.pv -and [version]$_.pv -gt [version]'0.0.0.0' })
    if (-not $RestoredWebView.registered_runtime_executables.Count -or -not $ValidRegistrations.Count) { throw 'The candidate embedded bootstrapper did not restore an actual registered WebView2 Runtime.' }
    $Receipt.webview_bootstrap.after_candidate_install = $RestoredWebView
    $Receipt.checks.genuine_webview2_absence_then_embedded_bootstrapper_install = $true
    $Runtime = Join-Path $InstallDirectory 'runtime/openecon-runtime/openecon-runtime.exe'
    $ManifestPath = Join-Path $InstallDirectory 'runtime/runtime-manifest.json'
    if (-not (Test-Path -LiteralPath $Runtime -PathType Leaf) -or -not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) { throw 'The NSIS installation lacks its frozen runtime or manifest.' }
    $Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
    if ($Manifest.platform -ne 'win32' -or $Manifest.openecon_version -ne $ExpectedSdk -or $Manifest.requires_system_python -ne $false -or $Manifest.packages.'openecon-charts' -ne $ExpectedCharts) { throw 'Installed runtime version/platform identity differs from the source.' }
    $Receipt.installed_runtime_manifest = $Manifest
    $Receipt.resource_hashes = [ordered]@{}
    $BuildFiles = [ordered]@{
        'openecon-desktop.exe' = $NativeReferencePath
        'runtime/runtime-manifest.json' = Join-Path $DesktopRoot 'runtime/runtime-manifest.json'
        'runtime/openecon-runtime/openecon-runtime.exe' = Join-Path $DesktopRoot 'runtime/openecon-runtime/openecon-runtime.exe'
        'runtime/openecon-runtime/_internal/tools/uv.exe' = Join-Path $DesktopRoot 'runtime/openecon-runtime/_internal/tools/uv.exe'
        'runtime/openecon-runtime/_internal/openecon/static/index.html' = Join-Path $DesktopRoot 'runtime/openecon-runtime/_internal/openecon/static/index.html'
    }
    foreach ($Relative in $BuildFiles.Keys) {
        $File = Join-Path $InstallDirectory $Relative
        if (-not (Test-Path -LiteralPath $File -PathType Leaf)) { throw "Installed resource is absent: $Relative" }
        $InstalledHash = (Get-FileHash -LiteralPath $File -Algorithm SHA256).Hash.ToLowerInvariant()
        $InstalledBytes = (Get-Item -LiteralPath $File).Length
        $BundlePatch = $null
        if ($null -ne $DiagnosticProducer) {
            # Fixed old production evidence is used only by the explicitly
            # diagnostic replay. Never compare the old binary to newer SDK source.
            if (-not $DiagnosticProducer.resource_hashes.ContainsKey($Relative)) { throw 'A diagnostic producer resource is absent.' }
            $ProducerResource = $DiagnosticProducer.resource_hashes[$Relative]
            if ($ProducerResource.exact_packaged_bytes_equal -ne $true -or $ProducerResource.sha256 -cne $ProducerResource.expected_packaged_sha256 -or $ProducerResource.bytes -ne $ProducerResource.build_bytes) { throw 'A fixed producer resource did not pass its original build identity check.' }
            $BuildHash = $ProducerResource.expected_packaged_sha256
            $OriginalBuildHash = $ProducerResource.build_sha256
            $BuildBytes = $ProducerResource.build_bytes
            $BundlePatch = $ProducerResource.bundle_type_patch
        }
        else {
            $BuildFile = $BuildFiles[$Relative]
            if (-not (Test-Path -LiteralPath $BuildFile -PathType Leaf)) { throw "The exact source build resource is absent: $Relative" }
            $BuildHash = (Get-FileHash -LiteralPath $BuildFile -Algorithm SHA256).Hash.ToLowerInvariant()
            $OriginalBuildHash = $BuildHash
            if ($Relative -eq 'openecon-desktop.exe') {
                $NativeIdentity = Get-NsisNativeIdentity -BuildFile $BuildFile
                $BuildHash = $NativeIdentity.expected_sha256
                $BundlePatch = $NativeIdentity.patch
            }
            $BuildBytes = (Get-Item -LiteralPath $BuildFile).Length
        }
        $Matches = $InstalledHash -eq $BuildHash -and $InstalledBytes -eq $BuildBytes
        $Receipt.resource_hashes[$Relative] = @{ sha256 = $InstalledHash; build_sha256 = $OriginalBuildHash; expected_packaged_sha256 = $BuildHash; bundle_type_patch = $BundlePatch; exact_build_bytes_equal = ($InstalledHash -eq $OriginalBuildHash -and $InstalledBytes -eq $BuildBytes); exact_packaged_bytes_equal = $Matches; bytes = $InstalledBytes; build_bytes = $BuildBytes }
        if (-not $Matches) { throw "NSIS installed bytes differ from the exact source build: $Relative" }
    }
    $Receipt.checks.nsis_installed_in_unicode_space_path = $true
    $Receipt.checks.critical_installed_resources_match_exact_source_build = $true
    Enable-OwnedFirewall -Programs @($Executable, $Runtime)
    $Smoke = New-OwnedProcess -Executable $Executable -Arguments @('--smoke-test')
    $SmokeText = Wait-OwnedProcess -Owned $Smoke -TimeoutSeconds 300 -Name 'installed-native-smoke'
    $SmokeJson = $SmokeText | ConvertFrom-Json
    if ($SmokeJson.status -ne 'ok' -or -not $SmokeJson.bundled -or -not $SmokeJson.stable_origin -or $SmokeJson.execution.status -ne 'ok') { throw 'The actual installed native shell did not exercise its bundled runtime successfully.' }
    # Do not retain complete native execution metadata or transient readiness credentials.
    $Receipt.native_smoke = @{ bundled = $SmokeJson.bundled; stable_origin = $SmokeJson.stable_origin; ready_seconds = $SmokeJson.ready_seconds; warm_ready_seconds = $SmokeJson.warm_ready_seconds }
    $Receipt.checks.installed_native_shell_bundled_python_and_restart = $true
    $Receipt.native_default_webview_startup = Invoke-DefaultWebViewStartup
    $Receipt.checks.default_installed_webview2_render_trusted_ipc_and_shutdown_without_profile_override = $true
    $Receipt.native_ui_create = Invoke-NormalUi -Mode 'create'
    $Receipt.native_window = @{ mode = 'normal application, actual visible terminal and model results'; screenshot = $Receipt.native_ui_create.screenshot }
    $Receipt.checks.actual_installed_webview2_render_and_trusted_ipc = $true
    # SHGetKnownFolderPath may expand the child's USERPROFILE instead of its APPDATA.
    # A browser cache directory alone is not evidence of the Python runtime's data root.
    $Candidates = @($KnownFolderDataRoot, (Join-Path $NativeProfile 'Roaming/org.openecon.desktop'), (Join-Path $NativeProfile 'AppData/Roaming/org.openecon.desktop'))
    $Receipt.native_profile_candidates = @(Get-NativeProfileEvidence -Candidates $Candidates)
    $CreatedRoots = @($Receipt.native_profile_candidates | Where-Object { $_.native_runtime_evidence })
    if ($CreatedRoots.Count -ne 1) { throw "The real native application profile cannot be attributed to exactly one fresh owned CI location; native runtime evidence count: $($CreatedRoots.Count)." }
    $ObservedNativeDataRoot = $CreatedRoots[0].path
    $Receipt.observed_native_data_root = $ObservedNativeDataRoot
    $Receipt.native_profile_used_knownfolder = $ObservedNativeDataRoot -eq $KnownFolderDataRoot
    $RegularizedCreatedFiles = $null
    $SavedBinomialCreatedFiles = $null
    if ($null -eq $DiagnosticProducer) {
        $CreatedSnapshot = Get-NativeProjectSnapshot -DataRoot $ObservedNativeDataRoot
        $RegularizedCreatedFiles = Assert-RegularizedUiFiles -UiReceiptFile $Receipt.native_ui_create.receipt -Snapshot $CreatedSnapshot
        $Receipt.native_regularized_files_after_create = $RegularizedCreatedFiles
        $SavedBinomialCreatedFiles = Assert-SavedBinomialUiFiles -UiReceiptFile $Receipt.native_ui_create.receipt -Snapshot $CreatedSnapshot
        $Receipt.native_saved_binomial_files_after_create = $SavedBinomialCreatedFiles
    }
    $ProfileSentinel = Join-Path $ObservedNativeDataRoot 'owned-reinstall-sentinel.txt'
    if (-not (Test-Path -LiteralPath (Split-Path -Parent $ProfileSentinel) -PathType Container)) { throw 'Native application did not create its expected isolated profile.' }
    [System.IO.File]::WriteAllText($ProfileSentinel, 'Only this disposable runner profile belongs to the installer test.', [System.Text.UTF8Encoding]::new($false))
    $SentinelHash = (Get-FileHash -LiteralPath $ProfileSentinel -Algorithm SHA256).Hash
    $BeforeReinstall = (Get-FileHash -LiteralPath $Executable -Algorithm SHA256).Hash
    $Reinstalled = Install-Owned
    if ($Reinstalled -ne $Executable -or (Get-FileHash -LiteralPath $Executable -Algorithm SHA256).Hash -ne $BeforeReinstall -or (Get-FileHash -LiteralPath $ProfileSentinel -Algorithm SHA256).Hash -ne $SentinelHash) { throw 'Same-version reinstall changed executable identity or deleted isolated application data.' }
    $Receipt.checks.same_version_reinstall_preserves_owned_application_data = $true
    $OfflineOutput = Join-Path $Output 'installed-frozen-offline-runtime.json'
    & $ControllerPython -I (Join-Path $DesktopRoot 'scripts/verify_windows_runtime.py') --runtime $Runtime --profile (Join-Path $OwnedRoot 'Çevrimdışı analiz projesi') --output $OfflineOutput --sdk-version $ExpectedSdk --charts-version $ExpectedCharts --source-sha $SourceCommit --offline-only
    if ($LASTEXITCODE -ne 0) { throw 'Installed frozen numerical acceptance under the actual Internet block failed.' }
    $OfflineReceipt = Get-Content -LiteralPath $OfflineOutput -Raw | ConvertFrom-Json
    if ($OfflineReceipt.status -cne 'passed' -or $OfflineReceipt.acceptance_mode -cne 'offline_numerical_only' -or -not $OfflineReceipt.owned_runtime_stopped -or $OfflineReceipt.source_sha -cne $SourceCommit -or $OfflineReceipt.runtime_sha256 -cne (Get-FileHash -LiteralPath $Runtime -Algorithm SHA256).Hash.ToLowerInvariant() -or $OfflineReceipt.checks.actual_installed_runtime_pypi_wheel_https_denied -ne $true -or $OfflineReceipt.checks.frozen_cpu_ols_hc3_json_latex_offline_chart -ne $true) { throw 'The actual blocked frozen runtime did not prove exact identity, numerical execution and denied wheel HTTPS.' }
    $Receipt.offline_frozen_receipt = @{ file = $OfflineOutput; sha256 = (Get-FileHash -LiteralPath $OfflineOutput -Algorithm SHA256).Hash.ToLowerInvariant(); checks = $OfflineReceipt.checks }
    $Receipt.checks.enabled_firewall_scoped_internet_block_and_loopback_compute = $true
    # Downloading new packages is an online operation. Preserve enabled Windows
    # profiles; temporarily disable only this run's exact admitted application rules.
    Set-OwnedFirewallRules -Records @($FirewallRuleRecords.ToArray()) -Enabled $false
    $Receipt.firewall.phases.Add(@{ phase = 'online_real_pypi_packages'; owned_rules_enabled = $false; exact_readback_verified = $true; three_profiles_enabled = $true })
    $RuntimeOutput = Join-Path $Output 'installed-frozen-runtime.json'
    & $ControllerPython -I (Join-Path $DesktopRoot 'scripts/verify_windows_runtime.py') --runtime $Runtime --profile (Join-Path $OwnedRoot 'Kalıcı analiz projesi') --output $RuntimeOutput --sdk-version $ExpectedSdk --charts-version $ExpectedCharts --source-sha $SourceCommit
    if ($LASTEXITCODE -ne 0) { throw 'Installed Windows frozen runtime acceptance failed. See its retained JSON receipt.' }
    $RuntimeReceipt = Get-Content -LiteralPath $RuntimeOutput -Raw | ConvertFrom-Json
    if ($RuntimeReceipt.status -ne 'passed' -or $RuntimeReceipt.acceptance_mode -cne 'online_packages_and_persistence' -or -not $RuntimeReceipt.owned_runtime_stopped -or $RuntimeReceipt.source_sha -cne $SourceCommit -or $RuntimeReceipt.sdk_version -cne $ExpectedSdk -or $RuntimeReceipt.charts_version -cne $ExpectedCharts -or $RuntimeReceipt.runtime_sha256 -cne $OfflineReceipt.runtime_sha256 -or $RuntimeReceipt.checks.actual_installed_runtime_pypi_wheel_https_available -ne $true) { throw 'Installed frozen online package acceptance did not retain exact identity, stop cleanly or prove actual HTTPS availability.' }
    $Receipt.checks.installed_frozen_numerical_packages_and_project_persistence = $true
    $Receipt.frozen_receipt = @{ file = $RuntimeOutput; sha256 = (Get-FileHash -LiteralPath $RuntimeOutput -Algorithm SHA256).Hash.ToLowerInvariant(); checks = $RuntimeReceipt.checks }
    Set-OwnedFirewallRules -Records @($FirewallRuleRecords.ToArray()) -Enabled $true
    $Receipt.firewall.phases.Add(@{ phase = 'offline_migration_and_native_cold_reopen'; owned_rules_enabled = $true; exact_readback_verified = $true; three_profiles_enabled = $true })
    if ($null -eq $DiagnosticProducer) {
        # Only this exact source build controller supplies archive inspection.
        # Every scientific fit/replay remains in the installed frozen worker.
        $BuildInspectorPython = (Resolve-Path -LiteralPath (Join-Path $DesktopRoot '.build-venv/Scripts/python.exe')).Path
        $RegularizedOutput = Join-Path $Output 'installed-regularized-all-family.json'
        & $BuildInspectorPython -I (Join-Path $ProjectRoot 'scripts/verify_regularized_all_family_runtime.py') --runtime $Runtime --output $RegularizedOutput --sdk-version $ExpectedSdk --source-sha $SourceCommit
        if ($LASTEXITCODE -ne 0) { throw 'Installed frozen four-family regularized acceptance failed.' }
        $RegularizedReceipt = Get-Content -LiteralPath $RegularizedOutput -Raw | ConvertFrom-Json -AsHashtable
        if ($RegularizedReceipt.status -cne 'passed' -or $RegularizedReceipt.source_sha -cne $SourceCommit -or $RegularizedReceipt.sdk_version -cne $ExpectedSdk -or $RegularizedReceipt.runtime_sha256 -cne (Get-FileHash -LiteralPath $Runtime -Algorithm SHA256).Hash.ToLowerInvariant() -or $RegularizedReceipt.compiled_modules_equal_source.Count -ne 12 -or $RegularizedReceipt.tables -ne 8 -or $RegularizedReceipt.rows_per_table -ne 4 -or $RegularizedReceipt.four_complete_models_equal_after_restart -ne $true -or $RegularizedReceipt.saved_replay_with_fit_disabled -ne $true -or $RegularizedReceipt.code_stdout_outputs_events_equal_after_restart -ne $true -or $RegularizedReceipt.editor_script_equal_after_restart -ne $true -or $RegularizedReceipt.no_worker_needed_for_history -ne $true -or $RegularizedReceipt.owned_runtime_stopped -ne $true -or $RegularizedReceipt.temporary_profile_removed -ne $true -or $RegularizedReceipt.source_path_injected -ne $false) { throw 'Frozen regularized receipt did not complete its exact source, four-model and cleanup gates.' }
        $Receipt.installed_regularized_all_family = @{ file = $RegularizedOutput; sha256 = (Get-FileHash -LiteralPath $RegularizedOutput -Algorithm SHA256).Hash.ToLowerInvariant(); archive_inspection_python = $BuildInspectorPython; all_calculations_in_installed_frozen_worker = $true; compiled_module_count = 12; tables = 8 }
        $Receipt.checks.installed_frozen_four_regularized_methods_full_state_and_cold_replay = $true
        $SavedBinomialOutput = Join-Path $Output 'installed-saved-binomial-suest.json'
        & $BuildInspectorPython -I (Join-Path $ProjectRoot 'scripts/verify_saved_binomial_runtime.py') --runtime $Runtime --output $SavedBinomialOutput --sdk-version $ExpectedSdk --source-sha $SourceCommit
        if ($LASTEXITCODE -ne 0) { throw 'Installed frozen saved-binomial acceptance failed.' }
        $SavedBinomialReceipt = Get-Content -LiteralPath $SavedBinomialOutput -Raw | ConvertFrom-Json -AsHashtable
        if ($SavedBinomialReceipt.status -cne 'passed' -or $SavedBinomialReceipt.source_sha -cne $SourceCommit -or $SavedBinomialReceipt.runtime_sha256 -cne (Get-FileHash -LiteralPath $Runtime -Algorithm SHA256).Hash.ToLowerInvariant() -or $SavedBinomialReceipt.compiled_modules_equal_source.Count -ne 26 -or $SavedBinomialReceipt.component_models -ne 16 -or $SavedBinomialReceipt.joint_systems -ne 8 -or $SavedBinomialReceipt.sixteen_complete_models_and_eight_joint_states_equal_after_restart -ne $true -or $SavedBinomialReceipt.saved_replay_with_fit_disabled -ne $true -or $SavedBinomialReceipt.code_stdout_outputs_events_equal_after_restart -ne $true -or $SavedBinomialReceipt.editor_script_equal_after_restart -ne $true -or $SavedBinomialReceipt.no_worker_needed_for_history -ne $true -or $SavedBinomialReceipt.owned_runtime_stopped -ne $true -or $SavedBinomialReceipt.temporary_profile_removed -ne $true -or $SavedBinomialReceipt.source_path_injected -ne $false) { throw 'Frozen saved-binomial receipt did not complete its exact compiled source/state/replay/cleanup gates.' }
        $Receipt.installed_saved_binomial_suest = @{ file = $SavedBinomialOutput; sha256 = (Get-FileHash -LiteralPath $SavedBinomialOutput -Algorithm SHA256).Hash.ToLowerInvariant(); compiled_module_count = 26; joint_systems = 8; all_calculations_in_installed_frozen_worker = $true }
        $Receipt.checks.installed_frozen_saved_binomial_full_state_and_cold_replay = $true
        $TwoStepOutput = Join-Path $Output 'installed-twostep-adaptive.json'
        & $BuildInspectorPython -I (Join-Path $ProjectRoot 'scripts/verify_twostep_adaptive_runtime.py') --runtime $Runtime --output $TwoStepOutput --sdk-version $ExpectedSdk --source-sha $SourceCommit
        if ($LASTEXITCODE -ne 0) { throw 'Installed frozen adaptive TwoStep acceptance failed.' }
        $TwoStepReceipt = Get-Content -LiteralPath $TwoStepOutput -Raw | ConvertFrom-Json -AsHashtable
        if ($TwoStepReceipt.status -cne 'passed' -or $TwoStepReceipt.source_sha -cne $SourceCommit -or $TwoStepReceipt.runtime_sha256 -cne (Get-FileHash -LiteralPath $Runtime -Algorithm SHA256).Hash.ToLowerInvariant() -or $TwoStepReceipt.compiled_modules_equal_source.Count -ne 10 -or $TwoStepReceipt.four_complete_saved_states.cases.Count -ne 4 -or $TwoStepReceipt.four_complete_saved_states.files.Count -ne 9 -or $TwoStepReceipt.fit_disabled_saved_replay -ne $true -or $TwoStepReceipt.cold_history_without_worker -ne $true -or $TwoStepReceipt.owned_runtime_stopped -ne $true -or $TwoStepReceipt.temporary_profile_removed -ne $true -or $TwoStepReceipt.source_path_injected -ne $false -or $TwoStepReceipt.native_window_verified -ne $false) { throw 'TwoStep did not prove exact installed source, full state, cold replay and cleanup.' }
        $Receipt.installed_twostep_adaptive = @{ file = $TwoStepOutput; sha256 = (Get-FileHash -LiteralPath $TwoStepOutput -Algorithm SHA256).Hash.ToLowerInvariant(); compiled_module_count = 10; cases = 4; all_calculations_in_installed_frozen_worker = $true }
        $Receipt.checks.installed_frozen_twostep_adaptive_full_state_and_cold_replay = $true
    }
    $PreviousNativeSha = 'c9ae7c7570e936a3f7d3233ab42c3848d20686ac7906f45d36ec2dc67efba647'
    $PreviousRuntimeSha = '8169d70dc79e4675b256591647e0a6ac1310fb1736c7f2cbb5ff3829255ce582'
    function Assert-PreviousInstalled {
        if ((Get-FileHash -LiteralPath $Executable -Algorithm SHA256).Hash.ToLowerInvariant() -ne $PreviousNativeSha -or (Get-FileHash -LiteralPath $Runtime -Algorithm SHA256).Hash.ToLowerInvariant() -ne $PreviousRuntimeSha) { throw 'The actual installed previous-version payload differs from the reviewed public0.3.44 package.' }
        $PreviousManifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
        if ($PreviousManifest.platform -ne 'win32' -or $PreviousManifest.openecon_version -ne '0.3.19a1' -or $PreviousManifest.packages.'openecon-charts' -ne '0.3.1a1' -or $PreviousManifest.requires_system_python -ne $false) { throw 'Previous installed frozen version/compute identity is incorrect.' }
    }
    function Assert-CandidateRestored {
        foreach ($Relative in $BuildFiles.Keys) {
            $File = Join-Path $InstallDirectory $Relative
            $Expected = $Receipt.resource_hashes[$Relative]
            if ((Get-FileHash -LiteralPath $File -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Expected.expected_packaged_sha256 -or (Get-Item -LiteralPath $File).Length -ne $Expected.bytes) { throw "Upgrade/restore did not install exact candidate resource: $Relative" }
        }
    }
    Install-Owned -Package $PreviousInstaller -Name 'nsis-previous-baseline' | Out-Null
    Assert-PreviousInstalled
    $Receipt.previous_version_migration = [ordered]@{ previous_version = '0.3.44'; candidate_version = $ExpectedDesktop; previous_source_sha = $PreviousSource; candidate_source_sha = $SourceCommit }
    $Receipt.previous_version_migration.prepare = Invoke-Migration -Phase 'prepare' -Name 'migration-previous-prepare' -Sdk '0.3.19a1' -Charts '0.3.1a1' -Sha $PreviousSource
    Install-Owned -Name 'nsis-actual-upgrade' | Out-Null
    Assert-CandidateRestored
    $Receipt.previous_version_migration.upgrade = Invoke-Migration -Phase 'verify' -Name 'migration-candidate-upgrade' -Sdk $ExpectedSdk -Charts $ExpectedCharts -Sha $SourceCommit
    $Receipt.checks.previous_public_version_upgrade_preserves_real_project = $true
    Install-Owned -Package $PreviousInstaller -Name 'nsis-actual-rollback' | Out-Null
    Assert-PreviousInstalled
    $Receipt.previous_version_migration.rollback = Invoke-Migration -Phase 'verify' -Name 'migration-previous-rollback' -Sdk '0.3.19a1' -Charts '0.3.1a1' -Sha $PreviousSource
    $Receipt.checks.rollback_to_previous_public_version_preserves_real_project = $true
    Install-Owned -Name 'nsis-final-candidate-restore' | Out-Null
    Assert-CandidateRestored
    $Receipt.previous_version_migration.final_candidate_restored = $true
    $Receipt.native_ui_reopen = Invoke-NormalUi -Mode 'reopen'
    $Receipt.checks.normal_native_ui_three_models_and_cold_history_reopen = $true
    if ($null -eq $DiagnosticProducer) { $Receipt.checks.normal_native_ui_four_regularized_methods_and_cold_history_reopen = $true }
    $BeforeUninstallProject = Get-NativeProjectSnapshot -DataRoot $ObservedNativeDataRoot
    $Receipt.native_project_before_uninstall = $BeforeUninstallProject
    if ($null -eq $DiagnosticProducer) {
        $Receipt.native_regularized_files_after_cold_reopen = Assert-RegularizedUiFiles -UiReceiptFile $Receipt.native_ui_reopen.receipt -Snapshot $BeforeUninstallProject -Previous $RegularizedCreatedFiles
        $Receipt.checks.native_four_regularized_saved_markers_match_actual_files_after_create_and_cold_reopen = $true
        $Receipt.native_saved_binomial_files_after_cold_reopen = Assert-SavedBinomialUiFiles -UiReceiptFile $Receipt.native_ui_reopen.receipt -Snapshot $BeforeUninstallProject -Previous $SavedBinomialCreatedFiles
        $Receipt.checks.native_eight_saved_binomial_tables_and_saved_files_match_after_cold_reopen = $true
    }
    $Uninstallers = @(Get-ChildItem -LiteralPath $InstallDirectory -File | Where-Object { $_.Name -match '^uninstall.*\.exe$' })
    if ($Uninstallers.Count -ne 1) { throw 'Exactly one installed NSIS uninstaller is required.' }
    # NSIS /S alone launches a temporary copy and returns before resource removal.
    # _?= keeps the uninstaller in place so WaitForExit waits for the actual work;
    # it must be last and unquoted, including for this Unicode path with spaces.
    $Uninstall = New-OwnedProcess -Executable $Uninstallers[0].FullName -RawArguments ('/S _?=' + $InstallDirectory)
    Wait-OwnedProcess -Owned $Uninstall -TimeoutSeconds 180 -Name 'nsis-uninstall' | Out-Null
    $RemovedResources = [ordered]@{}
    foreach ($Relative in $BuildFiles.Keys) {
        $Removed = -not (Test-Path -LiteralPath (Join-Path $InstallDirectory $Relative))
        $RemovedResources[$Relative] = $Removed
        if (-not $Removed) { throw "NSIS uninstall did not remove an owned application resource: $Relative" }
    }
    # The in-place uninstaller itself can remain until owned-root cleanup after exit.
    $Receipt.uninstall = @{ waited_in_place = $true; removed_application_resources = $RemovedResources }
    if ((Get-FileHash -LiteralPath $ProfileSentinel -Algorithm SHA256).Hash -ne $SentinelHash) { throw 'Uninstall unexpectedly deleted persisted application data.' }
    $AfterUninstallProject = Get-NativeProjectSnapshot -DataRoot $ObservedNativeDataRoot
    if ($BeforeUninstallProject.project_id -cne $AfterUninstallProject.project_id -or $BeforeUninstallProject.catalog_sha256 -cne $AfterUninstallProject.catalog_sha256 -or $BeforeUninstallProject.file_count -ne $AfterUninstallProject.file_count -or $BeforeUninstallProject.total_bytes -ne $AfterUninstallProject.total_bytes) { throw 'Uninstall changed the actual native project catalog or complete saved file set.' }
    foreach ($Relative in $BeforeUninstallProject.files.Keys) {
        if (-not $AfterUninstallProject.files.Contains($Relative) -or $BeforeUninstallProject.files[$Relative].sha256 -cne $AfterUninstallProject.files[$Relative].sha256 -or $BeforeUninstallProject.files[$Relative].bytes -ne $AfterUninstallProject.files[$Relative].bytes) { throw "Uninstall changed a saved native data/result/history/editor file: $Relative" }
    }
    $Receipt.uninstall.native_project_snapshot = $AfterUninstallProject
    $Receipt.uninstall.exact_native_project_file_set_and_bytes_preserved = $true
    $Receipt.checks.nsis_uninstall_removes_application_and_preserves_user_data = $true
    $Receipt.status = 'passed'
}
catch {
    $Failure = $_
    $Receipt.status = 'error'
    $Receipt.error = $Failure.Exception.Message
    if ($null -ne $Native -and -not $Native.disposed -and -not $Native.process.HasExited) {
        try { $Receipt.native_failure_screenshot = Capture-NativeWindow -Process $Native.process -Name 'installed-native-failure.png' }
        catch { $Receipt.native_failure_screenshot_error = $_.Exception.Message }
    }
    throw $Failure
}
finally {
    try {
        Remove-OwnedFirewallRules -Records @($FirewallRuleRecords.ToArray())
        if ($Receipt.Contains('firewall')) { $Receipt.firewall.owned_rules_removed = $true }
    }
    catch { $Receipt.status = 'error'; $Receipt.firewall_rule_cleanup_error = $_.Exception.Message }
    try {
        $RestoreFailures = [System.Collections.Generic.List[string]]::new()
        foreach ($Profile in $FirewallBefore) {
            try { Set-NetFirewallProfile -Name $Profile.Name -Enabled $Profile.Enabled }
            catch { $RestoreFailures.Add($_.Exception.Message) }
        }
        if ($RestoreFailures.Count) { throw 'An original Windows firewall profile failed restoration.' }
        if ($FirewallBefore.Count) {
            foreach ($Profile in $FirewallBefore) {
                if ((Get-NetFirewallProfile -Name $Profile.Name).Enabled -ne $Profile.Enabled) { throw 'An original Windows firewall profile was not restored.' }
            }
            if (-not $Receipt.Contains('firewall')) { $Receipt.firewall = @{} }
            $Receipt.firewall.previous_profiles_restored = $true
        }
    }
    catch { $Receipt.status = 'error'; $Receipt.firewall_cleanup_error = $_.Exception.Message }
    if ($null -ne $Native -and -not $Native.disposed) {
        try {
            if (-not $Native.process.HasExited) {
                $Native.process.CloseMainWindow() | Out-Null
                if (-not $Native.process.WaitForExit(10000)) { $Native.process.Kill($true); $Native.process.WaitForExit(10000) | Out-Null }
            }
        }
        finally { $Native.process.Dispose(); $Native.disposed = $true }
    }
    foreach ($OwnedBrowser in $DefaultOwnedBrowserProcesses) {
        $Process = Get-CimInstance Win32_Process -Filter "ProcessId=$($OwnedBrowser.pid)"
        if ($null -ne $Process -and ([DateTime]$Process.CreationDate).ToUniversalTime().ToString('o') -ceq $OwnedBrowser.creation_date) { Stop-Process -Id $Process.ProcessId -Force -ErrorAction SilentlyContinue }
    }
    # Kill only processes whose executable is inside this freshly created install.
    $OwnedProcesses = @(Get-CimInstance Win32_Process | Where-Object {
        ($_.ExecutablePath -and $_.ExecutablePath.StartsWith($InstallDirectory + '\', [StringComparison]::OrdinalIgnoreCase)) -or
        ($_.Name -eq 'msedgewebview2.exe' -and (Get-WebViewArgument -CommandLine $_.CommandLine -Name 'user-data-dir') -iin $BrowserProfileCandidates)
    })
    foreach ($Process in $OwnedProcesses) { Stop-Process -Id $Process.ProcessId -Force -ErrorAction SilentlyContinue }
    try { Remove-OwnedDebugPolicy; $Receipt.owned_debugger_policy_removed = $true }
    catch { $Receipt.status = 'error'; $Receipt.debugger_policy_cleanup_error = $_.Exception.Message }
    $Receipt.owned_processes_remaining_at_cleanup = $OwnedProcesses.Count
    if ($Receipt.status -eq 'passed' -and $OwnedProcesses.Count -gt 0) { $Receipt.status = 'error'; $Receipt.error = 'Installed application or worker processes were left behind.' }
    try {
        if (Test-Path -LiteralPath $OwnedRoot -PathType Container) { Remove-Item -LiteralPath $OwnedRoot -Recurse -Force }
        # KnownFolder was proved absent before any launch; the CI-only guard protects real user profiles.
        if ($NativeLaunchAttempted -and (Test-Path -LiteralPath $KnownFolderDataRoot -PathType Container)) { Remove-Item -LiteralPath $KnownFolderDataRoot -Recurse -Force }
        if ($NativeLaunchAttempted -and (Test-Path -LiteralPath $KnownFolderLocalRoot -PathType Container)) { Remove-Item -LiteralPath $KnownFolderLocalRoot -Recurse -Force }
        $Receipt.owned_directories_removed = $true
    }
    catch { $Receipt.status = 'error'; $Receipt.cleanup_error = $_.Exception.Message }
    if ($null -ne $DiagnosticProducer -and $Receipt.status -eq 'passed') { $Receipt.status = 'diagnostic_passed' }
    $ReceiptJson = $Receipt | ConvertTo-Json -Depth 30
    $ReceiptName = if ($null -ne $DiagnosticProducer) { 'windows-diagnostic-replay.json' } else { 'windows-installed-acceptance.json' }
    [System.IO.File]::WriteAllText((Join-Path $Output $ReceiptName), $ReceiptJson + "`n", [System.Text.UTF8Encoding]::new($false))
}
if ($Receipt.status -notin @('passed', 'diagnostic_passed')) { throw 'Windows installed acceptance failed its cleanup gate.' }
if ($Receipt.status -eq 'diagnostic_passed') { Write-Output 'Fixed older installer diagnostic replay passed; this is not release/current-source acceptance.'; return }
Write-Output "NSIS installation, native WebView2, frozen CPU analyses, package installation, project relaunch and uninstallation passed."
