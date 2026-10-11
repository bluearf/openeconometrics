param(
 [Parameter(Mandatory)][ValidateSet('prepare','owner-request','owner-phases','viewer-request','viewer-readback','cleanup')][string]$Mode,
 [Parameter(Mandatory)][string]$OwnedRoot,
 [Parameter(Mandatory)][string]$OutputDirectory,
 [string]$InstallerPath='', [string]$ProducerReceipt='',
 [ValidatePattern('^[a-f0-9]{40}$')][string]$InstallerSource,
 [string]$MailboxBranch=''
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
if (-not $IsWindows -or $env:GITHUB_ACTIONS -cne 'true' -or $env:GITHUB_REPOSITORY -cne 'bluearf/openecon' -or $env:RUNNER_ENVIRONMENT -cne 'github-hosted' -or $env:ImageOS -notmatch '^win' -or -not ([Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw 'Use only the disposable GitHub-hosted Windows administrator.' }
$Root=[IO.Path]::GetFullPath($OwnedRoot)
$Expected=Join-Path $env:RUNNER_TEMP "openecon-cloud-$($env:GITHUB_RUN_ID)-$($env:GITHUB_RUN_ATTEMPT)"
if ($Root -cne $Expected) { throw 'The QA root must be the exact disposable temporary directory for this run.' }
$DesktopRoot=Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
. (Join-Path $DesktopRoot 'scripts/windows_webview_attribution.ps1')
$Node=(Get-Command node).Source
$Controller=Join-Path $DesktopRoot 'scripts/verify_windows_cloud_ui.mjs'
$Mailbox=Join-Path $DesktopRoot 'scripts/qa_mailbox_cli.mjs'
$JournalPath=Join-Path $Root 'private-journal.json'
$Public=Join-Path $OutputDirectory 'mailbox-public.json'
$Private=Join-Path $Root 'mailbox-private.pem'
$Fixture=Join-Path $Root 'fixture.json'
$script:OwnedDebugPolicy=$null
$script:Journal=$null
$Utf8=[Text.UTF8Encoding]::new($false)
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


function Save-Journal { [IO.File]::WriteAllText($JournalPath,($script:Journal|ConvertTo-Json -Depth 30),$Utf8) }
function Read-Json([string]$Path) { Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json -AsHashtable }
function Invoke-Node([string[]]$Arguments,[int]$Timeout=720) {
 $Info=[Diagnostics.ProcessStartInfo]::new($Node); $Info.UseShellExecute=$false; $Info.WorkingDirectory=$Root
 $Info.RedirectStandardOutput=$true; $Info.RedirectStandardError=$true
 foreach($Argument in $Arguments){$Info.ArgumentList.Add($Argument)}
 $Info.Environment.Clear()
 foreach($Name in @('SYSTEMROOT','WINDIR','COMSPEC','GITHUB_ACTIONS','GITHUB_REPOSITORY','GITHUB_RUN_ID','GITHUB_RUN_ATTEMPT','GITHUB_SHA','GITHUB_TOKEN')) {
  $Value=[Environment]::GetEnvironmentVariable($Name); if($null -ne $Value){$Info.Environment[$Name]=$Value}
 }
 $Info.Environment['PATH']=Join-Path $env:SYSTEMROOT 'System32';$Info.Environment['TEMP']=$Root;$Info.Environment['TMP']=$Root
 $P=[Diagnostics.Process]::new();$P.StartInfo=$Info
 try {
  if(-not $P.Start()){throw 'The exact owned Node controller did not start.'}
  $Out=$P.StandardOutput.ReadToEndAsync();$Err=$P.StandardError.ReadToEndAsync()
  if(-not $P.WaitForExit($Timeout*1000)){$P.Kill($true);$P.WaitForExit(10000)|Out-Null;throw 'Owned controller timed out.'}
  # Auth inputs, code packets and transport text never enter workflow output.
  $Out.GetAwaiter().GetResult()|Out-Null;$Err.GetAwaiter().GetResult()|Out-Null
  if($P.ExitCode -ne 0){throw 'Owned native controller failed; inspect the sanitized phase receipt.'}
 } finally {$P.Dispose()}
}
if(-not ('QaKnownFolders' -as [type])) {
 Add-Type -TypeDefinition @'
using System; using System.Runtime.InteropServices;
public static class QaKnownFolders {
 [DllImport("shell32.dll",CharSet=CharSet.Unicode)] static extern int SHSetKnownFolderPath(ref Guid id,uint flags,IntPtr token,string path);
 [DllImport("shell32.dll")] static extern int SHGetKnownFolderPath(ref Guid id,uint flags,IntPtr token,out IntPtr path);
 public static string Get(string value){Guid id=new Guid(value);IntPtr p;Marshal.ThrowExceptionForHR(SHGetKnownFolderPath(ref id,0,IntPtr.Zero,out p));try{return Marshal.PtrToStringUni(p);}finally{Marshal.FreeCoTaskMem(p);}}
 public static void Set(string value,string path){Guid id=new Guid(value);Marshal.ThrowExceptionForHR(SHSetKnownFolderPath(ref id,0,IntPtr.Zero,path));}
}
'@
}
$FolderIds=@{Roaming='3EB685DB-65F9-4CF6-A03A-E3EF65729F3D';Local='F1B32785-6FBA-4FCF-9D55-7B8E7F157091'}
function Set-ActorFolders([string]$Actor) {
 foreach($Name in $FolderIds.Keys){
  $Path=Join-Path (Join-Path $Root $Actor) $Name;New-Item -ItemType Directory -Path $Path -Force|Out-Null
  [QaKnownFolders]::Set($FolderIds[$Name],$Path)
  if([QaKnownFolders]::Get($FolderIds[$Name]) -cne $Path){throw 'The actual native KnownFolder was not isolated.'}
 }
}
function Restore-Folders {
 if($null -eq $script:Journal){return}
 foreach($Name in $FolderIds.Keys){
  [QaKnownFolders]::Set($FolderIds[$Name],$script:Journal.original_folders[$Name])
  if([QaKnownFolders]::Get($FolderIds[$Name]) -cne $script:Journal.original_folders[$Name]){throw 'An original KnownFolder was not restored.'}
 }
}
function Native-Process {
 if(-not $script:Journal.native){return $null}
 $P=Get-Process -Id $script:Journal.native.pid -ErrorAction SilentlyContinue
 if($null -eq $P -or $P.StartTime.ToUniversalTime().ToString('o') -cne $script:Journal.native.created){return $null}
 if($P.Path -cne $script:Journal.executable){throw 'Owned native PID identity changed.'}
 return $P
}
function Close-Native {
 $P=Native-Process
 if($null -ne $P){
  if(-not $P.CloseMainWindow() -or -not $P.WaitForExit(45000)){
   if($Mode -ne 'cleanup'){throw 'The owned native window did not finish graceful shutdown.'}
   $P.Kill($true);$P.WaitForExit(10000)|Out-Null
  }
  $P.Dispose()
 }
 $script:Journal.native=$null;Save-Journal
 if($null -ne $script:OwnedDebugPolicy){Remove-OwnedDebugPolicy}
 $script:Journal.debug_policy=$null;Save-Journal
}
function Start-Native([string]$Actor) {
 if($null -ne (Native-Process)){throw 'A previous owned native process is still live.'}
 Set-ActorFolders $Actor
 $Listener=[Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback,0);$Listener.Start();$Port=$Listener.LocalEndpoint.Port;$Listener.Stop()
 $Browser=Join-Path (Join-Path $Root $Actor) 'Owned WebView2'
 Set-OwnedDebugPolicy -Port $Port -OwnedRoot $Root -NativeExecutable $script:Journal.executable -BrowserProfile $Browser -CloudActor $Actor
 $script:Journal.debug_policy=@($script:OwnedDebugPolicy.entries.ToArray());Save-Journal
 $Info=[Diagnostics.ProcessStartInfo]::new($script:Journal.executable);$Info.UseShellExecute=$false;$Info.WorkingDirectory=$Root
 $Info.Environment.Clear()
 foreach($Name in @('SYSTEMROOT','WINDIR','COMSPEC','HOMEDRIVE','LANG')){$Value=[Environment]::GetEnvironmentVariable($Name);if($null -ne $Value){$Info.Environment[$Name]=$Value}}
 $Profile=Join-Path $Root $Actor
 $Info.Environment['PATH']=Join-Path $env:SYSTEMROOT 'System32';$Info.Environment['USERPROFILE']=$Profile
 $Info.Environment['APPDATA']=Join-Path $Profile 'Roaming';$Info.Environment['LOCALAPPDATA']=Join-Path $Profile 'Local'
 $Info.Environment['TEMP']=$Root;$Info.Environment['TMP']=$Root;$Info.Environment['PYTHONNOUSERSITE']='1'
 $P=[Diagnostics.Process]::new();$P.StartInfo=$Info
 if(-not $P.Start()){throw 'The accepted installed native app did not start.'}
 $script:Journal.native=@{pid=$P.Id;created=$P.StartTime.ToUniversalTime().ToString('o');actor=$Actor;port=$Port};Save-Journal
 $Deadline=[DateTime]::UtcNow.AddSeconds(120);$Found=$false
 while([DateTime]::UtcNow -lt $Deadline){
  if($P.HasExited){throw 'The accepted app exited before its native window.'}
  $Processes=@{};foreach($Row in @(Get-CimInstance Win32_Process)){$Processes[[uint32]$Row.ProcessId]=$Row}
  foreach($Connection in @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)){
   $Evidence=Get-WebViewAttribution -Processes $Processes -BrowserPid $Connection.OwningProcess -NativePid $P.Id -NativeStarted $P.StartTime.ToUniversalTime() -NativeExecutable $script:Journal.executable -FreshProfileCandidates @($Browser,(Join-Path $Browser 'EBWebView')) -RuntimeExecutables $script:Journal.webview_executables -Port $Port
   if($Evidence.owned -and $Connection.LocalAddress -in @('127.0.0.1','::1')){$Found=$true;break}
  }
  if($Found){break};Start-Sleep -Milliseconds 250
 }
 if(-not $Found){throw 'The native CDP endpoint was not attributed to its exact Microsoft WebView2 descendant.'}
 $script:Journal.attributions+=@(@{actor=$Actor;port=$Port;native_pid=$P.Id;evidence=$Evidence});Save-Journal
 $P.Dispose()
}
function Invoke-Phase([string]$Phase,[string]$Baseline='',[string]$Online='',[string]$Challenge='') {
 if($null -eq (Native-Process)){throw 'An attributed real native window is required.'}
 $Result=Join-Path $OutputDirectory "$Phase.json"
 $State=Read-Json $Fixture
 $Args=@($Controller,'--port',"$($script:Journal.native.port)",'--phase',$Phase,'--state',$Fixture,'--output',$Result,'--project-id',$State.projects[0],'--project-name',$State.project_name)
 if($Baseline){$Args+=@('--baseline',$Baseline)};if($Online){$Args+=@('--online-identity',$Online)}
 if($Phase.StartsWith('email-request-')){$Args+=@('--mailbox-public',$Public)}
 if($Challenge){$Args+=@('--email-challenge',$Challenge,'--mailbox-public',$Public,'--mailbox-private',$Private)}
 Invoke-Node $Args
 $R=Read-Json $Result
 $ExpectedStatus=if($Phase.StartsWith('email-request-')){'awaiting_email_code'}else{'passed'}
 if($R.status -cne $ExpectedStatus -or $R.phase -cne $Phase){throw 'The exact native phase did not produce its expected receipt.'}
 $script:Journal.phases+=@(@{phase=$Phase;receipt_sha256=(Get-FileHash $Result -Algorithm SHA256).Hash.ToLowerInvariant()});Save-Journal
}
function Block-NativeInternet {
 Assert-OwnedFirewallHost
 $script:Journal.firewall_before=@(Get-NetFirewallProfile|Select-Object Name,Enabled);Save-Journal
 foreach($Profile in $script:Journal.firewall_before){Set-NetFirewallProfile -Name $Profile.Name -Enabled True}
 if(@(Get-NetFirewallProfile|Where-Object {$_.Enabled -ne 'True'}).Count){throw 'Actual Windows firewall profiles are not enabled.'}
 foreach($Program in @($script:Journal.executable,$script:Journal.runtime)){
  $Name='OpenEcon-owned-'+[Guid]::NewGuid().ToString('N')
  $Record=@{name=$Name;program=$Program;direction='Outbound';action='Block';remote_address='Internet';profile='Any';readback_verified=$false;enabled=$true}
  $script:Journal.firewall_rules+=@($Record);Save-Journal
  New-NetFirewallRule -Name $Name -DisplayName $Name -Direction Outbound -Action Block -Program $Program -RemoteAddress Internet -Profile Any -Enabled True|Out-Null
  $Rule=Get-NetFirewallRule -Name $Name
  Assert-OwnedFirewallRule $Record $Rule ($Rule|Get-NetFirewallAddressFilter) ($Rule|Get-NetFirewallApplicationFilter) 'True'
  $Record.readback_verified=$true;Save-Journal
 }
}
function Restore-Firewall {
 if($null -eq $script:Journal){return}
 if($script:Journal.firewall_rules.Count){Remove-OwnedFirewallRules -Records @($script:Journal.firewall_rules)}
 foreach($Profile in $script:Journal.firewall_before){Set-NetFirewallProfile -Name $Profile.Name -Enabled $Profile.Enabled}
 foreach($Profile in $script:Journal.firewall_before){if((Get-NetFirewallProfile -Name $Profile.Name).Enabled -ne $Profile.Enabled){throw 'An original firewall profile was not restored.'}}
 $script:Journal.firewall_rules=@();$script:Journal.firewall_before=@();Save-Journal
}
try {
 if($Mode -eq 'prepare'){
  if(Test-Path $Root){throw 'A fresh disposable run root is required.'}
  New-Item -ItemType Directory -Path $Root|Out-Null;New-Item -ItemType Directory -Path $OutputDirectory -Force|Out-Null
  $Acl=[Security.AccessControl.DirectorySecurity]::new();$Acl.SetAccessRuleProtection($true,$false)
  foreach($Sid in @([Security.Principal.WindowsIdentity]::GetCurrent().User,[Security.Principal.SecurityIdentifier]::new('S-1-5-18'))){
   $Acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new($Sid,'FullControl','ContainerInherit,ObjectInherit','None','Allow'))
  };Set-Acl -LiteralPath $Root -AclObject $Acl
  $Receipt=Read-Json $ProducerReceipt;$Installer=(Resolve-Path $InstallerPath).Path
  if($Receipt.status -cne 'passed' -or $Receipt.source_sha -cne $InstallerSource -or $Receipt.installer_sha256 -cne (Get-FileHash $Installer -Algorithm SHA256).Hash.ToLowerInvariant()){throw 'Only an exactly source-bound accepted producer installer is allowed.'}
  $script:Journal=@{schema=1;run_id=$env:GITHUB_RUN_ID;controller_source=$env:GITHUB_SHA;installer_source=$InstallerSource;installer_sha256=$Receipt.installer_sha256;native=$null;debug_policy=$null;phases=@();attributions=@();firewall_before=@();firewall_rules=@();original_folders=@{};webview_executables=@()}
  foreach($Name in $FolderIds.Keys){$script:Journal.original_folders[$Name]=[QaKnownFolders]::Get($FolderIds[$Name])};Save-Journal
  Invoke-Node @($Mailbox,'key',$Private,$Public,$MailboxBranch,$InstallerSource)
  $Install=Join-Path $Root 'Kurulum alanı';$P=Start-Process -FilePath $Installer -ArgumentList ('/S /D='+$Install) -PassThru
  if(-not $P.WaitForExit(600000)){$P.Kill($true);throw 'Owned NSIS install timed out.'};if($P.ExitCode -ne 0){throw 'Owned NSIS installation failed.'};$P.Dispose()
  $script:Journal.executable=Join-Path $Install 'openecon-desktop.exe';$script:Journal.runtime=Join-Path $Install 'runtime/openecon-runtime/openecon-runtime.exe'
  foreach($Pair in $Receipt.resource_hashes.GetEnumerator()){
   $File=Join-Path $Install $Pair.Key;$ExpectedHash=$Pair.Value.sha256
   if($Pair.Value.ContainsKey('expected_packaged_sha256')){$ExpectedHash=$Pair.Value.expected_packaged_sha256}
   if(-not (Test-Path $File -PathType Leaf) -or (Get-FileHash $File -Algorithm SHA256).Hash.ToLowerInvariant() -cne $ExpectedHash){throw 'Installed bytes differ from the accepted producer resource inventory.'}
  }
  $WebView=Get-WebViewState
  $script:Journal.webview_executables=@($WebView.registered_runtime_executables)
  foreach($File in $script:Journal.webview_executables){
   $Signature=Get-AuthenticodeSignature $File
   if($Signature.Status -ne 'Valid' -or $Signature.SignerCertificate.Subject -notmatch 'O=Microsoft Corporation'){throw 'The registered WebView2 executable is not signed by Microsoft.'}
  }
  if(-not $script:Journal.webview_executables.Count){throw 'A valid Microsoft WebView2 executable was not provisioned.'};Save-Journal
 } else {
  if(-not (Test-Path $JournalPath)) {if($Mode -eq 'cleanup'){return};throw 'The owned run journal is absent.'}
  $script:Journal=Read-Json $JournalPath
  if($script:Journal.run_id -cne $env:GITHUB_RUN_ID -or $script:Journal.controller_source -cne $env:GITHUB_SHA){throw 'The run journal belongs to another execution.'}
  if($script:Journal.debug_policy){$List=[Collections.Generic.List[object]]::new();foreach($Row in $script:Journal.debug_policy){$List.Add($Row)};$script:OwnedDebugPolicy=@{entries=$List}}
  switch($Mode){
   'owner-request' {Invoke-Node @($Mailbox,'bootstrap',$Private,$Public,$Fixture);Start-Native 'owner';Invoke-Phase 'email-request-owner'}
   'owner-phases' {
    Invoke-Phase 'email-open' -Challenge (Join-Path $OutputDirectory 'email-request-owner.json')
    Block-NativeInternet
    Invoke-Phase 'offline-run' -Online (Join-Path $OutputDirectory 'email-open.json')
    Close-Native;Start-Native 'owner'
    Invoke-Phase 'offline-reopen' -Baseline (Join-Path $OutputDirectory 'offline-run.json')
    Restore-Firewall
    Invoke-Phase 'retry-online' -Baseline (Join-Path $OutputDirectory 'offline-run.json')
    Close-Native
   }
   'viewer-request' {Start-Native 'viewer';Invoke-Phase 'email-request-viewer' -Baseline (Join-Path $OutputDirectory 'offline-run.json')}
   'viewer-readback' {
    Invoke-Phase 'viewer-readback' -Baseline (Join-Path $OutputDirectory 'offline-run.json') -Challenge (Join-Path $OutputDirectory 'email-request-viewer.json')
    Close-Native
    [IO.File]::WriteAllText((Join-Path $OutputDirectory 'WINDOWS-CLOUD-ACCEPTANCE.json'),(@{status='passed';installer_source=$script:Journal.installer_source;controller_source=$env:GITHUB_SHA;installer_sha256=$script:Journal.installer_sha256;phases=$script:Journal.phases;attributions=$script:Journal.attributions;actual_firewall_block=$true;separate_knownfolder_profiles=$true;password_used=$false;observed_cloud_execution_requests=0}|ConvertTo-Json -Depth 30),$Utf8)
   }
   'cleanup' {
    # Attempt every independent restoration even when an earlier one fails.
    $Clean=@{status='failed';owned_native_closed=$false;firewall_restored=$false;knownfolders_restored=$false;debug_policy_removed=$false;private_key_removed=$false;fixture_removed=$false}
    try {Close-Native;$Clean.owned_native_closed=$true} catch {}
    try {if($null -ne $script:OwnedDebugPolicy){Remove-OwnedDebugPolicy};$Clean.debug_policy_removed=$true} catch {}
    try {Restore-Firewall;$Clean.firewall_restored=$true} catch {}
    try {Restore-Folders;$Clean.knownfolders_restored=$true} catch {}
    foreach($Pair in @(@{path=$Private;key='private_key_removed'},@{path=$Fixture;key='fixture_removed'})){
     try {if(Test-Path $Pair.path){Remove-Item -LiteralPath $Pair.path -Force};$Clean[$Pair.key]=(-not(Test-Path $Pair.path))} catch {}
    }
    if(@($Clean.Keys|Where-Object {$_ -ne 'status' -and -not $Clean[$_]}).Count -eq 0){$Clean.status='passed'}
    [IO.File]::WriteAllText((Join-Path $OutputDirectory 'cleanup.json'),($Clean|ConvertTo-Json),$Utf8)
    if($Clean.status -ne 'passed'){throw 'An owned resource could not be restored; inspect the sanitized cleanup receipt.'}
   }
  }
 }
} finally {Restore-Folders}
