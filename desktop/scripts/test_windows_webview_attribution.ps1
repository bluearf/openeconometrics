$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'windows_webview_attribution.ps1')
$Checks = 0
function Assert-Attribution($Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
    $script:Checks++
}
# The native Windows parser treats argv[0] as the executable. Switch fixtures
# must therefore include that token, just like actual browser command lines.
foreach ($Line in @('app --user-data-dir="C:\Türkçe alan\org.openecon.desktop" --flag', 'app "--user-data-dir=C:\Türkçe alan\org.openecon.desktop"', 'app --user-data-dir=C:\plain\org.openecon.desktop')) {
    Assert-Attribution ((Get-WebViewArgument $Line 'user-data-dir') -like 'C:\*\org.openecon.desktop') 'A valid bounded profile argument was not parsed.'
}
foreach ($Line in @('app --user-data-dir=one --user-data-dir=two', 'app --user-data-dir="one" --user-data-dir "two"', 'app --user-data-dir "separate value"', 'app "--opaque=prefix --user-data-dir=C:\owned\org.openecon.desktop suffix" --remote-debugging-port=49680')) {
    Assert-Attribution ($null -eq (Get-WebViewArgument $Line 'user-data-dir')) 'An ambiguous profile argument was accepted.'
}
Assert-Attribution ((Get-WebViewArgument 'app --user-data-dir="value"suffix' 'user-data-dir') -ceq 'valuesuffix') 'Windows quoted segment concatenation was not parsed as one argument.'
Assert-Attribution ((Get-WebViewArgument 'app --REMOTE-DEBUGGING-PORT=49680' 'remote-debugging-port') -ceq '49680') 'Chromium argument case was not handled.'
Assert-Attribution ($null -eq (Get-WebViewArgument ('x' * 65537) 'user-data-dir')) 'Oversized process input was accepted.'
$Directory = Join-Path ([IO.Path]::GetTempPath()) ('openecon-attribution-' + [guid]::NewGuid().ToString('N'))
$Profile = Join-Path $Directory 'Türkçe browser profile'
[IO.Directory]::CreateDirectory($Profile) | Out-Null
try {
    $RegisteredRoots = @{ LocalMachine = @((Join-Path $Directory 'system32'), (Join-Path $Directory 'system64')); CurrentUser = @((Join-Path $Directory 'user')) }
    $SystemRuntime = Join-Path $RegisteredRoots.LocalMachine[0] 'Microsoft/EdgeWebView/Application/155.0.1.2/msedgewebview2.exe'
    $OrphanRuntime = Join-Path $RegisteredRoots.LocalMachine[0] 'Microsoft/EdgeWebView/Application/154.0.1.2/msedgewebview2.exe'
    $UserRuntime = Join-Path $RegisteredRoots.CurrentUser[0] 'Microsoft/EdgeWebView/Application/154.0.1.2/msedgewebview2.exe'
    $Discovered = @($SystemRuntime, $OrphanRuntime, $UserRuntime)
    $Registered = @(Get-RegisteredWebViewExecutables @(@{ hive='LocalMachine'; pv='155.0.1.2' }) $RegisteredRoots $Discovered)
    Assert-Attribution ($Registered.Count -eq 1 -and $Registered[0] -ceq $SystemRuntime) 'An orphaned version was attributed as registered.'
    $Registered = @(Get-RegisteredWebViewExecutables @(@{ hive='CurrentUser'; pv='154.0.1.2' }) $RegisteredRoots $Discovered)
    Assert-Attribution ($Registered.Count -eq 1 -and $Registered[0] -ceq $UserRuntime) 'A user registration admitted an unrelated machine runtime.'
    $Registered = @(Get-RegisteredWebViewExecutables @(@{ hive='LocalMachine'; pv='155.0.1.2' }, @{ hive='LocalMachine'; pv='155.0.1.2' }) $RegisteredRoots $Discovered)
    Assert-Attribution ($Registered.Count -eq 1) 'Duplicate registry views yielded duplicate runtime identities.'
    foreach ($Registration in @(@{ hive='LocalMachine'; pv='0.0.0.0' }, @{ hive='LocalMachine'; pv='../155.0.1.2' }, @{ hive='ForeignHive'; pv='155.0.1.2' })) {
        Assert-Attribution (@(Get-RegisteredWebViewExecutables @($Registration) $RegisteredRoots $Discovered).Count -eq 0) 'An invalid registration was attributed.'
    }
    $Started = [DateTime]::UtcNow.AddSeconds(-5)
    $NativeExe = Join-Path $Directory 'openecon-desktop.exe'
    $BrowserExe = Join-Path $Directory 'msedgewebview2.exe'
    $Processes = @{
        [uint32]11 = @{ Name = 'openecon-desktop.exe'; ExecutablePath = $NativeExe; CreationDate = $Started; ParentProcessId = 5 }
        [uint32]12 = @{ Name = 'msedgewebview2.exe'; ExecutablePath = $BrowserExe; CreationDate = $Started.AddSeconds(1); ParentProcessId = 11; CommandLine = "app --user-data-dir=`"$Profile`" --remote-debugging-port=49680" }
    }
    function Get-TestEvidence { Get-WebViewAttribution -Processes $Processes -BrowserPid 12 -NativePid 11 -NativeStarted $Started -NativeExecutable $NativeExe -FreshProfileCandidates @($Profile) -RuntimeExecutables @($BrowserExe) -Port 49680 }
    $Evidence = Get-TestEvidence
    Assert-Attribution $Evidence.owned 'A fresh exact native descendant was rejected.'
    Assert-Attribution ($Evidence.lineage.Count -eq 2) 'Native lineage is incomplete.'
    Assert-Attribution (-not $Evidence.ContainsKey('command_line')) 'The raw command line was disclosed.'
    $Original = $Processes[[uint32]12].CommandLine
    $Processes[[uint32]12].CommandLine = $Original.Replace('49680', '49681')
    Assert-Attribution (-not (Get-TestEvidence).owned) 'A foreign debugger port was accepted.'
    $Processes[[uint32]12].CommandLine = $Original.Replace($Profile, (Join-Path $Directory 'foreign-profile'))
    Assert-Attribution (-not (Get-TestEvidence).owned) 'A foreign profile was accepted.'
    $Processes[[uint32]12].CommandLine = 'app --user-data-dir="https://example.invalid/?token=SYNTHETIC_SECRET" --remote-debugging-port=49680'
    $Foreign = Get-TestEvidence
    Assert-Attribution ($Foreign.user_data_dir_present -and $null -eq $Foreign.user_data_dir -and (($Foreign | ConvertTo-Json -Depth 10) -notlike '*SYNTHETIC_SECRET*')) 'A foreign profile value was disclosed.'
    $Processes[[uint32]12].CommandLine = $Original
    $Processes[[uint32]12].ParentProcessId = 13
    Assert-Attribution (-not (Get-TestEvidence).owned) 'An unrelated browser was accepted.'
    $Processes[[uint32]13] = @{ Name = 'intermediate.exe'; ExecutablePath = 'owned intermediate'; CreationDate = $Started.AddMilliseconds(500); ParentProcessId = 11 }
    Assert-Attribution (Get-TestEvidence).owned 'A valid intermediate descendant was rejected.'
    $Processes[[uint32]13].CreationDate = $Started.AddSeconds(2)
    Assert-Attribution (-not (Get-TestEvidence).owned) 'A reused parent PID created after its child was accepted.'
    $Processes[[uint32]13].CreationDate = $Started.AddMilliseconds(500)
    $Processes[[uint32]13].ParentProcessId = 12
    Assert-Attribution (-not (Get-TestEvidence).owned) 'A cyclic lineage was accepted.'
    $Processes[[uint32]12].ParentProcessId = 11
    $Processes[[uint32]11].ExecutablePath = 'foreign.exe'
    Assert-Attribution (-not (Get-TestEvidence).owned) 'A reused native PID with different executable was accepted.'
    $Processes[[uint32]11].ExecutablePath = $NativeExe
    $Processes[[uint32]11].CreationDate = $Started.AddSeconds(1)
    Assert-Attribution (-not (Get-TestEvidence).owned) 'A reused native PID with different creation time was accepted.'
    $Processes[[uint32]11].CreationDate = $Started
    $Processes[[uint32]12].ExecutablePath = 'foreign-browser.exe'
    Assert-Attribution (-not (Get-TestEvidence).owned) 'An unregistered browser executable was accepted.'
    $Processes[[uint32]12].ExecutablePath = $BrowserExe
    $Processes[[uint32]12].CommandLine = $Original.Replace($Profile, $Profile.ToUpperInvariant())
    Assert-Attribution (Get-TestEvidence).owned 'Windows profile path case was not handled ordinally.'
    $Processes[[uint32]12].CommandLine = $Original
    [IO.Directory]::Delete($Profile)
    Assert-Attribution (-not (Get-TestEvidence).owned) 'A missing profile directory was accepted.'
    $ForeignDirectory = Join-Path $Directory 'foreign-real-directory'
    [IO.Directory]::CreateDirectory($ForeignDirectory) | Out-Null
    New-Item -ItemType SymbolicLink -Path $Profile -Target $ForeignDirectory | Out-Null
    Assert-Attribution (-not (Get-TestEvidence).owned -and (Get-TestEvidence).profile_reparse_point) 'A reparse-point browser profile was accepted.'
    Remove-Item -LiteralPath $Profile -Force
}
finally { if (Test-Path -LiteralPath $Directory) { Remove-Item -LiteralPath $Directory -Recurse -Force } }
$RegistryChecks = 0
if ($IsWindows -and $env:GITHUB_ACTIONS -ceq 'true' -and $env:RUNNER_ENVIRONMENT -ceq 'github-hosted') {
    $PolicyRoot = Join-Path $env:RUNNER_TEMP ('OpenEconometrics ölçüm alanı ' + [guid]::NewGuid().ToString('N'))
    $PolicyExecutable = Join-Path $PolicyRoot 'Kurulu uygulama Türkçe/openecon-desktop.exe'
    $PolicyProfile = Join-Path $PolicyRoot 'Yerel kullanıcı profili/Owned WebView2'
    [IO.Directory]::CreateDirectory($PolicyRoot) | Out-Null
    $script:OwnedDebugPolicy = $null
    $PolicyPaths = @('SOFTWARE\Policies\Microsoft\Edge\WebView2\AdditionalBrowserArguments', 'SOFTWARE\Policies\Microsoft\Edge\WebView2\UserDataFolder')
    $PolicyBase = [Microsoft.Win32.RegistryKey]::OpenBaseKey([Microsoft.Win32.RegistryHive]::LocalMachine, [Microsoft.Win32.RegistryView]::Registry64)
    $ExistingKeys = @{}
    $FixtureValues = @{}
    try {
        foreach ($Path in $PolicyPaths) {
            $Key = $PolicyBase.OpenSubKey($Path, $false)
            try {
                $ExistingKeys[$Path] = $null -ne $Key
                if ($null -ne $Key -and $Key.GetValueNames() -contains 'openecon-desktop.exe') { throw 'Hosted policy preflight refuses any preexisting exact-executable value.' }
            }
            finally { if ($null -ne $Key) { $Key.Dispose() } }
        }
        $Refused = $false
        try { Set-OwnedDebugPolicy -Port 49680 -OwnedRoot $PolicyRoot -NativeExecutable $PolicyExecutable -BrowserProfile (Join-Path $PolicyRoot 'foreign profile') }
        catch { $Refused = $true }
        Assert-Attribution ($Refused -and $null -eq $OwnedDebugPolicy) 'A policy setter accepted a profile outside its exact owned synthetic directory.'
        $RegistryChecks++
        Set-OwnedDebugPolicy -Port 49680 -OwnedRoot $PolicyRoot -NativeExecutable $PolicyExecutable -BrowserProfile $PolicyProfile
        Assert-Attribution ($OwnedDebugPolicy.entries.Count -eq 2 -and @($OwnedDebugPolicy.entries | Where-Object { -not $_.readback_verified }).Count -eq 0) 'Both actual HKLM policies failed exact readback.'
        $RegistryChecks++
        Assert-Attribution (-not (Test-Path -LiteralPath $PolicyProfile)) 'The policy setter fabricated or reset browser storage.'
        $RegistryChecks++
        $Refused = $false
        try { Set-OwnedDebugPolicy -Port 49681 -OwnedRoot $PolicyRoot -NativeExecutable $PolicyExecutable -BrowserProfile $PolicyProfile }
        catch { $Refused = $true }
        Assert-Attribution ($Refused -and $OwnedDebugPolicy.entries.Count -eq 2) 'A second active policy lifecycle was accepted.'
        $RegistryChecks++
        [IO.Directory]::CreateDirectory($PolicyProfile) | Out-Null
        $Marker = Join-Path $PolicyProfile 'owned-persistence-fixture.txt'
        [IO.File]::WriteAllText($Marker, 'Retain actual owned browser storage across registry cleanup and reinstall phases.')
        $MarkerSha = (Get-FileHash -LiteralPath $Marker -Algorithm SHA256).Hash
        Remove-OwnedDebugPolicy
        Assert-Attribution ($null -eq $OwnedDebugPolicy -and (Get-FileHash -LiteralPath $Marker -Algorithm SHA256).Hash -ceq $MarkerSha) 'Policy cleanup reset existing owned browser storage.'
        $RegistryChecks++
        $UserPath = $PolicyPaths[1]
        $Key = $PolicyBase.CreateSubKey($UserPath, $true)
        try { $FixtureValues[$UserPath] = 'owned-fixture-existing-value'; $Key.SetValue('openecon-desktop.exe', $FixtureValues[$UserPath], [Microsoft.Win32.RegistryValueKind]::String) }
        finally { $Key.Dispose() }
        $Refused = $false
        try { Set-OwnedDebugPolicy -Port 49680 -OwnedRoot $PolicyRoot -NativeExecutable $PolicyExecutable -BrowserProfile $PolicyProfile }
        catch { $Refused = $true }
        $Key = $PolicyBase.OpenSubKey($UserPath, $true)
        try {
            Assert-Attribution ($Refused -and $Key.GetValue('openecon-desktop.exe') -ceq $FixtureValues[$UserPath] -and $null -eq $OwnedDebugPolicy) 'An existing exact-executable profile policy was replaced or its partial setter leaked.'
            $RegistryChecks++
            $Key.DeleteValue('openecon-desktop.exe', $true)
            $FixtureValues.Remove($UserPath)
        }
        finally { $Key.Dispose() }
        Set-OwnedDebugPolicy -Port 49680 -OwnedRoot $PolicyRoot -NativeExecutable $PolicyExecutable -BrowserProfile $PolicyProfile
        $Key = $PolicyBase.OpenSubKey($UserPath, $true)
        try { $FixtureValues[$UserPath] = 'owned-fixture-tampered-value'; $Key.SetValue('openecon-desktop.exe', $FixtureValues[$UserPath], [Microsoft.Win32.RegistryValueKind]::String) }
        finally { $Key.Dispose() }
        $Refused = $false
        try { Remove-OwnedDebugPolicy }
        catch { $Refused = $true }
        $Key = $PolicyBase.OpenSubKey($UserPath, $true)
        try {
            Assert-Attribution ($Refused -and $Key.GetValue('openecon-desktop.exe') -ceq $FixtureValues[$UserPath]) 'Policy cleanup deleted a changed value.'
            $RegistryChecks++
            $Key.SetValue('openecon-desktop.exe', $PolicyProfile, [Microsoft.Win32.RegistryValueKind]::String)
            $FixtureValues.Remove($UserPath)
        }
        finally { $Key.Dispose() }
        Remove-OwnedDebugPolicy
        Assert-Attribution ((Get-FileHash -LiteralPath $Marker -Algorithm SHA256).Hash -ceq $MarkerSha) 'Cold policy re-admission changed browser storage.'
        $RegistryChecks++
    }
    finally {
        foreach ($Path in $FixtureValues.Keys) {
            $Key = $PolicyBase.OpenSubKey($Path, $true)
            try {
                if ($null -ne $Key -and $Key.GetValue('openecon-desktop.exe') -ceq $FixtureValues[$Path]) {
                    if ($null -ne $OwnedDebugPolicy) { $Key.SetValue('openecon-desktop.exe', $PolicyProfile, [Microsoft.Win32.RegistryValueKind]::String) }
                    else { $Key.DeleteValue('openecon-desktop.exe', $true) }
                }
            }
            finally { if ($null -ne $Key) { $Key.Dispose() } }
        }
        try {
            Remove-OwnedDebugPolicy
            foreach ($Path in $PolicyPaths) {
                $Key = $PolicyBase.OpenSubKey($Path, $true)
                try {
                    if ($null -ne $Key) {
                        if ($Key.GetValueNames() -contains 'openecon-desktop.exe') { throw 'A test-owned exact-executable policy survived hosted preflight.' }
                        $Empty = $Key.ValueCount -eq 0 -and $Key.SubKeyCount -eq 0
                        $Key.Dispose(); $Key = $null
                        if ($ExistingKeys.ContainsKey($Path) -and -not $ExistingKeys[$Path] -and $Empty) { $PolicyBase.DeleteSubKey($Path, $true) }
                    }
                }
                finally { if ($null -ne $Key) { $Key.Dispose() } }
            }
        }
        finally { $PolicyBase.Dispose(); Remove-Item -LiteralPath $PolicyRoot -Recurse -Force }
    }
}
$FirewallChecks = 0
$HostedFirewallChecks = 0
function Assert-FirewallFixture($Condition, [string]$Message, [switch]$Hosted) {
    if (-not $Condition) { throw $Message }
    if ($Hosted) { $script:HostedFirewallChecks++ } else { $script:FirewallChecks++ }
}
$FixtureRecord = @{ name = 'OpenEcon-owned-' + [guid]::NewGuid().ToString('N'); program = 'C:\owned\openecon-runtime.exe'; enabled = $true; readback_verified = $true }
$FixtureRule = @{ Name = $FixtureRecord.name; DisplayName = $FixtureRecord.name; Direction = 'Outbound'; Action = 'Block'; Profile = 'Any'; Enabled = 'True' }
$FixtureAddress = @{ RemoteAddress = @('Internet') }
$FixtureApplication = @{ Program = $FixtureRecord.program }
Assert-OwnedFirewallRule $FixtureRecord $FixtureRule $FixtureAddress $FixtureApplication 'True'
Assert-FirewallFixture $true 'The exact rule identity was refused.'
foreach ($Pair in @(@('Name', 'foreign'), @('DisplayName', 'foreign'), @('Direction', 'Inbound'), @('Action', 'Allow'), @('Profile', 'Public'), @('Enabled', 'Unknown'))) {
    $Changed = $FixtureRule.Clone(); $Changed[$Pair[0]] = $Pair[1]
    $Refused = $false
    try { Assert-OwnedFirewallRule $FixtureRecord $Changed $FixtureAddress $FixtureApplication }
    catch { $Refused = $true }
    Assert-FirewallFixture $Refused ('A changed firewall rule field was accepted: ' + $Pair[0])
}
foreach ($Addresses in @(@('Any'), @('Internet', 'LocalSubnet'))) {
    $Refused = $false
    try { Assert-OwnedFirewallRule $FixtureRecord $FixtureRule @{ RemoteAddress = $Addresses } $FixtureApplication }
    catch { $Refused = $true }
    Assert-FirewallFixture $Refused 'A broadened or changed address scope was accepted.'
}
$Refused = $false
try { Assert-OwnedFirewallRule $FixtureRecord $FixtureRule $FixtureAddress @{ Program = 'C:\foreign\openecon-runtime.exe' } }
catch { $Refused = $true }
Assert-FirewallFixture $Refused 'A foreign program rule was accepted.'
$Refused = $false
try { Assert-OwnedFirewallRule $FixtureRecord $FixtureRule $FixtureAddress $FixtureApplication 'False' }
catch { $Refused = $true }
Assert-FirewallFixture $Refused 'A missing disabled-state readback was accepted.'
$Refused = $false
try { Assert-OwnedFirewallRule $FixtureRecord @($FixtureRule, $FixtureRule) $FixtureAddress $FixtureApplication }
catch { $Refused = $true }
Assert-FirewallFixture $Refused 'An ambiguous duplicate rule lookup was accepted.'
$ForeignRecord = $FixtureRecord.Clone(); $ForeignRecord.name = 'OpenEcon-owned-not-a-uuid'
$Refused = $false
try { Assert-OwnedFirewallRule $ForeignRecord $FixtureRule $FixtureAddress $FixtureApplication }
catch { $Refused = $true }
Assert-FirewallFixture $Refused 'A non-UUID owned-rule record was accepted.'
Assert-OwnedFirewallRule $FixtureRecord $FixtureRule $FixtureAddress @{ Program = $FixtureRecord.program.ToUpperInvariant() } 'True'
Assert-FirewallFixture $true 'The actual Windows program path comparison lost ordinal case handling.'
if ($IsWindows -and $env:GITHUB_ACTIONS -ceq 'true' -and $env:RUNNER_ENVIRONMENT -ceq 'github-hosted') {
    Assert-OwnedFirewallHost
    $BeforeProfiles = @(Get-NetFirewallProfile | Select-Object Name, Enabled)
    if ($BeforeProfiles.Count -ne 3) { throw 'Hosted firewall fixtures require all three actual profiles.' }
    $FirewallFixtureRoot = Join-Path $env:RUNNER_TEMP ('openecon-firewall-fixture-' + [guid]::NewGuid().ToString('N'))
    $HostedRecords = @(
        @{ name = 'OpenEcon-owned-' + [guid]::NewGuid().ToString('N'); program = Join-Path $FirewallFixtureRoot 'fixture-native.exe'; enabled = $true; readback_verified = $false },
        @{ name = 'OpenEcon-owned-' + [guid]::NewGuid().ToString('N'); program = Join-Path $FirewallFixtureRoot 'fixture-runtime.exe'; enabled = $true; readback_verified = $false }
    )
    $Spectator = @{ name = 'OpenEcon-owned-' + [guid]::NewGuid().ToString('N'); program = Join-Path $FirewallFixtureRoot 'fixture-unowned.exe'; enabled = $true; readback_verified = $false }
    $Mutated = $null
    $FixtureCleaned = $false
    try {
        foreach ($Profile in $BeforeProfiles) { Set-NetFirewallProfile -Name $Profile.Name -Enabled True }
        foreach ($Record in @($HostedRecords) + @($Spectator)) {
            if (Get-NetFirewallRule -Name $Record.name -ErrorAction SilentlyContinue) { throw 'A fresh hosted fixture UUID collides with an existing firewall rule.' }
            New-NetFirewallRule -Name $Record.name -DisplayName $Record.name -Direction Outbound -Action Block -Program $Record.program -RemoteAddress Internet -Profile Any -Enabled True | Out-Null
            $Rule = Get-NetFirewallRule -Name $Record.name
            Assert-OwnedFirewallRule $Record $Rule ($Rule | Get-NetFirewallAddressFilter) ($Rule | Get-NetFirewallApplicationFilter) 'True'
            $Record.readback_verified = $true
        }
        Assert-FirewallFixture $true 'Real hosted firewall fixture admission failed.' -Hosted
        Set-OwnedFirewallRules $HostedRecords $false
        Assert-FirewallFixture (@(Get-NetFirewallProfile | Where-Object { $_.Enabled -ne 'True' }).Count -eq 0) 'Disabling owned program blocks disabled a Windows firewall profile.' -Hosted
        Set-OwnedFirewallRules $HostedRecords $true
        Assert-FirewallFixture $true 'Actual owned blocks failed re-enable/readback.' -Hosted
        $Mutated = 'program'
        Set-NetFirewallRule -Name $HostedRecords[0].name -Program (Join-Path $FirewallFixtureRoot 'changed-foreign.exe') | Out-Null
        $Refused = $false
        try { Set-OwnedFirewallRules $HostedRecords $false }
        catch { $Refused = $true }
        Assert-FirewallFixture ($Refused -and (Get-NetFirewallRule -Name $HostedRecords[1].name).Enabled -eq 'True') 'A changed program was modified or a partial foreign phase transition occurred.' -Hosted
        Set-NetFirewallRule -Name $HostedRecords[0].name -Program $HostedRecords[0].program | Out-Null
        $Mutated = 'direction'
        Set-NetFirewallRule -Name $HostedRecords[0].name -Direction Inbound | Out-Null
        $Refused = $false
        try { Remove-OwnedFirewallRules @($HostedRecords[0]) }
        catch { $Refused = $true }
        Assert-FirewallFixture ($Refused -and (Get-NetFirewallRule -Name $HostedRecords[0].name).Direction -eq 'Inbound') 'Cleanup removed a changed foreign rule.' -Hosted
        Set-NetFirewallRule -Name $HostedRecords[0].name -Direction Outbound | Out-Null
        $Mutated = $null
        Remove-OwnedFirewallRules $HostedRecords
        $FixtureCleaned = $true
        Assert-FirewallFixture (@(Get-NetFirewallRule -Name $HostedRecords.name -ErrorAction SilentlyContinue).Count -eq 0) 'Owned fixture rules survived cleanup.' -Hosted
        $Rule = Get-NetFirewallRule -Name $Spectator.name
        Assert-OwnedFirewallRule $Spectator $Rule ($Rule | Get-NetFirewallAddressFilter) ($Rule | Get-NetFirewallApplicationFilter) 'True'
        Assert-FirewallFixture $true 'The unowned spectator rule changed.' -Hosted
    }
    finally {
        try {
            # Restore only the exact deliberate synthetic mutation, then apply
            # the production cleanup helper. Never broaden a foreign identity.
            if ($Mutated -eq 'program') {
                $Rule = Get-NetFirewallRule -Name $HostedRecords[0].name
                $Application = $Rule | Get-NetFirewallApplicationFilter
                $ChangedRecord = $HostedRecords[0].Clone(); $ChangedRecord.program = Join-Path $FirewallFixtureRoot 'changed-foreign.exe'
                if ($Application.Program -ceq $ChangedRecord.program) {
                    Assert-OwnedFirewallRule $ChangedRecord $Rule ($Rule | Get-NetFirewallAddressFilter) $Application
                    Set-NetFirewallRule -Name $HostedRecords[0].name -Program $HostedRecords[0].program | Out-Null
                }
                else { Assert-OwnedFirewallRule $HostedRecords[0] $Rule ($Rule | Get-NetFirewallAddressFilter) $Application }
            }
            elseif ($Mutated -eq 'direction') {
                $Rule = Get-NetFirewallRule -Name $HostedRecords[0].name
                if ($Rule.Direction -eq 'Inbound') {
                    $DirectionRecord = [pscustomobject]@{ Name = $Rule.Name; DisplayName = $Rule.DisplayName; Direction = 'Outbound'; Action = $Rule.Action; Profile = $Rule.Profile; Enabled = $Rule.Enabled }
                    Assert-OwnedFirewallRule $HostedRecords[0] $DirectionRecord ($Rule | Get-NetFirewallAddressFilter) ($Rule | Get-NetFirewallApplicationFilter)
                    Set-NetFirewallRule -Name $HostedRecords[0].name -Direction Outbound | Out-Null
                }
                else { Assert-OwnedFirewallRule $HostedRecords[0] $Rule ($Rule | Get-NetFirewallAddressFilter) ($Rule | Get-NetFirewallApplicationFilter) }
            }
        }
        finally {
            try {
                $CleanupRecords = if ($FixtureCleaned) { @($Spectator) } else { @($HostedRecords) + @($Spectator) }
                Remove-OwnedFirewallRules $CleanupRecords
            }
            finally {
            $Failures = [System.Collections.Generic.List[string]]::new()
            foreach ($Profile in $BeforeProfiles) {
                try { Set-NetFirewallProfile -Name $Profile.Name -Enabled $Profile.Enabled }
                catch { $Failures.Add($_.Exception.Message) }
            }
            foreach ($Profile in $BeforeProfiles) {
                if ((Get-NetFirewallProfile -Name $Profile.Name).Enabled -ne $Profile.Enabled) { $Failures.Add('Original profile restoration readback differs.') }
            }
            if ($Failures.Count) { throw 'An original hosted fixture firewall profile was not restored.' }
            Assert-FirewallFixture $true 'Original profiles were not restored.' -Hosted
            }
        }
    }
}
Write-Output "$Checks bounded WebView2 attribution checks passed; $RegistryChecks actual hosted HKLM policy checks; $FirewallChecks portable firewall identity checks; $HostedFirewallChecks actual hosted firewall lifecycle checks."
