# Pure, bounded attribution helpers. Never retain a complete process command line.
if ($IsWindows -and -not ('OwnedWebViewArguments' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class OwnedWebViewArguments {
    [DllImport("shell32.dll", SetLastError=true, CharSet=CharSet.Unicode)]
    static extern IntPtr CommandLineToArgvW(string command, out int count);
    [DllImport("kernel32.dll")] static extern IntPtr LocalFree(IntPtr memory);
    public static string[] Read(string command) {
        int count;
        IntPtr memory = CommandLineToArgvW(command, out count);
        if (memory == IntPtr.Zero) throw new InvalidOperationException("Windows argument parsing failed.");
        try {
            if (count < 1 || count > 256) throw new InvalidOperationException("Windows argument count is outside its bound.");
            var result = new string[count];
            for (int i = 0; i < count; i++) result[i] = Marshal.PtrToStringUni(Marshal.ReadIntPtr(memory, i * IntPtr.Size));
            return result;
        } finally { LocalFree(memory); }
    }
}
'@
}

function Split-WebViewFixtureArguments([string]$CommandLine) {
    # Portable fixture adapter only. Real Windows attribution always uses the
    # operating system's CommandLineToArgvW above, not a substring/regex parser.
    $Arguments = [System.Collections.Generic.List[string]]::new()
    $Index = 0
    while ($Index -lt $CommandLine.Length) {
        while ($Index -lt $CommandLine.Length -and [char]::IsWhiteSpace($CommandLine[$Index])) { $Index++ }
        if ($Index -ge $CommandLine.Length) { break }
        $Argument = [Text.StringBuilder]::new()
        $Quoted = $false
        while ($Index -lt $CommandLine.Length -and ($Quoted -or -not [char]::IsWhiteSpace($CommandLine[$Index]))) {
            $Slashes = 0
            while ($Index -lt $CommandLine.Length -and $CommandLine[$Index] -eq '\') { $Slashes++; $Index++ }
            if ($Index -lt $CommandLine.Length -and $CommandLine[$Index] -eq '"') {
                $Argument.Append('\', [int][Math]::Floor($Slashes / 2)) | Out-Null
                if ($Slashes % 2) { $Argument.Append('"') | Out-Null; $Index++ }
                elseif ($Quoted -and $Index + 1 -lt $CommandLine.Length -and $CommandLine[$Index + 1] -eq '"') { $Argument.Append('"') | Out-Null; $Index += 2 }
                else { $Quoted = -not $Quoted; $Index++ }
            }
            else {
                $Argument.Append('\', $Slashes) | Out-Null
                if ($Index -lt $CommandLine.Length -and ($Quoted -or -not [char]::IsWhiteSpace($CommandLine[$Index]))) { $Argument.Append($CommandLine[$Index]) | Out-Null; $Index++ }
            }
        }
        $Arguments.Add($Argument.ToString())
        if ($Arguments.Count -gt 256) { return @() }
    }
    return $Arguments.ToArray()
}

function Get-WebViewArgument([string]$CommandLine, [string]$Name) {
    if ($Name -notin @('user-data-dir', 'remote-debugging-port')) { throw 'Unknown WebView2 argument summary.' }
    if (-not $CommandLine -or $CommandLine.Length -gt 65536) { return $null }
    try {
        $Arguments = if ($IsWindows) { [OwnedWebViewArguments]::Read($CommandLine) } else { @(Split-WebViewFixtureArguments $CommandLine) }
    }
    catch { return $null }
    $Prefix = '--' + $Name + '='
    $Named = @($Arguments | Where-Object { $_.StartsWith($Prefix, [StringComparison]::OrdinalIgnoreCase) -or $_.Equals('--' + $Name, [StringComparison]::OrdinalIgnoreCase) })
    if ($Named.Count -ne 1 -or -not $Named[0].StartsWith($Prefix, [StringComparison]::OrdinalIgnoreCase)) { return $null }
    $Value = $Named[0].Substring($Prefix.Length)
    if (-not $Value -or $Value.Contains('"') -or $Value.Contains("`r") -or $Value.Contains("`n")) { return $null }
    return $Value
}

function Get-WebViewLineage([hashtable]$Processes, [uint32]$BrowserPid, [uint32]$NativePid, [DateTime]$NativeStarted, [string]$NativeExecutable) {
    $Lineage = [System.Collections.Generic.List[object]]::new()
    $Seen = [System.Collections.Generic.HashSet[uint32]]::new()
    $Current = $BrowserPid
    $ChildCreated = [DateTime]::MaxValue
    for ($Depth = 0; $Depth -lt 16; $Depth++) {
        if (-not $Seen.Add($Current) -or -not $Processes.ContainsKey($Current)) { break }
        $Process = $Processes[$Current]
        if ($null -eq $Process.CreationDate) { break }
        $Created = ([DateTime]$Process.CreationDate).ToUniversalTime()
        # A reused parent PID created after the observed child cannot establish
        # ancestry. CIM and Process.StartTime can differ in submillisecond precision.
        if ($Created -gt $ChildCreated -or $Created -lt $NativeStarted.AddMilliseconds(-2)) { break }
        $Lineage.Add(@{ pid = $Current; parent_pid = [uint32]$Process.ParentProcessId; name = [string]$Process.Name; creation_date = $Created.ToString('o') })
        if ($Current -eq $NativePid) {
            $SameExecutable = $Process.ExecutablePath -and $Process.ExecutablePath.Equals($NativeExecutable, [StringComparison]::OrdinalIgnoreCase)
            $SameCreation = [Math]::Abs(($Created - $NativeStarted).TotalMilliseconds) -le 2
            return @{ owned_native_descendant = [bool]($SameExecutable -and $SameCreation -and $Lineage.Count -gt 1); lineage = @($Lineage.ToArray()) }
        }
        $ChildCreated = $Created
        $Current = [uint32]$Process.ParentProcessId
    }
    return @{ owned_native_descendant = $false; lineage = @($Lineage.ToArray()) }
}

function Get-WebViewAttribution([hashtable]$Processes, [uint32]$BrowserPid, [uint32]$NativePid, [DateTime]$NativeStarted, [string]$NativeExecutable, [string[]]$FreshProfileCandidates, [string[]]$RuntimeExecutables, [int]$Port) {
    $Result = @{ pid = $BrowserPid; owned = $false; user_data_dir = $null; user_data_dir_present = $false; exact_fresh_profile = $false; profile_directory_exists = $false; profile_reparse_point = $false; requested_debug_port_present = $false; exact_registered_runtime = $false; owned_native_descendant = $false; lineage = @() }
    if (-not $Processes.ContainsKey($BrowserPid)) { return $Result }
    $Browser = $Processes[$BrowserPid]
    if ($Browser.Name -ne 'msedgewebview2.exe') { return $Result }
    $Result.parent_pid = [uint32]$Browser.ParentProcessId
    $Result.creation_date = $Browser.CreationDate
    $Result.requested_debug_port_present = (Get-WebViewArgument -CommandLine $Browser.CommandLine -Name 'remote-debugging-port') -ceq "$Port"
    $Result.exact_registered_runtime = [bool]($Browser.ExecutablePath -and @($RuntimeExecutables | Where-Object { $_.Equals($Browser.ExecutablePath, [StringComparison]::OrdinalIgnoreCase) }).Count -eq 1)
    $Path = Get-WebViewArgument -CommandLine $Browser.CommandLine -Name 'user-data-dir'
    $Result.user_data_dir_present = $null -ne $Path
    if ($Path) {
        # Only a matched canonical owned candidate may enter a receipt. Foreign
        # user-data values can contain arbitrary URL/credential-like text.
        try {
            if ([IO.Path]::IsPathFullyQualified($Path)) {
                $FullPath = [IO.Path]::GetFullPath($Path).TrimEnd([IO.Path]::DirectorySeparatorChar)
                $Candidate = @($FreshProfileCandidates | Where-Object { $_.Equals($FullPath, [StringComparison]::OrdinalIgnoreCase) })
                if ($Candidate.Count -eq 1) {
                    $Result.exact_fresh_profile = $true
                    $Result.user_data_dir = $Candidate[0]
                    if (Test-Path -LiteralPath $Candidate[0] -PathType Container) {
                        $Item = Get-Item -LiteralPath $Candidate[0] -Force
                        $Result.profile_directory_exists = $true
                        $Result.profile_reparse_point = [bool]($Item.Attributes -band [IO.FileAttributes]::ReparsePoint)
                    }
                }
            }
        }
        catch { $Result.profile_path_invalid = $true }
    }
    $Identity = Get-WebViewLineage -Processes $Processes -BrowserPid $BrowserPid -NativePid $NativePid -NativeStarted $NativeStarted -NativeExecutable $NativeExecutable
    $Result.owned_native_descendant = $Identity.owned_native_descendant
    $Result.lineage = $Identity.lineage
    $Result.owned = $Result.requested_debug_port_present -and $Result.exact_registered_runtime -and $Result.exact_fresh_profile -and $Result.profile_directory_exists -and -not $Result.profile_reparse_point -and $Result.owned_native_descendant
    return $Result
}

function Get-RegisteredWebViewExecutables([object[]]$Registrations, [hashtable]$InstallationRoots, [string[]]$DiscoveredFiles) {
    $Registered = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($Registration in $Registrations) {
        $Version = [string]$Registration.pv
        $Hive = [string]$Registration.hive
        if (-not $InstallationRoots.ContainsKey($Hive) -or $Version -notmatch '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$') { continue }
        try { if ([version]$Version -le [version]'0.0.0.0') { continue } }
        catch { continue }
        foreach ($Root in $InstallationRoots[$Hive]) {
            $File = Join-Path $Root ('Microsoft/EdgeWebView/Application/' + $Version + '/msedgewebview2.exe')
            if (@($DiscoveredFiles | Where-Object { $_.Equals($File, [StringComparison]::OrdinalIgnoreCase) }).Count) { $Registered.Add($File) | Out-Null }
        }
    }
    return @($Registered | Sort-Object)
}

function Assert-OwnedWebViewPolicyProfile([string]$OwnedRoot, [string]$NativeExecutable, [string]$BrowserProfile) {
    $Root = [IO.Path]::GetFullPath($OwnedRoot).TrimEnd([IO.Path]::DirectorySeparatorChar)
    $Temporary = [IO.Path]::GetFullPath($env:RUNNER_TEMP).TrimEnd([IO.Path]::DirectorySeparatorChar)
    if ((Split-Path -Leaf $Root) -notmatch '^OpenEconometrics ölçüm alanı [0-9a-f]{32}$' -or -not (Split-Path -Parent $Root).Equals($Temporary, [StringComparison]::OrdinalIgnoreCase)) { throw 'The browser policy must use the exact owned disposable acceptance root.' }
    $ExpectedExecutable = Join-Path $Root 'Kurulu uygulama Türkçe/openecon-desktop.exe'
    if (-not [IO.Path]::GetFullPath($NativeExecutable).Equals($ExpectedExecutable, [StringComparison]::OrdinalIgnoreCase)) { throw 'The browser policy executable is outside its exact owned installation.' }
    $ExpectedProfile = Join-Path $Root 'Yerel kullanıcı profili/Owned WebView2'
    $Profile = [IO.Path]::GetFullPath($BrowserProfile).TrimEnd([IO.Path]::DirectorySeparatorChar)
    if (-not $Profile.Equals($ExpectedProfile, [StringComparison]::OrdinalIgnoreCase)) { throw 'The browser policy profile differs from its exact owned synthetic directory.' }
    $Current = $Profile
    while ($true) {
        if (Test-Path -LiteralPath $Current) {
            $Item = Get-Item -LiteralPath $Current -Force
            if (-not $Item.PSIsContainer -or ($Item.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'An owned browser policy directory must be a real non-reparse directory.' }
        }
        if ($Current.Equals($Root, [StringComparison]::OrdinalIgnoreCase)) { break }
        $Current = Split-Path -Parent $Current
    }
    if (-not (Test-Path -LiteralPath $Root -PathType Container)) { throw 'The owned browser policy root is absent.' }
    return $Profile
}

function Set-OwnedDebugPolicy([int]$Port, [string]$OwnedRoot, [string]$NativeExecutable, [string]$BrowserProfile) {
    # Supported exact-executable HKLM overrides survive elevated WebView2 hosts.
    # The profile location isolates real browser storage; it never changes native
    # project/data origins and is kept intact across cold reopen/migration phases.
    if (-not $IsWindows -or $env:GITHUB_ACTIONS -cne 'true' -or $env:RUNNER_ENVIRONMENT -cne 'github-hosted' -or $env:GITHUB_REPOSITORY -cne 'bluearf/openecon' -or $env:ImageOS -notmatch '^win' -or -not ([Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw 'Owned browser policy tests require the disposable GitHub-hosted Windows administrator.' }
    if ($Port -lt 1024 -or $Port -gt 65535 -or $null -ne $script:OwnedDebugPolicy) { throw 'Invalid owned WebView2 policy lifecycle.' }
    $Profile = Assert-OwnedWebViewPolicyProfile -OwnedRoot $OwnedRoot -NativeExecutable $NativeExecutable -BrowserProfile $BrowserProfile
    $Name = [IO.Path]::GetFileName($NativeExecutable)
    if ($Name -cne 'openecon-desktop.exe') { throw 'Only the installed acceptance executable may receive owned WebView2 policies.' }
    $Values = @(
        @{ path = 'SOFTWARE\Policies\Microsoft\Edge\WebView2\AdditionalBrowserArguments'; value = "--remote-debugging-address=127.0.0.1 --remote-debugging-port=$Port" },
        @{ path = 'SOFTWARE\Policies\Microsoft\Edge\WebView2\UserDataFolder'; value = $Profile }
    )
    $script:OwnedDebugPolicy = @{ entries = [System.Collections.Generic.List[object]]::new() }
    $Base = [Microsoft.Win32.RegistryKey]::OpenBaseKey([Microsoft.Win32.RegistryHive]::LocalMachine, [Microsoft.Win32.RegistryView]::Registry64)
    try {
        foreach ($Value in $Values) {
            $Key = $null
            try {
                $Key = $Base.OpenSubKey($Value.path, $true)
                $Existed = $null -ne $Key
                if (-not $Existed) { $Key = $Base.CreateSubKey($Value.path, $true) }
                if ($Key.GetValueNames() -contains $Name) { throw 'An existing exact-executable WebView2 policy must not be replaced.' }
                $Entry = @{ path = $Value.path; name = $Name; value = $Value.value; key_existed = $Existed; readback_verified = $false }
                $script:OwnedDebugPolicy.entries.Add($Entry)
                $Key.SetValue($Name, $Value.value, [Microsoft.Win32.RegistryValueKind]::String)
                if ($Key.GetValue($Name) -cne $Value.value -or $Key.GetValueKind($Name) -ne [Microsoft.Win32.RegistryValueKind]::String) { throw 'An actual exact-executable WebView2 policy failed readback.' }
                $Entry.readback_verified = $true
            }
            finally { if ($null -ne $Key) { $Key.Dispose() } }
        }
    }
    catch { Remove-OwnedDebugPolicy; throw }
    finally { $Base.Dispose() }
}

function Remove-OwnedDebugPolicy {
    if ($null -eq $script:OwnedDebugPolicy) { return }
    $Base = [Microsoft.Win32.RegistryKey]::OpenBaseKey([Microsoft.Win32.RegistryHive]::LocalMachine, [Microsoft.Win32.RegistryView]::Registry64)
    try {
        $Entries = @($script:OwnedDebugPolicy.entries.ToArray())
        [Array]::Reverse($Entries)
        foreach ($Policy in $Entries) {
            $Key = $null
            try {
                $Key = $Base.OpenSubKey($Policy.path, $true)
                if ($null -eq $Key) { throw 'An owned WebView2 policy key disappeared before cleanup.' }
                if ($Key.GetValueNames() -contains $Policy.name) {
                    if ($Key.GetValue($Policy.name) -cne $Policy.value -or $Key.GetValueKind($Policy.name) -ne [Microsoft.Win32.RegistryValueKind]::String) { throw 'The owned WebView2 policy identity changed before cleanup.' }
                    $Key.DeleteValue($Policy.name, $true)
                }
                elseif ($Policy.readback_verified) { throw 'An owned WebView2 policy value disappeared before cleanup.' }
                if ($Key.GetValueNames() -contains $Policy.name) { throw 'An owned WebView2 policy survived cleanup.' }
                $Empty = $Key.ValueCount -eq 0 -and $Key.SubKeyCount -eq 0
                $Key.Dispose(); $Key = $null
                if (-not $Policy.key_existed -and $Empty) { $Base.DeleteSubKey($Policy.path, $true) }
                $script:OwnedDebugPolicy.entries.Remove($Policy) | Out-Null
            }
            finally { if ($null -ne $Key) { $Key.Dispose() } }
        }
        $script:OwnedDebugPolicy = $null
    }
    finally { $Base.Dispose() }
}

function Assert-OwnedFirewallRule([object]$Expected, [object]$Rule, [object]$Address, [object]$Application, [string]$Enabled = '') {
    # These records describe only rules freshly created by this acceptance run.
    # Never disable, re-enable or remove an existing/changed foreign rule.
    if ($Expected.name -cnotmatch '^OpenEcon-owned-[0-9a-f]{32}$' -or @($Rule).Count -ne 1 -or @($Address).Count -ne 1 -or @($Application).Count -ne 1 -or
        $Rule.Name -cne $Expected.name -or $Rule.DisplayName -cne $Expected.name -or $Rule.Direction -ne 'Outbound' -or $Rule.Action -ne 'Block' -or $Rule.Profile -ne 'Any' -or
        @($Address.RemoteAddress).Count -ne 1 -or @($Address.RemoteAddress)[0] -cne 'Internet' -or -not ([string]$Application.Program).Equals($Expected.program, [StringComparison]::OrdinalIgnoreCase) -or
        [string]$Rule.Enabled -notin @('True', 'False') -or ($Enabled -and [string]$Rule.Enabled -cne $Enabled)) { throw 'An actual owned firewall rule differs from its exact UUID/program/outbound Internet-block identity.' }
}

function Set-OwnedFirewallRules([object[]]$Records, [bool]$Enabled) {
    Assert-OwnedFirewallHost
    $Profiles = @(Get-NetFirewallProfile)
    if ($Profiles.Count -ne 3 -or @($Profiles | Where-Object { $_.Enabled -ne 'True' }).Count -or @($Records).Count -ne 2 -or @($Records.name | Select-Object -Unique).Count -ne 2 -or @($Records.program | Select-Object -Unique).Count -ne 2) { throw 'The online/offline phase requires three enabled profiles and the two exact owned application rules.' }
    $State = if ($Enabled) { 'True' } else { 'False' }
    # Validate the complete set before changing even its first member.
    foreach ($Expected in $Records) {
        $Rule = Get-NetFirewallRule -Name $Expected.name -ErrorAction Stop
        Assert-OwnedFirewallRule $Expected $Rule ($Rule | Get-NetFirewallAddressFilter) ($Rule | Get-NetFirewallApplicationFilter)
    }
    foreach ($Expected in $Records) {
        Set-NetFirewallRule -Name $Expected.name -Enabled $State | Out-Null
        $Rule = Get-NetFirewallRule -Name $Expected.name -ErrorAction Stop
        Assert-OwnedFirewallRule $Expected $Rule ($Rule | Get-NetFirewallAddressFilter) ($Rule | Get-NetFirewallApplicationFilter) $State
        $Expected.enabled = $Enabled
    }
}

function Remove-OwnedFirewallRules([object[]]$Records) {
    Assert-OwnedFirewallHost
    $Failures = [System.Collections.Generic.List[string]]::new()
    foreach ($Expected in $Records) {
        try {
            $Rule = Get-NetFirewallRule -Name $Expected.name -ErrorAction SilentlyContinue
            if ($null -eq $Rule) {
                if ($Expected.readback_verified) { throw 'An admitted owned firewall rule disappeared before cleanup.' }
                continue
            }
            Assert-OwnedFirewallRule $Expected $Rule ($Rule | Get-NetFirewallAddressFilter) ($Rule | Get-NetFirewallApplicationFilter)
            Remove-NetFirewallRule -Name $Expected.name -ErrorAction Stop
            if (Get-NetFirewallRule -Name $Expected.name -ErrorAction SilentlyContinue) { throw 'An owned firewall rule survived cleanup.' }
        }
        catch { $Failures.Add($_.Exception.Message) }
    }
    if ($Failures.Count) { throw ($Failures -join '; ') }
}

function Assert-OwnedFirewallHost {
    if (-not $IsWindows -or $env:GITHUB_ACTIONS -cne 'true' -or $env:RUNNER_ENVIRONMENT -cne 'github-hosted' -or $env:GITHUB_REPOSITORY -cne 'bluearf/openecon' -or $env:ImageOS -notmatch '^win' -or -not ([Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw 'Owned firewall transitions require the disposable GitHub-hosted Windows administrator.' }
}
