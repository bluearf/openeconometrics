$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'windows_webview_attribution.ps1')
# Pure path-admission fixtures. These never apply registry/firewall policies or
# claim actual hosted/native acceptance.
$Names = @('RUNNER_TEMP','GITHUB_ACTIONS','RUNNER_ENVIRONMENT','GITHUB_REPOSITORY','GITHUB_WORKFLOW','GITHUB_RUN_ID','GITHUB_RUN_ATTEMPT')
$Original = @{}
foreach ($Name in $Names) { $Original[$Name] = [Environment]::GetEnvironmentVariable($Name) }
$Temporary = Join-Path ([IO.Path]::GetTempPath()) ('openecon-cloud-profile-fixture-' + [guid]::NewGuid().ToString('N'))
$Checks = 0
function Assert-Refused([scriptblock]$Action) {
    $Refused = $false
    try { & $Action | Out-Null } catch { $Refused = $true }
    if (-not $Refused) { throw 'A foreign cloud profile was admitted.' }
    $script:Checks++
}
try {
    [Environment]::SetEnvironmentVariable('RUNNER_TEMP', $Temporary)
    [Environment]::SetEnvironmentVariable('GITHUB_ACTIONS', 'true')
    [Environment]::SetEnvironmentVariable('RUNNER_ENVIRONMENT', 'github-hosted')
    [Environment]::SetEnvironmentVariable('GITHUB_REPOSITORY', 'bluearf/openecon')
    [Environment]::SetEnvironmentVariable('GITHUB_WORKFLOW', 'OpenEconometrics Windows email-code collaboration')
    [Environment]::SetEnvironmentVariable('GITHUB_RUN_ID', '12345678901')
    [Environment]::SetEnvironmentVariable('GITHUB_RUN_ATTEMPT', '1')
    $Root = Join-Path $Temporary 'openecon-cloud-12345678901-1'
    [IO.Directory]::CreateDirectory($Root) | Out-Null
    $Executable = Join-Path $Root 'Kurulum alanı/openecon-desktop.exe'
    foreach ($Actor in @('owner','viewer')) {
        $Profile = Join-Path $Root "$Actor/Owned WebView2"
        if ((Assert-OwnedWebViewPolicyProfile $Root $Executable $Profile $Actor) -cne $Profile) { throw 'An exact synthetic actor profile was rejected.' }
        $Checks++
        Assert-Refused { Assert-OwnedWebViewPolicyProfile $Root $Executable (Join-Path $Root 'foreign/Owned WebView2') $Actor }
        Assert-Refused { Assert-OwnedWebViewPolicyProfile $Root (Join-Path $Root 'foreign/openecon-desktop.exe') $Profile $Actor }
        Assert-Refused { Assert-OwnedWebViewPolicyProfile ($Root + '-other') $Executable $Profile $Actor }
    }
    foreach ($Pair in @(@('GITHUB_ACTIONS','false'), @('RUNNER_ENVIRONMENT','self-hosted'), @('GITHUB_REPOSITORY','foreign/repo'), @('GITHUB_WORKFLOW','another workflow'), @('GITHUB_RUN_ATTEMPT','2'))) {
        $Before = [Environment]::GetEnvironmentVariable($Pair[0])
        try {
            [Environment]::SetEnvironmentVariable($Pair[0], $Pair[1])
            Assert-Refused { Assert-OwnedWebViewPolicyProfile $Root $Executable (Join-Path $Root 'owner/Owned WebView2') 'owner' }
        } finally { [Environment]::SetEnvironmentVariable($Pair[0], $Before) }
    }
    Assert-Refused { Assert-OwnedWebViewPolicyProfile $Root $Executable (Join-Path $Root 'viewer/Owned WebView2') 'owner' }
    Assert-Refused { Assert-OwnedWebViewPolicyProfile $Root $Executable (Join-Path $Root 'owner/Owned WebView2') }
    $Profile = Join-Path $Root 'owner/Owned WebView2'
    [IO.Directory]::CreateDirectory((Split-Path -Parent $Profile)) | Out-Null
    [IO.Directory]::CreateDirectory((Join-Path $Temporary 'foreign')) | Out-Null
    New-Item -ItemType SymbolicLink -Path $Profile -Target (Join-Path $Temporary 'foreign') | Out-Null
    Assert-Refused { Assert-OwnedWebViewPolicyProfile $Root $Executable $Profile 'owner' }
    Remove-Item -LiteralPath $Profile -Force
    Write-Output "Cloud profile admission fixtures passed: $Checks"
} finally {
    foreach ($Name in $Names) { [Environment]::SetEnvironmentVariable($Name, $Original[$Name]) }
    if (Test-Path -LiteralPath $Temporary) { Remove-Item -LiteralPath $Temporary -Recurse -Force }
}
