[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$MutagenSession,

    [Parameter(Mandatory = $true)]
    [string]$PrimaryHost,

    [Parameter(Mandatory = $true)]
    [string]$RemoteRoot,

    [string]$SshUser = "root",
    [string]$CannPath = "/home/pkg/b061/cann-9.1.T560",
    [string]$Hostfile = "",
    [long]$Bs = 0,
    [string]$BsList = "",
    [int]$Warmup = 20,
    [int]$Iterations = 80,
    [int]$Experts = 64,
    [int]$CommDomain = 141,
    [int]$CommPort = 10067,
    [int]$WaitSeconds = 120,
    [int]$RetrySeconds = 15,
    [int]$RankTimeout = 600,
    [int]$BuildJobs = 16,
    [string]$LogFile = ""
)

$ErrorActionPreference = "Stop"

function Assert-SafeShellValue {
    param([string]$Name, [string]$Value)
    if ([string]::IsNullOrWhiteSpace($Value) -or
        $Value.Contains("'") -or $Value.Contains("`n") -or
        $Value.Contains("`r")) {
        throw "$Name cannot be empty or contain single quotes or newlines"
    }
}

function Quote-Bash {
    param([string]$Value)
    Assert-SafeShellValue -Name "remote argument" -Value $Value
    return "'$Value'"
}

function Invoke-NativeChecked {
    param(
        [string]$Step,
        [string]$FilePath,
        [string[]]$ArgumentList
    )
    Write-Host "==> $Step"
    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE"
    }
}

function Invoke-Primary {
    param([string]$Step, [string[]]$RemoteArguments)
    $remoteCommand = ($RemoteArguments | ForEach-Object { Quote-Bash $_ }) -join " "
    Invoke-NativeChecked -Step $Step -FilePath "ssh" -ArgumentList @(
        "-o", "BatchMode=yes", "$SshUser@$PrimaryHost", $remoteCommand)
}

Assert-SafeShellValue -Name "MutagenSession" -Value $MutagenSession
Assert-SafeShellValue -Name "PrimaryHost" -Value $PrimaryHost
Assert-SafeShellValue -Name "RemoteRoot" -Value $RemoteRoot
Assert-SafeShellValue -Name "SshUser" -Value $SshUser
Assert-SafeShellValue -Name "CannPath" -Value $CannPath
if (-not $RemoteRoot.StartsWith("/") -or -not $CannPath.StartsWith("/")) {
    throw "RemoteRoot and CannPath must be absolute remote paths"
}
if ($Bs -gt 0 -and -not [string]::IsNullOrEmpty($BsList)) {
    throw "Bs and BsList are mutually exclusive"
}
if ($Bs -lt 0) {
    throw "Bs cannot be negative"
}
if ($Bs -le 0 -and [string]::IsNullOrEmpty($BsList)) {
    $Bs = 128
}
if (-not [string]::IsNullOrEmpty($BsList) -and
    $BsList -notmatch '^[1-9][0-9]*(,[1-9][0-9]*)*$') {
    throw "BsList must be a comma-separated list of positive integers"
}
foreach ($value in @($Warmup, $Iterations, $Experts, $CommDomain, $CommPort,
        $WaitSeconds, $RetrySeconds, $RankTimeout, $BuildJobs)) {
    if ($value -lt 0) {
        throw "numeric arguments cannot be negative"
    }
}
if ($Iterations -eq 0 -or $Experts -eq 0 -or $CommDomain -eq 0 -or $CommPort -eq 0 -or
    $CommPort -gt 65535 -or $RetrySeconds -eq 0 -or
    $RankTimeout -eq 0 -or $BuildJobs -eq 0) {
    throw "iterations, domain, port, retry, timeout, and build jobs must be positive"
}

if ([string]::IsNullOrEmpty($Hostfile)) {
    $Hostfile = "$RemoteRoot/hostfile"
}
if ([string]::IsNullOrEmpty($LogFile)) {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $LogFile = "$RemoteRoot/logs/combine_v2_cluster_$timestamp.log"
}
Assert-SafeShellValue -Name "Hostfile" -Value $Hostfile
Assert-SafeShellValue -Name "LogFile" -Value $LogFile
if (-not $Hostfile.StartsWith("/") -or -not $LogFile.StartsWith("/")) {
    throw "Hostfile and LogFile must be absolute remote paths"
}

$sessionText = (& mutagen sync list $MutagenSession 2>&1 | Out-String)
if ($LASTEXITCODE -ne 0) {
    throw "cannot inspect Mutagen session $MutagenSession"
}
$expectedTarget = "$SshUser@$PrimaryHost`:$RemoteRoot/source"
if (-not $sessionText.Contains($expectedTarget)) {
    throw "Mutagen session $MutagenSession does not target $expectedTarget"
}
if ($sessionText -match 'Status:\s+.*problem') {
    throw "Mutagen session $MutagenSession has a synchronization problem"
}
Invoke-NativeChecked -Step "Flush source to primary with Mutagen" -FilePath "mutagen" -ArgumentList @("sync", "flush", $MutagenSession)

$sourceDir = "$RemoteRoot/source"
$buildDir = "$RemoteRoot/build"
$installDir = "$RemoteRoot/install"
$buildScript = "$sourceDir/tools/moonep/build_combine_v2_perf.sh"
$syncScript = "$sourceDir/tools/moonep/sync_combine_v2_perf_runtime.sh"
$runScript = "$sourceDir/tools/moonep/run_combine_v2_perf_multihost.sh"

Invoke-Primary -Step "Build Combine V2 benchmark on primary" -RemoteArguments @(
    "bash", $buildScript,
    "--source-dir", $sourceDir,
    "--build-dir", $buildDir,
    "--install-dir", $installDir,
    "--cann-path", $CannPath,
    "--jobs", "$BuildJobs")

Invoke-Primary -Step "Flat-sync runtime from primary to every host" -RemoteArguments @(
    "bash", $syncScript,
    "--hostfile", $Hostfile,
    "--install-dir", $installDir,
    "--ssh-user", $SshUser)

$runArguments = @(
    "bash", $runScript,
    "--hostfile", $Hostfile,
    "--install-dir", $installDir,
    "--cann-path", $CannPath,
    "--ssh-user", $SshUser,
    "--warmup", "$Warmup",
    "--iterations", "$Iterations",
    "--experts", "$Experts",
    "--comm-domain", "$CommDomain",
    "--comm-id", "$PrimaryHost`:$CommPort",
    "--wait-seconds", "$WaitSeconds",
    "--retry-seconds", "$RetrySeconds",
    "--timeout", "$RankTimeout",
    "--log-file", $LogFile)
if (-not [string]::IsNullOrEmpty($BsList)) {
    $runArguments += @("--bs-list", $BsList)
} else {
    $runArguments += @("--bs", "$Bs")
}
Invoke-Primary -Step "Run direct-SSH Combine V2 benchmark" -RemoteArguments $runArguments

Write-Host "Completed. Primary log: $LogFile"
