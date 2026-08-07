[CmdletBinding()]
param(
    [switch]$SkipBuild,
    [switch]$SkipCrossNode2,
    [switch]$Skip8,
    [switch]$Skip32,
    [int]$TimeoutSeconds = 1200
)

$ErrorActionPreference = 'Stop'
$Hosts = @('141.61.49.226', '141.61.49.223', '141.61.49.198', '141.61.49.195')
$Aliases = @('tilexr-226', 'tilexr-223', 'tilexr-198', 'tilexr-195')
$SourceSessions = @(
    'tilexr-pr96-9b7eb6c-49-226-20260805',
    'tilexr-pr96-9b7eb6c-49-223-20260805',
    'tilexr-pr96-9b7eb6c-49-198-20260805',
    'tilexr-pr96-9b7eb6c-49-195-20260805'
)
$RemoteRoot = '/home/l00929943/TileXR-pr96-9b7eb6c-20260805'
$RemoteEvidence = '/home/l00929943/tilexr-plan-evidence-49-20260803'
$CoordinatorIp = $Hosts[0]
$RunId = "$(Get-Date -Format 'yyyyMMdd-HHmmss-fff')-$PID"

function ConvertTo-RemoteRunner([string]$Command) {
    $normalized = $Command.Replace("`r`n", "`n").Replace("`r", "`n")
    $bytes = [Text.Encoding]::UTF8.GetBytes($normalized)
    $encoded = [Convert]::ToBase64String($bytes)
    return "echo $encoded | base64 -d | bash"
}

function Invoke-Remote([string]$Alias, [string]$Command) {
    $runner = ConvertTo-RemoteRunner $Command
    & ssh $Alias $runner | Out-Host
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "remote command failed on $Alias with exit code $exitCode"
    }
}

function Start-Remote([string]$Alias, [string]$Command) {
    $startInfo = [Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = 'ssh'
    $startInfo.UseShellExecute = $false
    [void]$startInfo.ArgumentList.Add($Alias)
    [void]$startInfo.ArgumentList.Add((ConvertTo-RemoteRunner $Command))
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    if (-not $process.Start()) { throw "failed to start ssh process for $Alias" }
    return $process
}

function Flush-SourceSessions {
    foreach ($session in $SourceSessions) {
        Write-Host "mutagen sync flush $session"
        & mutagen sync flush $session | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "Mutagen source flush failed: $session" }
    }
}

function Build-OnHost([string]$Alias, [string]$HostIp) {
    $command = @"
set -euo pipefail
root='$RemoteRoot'
evidence='$RemoteEvidence'
mkdir -p "`$evidence"
cd "`$root"
source scripts/common_env.sh
cmake_bin="`$(command -v cmake || command -v cmake3 || true)"
if [[ -z "`$cmake_bin" ]]; then
  cmake_bin="`$(find /usr/local/Ascend -type f -name cmake -perm -111 2>/dev/null | head -n 1)"
fi
if [[ -z "`$cmake_bin" || ! -x "`$cmake_bin" ]]; then
  echo 'no usable cmake found' >&2
  exit 20
fi
chmod +x tests/ep/integration/run_tilexr_ep_plan_rank.sh
"`$cmake_bin" -S . -B work/build-plan-multirank-runtime \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_INSTALL_PREFIX="`$root/work/install-plan-multirank" \
  -DTILEXR_BUILD_EP=ON -DTILEXR_BUILD_MOONEP_PLANNER=ON -DBUILD_TESTING=OFF -DTILEXR_BUILD_TESTS=OFF
"`$cmake_bin" --build work/build-plan-multirank-runtime -j8
"`$cmake_bin" --install work/build-plan-multirank-runtime
"`$cmake_bin" -S tests/ep -B work/build-plan-multirank-test \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBUILD_TILEXR_EP_PLAN_MULTIRANK_TEST=ON \
  -DTILEXR_INSTALL_PREFIX="`$root/work/install-plan-multirank" \
  -DCMAKE_INSTALL_PREFIX="`$root/work/install-plan-multirank"
"`$cmake_bin" --build work/build-plan-multirank-test --target test_tilexr_ep_plan_multirank -j8
binary="`$root/work/build-plan-multirank-test/test_tilexr_ep_plan_multirank"
test -x "`$binary"
readelf -d "`$binary" >"`$evidence/build-$HostIp-readelf.txt"
if grep -q '/devlib' "`$evidence/build-$HostIp-readelf.txt"; then
  echo 'runtime path contains forbidden CANN devlib' >&2
  exit 21
fi
{
  echo "host=`$(hostname)"
  echo "ip=$HostIp"
  echo "cmake=`$cmake_bin"
  echo "binary=`$binary"
  echo "built_at=`$(date -Iseconds)"
} >"`$evidence/build-$HostIp.txt"
"@
    Invoke-Remote $Alias $command
}

function Get-FreePorts {
    for ($attempt = 0; $attempt -lt 40; ++$attempt) {
        $base = Get-Random -Minimum 22000 -Maximum 50000
        $commPort = $base
        $controlPort = $base + 131
        if ($controlPort -gt 65535) { continue }
        $check = "if ss -ltn | awk '{print `$4}' | grep -Eq '[:.]($commPort|$controlPort)`$'; then exit 1; fi"
        $runner = ConvertTo-RemoteRunner $check
        & ssh $Aliases[0] $runner | Out-Host
        if ($LASTEXITCODE -eq 0) { return @($commPort, $controlPort) }
    }
    throw 'unable to find two free coordinator ports'
}

function Wait-RemoteLog([string]$Alias, [string]$Path, [string]$Needle, [int]$WaitSeconds) {
    $command = @"
set -euo pipefail
path='$Path'
needle='$Needle'
deadline=`$((SECONDS + $WaitSeconds))
while (( SECONDS < deadline )); do
  if [[ -f "`$path" ]] && grep -Fq -- "`$needle" "`$path"; then
    exit 0
  fi
  sleep 1
done
echo "timed out waiting for '`$needle' in `$path" >&2
if [[ -f "`$path" ]]; then
  tail -n 80 "`$path" >&2 || true
fi
exit 1
"@
    Invoke-Remote $Alias $command
}

function Stop-RecordedPhase([string]$Phase, [int]$RankSize, [int]$DevicesPerHost, [int]$HostCount) {
    for ($hostIndex = 0; $hostIndex -lt $HostCount; ++$hostIndex) {
        $ranks = @()
        for ($device = 0; $device -lt $DevicesPerHost; ++$device) {
            $ranks += $hostIndex * $DevicesPerHost + $device
        }
        $rankList = ($ranks -join ' ')
        $command = @"
set +e
for rank in $rankList; do
  pid_file='$RemoteEvidence/$Phase-rank-'"`$rank"'.pid'
  [[ -f "`$pid_file" ]] || continue
  mapfile -t recorded_pids <"`$pid_file"
  runner_pid="`${recorded_pids[0]:-}"
  if [[ ! "`$runner_pid" =~ ^[0-9]+`$ || ! -r "/proc/`$runner_pid/cmdline" ]]; then
    echo "skip unverified planner pid `$runner_pid for phase $Phase" >&2
    continue
  fi
  pid="`$runner_pid"
  cmdline="`$(tr '\0' ' ' <"/proc/`$pid/cmdline" 2>/dev/null || true)"
  if [[ "`$cmdline" != *run_tilexr_ep_plan_rank.sh* || "`$cmdline" != *'$Phase'* ]]; then
    echo "skip unverified planner pid `$runner_pid for phase $Phase" >&2
    continue
  fi
  for child_pid in "`${recorded_pids[@]:1}"; do
    [[ "`$child_pid" =~ ^[0-9]+`$ && -r "/proc/`$child_pid/cmdline" ]] || continue
    pid="`$child_pid"
    child_cmdline="`$(tr '\0' ' ' <"/proc/`$pid/cmdline" 2>/dev/null || true)"
    child_ppid="`$(awk '/^PPid:/ { print `$2 }' "/proc/`$child_pid/status" 2>/dev/null || true)"
    if [[ "`$child_ppid" == "`$runner_pid" && "`$child_cmdline" == *test_tilexr_ep_plan_multirank* ]]; then
      kill -TERM "`$child_pid" 2>/dev/null || true
    else
      echo "skip unverified planner pid `$child_pid for phase $Phase" >&2
    fi
  done
  kill -TERM "`$runner_pid" 2>/dev/null || true
done
"@
        try { Invoke-Remote $Aliases[$hostIndex] $command } catch { Write-Warning $_ }
    }
}

function Invoke-Phase([int]$RankSize, [int]$DevicesPerHost, [int]$HostCount, [string]$Phase) {
    if ($HostCount -lt 1 -or $HostCount -gt $Aliases.Count) { throw "invalid host count: $HostCount" }
    if ($RankSize -ne $HostCount * $DevicesPerHost) {
        throw "rank size $RankSize does not match $HostCount hosts x $DevicesPerHost devices"
    }
    $phaseRun = "$Phase-$RunId"
    $ports = Get-FreePorts
    $commAddress = "${CoordinatorIp}:$($ports[0])"
    $controlAddress = "${CoordinatorIp}:$($ports[1])"
    Write-Host "Starting $phaseRun with TILEXR_COMM_ID=$commAddress control=$controlAddress"

    for ($hostIndex = 0; $hostIndex -lt $HostCount; ++$hostIndex) {
        Invoke-Remote $Aliases[$hostIndex] "mkdir -p '$RemoteEvidence'"
    }

    $entries = @()
    for ($hostIndex = 0; $hostIndex -lt $HostCount; ++$hostIndex) {
        for ($device = 0; $device -lt $DevicesPerHost; ++$device) {
            $rank = $hostIndex * $DevicesPerHost + $device
            $entries += [pscustomobject]@{
                Alias = $Aliases[$hostIndex]
                Rank = $rank
                Device = $device
            }
        }
    }

    $started = @()
    try {
        $rankZero = $entries | Where-Object Rank -eq 0
        $rankZeroCommand = "bash '$RemoteRoot/tests/ep/integration/run_tilexr_ep_plan_rank.sh' $RankSize 0 0 '$commAddress' '$controlAddress' '$RemoteEvidence' '$phaseRun'"
        $started += [pscustomobject]@{ Entry = $rankZero; Process = (Start-Remote $rankZero.Alias $rankZeroCommand) }
        $listenerTimeout = [Math]::Min($TimeoutSeconds, 120)
        Write-Host "Waiting for rank 0 listener in $phaseRun"
        Wait-RemoteLog $rankZero.Alias "$RemoteEvidence/$phaseRun-rank-0.log" 'The server is listening' $listenerTimeout

        foreach ($entry in ($entries | Where-Object Rank -ne 0)) {
            $command = "bash '$RemoteRoot/tests/ep/integration/run_tilexr_ep_plan_rank.sh' $RankSize $($entry.Rank) $($entry.Device) '$commAddress' '$controlAddress' '$RemoteEvidence' '$phaseRun'"
            $started += [pscustomobject]@{ Entry = $entry; Process = (Start-Remote $entry.Alias $command) }
        }

        $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
        foreach ($item in $started) {
            $remainingMs = [Math]::Max(1, [int]($deadline - [DateTime]::UtcNow).TotalMilliseconds)
            if (-not $item.Process.WaitForExit($remainingMs)) {
                throw "$phaseRun timed out while waiting for rank $($item.Entry.Rank)"
            }
        }

        $failures = @($started | Where-Object { $_.Process.ExitCode -ne 0 })
        for ($hostIndex = 0; $hostIndex -lt $HostCount; ++$hostIndex) {
            $expectedRanks = @()
            for ($device = 0; $device -lt $DevicesPerHost; ++$device) {
                $expectedRanks += $hostIndex * $DevicesPerHost + $device
            }
            $checks = foreach ($rank in $expectedRanks) {
                "grep -qx '0' '$RemoteEvidence/$phaseRun-rank-$rank.exit'"
                "grep -q 'PLAN_VALIDATION_PASS rank=$rank rankSize=$RankSize' '$RemoteEvidence/$phaseRun-rank-$rank.log'"
            }
            try { Invoke-Remote $Aliases[$hostIndex] (($checks -join "`n")) }
            catch { $failures += [pscustomobject]@{ Process = [pscustomobject]@{ ExitCode = 1 }; Entry = [pscustomobject]@{ Rank = -1 } } }
        }
        if ($failures.Count -ne 0) { return $false }
        Write-Host "$phaseRun passed on all $RankSize ranks"
        return $true
    }
    finally {
        foreach ($item in $started) {
            if (-not $item.Process.HasExited) { try { $item.Process.Kill($true) } catch {} }
            $item.Process.Dispose()
        }
        Stop-RecordedPhase $phaseRun $RankSize $DevicesPerHost $HostCount
    }
}

Flush-SourceSessions
if (-not $SkipBuild) {
    for ($index = 0; $index -lt $Aliases.Count; ++$index) {
        Build-OnHost $Aliases[$index] $Hosts[$index]
    }
}

if (-not $SkipCrossNode2) {
    $crossNodeTwoResults = @(Invoke-Phase 2 1 2 '2rank-crossnode')
    if ($crossNodeTwoResults.Count -ne 1 -or $crossNodeTwoResults[0] -ne $true) {
        throw 'cross-node 2 Rank validation failed; 8 Rank and 32 Rank phases are blocked'
    }
}
if (-not $Skip8) {
    $eightResults = @(Invoke-Phase 8 2 4 '8rank')
    if ($eightResults.Count -ne 1 -or $eightResults[0] -ne $true) {
        throw '8 Rank validation failed; 32 Rank phase is blocked'
    }
}
if (-not $Skip32) {
    if ($Skip8) { throw '32 Rank validation requires the 8 Rank phase in the same invocation' }
    $thirtyTwoResults = @(Invoke-Phase 32 8 4 '32rank')
    if ($thirtyTwoResults.Count -ne 1 -or $thirtyTwoResults[0] -ne $true) {
        throw '32 Rank validation failed'
    }
}
Write-Host 'ALL_RANKS_PASS'
