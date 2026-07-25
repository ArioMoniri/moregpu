# MoreGPU worker — one-liner install for Windows (PowerShell).
#
#   $env:MOREGPU_SERVER="wss://ADMIN:8787/ws"; $env:MOREGPU_TOKEN="<join-token>"
#   irm https://raw.githubusercontent.com/ArioMoniri/moregpu/main/scripts/install.ps1 | iex
#
# Env: MOREGPU_SERVER, MOREGPU_TOKEN, MOREGPU_NAME, MOREGPU_THROTTLE,
#      MOREGPU_SERVICE=1 (install a logon scheduled task that survives reboot + self-heals)
$ErrorActionPreference = "Stop"

$repo   = if ($env:MOREGPU_REPO)   { $env:MOREGPU_REPO }   else { "ArioMoniri/moregpu" }
$branch = if ($env:MOREGPU_BRANCH) { $env:MOREGPU_BRANCH } else { "main" }
$server = if ($env:MOREGPU_SERVER) { $env:MOREGPU_SERVER } else { "ws://localhost:8787/ws" }
$token  = $env:MOREGPU_TOKEN
$workerUrl = "https://raw.githubusercontent.com/$repo/$branch/apps/worker/worker.ts"
$mgDir = Join-Path $env:USERPROFILE ".moregpu"
New-Item -ItemType Directory -Force -Path $mgDir | Out-Null

function Ensure-Deno {
  if (Get-Command deno -ErrorAction SilentlyContinue) { return }
  for ($i=0; $i -lt 3; $i++) {
    Write-Host "[moregpu] installing Deno runtime (try $($i+1))…"
    try { irm https://deno.land/install.ps1 | iex; $env:Path = "$env:USERPROFILE\.deno\bin;$env:Path"
          if (Get-Command deno -ErrorAction SilentlyContinue) { return } } catch { Start-Sleep 3 }
  }
  throw "[moregpu] could not install Deno — see https://deno.land/#installation"
}
Ensure-Deno
$deno = (Get-Command deno).Source

Write-Host "[moregpu] fetching worker…"
Invoke-WebRequest -UseBasicParsing $workerUrl -OutFile (Join-Path $mgDir "worker.ts")

$runArgs = @("run","--unstable-webgpu","--allow-net","--allow-env","--allow-sys",(Join-Path $mgDir "worker.ts"),"--server",$server)
if ($token) { $runArgs += @("--token",$token) }
if ($env:MOREGPU_NAME) { $runArgs += @("--name",$env:MOREGPU_NAME) }
if ($env:MOREGPU_THROTTLE) { $runArgs += @("--throttle",$env:MOREGPU_THROTTLE) }

# Self-healing supervisor wrapper.
$sup = Join-Path $mgDir "run.ps1"
@"
`$env:Path = "`$env:USERPROFILE\.deno\bin;`$env:Path"
while (`$true) {
  Write-Host "[moregpu] starting worker"
  & "$deno" $($runArgs | ForEach-Object { '"' + $_ + '"' }) -join ' '
  Write-Host "[moregpu] worker exited; restarting in 5s"; Start-Sleep 5
}
"@ | Set-Content -Encoding UTF8 $sup

if ($env:MOREGPU_SERVICE -eq "1") {
  $action  = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$sup`""
  $trigger = New-ScheduledTaskTrigger -AtLogOn
  $settings= New-ScheduledTaskSettingsSet -RestartInterval (New-TimeSpan -Minutes 1) -RestartCount 999 -StartWhenAvailable
  Register-ScheduledTask -TaskName "MoreGPUWorker" -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
  Start-ScheduledTask -TaskName "MoreGPUWorker"
  Write-Host "[moregpu] installed as a scheduled task 'MoreGPUWorker' (runs at logon, self-heals, survives reboot)."
} else {
  Write-Host "[moregpu] joining pool at $server (GPU if available, else CPU). Self-healing; Ctrl-C to stop."
  & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $sup
}
