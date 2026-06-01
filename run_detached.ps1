param(
    [string]$Target = "test-all",
    [string]$ExtraArgs = "--chunk-preview-chars 0",
    [string]$MakeExe = "make.exe"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$StdoutLog = Join-Path $LogDir "detached-$Target-$Stamp.out.log"
$StderrLog = Join-Path $LogDir "detached-$Target-$Stamp.err.log"

$env:PYTHONUNBUFFERED = "1"
$Command = "/c cd /d `"$Root`" && $MakeExe $Target EXTRA_ARGS=`"$ExtraArgs`""

$Process = Start-Process `
    -FilePath "cmd.exe" `
    -ArgumentList $Command `
    -WorkingDirectory $Root `
    -RedirectStandardOutput $StdoutLog `
    -RedirectStandardError $StderrLog `
    -WindowStyle Hidden `
    -PassThru

Write-Host "Started PID $($Process.Id)"
Write-Host "stdout: $StdoutLog"
Write-Host "stderr: $StderrLog"
Write-Host "error log: $(Join-Path $LogDir 'translation_errors.log')"
