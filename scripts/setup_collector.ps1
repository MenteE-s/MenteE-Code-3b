# setup_collector.ps1 - one-command setup for a new team member.
# Run from the repo root:
#     powershell -ExecutionPolicy Bypass -File scripts\setup_collector.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))

$name = Read-Host "Your collector name (lowercase, e.g. syab)"
$key  = Read-Host "Your OpenRouter API key (sk-or-...)" -AsSecureString
$keyPlain = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($key))
$shard = Read-Host "Which shard are you running (1-4)"

$py = ".\.venv312\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "Creating .venv312 (one-time)..."
    py -3.12 -m venv .venv312
    & $py -m pip install -q -r requirements-collect.txt
}

& $py scripts\team_setup.py --collector $name --key $keyPlain --shard $shard
if ($LASTEXITCODE -ne 0) { throw "setup failed" }

Write-Host ""
Write-Host "Setup complete. Start collecting with:"
Write-Host "    powershell -ExecutionPolicy Bypass -File scripts\collect_until_limit.ps1 -Collector $name"
"