# collect_until_limit.ps1 - run the collector until the API key dries up or Ctrl+C.
#     powershell -ExecutionPolicy Bypass -File scripts\collect_until_limit.ps1 -Collector syab
param(
    [Parameter(Mandatory=$true)][string]$Collector,
    [int]$SleepSeconds = 5,
    [int]$MaxHours = 0
)

$ErrorActionPreference = "Continue"
Set-Location (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$py = ".\.venv312\Scripts\python.exe"
$cfg = "configs\collectors\$Collector.yaml"

$start = Get-Date
$iter = 0
while ($true) {
    if ($MaxHours -gt 0 -and ((Get-Date) - $start).TotalHours -ge $MaxHours) {
        Write-Host "Reached MaxHours=$MaxHours - stopping."
        break
    }
    $iter++
    Write-Host ""
    Write-Host "=== iteration $iter ($(Get-Date -Format 'HH:mm:ss')) ==="
    & $py -m src.collect run --config $cfg --collector $Collector --limit 1
    $code = $LASTEXITCODE
    if ($code -eq 3) {
        Write-Host "Provider rate limit reached - key is dry. Stopping."
        break
    }
    if ($code -ne 0) {
        Write-Host "Run exited with code $code. Sleeping $SleepSeconds s then retrying."
        Start-Sleep -Seconds $SleepSeconds
        continue
    }
    # detect 'no tasks left' by checking generated count in the report
    $reportPath = "datasets\collectors\$Collector\train.report.json"
    if (Test-Path $reportPath) {
        $rep = Get-Content $reportPath -Raw | ConvertFrom-Json
        if ($rep.generated -eq 0) {
            Write-Host "No pending tasks left - done."
            break
        }
    }
    Start-Sleep -Seconds $SleepSeconds
}
Write-Host "Finished after $iter iterations."