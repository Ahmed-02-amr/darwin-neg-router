param(
    [string]$Output = "benchmarks/results/gpqa-diamond-v4-general-router20-adaptive-voting-full.jsonl",
    [int]$MaxRestarts = 8
)

$ErrorActionPreference = "Continue"
$projectRoot = Split-Path -Parent $PSScriptRoot
$outputPath = if ([System.IO.Path]::IsPathRooted($Output)) {
    $Output
} else {
    Join-Path $projectRoot $Output
}

for ($attempt = 1; $attempt -le $MaxRestarts; $attempt++) {
    Write-Output "Starting GPQA general-router benchmark attempt $attempt/$MaxRestarts at $(Get-Date -Format o)"
    & python -u (Join-Path $projectRoot "benchmarks/gpqa_reconstruction.py") `
        --backend router-general `
        --model darwin-neg-agent20 `
        --mode router20 `
        --routing-profile exact `
        --solver-tokens 6144 `
        --review-tokens 3072 `
        --output $outputPath `
        --resume
    if ($LASTEXITCODE -eq 0) {
        Write-Output "GPQA general-router benchmark completed at $(Get-Date -Format o)"
        exit 0
    }
    Write-Output "Benchmark exited with code $LASTEXITCODE; completed rows remain resumable."
    if ($attempt -lt $MaxRestarts) {
        Start-Sleep -Seconds 10
    }
}

Write-Error "GPQA general-router benchmark failed after $MaxRestarts attempts"
exit 1
