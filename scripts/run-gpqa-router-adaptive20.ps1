param(
    [string]$Output = "benchmarks/results/gpqa-diamond-v3-router-adaptive20-full.jsonl",
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
    Write-Output "Starting GPQA router benchmark attempt $attempt/$MaxRestarts at $(Get-Date -Format o)"
    & python -u (Join-Path $projectRoot "benchmarks/gpqa_reconstruction.py") `
        --backend router `
        --model darwin-neg-gpqa `
        --mode adaptive20 `
        --solver-tokens 6144 `
        --review-tokens 6144 `
        --output $outputPath `
        --resume
    if ($LASTEXITCODE -eq 0) {
        Write-Output "GPQA router benchmark completed at $(Get-Date -Format o)"
        exit 0
    }
    Write-Output "Benchmark exited with code $LASTEXITCODE; completed rows remain resumable."
    if ($attempt -lt $MaxRestarts) {
        Start-Sleep -Seconds 10
    }
}

Write-Error "GPQA router benchmark failed after $MaxRestarts attempts"
exit 1
