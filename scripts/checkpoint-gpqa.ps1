param(
    [Parameter(Mandatory = $true)]
    [string]$Results,

    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 1000000)]
    [int]$ResumeAt
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$resultsPath = if ([System.IO.Path]::IsPathRooted($Results)) {
    [System.IO.Path]::GetFullPath($Results)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $projectRoot $Results))
}

if (-not (Test-Path -LiteralPath $resultsPath -PathType Leaf)) {
    throw "Benchmark results file does not exist: $resultsPath"
}

$rawLines = @(
    Get-Content -LiteralPath $resultsPath -Encoding UTF8 |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
)
$resumeIndex = $ResumeAt - 1
$kept = [System.Collections.Generic.List[string]]::new()
$deferred = [System.Collections.Generic.List[string]]::new()

foreach ($line in $rawLines) {
    $record = $line | ConvertFrom-Json
    if ([int]$record.index -lt $resumeIndex) {
        $kept.Add($line)
    } else {
        $deferred.Add($line)
    }
}

if ($deferred.Count -eq 0) {
    Write-Output "Checkpoint already resumes at or before Q$ResumeAt; no records moved."
    exit 0
}

$checkpointDirectory = Join-Path (Split-Path -Parent $resultsPath) "checkpoints"
New-Item -ItemType Directory -Path $checkpointDirectory -Force | Out-Null
$baseName = [System.IO.Path]::GetFileNameWithoutExtension($resultsPath)
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$archivePath = Join-Path $checkpointDirectory "$baseName.deferred-from-q$ResumeAt.$timestamp.jsonl"
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
[System.IO.File]::WriteAllLines($archivePath, $deferred, $utf8NoBom)

$temporaryPath = "$resultsPath.checkpoint.tmp"
[System.IO.File]::WriteAllLines($temporaryPath, $kept, $utf8NoBom)
Move-Item -LiteralPath $temporaryPath -Destination $resultsPath -Force

Write-Output "Checkpoint saved: next question Q$ResumeAt"
Write-Output "Kept records: $($kept.Count)"
Write-Output "Deferred records: $($deferred.Count)"
Write-Output "Deferred-record archive: $archivePath"
