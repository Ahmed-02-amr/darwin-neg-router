param(
    [string]$BuildRoot = (Join-Path $env:LOCALAPPDATA 'DarwinNEG\native-build'),
    [string]$OllamaRoot = '',
    [string]$RuntimeRoot = '',
    [int]$Jobs = [Math]::Max(1, [Environment]::ProcessorCount - 2)
)

$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent $PSScriptRoot
if (-not $RuntimeRoot) {
    $RuntimeRoot = Join-Path $projectDir 'runtime\native-neg'
}

$ollamaCommit = 'b7871fc0d1d82fe109536efa3e0e8e411c766c75'
$llamaCommit = '9d77fa17254e1dee4b9e92504c91611a60b1359f'
$resolvedOllamaRoot = if ($OllamaRoot) { $OllamaRoot } else { Join-Path $BuildRoot 'ollama-0.32.15' }
$serverSource = Join-Path $resolvedOllamaRoot 'llama\server'
$serverBuild = Join-Path $resolvedOllamaRoot 'build\llama-server-cpu'
$llamaSource = Join-Path $serverBuild '_deps\llama_cpp-src'
$patchPath = Join-Path $projectDir 'native\llama-b10488-neg.patch'
$headCpp = Join-Path $projectDir 'native\neg-head.cpp'
$headHeader = Join-Path $projectDir 'native\neg-head.h'

foreach ($command in @('git', 'cmake')) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "$command is required and was not found on PATH"
    }
}

New-Item -ItemType Directory -Force -Path $BuildRoot | Out-Null
if (-not (Test-Path -LiteralPath (Join-Path $resolvedOllamaRoot '.git'))) {
    New-Item -ItemType Directory -Force -Path $resolvedOllamaRoot | Out-Null
    & git -C $resolvedOllamaRoot init
    & git -C $resolvedOllamaRoot remote add origin https://github.com/ollama/ollama.git
    & git -C $resolvedOllamaRoot -c core.longpaths=true fetch --depth 1 origin $ollamaCommit
    & git -C $resolvedOllamaRoot -c core.longpaths=true checkout --detach FETCH_HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Failed to materialize the pinned Ollama source' }
}

$actualOllama = (& git -C $resolvedOllamaRoot rev-parse HEAD).Trim()
if ($actualOllama -ne $ollamaCommit) {
    throw "Build source is Ollama $actualOllama; expected $ollamaCommit"
}

Push-Location $serverSource
try {
    & cmake --preset cpu
    if ($LASTEXITCODE -ne 0) { throw 'Initial llama-server configure failed' }
} finally {
    Pop-Location
}

$actualLlama = (& git -C $llamaSource rev-parse HEAD).Trim()
if ($actualLlama -ne $llamaCommit) {
    throw "Fetched llama.cpp is $actualLlama; expected $llamaCommit"
}

Copy-Item -LiteralPath $headCpp -Destination (Join-Path $llamaSource 'tools\server\neg-head.cpp') -Force
Copy-Item -LiteralPath $headHeader -Destination (Join-Path $llamaSource 'tools\server\neg-head.h') -Force

& git -C $llamaSource apply --check $patchPath 2>$null
if ($LASTEXITCODE -eq 0) {
    & git -C $llamaSource apply $patchPath
    if ($LASTEXITCODE -ne 0) { throw 'Failed to apply the native NEG patch' }
} else {
    & git -C $llamaSource apply --reverse --check $patchPath 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw 'Native NEG patch is neither cleanly applicable nor already applied'
    }
}

Push-Location $serverSource
try {
    & cmake --preset cpu
    if ($LASTEXITCODE -ne 0) { throw 'Patched llama-server configure failed' }
} finally {
    Pop-Location
}

& cmake --build $serverBuild --target llama-server --parallel $Jobs
if ($LASTEXITCODE -ne 0) { throw 'Native NEG runner build failed' }

New-Item -ItemType Directory -Force -Path $RuntimeRoot | Out-Null
Copy-Item -Path (Join-Path $serverBuild 'bin\*') -Destination $RuntimeRoot -Force

$runner = Join-Path $RuntimeRoot 'llama-server.exe'
if (-not (Test-Path -LiteralPath $runner)) {
    throw "Build finished without $runner"
}
$digest = (Get-FileHash -Algorithm SHA256 -LiteralPath $runner).Hash.ToLowerInvariant()
Write-Host "Native NEG runner: $runner"
Write-Host "llama.cpp commit: $llamaCommit"
Write-Host "runner sha256: $digest"
