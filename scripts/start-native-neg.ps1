param(
    [string]$ModelPath = '',
    [string]$RunnerPath = '',
    [string]$HeadPath = '',
    [int]$Port = 11436,
    [int]$ContextSize = 65536
)

$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent $PSScriptRoot
if (-not $ModelPath) {
    $ModelPath = Join-Path $projectDir 'models\gguf\Darwin-9B-NEG.i1-Q6_K.gguf'
}
if (-not $RunnerPath) {
    $RunnerPath = Join-Path $projectDir 'runtime\native-neg\llama-server.exe'
}
if (-not $HeadPath) {
    $HeadPath = Join-Path $projectDir 'models\neg-head.fp32.bin'
}

foreach ($required in @($ModelPath, $RunnerPath, $HeadPath)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required file not found: $required"
    }
}

$ollamaCommand = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollamaCommand) {
    throw 'Ollama is required for its installed CUDA 12 backend but was not found on PATH'
}
$ollamaRoot = Split-Path -Parent $ollamaCommand.Source
$cudaDir = Join-Path $ollamaRoot 'lib\ollama\cuda_v12'
$cudaBackend = Join-Path $cudaDir 'ggml-cuda.dll'
if (-not (Test-Path -LiteralPath $cudaBackend)) {
    throw "Ollama CUDA 12 backend not found: $cudaBackend"
}

$runnerDir = Split-Path -Parent $RunnerPath
$env:DARWIN_NEG_HEAD = $HeadPath
$env:GGML_BACKEND_PATH = $cudaBackend
$env:PATH = "$cudaDir;$runnerDir;$env:PATH"

& $RunnerPath `
    --model $ModelPath `
    --alias darwin-9b-neg-native `
    --host 127.0.0.1 `
    --port $Port `
    --ctx-size $ContextSize `
    --parallel 1 `
    --n-gpu-layers 99 `
    --flash-attn on `
    --jinja `
    --reasoning on `
    --reasoning-format deepseek `
    --no-webui

