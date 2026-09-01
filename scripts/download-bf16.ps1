$ErrorActionPreference = 'Stop'

$projectDir = Split-Path -Parent $PSScriptRoot
$modelsDir = if ($env:DARWIN_MODELS_DIR) { $env:DARWIN_MODELS_DIR } else { Join-Path $projectDir 'models' }
$target = Join-Path $modelsDir 'Darwin-9B-NEG-BF16'
New-Item -ItemType Directory -Force -Path $target | Out-Null

$env:HF_XET_HIGH_PERFORMANCE = '1'
hf download FINAL-Bench/Darwin-9B-NEG --local-dir $target
Write-Output "BF16 checkpoint: $target"
