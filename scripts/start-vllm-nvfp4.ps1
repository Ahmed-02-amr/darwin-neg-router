$ErrorActionPreference = 'Stop'

$projectDir = Split-Path -Parent $PSScriptRoot
$modelsDir = if ($env:DARWIN_MODELS_DIR) { $env:DARWIN_MODELS_DIR } else { Join-Path $projectDir 'models' }
$modelDir = Join-Path $modelsDir 'Darwin-9B-NEG-NVFP4'
$image = if ($env:DARWIN_VLLM_IMAGE) { $env:DARWIN_VLLM_IMAGE } else { 'vllm/vllm-openai:v0.25.1' }

if (-not (Test-Path -LiteralPath (Join-Path $modelDir 'config.json'))) {
    throw 'NVFP4 checkpoint is missing. Run scripts\quantize-nvfp4.ps1 first.'
}

docker run --rm --gpus all --ipc=host -p 8000:8000 `
    --mount "type=bind,source=$modelsDir,target=/models,readonly" `
    $image /models/Darwin-9B-NEG-NVFP4 `
    --served-model-name darwin-9b-neg-nvfp4 `
    --host 0.0.0.0 `
    --port 8000 `
    --quantization modelopt_fp4 `
    --linear-backend cutlass `
    --dtype bfloat16 `
    --kv-cache-dtype fp8 `
    --max-model-len 32768 `
    --max-num-seqs 1 `
    --max-num-batched-tokens 4096 `
    --gpu-memory-utilization 0.95 `
    --max-logprobs 20 `
    --reasoning-parser qwen3 `
    --enable-auto-tool-choice `
    --tool-call-parser qwen3 `
    --generation-config vllm

