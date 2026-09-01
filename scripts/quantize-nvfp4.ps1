$ErrorActionPreference = 'Stop'

$projectDir = Split-Path -Parent $PSScriptRoot
$modelsDir = if ($env:DARWIN_MODELS_DIR) { $env:DARWIN_MODELS_DIR } else { Join-Path $projectDir 'models' }
$modelOptDir = Join-Path $projectDir 'vendor\Model-Optimizer'
$sourceDir = Join-Path $modelsDir 'Darwin-9B-NEG-BF16'
$targetDir = Join-Path $modelsDir 'Darwin-9B-NEG-NVFP4'
$image = if ($env:DARWIN_VLLM_IMAGE) { $env:DARWIN_VLLM_IMAGE } else { 'vllm/vllm-openai:v0.25.1' }

if (-not (Test-Path -LiteralPath (Join-Path $sourceDir 'model.safetensors.index.json'))) {
    throw 'BF16 checkpoint is missing. Run scripts\download-bf16.ps1 first.'
}
if (-not (Test-Path -LiteralPath $modelOptDir)) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $modelOptDir) | Out-Null
    git clone --depth 1 https://github.com/NVIDIA/Model-Optimizer.git $modelOptDir
}
New-Item -ItemType Directory -Force -Path $targetDir | Out-Null

$command = @'
set -euo pipefail
python -m pip install --no-cache-dir -e '/workspace/modelopt[hf]'
cd /workspace/modelopt/examples/hf_ptq
python hf_ptq.py \
  --pyt_ckpt_path /models/Darwin-9B-NEG-BF16 \
  --export_path /models/Darwin-9B-NEG-NVFP4 \
  --qformat nvfp4 \
  --kv_cache_qformat fp8_cast \
  --dataset cnn_nemotron_v2_mix \
  --calib_size 128 \
  --calib_seq 2048 \
  --batch_size 1 \
  --low_memory_mode \
  --gpu_max_mem_percentage 0.78
cp /models/Darwin-9B-NEG-BF16/neg_modules.safetensors /models/Darwin-9B-NEG-NVFP4/
cp /models/Darwin-9B-NEG-BF16/modeling_darwin_neg.py /models/Darwin-9B-NEG-NVFP4/
'@

docker run --rm --gpus all --ipc=host `
    --mount "type=bind,source=$modelsDir,target=/models" `
    --mount "type=bind,source=$modelOptDir,target=/workspace/modelopt" `
    --entrypoint bash $image -lc $command

Write-Output "NVFP4 checkpoint: $targetDir"
