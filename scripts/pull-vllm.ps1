$ErrorActionPreference = 'Stop'

$image = if ($env:DARWIN_VLLM_IMAGE) { $env:DARWIN_VLLM_IMAGE } else { 'vllm/vllm-openai:v0.25.1' }
docker pull $image

