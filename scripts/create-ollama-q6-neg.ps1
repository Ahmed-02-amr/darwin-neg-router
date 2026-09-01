$ErrorActionPreference = 'Stop'

$projectDir = Split-Path -Parent $PSScriptRoot
$gguf = Join-Path $projectDir 'models\gguf\Darwin-9B-NEG.i1-Q6_K.gguf'
$modelfile = Join-Path $projectDir 'Modelfile.ollama-q6-neg'

if (-not (Test-Path -LiteralPath $gguf)) {
    throw "Missing language-model GGUF: $gguf. The 624 MB mmproj file is only the vision projector."
}

& ollama create 'darwin-9b-neg:q6-neg' -f $modelfile
& ollama show 'darwin-9b-neg:q6-neg' --verbose
