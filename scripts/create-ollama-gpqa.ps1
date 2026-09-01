$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$modelFile = Join-Path $projectRoot 'Modelfile.ollama-gpqa'

ollama create 'darwin-9b-neg:gpqa' --file $modelFile
ollama show 'darwin-9b-neg:gpqa'
