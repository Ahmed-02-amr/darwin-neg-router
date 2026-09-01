$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$modelFile = Join-Path $projectRoot 'Modelfile.ollama-agent-q6'

ollama create 'darwin-9b-neg:agent-q6' --file $modelFile
ollama show 'darwin-9b-neg:agent-q6'
