$ErrorActionPreference = 'Stop'

$projectDir = Split-Path -Parent $PSScriptRoot
$env:DARWIN_PRIMARY_BACKEND = 'ollama'
$env:DARWIN_PRIMARY_MODEL = 'darwin-9b-neg:q6-neg'
$env:DARWIN_VERIFIER_BACKEND = 'ollama'
$env:DARWIN_VERIFIER_MODEL = 'darwin-9b-neg:q6-neg'
$env:DARWIN_MAX_ENSEMBLE_INFERENCES = '20'

$venvPython = Join-Path $projectDir '.venv\Scripts\python.exe'
$python = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { 'python' }
& $python -m darwin_neg_router
