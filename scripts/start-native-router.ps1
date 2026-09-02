param(
    [int]$NativePort = 11436,
    [int]$RouterPort = 11435
)

$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent $PSScriptRoot
$env:DARWIN_PRIMARY_BACKEND = 'native'
$env:DARWIN_PRIMARY_MODEL = 'darwin-9b-neg-native'
$env:DARWIN_VERIFIER_BACKEND = 'native'
$env:DARWIN_VERIFIER_MODEL = 'darwin-9b-neg-native'
$env:DARWIN_NATIVE_URL = "http://127.0.0.1:$NativePort/v1"
$env:DARWIN_NATIVE_MODEL = 'darwin-9b-neg-native'
$env:DARWIN_PORT = "$RouterPort"
$env:DARWIN_MAX_CONTEXT = '163840'
$env:DARWIN_MAX_TOKENS = '43008'
$env:DARWIN_REVIEW_MAX_TOKENS = '3072'
$env:DARWIN_GPQA_SOLVER_TOKENS = '6144'
$env:DARWIN_GPQA_REVIEW_TOKENS = '6144'
$env:DARWIN_TRUNCATION_RECOVERY_TOKENS = '2048'
$env:DARWIN_MAX_ENSEMBLE_INFERENCES = '20'
$env:DARWIN_NEG_ACTIVATION_THRESHOLD = '0.05'
$env:DARWIN_NEG_MIN_ACTIVATIONS = '16'

$venvPython = Join-Path $projectDir '.venv\Scripts\python.exe'
$python = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { 'python' }
& $python -m darwin_neg_router
