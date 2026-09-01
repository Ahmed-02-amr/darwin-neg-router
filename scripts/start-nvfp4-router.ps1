$ErrorActionPreference = 'Stop'

$projectDir = Split-Path -Parent $PSScriptRoot
$env:DARWIN_PRIMARY_BACKEND = 'nvfp4'
$env:DARWIN_PRIMARY_MODEL = 'darwin-9b-neg-nvfp4'
$env:DARWIN_UPSTREAM_URL = 'http://127.0.0.1:8000/v1'
$env:DARWIN_UPSTREAM_API_KEY = 'EMPTY'
$env:DARWIN_VERIFIER_BACKEND = 'nvfp4'
$env:DARWIN_VERIFIER_MODEL = 'darwin-9b-neg-nvfp4'

python -m darwin_neg_router
