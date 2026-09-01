param(
    [int]$NativePort = 11436,
    [int]$RouterPort = 11435,
    [int]$StartupTimeoutSeconds = 60
)

$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent $PSScriptRoot
$nativeScript = Join-Path $PSScriptRoot 'start-native-neg.ps1'
$routerScript = Join-Path $PSScriptRoot 'start-native-router.ps1'
$nativeProcess = $null

try {
    try {
        Invoke-RestMethod -Uri "http://127.0.0.1:$NativePort/health" -TimeoutSec 2 | Out-Null
    } catch {
        $arguments = @(
            '-NoProfile',
            '-ExecutionPolicy', 'Bypass',
            '-File', "`"$nativeScript`"",
            '-Port', "$NativePort"
        )
        $nativeProcess = Start-Process `
            -FilePath 'powershell.exe' `
            -ArgumentList $arguments `
            -WindowStyle Hidden `
            -PassThru

        $deadline = (Get-Date).AddSeconds($StartupTimeoutSeconds)
        do {
            Start-Sleep -Milliseconds 500
            if ($nativeProcess.HasExited) {
                throw "Native NEG runner exited during startup with code $($nativeProcess.ExitCode)"
            }
            try {
                Invoke-RestMethod -Uri "http://127.0.0.1:$NativePort/health" -TimeoutSec 2 | Out-Null
                break
            } catch {
                if ((Get-Date) -ge $deadline) {
                    throw "Native NEG runner did not become ready within $StartupTimeoutSeconds seconds"
                }
            }
        } while ($true)
    }

    & $routerScript -NativePort $NativePort -RouterPort $RouterPort
} finally {
    if ($nativeProcess -and -not $nativeProcess.HasExited) {
        Stop-Process -Id $nativeProcess.Id
    }
}

