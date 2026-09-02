param(
    [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BuildPath = Join-Path $ProjectRoot "build"
$DistPath = Join-Path $ProjectRoot "dist"
$RuntimePath = Join-Path $ProjectRoot "runtime\native-neg"
$HeadPath = Join-Path $ProjectRoot "models\neg-head.fp32.bin"
$SpecPath = Join-Path $ProjectRoot "desktop_app\DarwinNEGControl.spec"
$InstallerScript = Join-Path $ProjectRoot "installer\DarwinNEGControl.iss"
$Iscc = Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$BuildPython = if (Test-Path -LiteralPath $VenvPython) { $VenvPython } else { "python" }
$Version = "0.4.4"

foreach ($RequiredPath in @($RuntimePath, $HeadPath, $SpecPath)) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) {
        throw "Required release input is missing: $RequiredPath"
    }
}

foreach ($Target in @($BuildPath, $DistPath)) {
    $FullTarget = [IO.Path]::GetFullPath($Target)
    if (-not $FullTarget.StartsWith($ProjectRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to clean a path outside the project: $FullTarget"
    }
    if (Test-Path -LiteralPath $FullTarget) {
        Remove-Item -LiteralPath $FullTarget -Recurse -Force
    }
}

Push-Location $ProjectRoot
try {
    & $BuildPython -c "import PyInstaller" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller is missing. Install the desktop extra with: python -m pip install -e '.[desktop]'"
    }
    & $BuildPython -m PyInstaller --noconfirm --clean $SpecPath
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE"
    }

    $PortableRoot = Join-Path $DistPath "DarwinNEGControl"
    $PortableZip = Join-Path $DistPath "DarwinNEGControl-$Version-win64-portable.zip"
    Compress-Archive -LiteralPath $PortableRoot -DestinationPath $PortableZip -CompressionLevel Optimal

    if (-not $SkipInstaller) {
        if (-not (Test-Path -LiteralPath $Iscc)) {
            throw "Inno Setup 6 was not found at $Iscc"
        }
        & $Iscc $InstallerScript
        if ($LASTEXITCODE -ne 0) {
            throw "Inno Setup failed with exit code $LASTEXITCODE"
        }
    }

    $Artifacts = @($PortableZip)
    $Installer = Join-Path $DistPath "installer\DarwinNEGControl-$Version-Setup.exe"
    if (Test-Path -LiteralPath $Installer) {
        $Artifacts += $Installer
    }
    $ChecksumPath = Join-Path $DistPath "SHA256SUMS.txt"
    $ChecksumLines = foreach ($Artifact in $Artifacts) {
        $Hash = Get-FileHash -LiteralPath $Artifact -Algorithm SHA256
        "{0}  {1}" -f $Hash.Hash.ToLowerInvariant(), (Split-Path $Artifact -Leaf)
    }
    Set-Content -LiteralPath $ChecksumPath -Value $ChecksumLines -Encoding ascii

    Get-Item -LiteralPath ($Artifacts + $ChecksumPath) |
        Select-Object Name, Length, LastWriteTime |
        Format-Table -AutoSize
}
finally {
    Pop-Location
}
