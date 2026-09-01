from pathlib import Path

project_root = Path(SPEC).resolve().parent.parent
hidden_imports = [
    "darwin_neg_router.config",
    "darwin_neg_router.server",
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
]

analysis = Analysis(
    [str(project_root / "desktop_app" / "darwin_neg_control.py")],
    pathex=[str(project_root)],
    binaries=[],
    datas=[
        (str(project_root / "runtime" / "native-neg"), "runtime/native-neg"),
        (str(project_root / "models" / "neg-head.fp32.bin"), "models"),
    ],
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "IPython",
        "PIL",
        "boto3",
        "botocore",
        "duckdb",
        "fsspec",
        "jupyter",
        "matplotlib",
        "nbformat",
        "numpy",
        "pandas",
        "pyarrow",
        "pytest",
        "torch",
        "transformers",
        "zmq",
    ],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="DarwinNEGControl",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

bundle = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="DarwinNEGControl",
)
