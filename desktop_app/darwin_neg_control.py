from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
import urllib.error
import urllib.request
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any


APP_NAME = "Darwin NEG Control"
APP_VERSION = "0.4.4"
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
DEFAULT_CONTEXT_SIZE = 163840
DEFAULT_MAX_TOKENS = 43008
KV_CACHE_TYPE_K = "q8_0"
KV_CACHE_TYPE_V = "q8_0"
BG = "#080d18"
HEADER = "#0a1220"
PANEL = "#101827"
PANEL_ALT = "#172033"
INPUT = "#0c1422"
BORDER = "#24324a"
TEXT = "#f3f7ff"
MUTED = "#93a4bc"
SUBTLE = "#66758c"
VIOLET = "#8b5cf6"
CYAN = "#22d3ee"
ACCENT = "#34d399"
BLUE = "#60a5fa"
RED = "#fb7185"
YELLOW = "#fbbf24"


def resource_root() -> Path:
    frozen = getattr(sys, "_MEIPASS", None)
    return Path(frozen) if frozen else Path(__file__).resolve().parent.parent


def state_root() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    root = base / "DarwinNEGControl"
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    return root


def router_child() -> None:
    base = resource_root()
    if str(base) not in sys.path:
        sys.path.insert(0, str(base))
    import uvicorn

    from darwin_neg_router.config import Settings
    from darwin_neg_router.server import create_app

    settings = Settings.from_env()
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="info")


def get_json(url: str, timeout: float = 1.5) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def endpoint_ready(url: str, timeout: float = 1.0) -> bool:
    try:
        return get_json(url, timeout).get("status") == "ok"
    except (OSError, ValueError, urllib.error.URLError):
        return False


def format_tokens(value: int | float) -> str:
    number = float(value or 0)
    if number >= 1_000_000:
        return f"{number / 1_000_000:.2f}M"
    if number >= 1_000:
        return f"{number / 1_000:.1f}K"
    return str(int(number))


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds or 0))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def process_snapshot(pid: int) -> dict[str, Any] | None:
    """Return a stable Windows process identity without adding a psutil dependency."""
    if os.name != "nt" or pid <= 0:
        return None
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.QueryFullProcessImageNameW.argtypes = (
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    )
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel32.GetProcessTimes.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
    )
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return None
    try:
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)) or exit_code.value != 259:
            return None
        size = wintypes.DWORD(32768)
        buffer = ctypes.create_unicode_buffer(size.value)
        if not kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            return None
        created = wintypes.FILETIME()
        exited = wintypes.FILETIME()
        kernel = wintypes.FILETIME()
        user = wintypes.FILETIME()
        if not kernel32.GetProcessTimes(
            handle,
            ctypes.byref(created),
            ctypes.byref(exited),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            return None
        created_value = (created.dwHighDateTime << 32) | created.dwLowDateTime
        return {"pid": pid, "image": buffer.value, "created": created_value}
    finally:
        kernel32.CloseHandle(handle)


def same_process(record: dict[str, Any], snapshot: dict[str, Any] | None) -> bool:
    if not snapshot:
        return False
    try:
        image_matches = os.path.normcase(os.path.abspath(str(record["image"]))) == os.path.normcase(
            os.path.abspath(str(snapshot["image"]))
        )
        return (
            int(record["pid"]) == int(snapshot["pid"])
            and int(record["created"]) == int(snapshot["created"])
            and image_matches
        )
    except (KeyError, TypeError, ValueError):
        return False


def listening_pid(port: int) -> int | None:
    """Resolve the PID bound to a local TCP port using the Windows system utility."""
    if os.name != "nt":
        return None
    result = subprocess.run(
        ["netstat", "-ano", "-p", "tcp"],
        text=True,
        capture_output=True,
        timeout=3,
        creationflags=CREATE_NO_WINDOW,
        check=False,
    )
    if result.returncode != 0:
        return None
    pattern = re.compile(rf"^\s*TCP\s+\S*:{port}\s+\S+\s+LISTENING\s+(\d+)\s*$", re.IGNORECASE)
    for line in result.stdout.splitlines():
        match = pattern.match(line)
        if match:
            return int(match.group(1))
    return None


class DarwinControlApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.base = resource_root()
        self.state_dir = state_root()
        self.config_path = self.state_dir / "config.json"
        self.process_state_path = self.state_dir / "managed-stack.json"
        self.native_log_path = self.state_dir / "logs" / "native.log"
        self.router_log_path = self.state_dir / "logs" / "router.log"
        self.native_process: subprocess.Popen[bytes] | None = None
        self.router_process: subprocess.Popen[bytes] | None = None
        self.native_log_handle: Any = None
        self.router_log_handle: Any = None
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.poll_inflight = False
        self.closing = False
        self.external_stack = False
        self.process_lock = threading.Lock()
        self.managed_processes: dict[str, dict[str, Any]] = self._load_managed_processes()
        self.running_since: float | None = None
        self.config = self._load_config()

        self.model_var = tk.StringVar(value=self.config["model_path"])
        self.native_port_var = tk.StringVar(value=str(self.config["native_port"]))
        self.router_port_var = tk.StringVar(value=str(self.config["router_port"]))
        self.context_var = tk.StringVar(value=str(self.config["context_size"]))
        self.output_var = tk.StringVar(value=str(self.config["max_tokens"]))
        self.status_var = tk.StringVar(value="Stopped")
        self.detail_var = tk.StringVar(value="Configure the GGUF path, then start the stack.")
        self.health_var = tk.StringVar(value="Service offline")
        self.uptime_var = tk.StringVar(value="Uptime —")
        self.pid_var = tk.StringVar(value="PID —")
        self.card_values: dict[str, tk.StringVar] = {}

        self._configure_window()
        self._build_ui()
        if self.managed_processes:
            self._set_status("Recovering", "Checking services left running by the previous app session…", YELLOW)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(150, self._drain_events)
        self.root.after(500, self._poll_tick)

    def _load_managed_processes(self) -> dict[str, dict[str, Any]]:
        try:
            saved = json.loads(self.process_state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {}
        valid: dict[str, dict[str, Any]] = {}
        if isinstance(saved, dict):
            for name in ("native", "router"):
                record = saved.get(name)
                if isinstance(record, dict) and same_process(record, process_snapshot(int(record.get("pid", 0)))):
                    valid[name] = record
        if valid != saved:
            self._persist_managed_processes(valid)
        return valid

    def _persist_managed_processes(self, records: dict[str, dict[str, Any]] | None = None) -> None:
        value = records if records is not None else self.managed_processes
        try:
            if value:
                temporary = self.process_state_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
                temporary.replace(self.process_state_path)
            elif self.process_state_path.exists():
                self.process_state_path.unlink()
        except OSError:
            pass

    def _remember_process(self, name: str, process: subprocess.Popen[bytes]) -> None:
        snapshot = process_snapshot(process.pid)
        if not snapshot:
            return
        with self.process_lock:
            self.managed_processes[name] = snapshot
            self._persist_managed_processes()

    def _remember_listener(self, name: str, port: int) -> None:
        snapshot = process_snapshot(listening_pid(port) or 0)
        if not snapshot:
            return
        with self.process_lock:
            self.managed_processes[name] = snapshot
            self._persist_managed_processes()

    @staticmethod
    def _normalized_image(path: str | Path) -> str:
        return os.path.normcase(os.path.abspath(str(path)))

    def _router_process_matches(self, snapshot: dict[str, Any], port: int) -> bool:
        image = self._normalized_image(snapshot["image"])
        current = process_snapshot(os.getpid())
        expected = {self._normalized_image(sys.executable)}
        if current:
            expected.add(self._normalized_image(current["image"]))
        if image in expected:
            return True
        if getattr(sys, "frozen", False) or Path(image).name.lower() not in {"python.exe", "pythonw.exe"}:
            return False
        try:
            models = get_json(f"http://127.0.0.1:{port}/v1/models", 0.8).get("data") or []
        except (OSError, ValueError, urllib.error.URLError):
            return False
        model_ids = {str(model.get("id", "")) for model in models if isinstance(model, dict)}
        return "darwin-neg-agent" in model_ids and "darwin-neg-agent20" in model_ids

    def _discover_recoverable(self, *, native_ok: bool, router_ok: bool) -> dict[str, dict[str, Any]]:
        """Adopt only Darwin processes whose listening PID resolves to an expected executable."""
        expected_native = self._normalized_image(self.base / "runtime" / "native-neg" / "llama-server.exe")
        ports = {
            "native": int(self.native_port_var.get()),
            "router": int(self.router_port_var.get()),
        }
        health = {"native": native_ok, "router": router_ok}
        recovered: dict[str, dict[str, Any]] = {}
        with self.process_lock:
            for name, record in list(self.managed_processes.items()):
                snapshot = process_snapshot(int(record.get("pid", 0)))
                if health.get(name) and same_process(record, snapshot):
                    recovered[name] = record
            for name in ("native", "router"):
                if not health[name] or name in recovered:
                    continue
                pid = listening_pid(ports[name])
                snapshot = process_snapshot(pid or 0)
                matches = bool(
                    snapshot
                    and (
                        self._normalized_image(snapshot["image"]) == expected_native
                        if name == "native"
                        else self._router_process_matches(snapshot, ports[name])
                    )
                )
                if snapshot and matches:
                    recovered[name] = snapshot
        return recovered

    def _load_config(self) -> dict[str, Any]:
        defaults = {
            "model_path": self._discover_model(),
            "native_port": 11436,
            "router_port": 11435,
            "context_size": DEFAULT_CONTEXT_SIZE,
            "max_tokens": DEFAULT_MAX_TOKENS,
        }
        try:
            value = json.loads(self.config_path.read_text(encoding="utf-8"))
            defaults.update({key: value[key] for key in defaults if key in value})
            if (
                defaults["context_size"] == 65536
                and defaults["max_tokens"] == 16384
            ):
                defaults["context_size"] = DEFAULT_CONTEXT_SIZE
                defaults["max_tokens"] = DEFAULT_MAX_TOKENS
        except (OSError, ValueError, TypeError):
            pass
        return defaults

    def _discover_model(self) -> str:
        candidates = [
            self.base / "models" / "gguf" / "Darwin-9B-NEG.i1-Q6_K.gguf",
            Path.home()
            / "Desktop"
            / "grindset"
            / "personal-projects"
            / "Sandbox"
            / "darwin-neg-router"
            / "models"
            / "gguf"
            / "Darwin-9B-NEG.i1-Q6_K.gguf",
        ]
        explicit = os.environ.get("DARWIN_MODEL_PATH")
        if explicit:
            candidates.insert(0, Path(explicit))
        return str(next((path for path in candidates if path.exists()), ""))

    def _save_config(self) -> dict[str, Any]:
        value = {
            "model_path": self.model_var.get().strip(),
            "native_port": int(self.native_port_var.get()),
            "router_port": int(self.router_port_var.get()),
            "context_size": int(self.context_var.get()),
            "max_tokens": int(self.output_var.get()),
        }
        if not 1024 <= value["context_size"] <= 262144:
            raise ValueError("Context size must be between 1,024 and 262,144")
        if not 256 <= value["max_tokens"] <= value["context_size"]:
            raise ValueError("Output allowance must be at least 256 and no larger than context")
        for key in ("native_port", "router_port"):
            if not 1024 <= value[key] <= 65535:
                raise ValueError("Ports must be between 1,024 and 65,535")
        if value["native_port"] == value["router_port"]:
            raise ValueError("Native and router ports must differ")
        self.config_path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        self.config = value
        return value

    def _configure_window(self) -> None:
        self.root.title(f"{APP_NAME} {APP_VERSION}")
        self.root.geometry("1320x840")
        self.root.minsize(1120, 720)
        self.root.configure(bg=BG)
        self.root.option_add("*Font", ("Segoe UI", 10))
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("Header.TFrame", background=HEADER)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("Alt.TFrame", background=PANEL_ALT)
        style.configure("TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Header.TLabel", background=HEADER, foreground=TEXT)
        style.configure("Muted.TLabel", foreground=MUTED)
        style.configure("HeaderMuted.TLabel", background=HEADER, foreground=MUTED)
        style.configure("Title.TLabel", background=HEADER, foreground=TEXT, font=("Segoe UI Semibold", 22))
        style.configure("Section.TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI Semibold", 12))
        style.configure("SectionMuted.TLabel", background=PANEL, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Field.TLabel", background=PANEL, foreground=MUTED, font=("Segoe UI Semibold", 9))
        style.configure("CardTitle.TLabel", background=PANEL, foreground=MUTED, font=("Segoe UI Semibold", 9))
        style.configure("CardValue.TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI Semibold", 19))
        style.configure("Status.TLabel", background=HEADER, foreground=ACCENT, font=("Segoe UI Semibold", 11))
        style.configure("Footer.TLabel", background=HEADER, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("FooterStrong.TLabel", background=HEADER, foreground=TEXT, font=("Segoe UI Semibold", 9))
        style.configure("TButton", background=PANEL_ALT, foreground=TEXT, bordercolor=BORDER, relief="flat", font=("Segoe UI Semibold", 9), padding=(13, 8))
        style.map("TButton", background=[("active", "#22304a"), ("disabled", PANEL)], foreground=[("disabled", SUBTLE)])
        style.configure("Accent.TButton", background=VIOLET, foreground=TEXT, bordercolor=VIOLET)
        style.map("Accent.TButton", background=[("active", "#9d78f8"), ("disabled", "#3a3155")], foreground=[("disabled", "#80759b")])
        style.configure("Danger.TButton", background=PANEL_ALT, foreground=RED, bordercolor="#643044")
        style.map("Danger.TButton", background=[("active", "#3a1d2a"), ("disabled", PANEL)], foreground=[("disabled", "#70505c")])
        style.configure("TEntry", fieldbackground=INPUT, foreground=TEXT, insertcolor=TEXT, bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER, padding=(9, 7))
        style.map("TEntry", bordercolor=[("focus", VIOLET)])
        style.configure("TNotebook", background=PANEL, borderwidth=0, bordercolor=PANEL, lightcolor=PANEL, darkcolor=PANEL, tabmargins=(8, 6, 0, 0))
        style.configure("TNotebook.Tab", background=PANEL, foreground=MUTED, borderwidth=1, bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER, padding=(17, 10))
        style.map("TNotebook.Tab", background=[("selected", PANEL_ALT), ("active", PANEL_ALT)], foreground=[("selected", TEXT), ("active", TEXT)])
        style.configure(
            "Treeview",
            background=INPUT,
            fieldbackground=INPUT,
            foreground=TEXT,
            rowheight=32,
            borderwidth=0,
            bordercolor=INPUT,
            lightcolor=INPUT,
            darkcolor=INPUT,
            relief="flat",
            font=("Segoe UI", 9),
        )
        style.configure("Treeview.Heading", background=PANEL_ALT, foreground=MUTED, relief="flat", font=("Segoe UI Semibold", 9), padding=(8, 8))
        style.map("Treeview", background=[("selected", "#2b2450")], foreground=[("selected", TEXT)])
        style.configure("Vertical.TScrollbar", background=PANEL_ALT, troughcolor=INPUT, bordercolor=INPUT, arrowcolor=MUTED)

    def _build_ui(self) -> None:
        header = ttk.Frame(self.root, style="Header.TFrame", padding=(24, 16, 24, 14))
        header.pack(fill="x")
        header.columnconfigure(1, weight=1)

        mark = tk.Canvas(header, width=46, height=46, bg=HEADER, highlightthickness=0)
        mark.grid(row=0, column=0, rowspan=2, sticky="w", padx=(0, 14))
        mark.create_polygon(23, 2, 43, 13, 43, 34, 23, 45, 3, 34, 3, 13, outline=VIOLET, width=3, fill="")
        mark.create_line(4, 13, 23, 25, 42, 13, fill=CYAN, width=2)
        mark.create_line(23, 25, 23, 44, fill=BLUE, width=2)

        title_row = ttk.Frame(header, style="Header.TFrame")
        title_row.grid(row=0, column=1, sticky="sw")
        ttk.Label(title_row, text="Darwin NEG Control", style="Title.TLabel").pack(side="left")
        tk.Label(
            title_row,
            text=f"v{APP_VERSION}",
            bg="#211b3d",
            fg="#bda7ff",
            font=("Segoe UI Semibold", 9),
            padx=8,
            pady=3,
            highlightthickness=1,
            highlightbackground="#4c3b7b",
        ).pack(side="left", padx=(10, 0), pady=(4, 0))
        ttk.Label(
            header,
            text="Native entropy gating · selective verification · CodePilot gateway",
            style="HeaderMuted.TLabel",
        ).grid(row=1, column=1, sticky="nw", pady=(2, 0))

        actions = ttk.Frame(header, style="Header.TFrame")
        actions.grid(row=0, column=2, rowspan=2, sticky="e")
        status_wrap = ttk.Frame(actions, style="Header.TFrame")
        status_wrap.pack(side="left", padx=(0, 18))
        status_line = ttk.Frame(status_wrap, style="Header.TFrame")
        status_line.pack(anchor="e")
        self.status_dot = tk.Canvas(status_line, width=12, height=12, bg=HEADER, highlightthickness=0)
        self.status_dot.pack(side="left", padx=(0, 7), pady=(3, 0))
        self.status_dot_id = self.status_dot.create_oval(2, 2, 10, 10, fill=SUBTLE, outline="")
        self.status_label = ttk.Label(status_line, textvariable=self.status_var, style="Status.TLabel")
        self.status_label.pack(side="left")
        self.detail_label = ttk.Label(status_wrap, textvariable=self.detail_var, style="HeaderMuted.TLabel")
        self.detail_label.pack(anchor="e", pady=(2, 0))
        self.start_button = ttk.Button(actions, text="▶  Start stack", style="Accent.TButton", command=self.start)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(actions, text="■  Stop", style="Danger.TButton", command=self.stop)
        self.stop_button.pack(side="left", padx=(8, 0))
        self.restart_button = ttk.Button(actions, text="↻  Restart", command=self.restart)
        self.restart_button.pack(side="left", padx=(8, 0))

        accent_line = tk.Canvas(self.root, height=2, bg=VIOLET, highlightthickness=0)
        accent_line.pack(fill="x")
        accent_line.bind("<Configure>", lambda event: self._draw_accent_line(accent_line, event.width))

        outer = ttk.Frame(self.root, padding=(24, 16, 24, 10))
        outer.pack(fill="both", expand=True)

        config_border = tk.Frame(outer, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
        config_border.pack(fill="x", pady=(0, 14))
        config_panel = ttk.Frame(config_border, style="Panel.TFrame", padding=(16, 12, 16, 14))
        config_panel.pack(fill="x")
        config_panel.columnconfigure(0, weight=5)
        config_panel.columnconfigure(1, weight=0)
        for index in range(2, 6):
            config_panel.columnconfigure(index, weight=1)
        ttk.Label(config_panel, text="Runtime configuration", style="Section.TLabel").grid(row=0, column=0, columnspan=5, sticky="w")
        ttk.Button(config_panel, text="CodePilot setup", command=self._open_codepilot_setup).grid(row=0, column=5, sticky="e")
        ttk.Label(config_panel, text="Language model", style="Field.TLabel").grid(row=1, column=0, sticky="w", pady=(10, 5))
        ttk.Entry(config_panel, textvariable=self.model_var).grid(row=2, column=0, sticky="ew")
        ttk.Button(config_panel, text="Browse…", command=self._browse_model).grid(row=2, column=1, padx=(8, 18))
        fields = (
            ("Native port", self.native_port_var),
            ("Gateway port", self.router_port_var),
            ("Context", self.context_var),
            ("Max output", self.output_var),
        )
        for offset, (label, variable) in enumerate(fields, start=2):
            ttk.Label(config_panel, text=label, style="Field.TLabel").grid(row=1, column=offset, sticky="w", padx=(0, 8), pady=(10, 5))
            ttk.Entry(config_panel, textvariable=variable, width=11).grid(row=2, column=offset, sticky="ew", padx=(0, 8))

        cards = ttk.Frame(outer)
        cards.pack(fill="x", pady=(0, 14))
        card_specs = (
            ("Requests", "requests", "0", "Total", VIOLET),
            ("Model calls", "calls", "0", "Inferences", "#a78bfa"),
            ("Generated", "tokens", "0", "Tokens", CYAN),
            ("NEG active", "neg", "0.0%", "Gate rate", ACCENT),
            ("Routed", "routed", "0.0%", "Verifier rate", BLUE),
            ("Throughput", "throughput", "0.0 tok/s", "Generation", CYAN),
            ("GPU memory", "gpu_memory", "—", "VRAM", YELLOW),
            ("GPU load", "gpu", "—", "Utilization · temp", "#f59e0b"),
        )
        for index, (title, key, initial, caption, color) in enumerate(card_specs):
            cards.columnconfigure(index, weight=1)
            card = tk.Frame(cards, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
            card.grid(row=0, column=index, sticky="nsew", padx=(0 if index == 0 else 6, 0))
            tk.Frame(card, bg=color, height=2).pack(fill="x")
            body = ttk.Frame(card, style="Panel.TFrame", padding=(12, 9, 12, 10))
            body.pack(fill="both", expand=True)
            ttk.Label(body, text=title, style="CardTitle.TLabel").pack(anchor="w")
            variable = tk.StringVar(value=initial)
            self.card_values[key] = variable
            ttk.Label(body, textvariable=variable, style="CardValue.TLabel").pack(anchor="w", pady=(3, 0))
            ttk.Label(body, text=caption, style="SectionMuted.TLabel").pack(anchor="w")

        notebook_border = tk.Frame(outer, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
        notebook_border.pack(fill="both", expand=True)
        notebook = ttk.Notebook(notebook_border)
        notebook.pack(fill="both", expand=True)
        history_panel = ttk.Frame(notebook, style="Panel.TFrame", padding=(10, 4, 10, 10))
        logs_panel = ttk.Frame(notebook, style="Panel.TFrame", padding=(10, 4, 10, 10))
        notebook.add(history_panel, text="Recent requests")
        notebook.add(logs_panel, text="Service logs")

        columns = ("time", "model", "calls", "tokens", "neg", "latency", "route", "finish")
        self.history = ttk.Treeview(history_panel, columns=columns, show="headings")
        headings = {
            "time": "Time",
            "model": "Model",
            "calls": "Calls",
            "tokens": "Tokens",
            "neg": "NEG",
            "latency": "Latency",
            "route": "Route",
            "finish": "Finish",
        }
        widths = {"time": 90, "model": 165, "calls": 55, "tokens": 85, "neg": 75, "latency": 75, "route": 180, "finish": 75}
        for column in columns:
            self.history.heading(column, text=headings[column])
            self.history.column(column, width=widths[column], minwidth=50, anchor="w")
        self.history.tag_configure("even", background=INPUT)
        self.history.tag_configure("odd", background="#0f1929")
        history_scroll = ttk.Scrollbar(history_panel, orient="vertical", command=self.history.yview)
        self.history.configure(yscrollcommand=history_scroll.set)
        self.history.pack(side="left", fill="both", expand=True)
        history_scroll.pack(side="right", fill="y")

        log_toolbar = ttk.Frame(logs_panel, style="Panel.TFrame")
        log_toolbar.pack(fill="x", pady=(0, 8))
        ttk.Button(log_toolbar, text="Refresh", command=self._refresh_logs).pack(side="left")
        ttk.Button(log_toolbar, text="Open log folder", command=self._open_logs).pack(side="left", padx=8)
        self.log_text = tk.Text(
            logs_panel,
            bg=INPUT,
            fg="#cbd6e7",
            insertbackground=TEXT,
            relief="flat",
            borderwidth=0,
            padx=12,
            pady=10,
            font=("Cascadia Mono", 9),
            wrap="none",
        )
        log_scroll = ttk.Scrollbar(logs_panel, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

        footer = ttk.Frame(self.root, style="Header.TFrame", padding=(24, 9, 24, 10))
        footer.pack(fill="x", side="bottom", before=outer)
        footer.columnconfigure(7, weight=1)
        self.footer_dot = tk.Canvas(footer, width=10, height=10, bg=HEADER, highlightthickness=0)
        self.footer_dot.grid(row=0, column=0, padx=(0, 6))
        self.footer_dot_id = self.footer_dot.create_oval(1, 1, 9, 9, fill=SUBTLE, outline="")
        ttk.Label(footer, textvariable=self.health_var, style="FooterStrong.TLabel").grid(row=0, column=1, sticky="w")
        ttk.Label(footer, text="│", style="Footer.TLabel").grid(row=0, column=2, padx=12)
        ttk.Label(footer, text="Protocols", style="Footer.TLabel").grid(row=0, column=3, sticky="w")
        ttk.Label(footer, text="●  Anthropic /v1/messages", style="Footer.TLabel", foreground="#b79cff").grid(row=0, column=4, padx=(10, 14))
        ttk.Label(footer, text="●  OpenAI /v1/chat/completions", style="Footer.TLabel", foreground=CYAN).grid(row=0, column=5)
        ttk.Label(footer, textvariable=self.uptime_var, style="Footer.TLabel").grid(row=0, column=8, padx=(14, 0), sticky="e")
        ttk.Label(footer, textvariable=self.pid_var, style="Footer.TLabel").grid(row=0, column=9, padx=(14, 0), sticky="e")
        self._set_buttons(running=False, busy=False)

    @staticmethod
    def _draw_accent_line(canvas: tk.Canvas, width: int) -> None:
        canvas.delete("all")
        midpoint = max(1, width // 2)
        canvas.create_rectangle(0, 0, midpoint, 2, fill=VIOLET, outline="")
        canvas.create_rectangle(midpoint, 0, width, 2, fill=CYAN, outline="")

    def _browse_model(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select Darwin language-model GGUF",
            filetypes=(("GGUF model", "*.gguf"), ("All files", "*.*")),
            initialdir=str(Path(self.model_var.get()).parent) if self.model_var.get() else str(Path.home()),
        )
        if selected:
            self.model_var.set(selected)

    def _set_buttons(self, *, running: bool, busy: bool) -> None:
        controllable = bool(
            self.native_process
            or self.router_process
            or self.managed_processes
        )
        self.start_button.configure(state="disabled" if running or busy else "normal")
        control_state = "normal" if running and not busy and controllable else "disabled"
        self.stop_button.configure(state=control_state)
        self.restart_button.configure(state=control_state)

    def _set_status(self, status: str, detail: str, color: str = ACCENT) -> None:
        self.status_var.set(status)
        self.detail_var.set(detail)
        self.status_label.configure(foreground=color)
        self.status_dot.itemconfigure(self.status_dot_id, fill=color)
        healthy = status == "Running"
        self.footer_dot.itemconfigure(self.footer_dot_id, fill=ACCENT if healthy else color)
        self.health_var.set("Service healthy" if healthy else ("Service recovering" if status in {"Starting", "Restarting", "Recovering"} else "Service offline"))

    def start(self) -> None:
        try:
            config = self._save_config()
        except (OSError, ValueError) as exc:
            messagebox.showerror(APP_NAME, str(exc))
            return
        model = Path(config["model_path"])
        if not model.is_file() or model.suffix.lower() != ".gguf":
            messagebox.showerror(APP_NAME, "Select the Darwin language-model .gguf file first.")
            return
        self._set_buttons(running=False, busy=True)
        self._set_status("Starting", "Loading Q6_K and attaching the released NEG head…", YELLOW)
        threading.Thread(target=self._start_worker, args=(config,), daemon=True).start()

    def _start_worker(self, config: dict[str, Any]) -> None:
        native_health = f"http://127.0.0.1:{config['native_port']}/health"
        router_health = f"http://127.0.0.1:{config['router_port']}/health"
        if endpoint_ready(native_health) and endpoint_ready(router_health):
            recovered = self._discover_recoverable(native_ok=True, router_ok=True)
            if recovered:
                with self.process_lock:
                    self.managed_processes.update(recovered)
                    self._persist_managed_processes()
                self.external_stack = False
                self.events.put(("recovered", "Reconnected to services from a previous Darwin Control session."))
            else:
                self.external_stack = True
                self.events.put(("started", "Connected to an already-running external stack; process ownership is external."))
            return
        if endpoint_ready(native_health) or endpoint_ready(router_health):
            native_ok = endpoint_ready(native_health)
            router_ok = endpoint_ready(router_health)
            recovered = self._discover_recoverable(native_ok=native_ok, router_ok=router_ok)
            if recovered and len(recovered) == int(native_ok) + int(router_ok):
                with self.process_lock:
                    self.managed_processes.update(recovered)
                    self._persist_managed_processes()
                self._stop_owned_processes()
                for url in (native_health, router_health):
                    deadline = time.monotonic() + 8
                    while endpoint_ready(url, 0.3) and time.monotonic() < deadline:
                        time.sleep(0.2)
            else:
                self.events.put(("error", "A non-Darwin process occupies one service port. Free that port before starting."))
                return
        try:
            runner = self.base / "runtime" / "native-neg" / "llama-server.exe"
            head = self.base / "models" / "neg-head.fp32.bin"
            if not runner.is_file():
                raise FileNotFoundError(f"Native runner is missing: {runner}")
            if not head.is_file():
                raise FileNotFoundError(f"Released NEG head is missing: {head}")
            cuda_dir, cuda_backend = self._find_ollama_cuda()
            native_env = os.environ.copy()
            native_env["DARWIN_NEG_HEAD"] = str(head)
            native_env["GGML_BACKEND_PATH"] = str(cuda_backend)
            native_env["PATH"] = os.pathsep.join((str(cuda_dir), str(runner.parent), native_env.get("PATH", "")))
            self.native_log_handle = self.native_log_path.open("ab", buffering=0)
            self.native_process = subprocess.Popen(
                [
                    str(runner),
                    "--model", str(config["model_path"]),
                    "--alias", "darwin-9b-neg-native",
                    "--host", "127.0.0.1",
                    "--port", str(config["native_port"]),
                    "--ctx-size", str(config["context_size"]),
                    "--parallel", "1",
                    "--n-gpu-layers", "99",
                    "--cache-type-k", KV_CACHE_TYPE_K,
                    "--cache-type-v", KV_CACHE_TYPE_V,
                    "--flash-attn", "on",
                    "--jinja",
                    "--reasoning", "on",
                    "--reasoning-format", "deepseek",
                    "--no-webui",
                ],
                cwd=str(self.base),
                env=native_env,
                stdout=self.native_log_handle,
                stderr=subprocess.STDOUT,
                creationflags=CREATE_NO_WINDOW,
            )
            self._remember_process("native", self.native_process)
            self._wait_for(native_health, self.native_process, 120, "native NEG runner")
            self._remember_listener("native", config["native_port"])

            router_env = os.environ.copy()
            router_env.update(
                {
                    "DARWIN_PRIMARY_BACKEND": "native",
                    "DARWIN_PRIMARY_MODEL": "darwin-9b-neg-native",
                    "DARWIN_VERIFIER_BACKEND": "native",
                    "DARWIN_VERIFIER_MODEL": "darwin-9b-neg-native",
                    "DARWIN_NATIVE_URL": f"http://127.0.0.1:{config['native_port']}/v1",
                    "DARWIN_NATIVE_MODEL": "darwin-9b-neg-native",
                    "DARWIN_HOST": "127.0.0.1",
                    "DARWIN_PORT": str(config["router_port"]),
                    "DARWIN_MAX_CONTEXT": str(config["context_size"]),
                    "DARWIN_MAX_TOKENS": str(config["max_tokens"]),
                    "DARWIN_REVIEW_MAX_TOKENS": "3072",
                    "DARWIN_GPQA_SOLVER_TOKENS": "6144",
                    "DARWIN_GPQA_REVIEW_TOKENS": "6144",
                    "DARWIN_TRUNCATION_RECOVERY_TOKENS": "2048",
                    "DARWIN_MAX_ENSEMBLE_INFERENCES": "20",
                    "DARWIN_NEG_ACTIVATION_THRESHOLD": "0.05",
                    "DARWIN_NEG_MIN_ACTIVATIONS": "16",
                }
            )
            self.router_log_handle = self.router_log_path.open("ab", buffering=0)
            command = [sys.executable, "--router-child"] if getattr(sys, "frozen", False) else [sys.executable, str(Path(__file__).resolve()), "--router-child"]
            self.router_process = subprocess.Popen(
                command,
                cwd=str(self.base),
                env=router_env,
                stdout=self.router_log_handle,
                stderr=subprocess.STDOUT,
                creationflags=CREATE_NO_WINDOW,
            )
            self._remember_process("router", self.router_process)
            self._wait_for(router_health, self.router_process, 45, "routing gateway")
            self._remember_listener("router", config["router_port"])
            self.external_stack = False
            self.events.put(
                (
                    "started",
                    f"OpenAI and Anthropic endpoints ready on 127.0.0.1:{config['router_port']}",
                )
            )
        except Exception as exc:
            self._stop_owned_processes()
            self.events.put(("error", str(exc)))

    def _find_ollama_cuda(self) -> tuple[Path, Path]:
        candidates: list[Path] = []
        command = shutil.which("ollama")
        if command:
            candidates.append(Path(command).resolve().parent)
        local = Path(os.environ.get("LOCALAPPDATA", ""))
        candidates.extend((local / "Programs" / "Ollama", Path("C:/Program Files/Ollama")))
        for root in candidates:
            cuda_dir = root / "lib" / "ollama" / "cuda_v12"
            backend = cuda_dir / "ggml-cuda.dll"
            if backend.is_file():
                return cuda_dir, backend
        raise FileNotFoundError("Ollama's CUDA 12 backend was not found. Install Ollama 0.32.15 first.")

    @staticmethod
    def _wait_for(url: str, process: subprocess.Popen[bytes], timeout: int, name: str) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"The {name} exited during startup with code {process.returncode}")
            if endpoint_ready(url):
                return
            time.sleep(0.4)
        raise TimeoutError(f"The {name} did not become ready within {timeout} seconds")

    def stop(self) -> None:
        if self.external_stack and not self.managed_processes:
            messagebox.showinfo(APP_NAME, "This app did not start the connected services and will not terminate them.")
            return
        self._set_buttons(running=True, busy=True)
        self._set_status("Stopping", "Closing router and native runner…", YELLOW)
        threading.Thread(target=self._stop_worker, daemon=True).start()

    def _stop_worker(self) -> None:
        self._stop_owned_processes()
        self.events.put(("stopped", None))

    def _stop_owned_processes(self) -> None:
        popen_pids: set[int] = set()
        for process in (self.router_process, self.native_process):
            if process:
                popen_pids.add(process.pid)
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
        self.router_process = None
        self.native_process = None
        with self.process_lock:
            records = dict(self.managed_processes)
        for record in records.values():
            pid = int(record.get("pid", 0))
            if pid in popen_pids:
                continue
            snapshot = process_snapshot(pid)
            if not same_process(record, snapshot):
                continue
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                timeout=8,
                creationflags=CREATE_NO_WINDOW,
                check=False,
            )
        with self.process_lock:
            self.managed_processes.clear()
            self._persist_managed_processes()
        for handle_name in ("router_log_handle", "native_log_handle"):
            handle = getattr(self, handle_name)
            if handle:
                try:
                    handle.close()
                except OSError:
                    pass
                setattr(self, handle_name, None)

    def restart(self) -> None:
        if self.external_stack and not self.managed_processes:
            return
        try:
            config = self._save_config()
        except (OSError, ValueError) as exc:
            messagebox.showerror(APP_NAME, str(exc))
            return
        self._set_buttons(running=True, busy=True)
        self._set_status("Restarting", "Recycling both services…", YELLOW)

        def worker() -> None:
            self._stop_owned_processes()
            self._start_worker(config)

        threading.Thread(target=worker, daemon=True).start()

    def _poll_tick(self) -> None:
        if not self.poll_inflight:
            self.poll_inflight = True
            threading.Thread(target=self._poll_worker, daemon=True).start()
        if not self.closing:
            self.root.after(1000, self._poll_tick)

    def _poll_worker(self) -> None:
        try:
            router_port = int(self.router_port_var.get())
            native_port = int(self.native_port_var.get())
            router_ok = endpoint_ready(f"http://127.0.0.1:{router_port}/health", 0.7)
            native_ok = endpoint_ready(f"http://127.0.0.1:{native_port}/health", 0.7)
            telemetry = get_json(f"http://127.0.0.1:{router_port}/telemetry", 0.9) if router_ok else None
            gpu = self._gpu_stats() if native_ok else None
            recoverable = self._discover_recoverable(native_ok=native_ok, router_ok=router_ok)
            self.events.put(("telemetry", {"router": router_ok, "native": native_ok, "data": telemetry, "gpu": gpu, "recoverable": recoverable}))
        except Exception:
            self.events.put(("telemetry", {"router": False, "native": False, "data": None, "gpu": None}))
        finally:
            self.poll_inflight = False

    @staticmethod
    def _gpu_stats() -> dict[str, float] | None:
        command = shutil.which("nvidia-smi")
        if not command:
            return None
        result = subprocess.run(
            [
                command,
                "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            timeout=2,
            creationflags=CREATE_NO_WINDOW,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        values = [float(item.strip()) for item in result.stdout.splitlines()[0].split(",")]
        return {"memory_used": values[0], "memory_total": values[1], "utilization": values[2], "temperature": values[3], "power": values[4]}

    def _drain_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "started":
                    if self.running_since is None:
                        self.running_since = time.monotonic()
                    self._set_status("Running", payload, ACCENT)
                    self._set_buttons(running=True, busy=False)
                elif kind == "recovered":
                    if self.running_since is None:
                        self.running_since = time.monotonic()
                    self.external_stack = False
                    self._set_status("Running", payload, ACCENT)
                    self._set_buttons(running=True, busy=False)
                elif kind == "stopped":
                    self.external_stack = False
                    self.running_since = None
                    self.uptime_var.set("Uptime —")
                    self.pid_var.set("PID —")
                    self._set_status("Stopped", "Services are offline.", MUTED)
                    self._set_buttons(running=False, busy=False)
                    if self.closing:
                        self.root.destroy()
                        return
                elif kind == "error":
                    self._set_status("Error", str(payload), RED)
                    self._set_buttons(running=False, busy=False)
                    messagebox.showerror(APP_NAME, str(payload))
                elif kind == "telemetry":
                    self._apply_telemetry(payload)
        except queue.Empty:
            pass
        self.root.after(150, self._drain_events)

    def _apply_telemetry(self, payload: dict[str, Any]) -> None:
        router_ok, native_ok = payload["router"], payload["native"]
        data = payload.get("data")
        recoverable = payload.get("recoverable") or {}
        transitional = {"Starting", "Restarting", "Stopping"}
        if self.status_var.get() not in transitional:
            with self.process_lock:
                changed = recoverable != self.managed_processes
                self.managed_processes = dict(recoverable)
                if changed:
                    self._persist_managed_processes()
        if router_ok and native_ok and self.status_var.get() not in transitional:
            if not self.native_process and not self.router_process and not self.managed_processes:
                self.external_stack = True
            elif self.managed_processes:
                self.external_stack = False
            if self.running_since is None:
                self.running_since = time.monotonic()
            self._set_status("Running", f"CodePilot endpoint · 127.0.0.1:{self.router_port_var.get()}", ACCENT)
            self._set_buttons(running=True, busy=False)
        elif not router_ok and not native_ok and self.status_var.get() not in transitional and self.status_var.get() != "Stopped":
            self.external_stack = False
            self.running_since = None
            self._set_status("Stopped", "Services are offline.", MUTED)
            self._set_buttons(running=False, busy=False)
        elif router_ok != native_ok and self.status_var.get() not in transitional:
            if self.managed_processes:
                self.external_stack = False
                self._set_status("Partial", "A previous Darwin service survived; Stop or Restart will recover it.", YELLOW)
                self._set_buttons(running=True, busy=False)
            else:
                self._set_status("Partial", "One external service is online; its process could not be safely identified.", YELLOW)
                self._set_buttons(running=True, busy=False)

        if router_ok or native_ok:
            if self.running_since is None:
                self.running_since = time.monotonic()
            self.uptime_var.set(f"Uptime {format_duration(time.monotonic() - self.running_since)}")
            pids = [str(record.get("pid")) for record in self.managed_processes.values() if record.get("pid")]
            self.pid_var.set(f"PID {' / '.join(pids)}" if pids else "PID external")
        else:
            self.uptime_var.set("Uptime —")
            self.pid_var.set("PID —")

        if data:
            self.card_values["requests"].set(format_tokens(data.get("requests", 0)))
            self.card_values["calls"].set(format_tokens(data.get("inference_calls", 0)))
            self.card_values["tokens"].set(format_tokens(data.get("completion_tokens", 0)))
            self.card_values["neg"].set(f"{100 * float(data.get('neg_activation_rate', 0)):.1f}%")
            self.card_values["routed"].set(f"{100 * float(data.get('routing_rate', 0)):.1f}%")
            self.card_values["throughput"].set(f"{float(data.get('tokens_per_second', 0)):.1f} tok/s")
            self._update_history(data.get("recent") or [])
        gpu = payload.get("gpu")
        if gpu:
            self.card_values["gpu_memory"].set(f"{gpu['memory_used'] / 1024:.1f}/{gpu['memory_total'] / 1024:.1f}G")
            self.card_values["gpu"].set(f"{gpu['utilization']:.0f}% · {gpu['temperature']:.0f}°")
        elif not native_ok:
            self.card_values["gpu_memory"].set("—")
            self.card_values["gpu"].set("—")

    def _update_history(self, records: list[dict[str, Any]]) -> None:
        self.history.delete(*self.history.get_children())
        for index, record in enumerate(records[:100]):
            timestamp = time.strftime("%H:%M:%S", time.localtime(float(record.get("timestamp", 0))))
            reasons = ", ".join(record.get("route_reasons") or []) or "single"
            guard_events = []
            for label, field in (
                ("dedup", "tool_duplicates_removed"),
                ("stalled", "tool_stalled_calls_blocked"),
                ("capped", "tool_parallel_overflow_removed"),
                ("recovered", "tool_recovery_inferences"),
            ):
                count = int(record.get(field, 0) or 0)
                if count:
                    guard_events.append(f"{label}:{count}")
            if guard_events:
                reasons = f"{reasons} · {' '.join(guard_events)}"
            self.history.insert(
                "",
                "end",
                values=(
                    timestamp,
                    record.get("model", ""),
                    record.get("inference_calls", 1),
                    format_tokens(record.get("completion_tokens", 0)),
                    f"{100 * float(record.get('neg_activation_rate', 0)):.1f}%",
                    f"{float(record.get('latency_seconds', 0)):.1f}s",
                    reasons,
                    record.get("finish_reason", ""),
                ),
                tags=("even" if index % 2 == 0 else "odd",),
            )

    def _refresh_logs(self) -> None:
        blocks: list[str] = []
        for title, path in (("NATIVE", self.native_log_path), ("ROUTER", self.router_log_path)):
            blocks.append(f"===== {title} =====\n{self._tail(path, 24000)}")
        self.log_text.delete("1.0", "end")
        self.log_text.insert("1.0", "\n\n".join(blocks))
        self.log_text.see("end")

    @staticmethod
    def _tail(path: Path, limit: int) -> str:
        try:
            with path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - limit), os.SEEK_SET)
                return handle.read().decode("utf-8", errors="replace")
        except OSError:
            return "No log output yet."

    def _open_logs(self) -> None:
        os.startfile(self.state_dir / "logs")

    def _open_codepilot_setup(self) -> None:
        port = self.router_port_var.get().strip() or "11435"
        context = self.context_var.get().strip() or str(DEFAULT_CONTEXT_SIZE)
        output = self.output_var.get().strip() or str(DEFAULT_MAX_TOKENS)
        key_value = "EMPTY" if not os.environ.get("DARWIN_API_KEY") else "Use the DARWIN_API_KEY value from your environment"
        anthropic_config = "\n".join(
            (
                "Provider type: Claude Code → third-party Anthropic",
                f"Base URL: http://127.0.0.1:{port}",
                f"API key: {key_value}",
                "Model: darwin-neg-agent",
                "Ensemble model: darwin-neg-agent20",
                f"Context window: {context}",
                f"Max output: {output}",
                f"Native KV cache: {KV_CACHE_TYPE_K}/{KV_CACHE_TYPE_V} · Flash Attention on",
            )
        )
        openai_config = "\n".join(
            (
                "Provider type: Custom API / OpenAI-compatible",
                f"Base URL: http://127.0.0.1:{port}/v1",
                f"API key: {key_value}",
                "Model: darwin-neg-agent",
                "Ensemble model: darwin-neg-agent20",
                f"Context window: {context}",
                f"Max output: {output}",
                f"Native KV cache: {KV_CACHE_TYPE_K}/{KV_CACHE_TYPE_V} · Flash Attention on",
                "Temperature: 0 (candidate diversity is routed internally)",
            )
        )

        dialog = tk.Toplevel(self.root)
        dialog.title("CodePilot setup · Darwin NEG")
        dialog.geometry("780x720")
        dialog.minsize(720, 650)
        dialog.configure(bg=BG)
        dialog.transient(self.root)

        heading = ttk.Frame(dialog, style="Header.TFrame", padding=(22, 18))
        heading.pack(fill="x")
        ttk.Label(heading, text="Connect Darwin NEG to CodePilot", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            heading,
            text="Use either provider profile below. The Anthropic profile is preferred for Claude Code skills and tool events.",
            style="HeaderMuted.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        content = ttk.Frame(dialog, padding=(22, 18, 22, 16))
        content.pack(fill="both", expand=True)
        steps_border = tk.Frame(content, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
        steps_border.pack(fill="x", pady=(0, 14))
        steps = ttk.Frame(steps_border, style="Panel.TFrame", padding=(15, 12))
        steps.pack(fill="x")
        ttk.Label(steps, text="In CodePilot", style="Section.TLabel").pack(anchor="w")
        ttk.Label(
            steps,
            text=(
                "1. Open Settings → Providers → Add provider.\n"
                "2. Choose Claude Code and enable its third-party Anthropic provider (recommended), or choose Custom API.\n"
                "3. Enter the matching values below, enable the model, and select it in a chat.\n"
                "4. Start this stack and wait for the green Running status before sending a request."
            ),
            style="SectionMuted.TLabel",
            justify="left",
        ).pack(anchor="w", pady=(7, 0))

        notebook = ttk.Notebook(content)
        notebook.pack(fill="both", expand=True)
        anthropic_panel = self._setup_panel(notebook, anthropic_config, "Copy Anthropic config")
        openai_panel = self._setup_panel(notebook, openai_config, "Copy OpenAI config")
        notebook.add(anthropic_panel, text="Anthropic (recommended)")
        notebook.add(openai_panel, text="OpenAI-compatible")

        note = ttk.Label(
            content,
            text=(
                "Web search is a CodePilot/MCP capability, not part of the model endpoint. "
                "Add your Tavily or other search MCP server separately in CodePilot."
            ),
            style="Muted.TLabel",
            wraplength=700,
            justify="left",
        )
        note.pack(fill="x", pady=(13, 0))
        ttk.Button(content, text="Close", command=dialog.destroy).pack(anchor="e", pady=(12, 0))
        dialog.bind("<Escape>", lambda _event: dialog.destroy())
        dialog.update_idletasks()
        width, height = dialog.winfo_width(), dialog.winfo_height()
        x = self.root.winfo_rootx() + max(0, (self.root.winfo_width() - width) // 2)
        y = self.root.winfo_rooty() + max(0, (self.root.winfo_height() - height) // 2)
        dialog.geometry(f"{width}x{height}+{x}+{y}")
        dialog.after(50, dialog.focus_force)

    def _setup_panel(self, parent: ttk.Notebook, config: str, button_text: str) -> ttk.Frame:
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=(14, 12))
        text = tk.Text(
            panel,
            height=10,
            bg=INPUT,
            fg=TEXT,
            selectbackground="#3b2f6d",
            selectforeground=TEXT,
            relief="flat",
            borderwidth=0,
            padx=14,
            pady=12,
            font=("Cascadia Mono", 10),
            wrap="none",
        )
        text.insert("1.0", config)
        text.configure(state="disabled")
        text.pack(fill="both", expand=True)
        ttk.Button(panel, text=button_text, command=lambda: self._copy_to_clipboard(config)).pack(anchor="e", pady=(10, 0))
        return panel

    def _copy_to_clipboard(self, value: str) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(value)
        self.root.update_idletasks()

    def _on_close(self) -> None:
        if self.native_process or self.router_process or self.managed_processes:
            if not messagebox.askyesno(APP_NAME, "Stop the Darwin services and exit?"):
                return
            self.closing = True
            self._set_status("Stopping", "Closing services before exit…", YELLOW)
            threading.Thread(target=self._stop_worker, daemon=True).start()
            return
        self.closing = True
        self.root.destroy()


def main() -> None:
    if "--router-child" in sys.argv:
        router_child()
        return
    root = tk.Tk()
    DarwinControlApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
