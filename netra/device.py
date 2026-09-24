"""Hardware detection and ONNX Runtime sessions on the Snapdragon Hexagon NPU.

`open_npu_session` follows Qualcomm's ai-hub-apps helper (BSD-3-Clause,
qai_hub_apps_utils/onnxruntime_qnn.py) and supports both onnxruntime-qnn lines:
the >=2.0 plugin EP and the classic built-in QNNExecutionProvider.
"""

from __future__ import annotations

import functools
import platform
import subprocess
import sys


@functools.cache
def cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            return platform.processor()
    if sys.platform == "darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return platform.processor() or platform.machine()


@functools.cache
def is_snapdragon() -> bool:
    """True on Windows-on-Snapdragon (X Elite / X Plus / X2) with native ARM64 Python."""
    return sys.platform == "win32" and platform.machine().upper() == "ARM64" and (
        "snapdragon" in cpu_name().lower() or "qualcomm" in cpu_name().lower()
    )


def python_is_emulated() -> bool:
    """x64 Python under Prism emulation on an ARM64 PC cannot reach the NPU."""
    if sys.platform != "win32":
        return False
    import os

    native = os.environ.get("PROCESSOR_ARCHITEW6432") or os.environ.get("PROCESSOR_ARCHITECTURE", "")
    return platform.machine().upper() != "ARM64" and "ARM64" in (native.upper(), cpu_name().upper())


# EP options mirroring qai_hub_models' defaults for the QNN EP.
_QNN_OPTIONS: dict[str, str] = {
    "enable_htp_fp16_precision": "1",
    "htp_performance_mode": "burst",
    "htp_graph_finalization_optimization_mode": "3",
    "offload_graph_io_quantization": "1",
}


@functools.cache
def _register_plugin_ep() -> str:
    import onnxruntime as ort
    import onnxruntime_qnn as qnn

    ort.register_execution_provider_library(qnn.EP_NAME, qnn.get_library_path())
    return qnn.EP_NAME


def open_npu_session(model_path: str):
    """Open an ORT session on the Hexagon NPU (HTP backend).

    AI Hub `precompiled_qnn_onnx` exports are a single EPContext node wrapping a QNN
    context binary, so the whole graph lands on the NPU.
    """
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        import onnxruntime_qnn  # noqa: F401  (plugin EP, onnxruntime-qnn >= 2.0)
    except ImportError:
        return ort.InferenceSession(
            model_path,
            sess_options=so,
            providers=["QNNExecutionProvider"],
            provider_options=[{**_QNN_OPTIONS, "backend_path": "QnnHtp.dll"}],
        )
    ep_name = _register_plugin_ep()
    devices = [d for d in ort.get_ep_devices() if d.ep_name == ep_name]
    so.add_provider_for_devices(devices, _QNN_OPTIONS)
    return ort.InferenceSession(model_path, sess_options=so)


def npu_status() -> tuple[bool, str]:
    """(available, human-readable reason) for the QNN HTP path."""
    if not is_snapdragon():
        if python_is_emulated():
            return False, "x64 Python is running under emulation; install ARM64 Python to reach the NPU"
        return False, f"not a Snapdragon PC ({cpu_name()})"
    try:
        import onnxruntime as ort
    except ImportError:
        return False, "onnxruntime-qnn is not installed"
    try:
        ep = _register_plugin_ep()
        if any(d.ep_name == ep for d in ort.get_ep_devices()):
            return True, f"QNN plugin EP ({ep}) found an HTP device"
        return False, "QNN plugin EP registered but no HTP device was enumerated"
    except ImportError:
        if "QNNExecutionProvider" in ort.get_available_providers():
            return True, "classic QNNExecutionProvider available"
        return False, "QNN execution provider not available in this onnxruntime build"
    except Exception as e:  # driver / DLL problems surface here
        return False, f"QNN EP failed to load: {e}"
