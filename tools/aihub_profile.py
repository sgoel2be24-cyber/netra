"""Compile + profile Netra's speech models on real Snapdragon NPUs hosted by Qualcomm AI Hub.

No Snapdragon laptop needed: AI Hub runs the jobs on its device farm and reports on-device
NPU latency and memory. Logs land in docs/aihub/ for the README / deck.

One-time setup (free account at https://aihub.qualcomm.com):
    uvx qai-hub configure --api_token <YOUR_TOKEN>

Run (x64 or macOS Python; qai-hub-models does not install on Windows ARM64 Python):
    uv run --python 3.12 --with "qai-hub-models[whisper-base,whisper-small]" python tools/aihub_profile.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "aihub"
MODELS = ["whisper_base", "whisper_small"]
DEVICES = ["Snapdragon X Elite CRD", "Snapdragon X2 Elite CRD"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        for device in DEVICES:
            tag = f"{model}__{device.lower().replace(' ', '_')}"
            log = OUT / f"{tag}.log"
            cmd = [
                "qai-hub-models", "export", model,
                "--target-runtime", "precompiled_qnn_onnx",
                "--device", device,
                "--output-dir", str(OUT / tag),
            ]
            print(f"\n=== {model} on {device}\n$ {' '.join(cmd)}", flush=True)
            with log.open("w", encoding="utf-8") as f:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in proc.stdout:  # tee to console and log
                    sys.stdout.write(line)
                    f.write(line)
                proc.wait()
            print(f"--> {log} (exit {proc.returncode})")


if __name__ == "__main__":
    main()
