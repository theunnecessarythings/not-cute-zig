from __future__ import annotations

import subprocess
from pathlib import Path

import modal


APP_NAME = "not-cute-zig"
LOCAL_BINARY = Path(__file__).parent / "zig-out" / "bin" / "not-cute-zig"
REMOTE_BINARY = "/workspace/not-cute-zig"


image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .add_local_file(LOCAL_BINARY, remote_path=REMOTE_BINARY)
)

app = modal.App(APP_NAME, image=image)


@app.function(gpu="A10G", timeout=900)
def run_demos() -> None:
    subprocess.run(
        ["nvidia-smi"],
        check=True,
    )
    subprocess.run(
        ["chmod", "+x", REMOTE_BINARY],
        check=True,
    )
    subprocess.run(
        [REMOTE_BINARY],
        check=True,
    )


@app.local_entrypoint()
def main() -> None:
    run_demos.remote()
