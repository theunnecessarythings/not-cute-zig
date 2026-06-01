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
def run_demos(args: list[str]) -> None:
    subprocess.run(
        ["nvidia-smi"],
        check=True,
    )
    subprocess.run(
        ["chmod", "+x", REMOTE_BINARY],
        check=True,
    )
    
    cmd = [REMOTE_BINARY] + args
    subprocess.run(
        cmd,
        check=True,
    )


@app.local_entrypoint()
def main(demo: str = "all") -> None:
    """
    Run the not-cute-zig demos on an A10G GPU.
    
    Args:
        demo: The specific demo to run. Options: "vector-add", "transpose", "ownership", "mma", "streams", "reduction", "batched-mma", "pipeline", "epilogue", "flash", "occupancy", "benchmark", "all"
    """
    run_demos.remote([demo])
