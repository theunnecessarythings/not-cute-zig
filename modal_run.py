from __future__ import annotations

import subprocess
import json
import shutil
from pathlib import Path

import modal


APP_NAME = "not-cute-zig"
LOCAL_BINARY = Path(__file__).parent / "zig-out" / "bin" / "not-cute-zig"
LOCAL_CUDA_BASELINES = Path(__file__).parent / "tools" / "cuda_baselines.cu"
LOCAL_PYTORCH_BASELINES = Path(__file__).parent / "tools" / "pytorch_baselines.py"
REMOTE_BINARY = "/workspace/not-cute-zig"
REMOTE_CUDA_BASELINES = "/workspace/cuda_baselines.cu"
REMOTE_CUDA_BASELINES_BIN = "/workspace/cuda_baselines"
REMOTE_PYTORCH_BASELINES = "/workspace/pytorch_baselines.py"


def _parse_jsonl(text: str) -> list[dict]:
    records: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            print(f"skipping malformed JSONL record: {exc}: {line}", flush=True)
    return records


def _print_compare_table(records: list[dict]) -> None:
    by_case: dict[tuple[str, str], dict[str, dict]] = {}
    for record in records:
        key = (record["kernel"], record["shape"])
        by_case.setdefault(key, {})[record["implementation"]] = record

    print("\n=== CUDA Toolkit vs not-cute ===")
    print(
        f"{'kernel':<18} {'shape':<40} {'baseline':<24} "
        f"{'not-cute ms':>12} {'base ms':>12} {'speedup':>9} {'ok':>5}"
    )
    for (kernel, shape), impls in sorted(by_case.items()):
        not_cute = impls.get("not_cute")
        if not not_cute:
            continue
        for implementation in (
            "cuda_wmma",
            "cuda_tiled",
            "cublas_materialized",
            "torch_sdpa_flash",
            "torch_sdpa_mem_efficient",
            "torch_sdpa_math",
            "cuda_cpp",
            "cublas",
            "cub",
        ):
            cuda = impls.get(implementation)
            if not cuda:
                continue
            speedup = cuda["mean_ms"] / not_cute["mean_ms"] if not_cute["mean_ms"] else 0
            ok = "yes" if not_cute["correct"] and cuda["correct"] else "no"
            print(
                f"{kernel:<18} {shape:<40} {implementation:<24} "
                f"{not_cute['mean_ms']:>12.6f} {cuda['mean_ms']:>12.6f} "
                f"{speedup:>8.2f}x {ok:>5}"
            )


def _run_captured(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result


image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install("torch", "numpy")
    .add_local_file(LOCAL_BINARY, remote_path=REMOTE_BINARY)
    .add_local_file(LOCAL_CUDA_BASELINES, remote_path=REMOTE_CUDA_BASELINES)
    .add_local_file(LOCAL_PYTORCH_BASELINES, remote_path=REMOTE_PYTORCH_BASELINES)
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

    if args and args[0] == "profile-flash":
        ncu = shutil.which("ncu")
        if ncu is None:
            raise RuntimeError("Nsight Compute CLI 'ncu' was not found in the CUDA image")
        profile_cmd = [
            ncu,
            "--kernel-name-base",
            "function",
            "--kernel-name",
            "regex:flash_attention_fwd_(opt|v2|h64|h64_causal)",
            "--launch-skip",
            "5",
            "--launch-count",
            "1",
            "--set",
            "speedOfLight",
            "--section",
            "Occupancy",
            "--section",
            "SchedulerStats",
            "--section",
            "WarpStateStats",
            "--section",
            "MemoryWorkloadAnalysis",
            REMOTE_BINARY,
            "profile-flash",
        ]
        result = subprocess.run(profile_cmd, text=True, capture_output=True)
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="")
        if result.returncode != 0:
            print(
                f"ncu failed with exit code {result.returncode}; running the focused flash benchmark without profiler",
                flush=True,
            )
            subprocess.run([REMOTE_BINARY, "profile-flash"], check=True)
        return

    if args and args[0] == "compare":
        subprocess.run(
            [
                "nvcc",
                "-O3",
                "-std=c++17",
                "-arch=sm_86",
                REMOTE_CUDA_BASELINES,
                "-lcublas",
                "-o",
                REMOTE_CUDA_BASELINES_BIN,
            ],
            check=True,
        )

    if args and args[0] == "compare":
        not_cute = _run_captured([REMOTE_BINARY] + args)
        cuda = _run_captured([REMOTE_CUDA_BASELINES_BIN])
        pytorch = _run_captured(["python", REMOTE_PYTORCH_BASELINES])
        _print_compare_table(_parse_jsonl(not_cute.stdout) + _parse_jsonl(cuda.stdout) + _parse_jsonl(pytorch.stdout))
    else:
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
        demo: The specific demo to run. Options: "vector-add", "transpose", "ownership", "mma", "streams", "reduction", "batched-mma", "pipeline", "epilogue", "flash", "occupancy", "benchmark", "compare", "profile-flash", "all"
    """
    run_demos.remote([demo])
