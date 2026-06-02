from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F


FLASH_CASES = (
    (2, 32, 16, False),
    (2, 32, 32, True),
    (2, 64, 16, False),
    (2, 64, 32, True),
    (2, 128, 32, True),
    (8, 128, 16, False),
    (8, 128, 32, True),
    (8, 256, 16, False),
    (8, 256, 32, True),
    (16, 256, 32, False),
    (16, 512, 32, True),
    (8, 256, 64, False),
    (8, 512, 64, True),
    (16, 512, 64, False),
    (8, 1024, 64, True),
)


@dataclass(frozen=True)
class Backend:
    name: str
    attr: str


BACKENDS = (
    Backend("torch_sdpa_flash", "FLASH_ATTENTION"),
    Backend("torch_sdpa_mem_efficient", "EFFICIENT_ATTENTION"),
    Backend("torch_sdpa_math", "MATH"),
)


def shape_name(batch_heads: int, seq_len: int, head_dim: int, causal: bool) -> str:
    return f"batch_heads={batch_heads},seq={seq_len},head={head_dim},causal={str(causal).lower()}"


def make_inputs(batch_heads: int, seq_len: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = batch_heads * seq_len * head_dim
    idx = torch.arange(n, device="cuda", dtype=torch.int64)
    q = (((idx * 3 + 1) % 17).to(torch.float32) * 0.125).to(torch.float16)
    k = (((idx * 5 + 2) % 17).to(torch.float32) * 0.0625).to(torch.float16)
    v = (((idx * 7 + 3) % 17).to(torch.float32) * 0.03125).to(torch.float16)
    return (
        q.reshape(batch_heads, 1, seq_len, head_dim),
        k.reshape(batch_heads, 1, seq_len, head_dim),
        v.reshape(batch_heads, 1, seq_len, head_dim),
    )


def sdpa_context(attr: str):
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return sdpa_kernel(getattr(SDPBackend, attr))
    except Exception:
        pass

    try:
        flags = {
            "enable_flash": attr == "FLASH_ATTENTION",
            "enable_mem_efficient": attr == "EFFICIENT_ATTENTION",
            "enable_math": attr == "MATH",
        }
        return torch.backends.cuda.sdp_kernel(**flags)
    except Exception:
        return contextlib.nullcontext()


def run_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=causal)


def time_backend(
    backend: Backend,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    warmups: int,
    iters: int,
) -> tuple[float, float, float, torch.Tensor]:
    with sdpa_context(backend.attr):
        for _ in range(warmups):
            out = run_sdpa(q, k, v, causal)
        torch.cuda.synchronize()

        samples: list[float] = []
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        for _ in range(iters):
            start.record()
            out = run_sdpa(q, k, v, causal)
            stop.record()
            stop.synchronize()
            samples.append(float(start.elapsed_time(stop)))
        torch.cuda.synchronize()

    return sum(samples) / len(samples), min(samples), max(samples), out


def emit(
    device_name: str,
    sm: tuple[int, int],
    sm_count: int,
    implementation: str,
    shape: str,
    correct: bool,
    mean_ms: float,
    min_ms: float,
    max_ms: float,
    bytes_processed: int,
    flops: int,
    warmups: int,
    iters: int,
    max_abs_error: float,
) -> None:
    gbps = bytes_processed / (mean_ms * 1.0e6) if bytes_processed else 0.0
    tflops = flops / (mean_ms * 1.0e9) if flops else 0.0
    print(
        json.dumps(
            {
                "device": device_name,
                "sm": f"{sm[0]}{sm[1]}",
                "sm_count": sm_count,
                "kernel": "flash_attention",
                "implementation": implementation,
                "shape": shape,
                "correct": correct,
                "warmups": warmups,
                "iterations": iters,
                "mean_ms": mean_ms,
                "min_ms": min_ms,
                "max_ms": max_ms,
                "gbps": gbps,
                "tflops": tflops,
                "max_abs_error": max_abs_error,
            },
            separators=(",", ":"),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for PyTorch baselines")

    props = torch.cuda.get_device_properties(0)
    torch.backends.cuda.matmul.allow_tf32 = False

    for batch_heads, seq_len, head_dim, causal in FLASH_CASES:
        q, k, v = make_inputs(batch_heads, seq_len, head_dim)
        shape = shape_name(batch_heads, seq_len, head_dim, causal)
        effective_seq = seq_len * (seq_len + 1) // 2 if causal else seq_len * seq_len
        flops = batch_heads * 4 * effective_seq * head_dim
        bytes_processed = batch_heads * seq_len * head_dim * (2 * 3 + 4)

        try:
            mean, min_ms, max_ms, ref = time_backend(BACKENDS[-1], q, k, v, causal, args.warmups, args.iters)
            emit(
                props.name,
                (props.major, props.minor),
                props.multi_processor_count,
                BACKENDS[-1].name,
                shape,
                True,
                mean,
                min_ms,
                max_ms,
                bytes_processed,
                flops,
                args.warmups,
                args.iters,
                0.0,
            )
        except Exception as exc:
            print(f"pytorch baseline {BACKENDS[-1].name} failed for {shape}: {exc}", file=sys.stderr)
            continue

        ref_f32 = ref.to(torch.float32)
        for backend in BACKENDS[:-1]:
            try:
                mean, min_ms, max_ms, out = time_backend(backend, q, k, v, causal, args.warmups, args.iters)
                max_abs_error = float((out.to(torch.float32) - ref_f32).abs().max().item())
                emit(
                    props.name,
                    (props.major, props.minor),
                    props.multi_processor_count,
                    backend.name,
                    shape,
                    max_abs_error <= 0.08,
                    mean,
                    min_ms,
                    max_ms,
                    bytes_processed,
                    flops,
                    args.warmups,
                    args.iters,
                    max_abs_error,
                )
            except Exception as exc:
                print(f"pytorch baseline {backend.name} failed for {shape}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
