import argparse
import time
from enum import IntEnum
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import triton
import triton.language as tl
import triton.testing as ttesting
from torch.utils.cpp_extension import load


cublas_ext = load(
    name="cublas_ext",
    sources=["cublas_ext.cu"],
    extra_cuda_cflags=["-O3"], verbose=True
)


class Activation(IntEnum):
    NONE = 0
    RELU = 1
    GELU = 2
    SILU = 3

ACT_MAP = {
    "none": Activation.NONE,
    "relu": Activation.RELU,
    "gelu": Activation.GELU,
    "silu": Activation.SILU,
}

DTYPE_MAP = {
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def sgemm_get_configs():
    cfgs = []
    for BM in [32, 64, 128]:
        for BN in [32, 64, 128]:
            for BK in [32, 64, 128]:
                for num_warps in [1, 2, 4, 8, 16]:
                    for stages in [1, 2, 3, 4, 5]:
                        cfgs.append(
                            triton.Config(
                                kwargs={
                                    "BLOCK_M": BM,
                                    "BLOCK_N": BN,
                                    "BLOCK_K": BK,
                                    "GROUP_SIZE_M": 8,
                                },
                                num_warps=num_warps,
                                num_stages=stages,
                            )
                        )
    return cfgs


def _sgemm_launch_metadata(grid, _kernel, meta):
    M, N, K = meta["M"], meta["N"], meta["K"]
    bm, bn, bk = meta["BLOCK_M"], meta["BLOCK_N"], meta["BLOCK_K"]
    name = f"sgemm_{bm}x{bn}x{bk}_A{meta['ACTIVATION']}_D{meta['DTYPE']}"
    flops = 2.0 * M * N * K
    bytes_accessed = (M * K + K * N + M * N) * torch.finfo(meta["DTYPE_TORCH"]).bits / 8
    return {"name": name, "flops32": flops, "bytes": bytes_accessed}


def make_sgemm_kernel(dtype: torch.dtype):
    DTYPE = tl.float16 if dtype == torch.float16 else tl.float32

    @triton.autotune(configs=sgemm_get_configs(), key=["M", "N", "K", "DTYPE"])
    @triton.heuristics({"EVEN_K": lambda args: args["K"] % args["BLOCK_K"] == 0})
    @triton.jit(launch_metadata=_sgemm_launch_metadata)
    def _sgemm_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        alpha, beta,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr, EVEN_K: tl.constexpr,
        ACTIVATION: tl.constexpr, DTYPE: tl.constexpr, DTYPE_TORCH: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
        offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)

        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for _ in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0).to(DTYPE)
            b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0).to(DTYPE)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        result = alpha * acc

        if ACTIVATION == 1:
            result = tl.maximum(result, 0)
        elif ACTIVATION == 2:
            result = 0.5 * result * (1.0 + tl.erf(result * 0.70710678))
        elif ACTIVATION == 3:
            result = result * tl.sigmoid(result)

        c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, result.to(DTYPE), mask=mask)

    return _sgemm_kernel

_KERNEL_CACHE = {}

def sgemm(a: torch.Tensor, b: torch.Tensor, *, activation: Activation, alpha: float = 1.0):
    assert a.dtype == b.dtype, "Input dtypes must match"
    dtype = a.dtype
    key = dtype
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = make_sgemm_kernel(dtype)
    kernel = _KERNEL_CACHE[key]

    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "Inner dimensions must match"
    out = torch.empty((M, N), device=a.device, dtype=dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    kernel[grid](
        a, b, out,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        alpha, 0.0,
        ACTIVATION=activation,
        DTYPE=tl.float16 if dtype == torch.float16 else tl.float32,
        DTYPE_TORCH=dtype,
    )
    return out


def _torch_mm_act(a, b, act: Activation):
    c = torch.mm(a, b)
    if act == Activation.RELU:
        return torch.nn.functional.relu(c)
    elif act == Activation.GELU:
        return torch.nn.functional.gelu(c)
    elif act == Activation.SILU:
        return torch.nn.functional.silu(c)
    return c

def _torch_lin_act(a, w, act: Activation):
    out = torch.nn.functional.linear(a, w, bias=None)
    if act == Activation.RELU:
        return torch.nn.functional.relu(out)
    elif act == Activation.GELU:
        return torch.nn.functional.gelu(out)
    elif act == Activation.SILU:
        return torch.nn.functional.silu(out)
    return out


def bench(size, dtype, act, dumpdir = None) -> dict:
    M = N = K = size
    a = torch.randn((M, K), device="cuda", dtype=dtype)
    b = torch.randn((K, N), device="cuda", dtype=dtype)
    
    if dumpdir is not None:
        dumpdir.mkdir(parents=True, exist_ok=True)
        torch.save(a.cpu(), dumpdir / f"A_{size}.pt")  
        torch.save(b.cpu(), dumpdir / f"B_{size}.pt")

    sgemm(a, b, activation=act); torch.cuda.synchronize()

    fused_ms = ttesting.do_bench(lambda: sgemm(a, b, activation=act))
    mm_ms    = ttesting.do_bench(lambda: _torch_mm_act(a, b, act))
    cublas_ms = cublas_ext.gemm_act(a, b, act)
    w = b.T.contiguous()
    lin_ms   = ttesting.do_bench(lambda: _torch_lin_act(a, w, act))

    return {
        "size": size,
        "fused_ms": fused_ms,
        "mm_act_ms": mm_ms,
        "lin_act_ms": lin_ms,
        "cublas_act_ms": cublas_ms
    }


def make_plot(df: pd.DataFrame, outfile: Path, show: bool):
    plt.figure(figsize=(8, 5))
    plt.plot(df["size"], df["fused_ms"], "o-", label="Triton fused")
    plt.plot(df["size"], df["mm_act_ms"], "s--", label="torch.mm + act")
    plt.plot(df["size"], df["lin_act_ms"], "d--", label="Linear + act")
    plt.plot(df["size"], df["cublas_act_ms"], "d--", label="CUBLAS + act")
    plt.xlabel("Matrix size N (N×N)")
    plt.ylabel("Runtime [ms]")
    plt.title("GEMM + {} (dtype: {})".format(args.act.upper(), args.dtype.upper()))
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.grid(True, which="both", ls=":", lw=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile.with_suffix(".png"))
    if show:
        plt.show()
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Triton GEMM kernels with optional activation.")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16", help="Data type")
    parser.add_argument("--act", choices=["none", "relu", "gelu", "silu"], default="silu", help="Activation function")
    parser.add_argument("--sizes", nargs="*", type=int, default=[1024, 2048, 4096, 8192, 16384], help="Square matrix sizes to test")
    parser.add_argument("--outfile", default="benchmark", help="Base filename for CSV/PNG output")
    parser.add_argument("--show", action="store_true", help="Show plot interactively")
    parser.add_argument("--dumpdir", type=Path, help="Directory in which to save A_{N}.pt / B_{N}.pt")

    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    dtype = DTYPE_MAP[args.dtype]
    act = ACT_MAP[args.act]

    records = []
    for sz in args.sizes:
        rec = bench(sz, dtype, act)
        print(f"N={sz:6d}: fused={rec['fused_ms']:.3f} ms | mm+act={rec['mm_act_ms']:.3f} ms | lin+act={rec['lin_act_ms']:.3f} ms | cublas+act={rec['cublas_act_ms']:.3f} ms")
        records.append(rec)

    df = pd.DataFrame(records)
    csv_path = Path(f"{args.outfile}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV → {csv_path}")

    make_plot(df, Path(args.outfile), show=args.show)
    print(f"Saved plot → {args.outfile}.png")

