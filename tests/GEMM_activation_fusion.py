import time
from typing import Optional
from enum import IntEnum
import torch
import triton
import triton.language as tl

class Activation(IntEnum):
    NONE       = 0
    RELU       = 1
    GELU       = 2
    SILU       = 3
    LEAKY_RELU = 4


def sgemm_get_configs():
    cfgs = []
    for BM in [64, 128]:
        for BN in [64, 128]:
            for BK in [32, 64, 128]:
                for num_warps in ([4, 8]):
                    for stages in [3, 4]:
                        for group_size_m in [8]:
                            cfgs.append(
                                triton.Config(
                                    kwargs={
                                        "BLOCK_M": BM,
                                        "BLOCK_N": BN,
                                        "BLOCK_K": BK,
                                        "GROUP_SIZE_M": group_size_m,
                                    },
                                    num_warps=num_warps,
                                    num_stages=stages,
                                )
                            )
    return cfgs


def _sgemm_launch_metadata(grid, kernel, meta_args):
    M, N, K = meta_args["M"], meta_args["N"], meta_args["K"]
    bm, bn, bk = meta_args["BLOCK_M"], meta_args["BLOCK_N"], meta_args["BLOCK_K"]
    name = f"sgemm_{bm}x{bn}x{bk}+[act={meta_args['ACTIVATION']}]"
    flops = 2.0 * M * N * K
    bytes_accessed = (M * K + K * N + M * N) * 4  # FP32 bytes
    return {"name": name, "flops32": flops, "bytes": bytes_accessed}

def act_none(x):
    return x

def act_relu(x):
    return tl.maximum(x, 0.0)

def act_gelu(x):
    return 0.5 * x * (1.0 + tl.erf(x * 0.70710678))

def act_silu(x):
    return x * tl.sigmoid(x)

def act_leaky_relu(x, slope):
    return tl.where(x > 0.0, x, x * slope)


@triton.autotune(configs=sgemm_get_configs(), key=["M", "N", "K"])
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
    GROUP_SIZE_M: tl.constexpr, EVEN_K: tl.constexpr, ACTIVATION: tl.constexpr, LEAKY_SLOPE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
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
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0.0)
            b = tl.load(b_ptrs, mask=offs_n[None, :] < N, other=0.0)
        else:
            mask_k = offs_k < K - k * BLOCK_K
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None] & (offs_n[None, :] < N), other=0.0)
        
        acc = tl.dot(a, b, acc)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    c = acc.to(tl.float32)  
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    if beta != 0.0:
        c_prev = tl.load(c_ptrs, mask=mask, other=0.0)
        result = alpha * c + beta * c_prev
    else:
        result = alpha * c

    fused = result
    # literal codes, matching your Python enum:
    #   0 = NONE, 1 = RELU, 2 = GELU, 3 = SILU, 4 = LEAKY_RELU 
    if ACTIVATION == 1:
        fused = tl.maximum(result, 0.0)
    elif ACTIVATION == 2:
        fused = 0.5 * result * (1.0 + tl.erf(result * 0.70710678))
    elif ACTIVATION == 3:
        fused = result * tl.sigmoid(result)
    elif ACTIVATION == 4:
        fused = tl.where(result > 0.0, result, result * LEAKY_SLOPE)

    tl.store(c_ptrs, fused, mask=mask)

def sgemm(a, b, *, alpha = 1.0, beta = 0.0, activation = Activation.NONE, leaky_slope = 0.01, out = None):
    assert a.dtype == b.dtype == torch.float32, "FP32 only"
    assert a.is_contiguous() and b.is_contiguous(), "Row‑major contiguous required"

    M, K = a.shape
    Kb, N = b.shape
    assert K == Kb, "Inner dimensions must match"

    if out is None:
        out = torch.empty((M, N), device=a.device, dtype=a.dtype)
    else:
        assert out.shape == (M, N)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    _sgemm_kernel[grid](
        a, b, out,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
                alpha, beta, ACTIVATION=activation, LEAKY_SLOPE=leaky_slope
    )
    return out

def check_activation(act, name):
    torch.manual_seed(0)
    M, K, N = 1024, 1024, 1024
    a = torch.randn((M, K), device="cuda", dtype=torch.float32)
    b = torch.randn((K, N), device="cuda", dtype=torch.float32)

    c_fused = sgemm(a, b, activation=act)

    c_ref = torch.mm(a, b)
    if act == Activation.GELU:
        c_ref = torch.nn.functional.gelu(c_ref)
    elif act == Activation.SILU:
        c_ref = torch.nn.functional.silu(c_ref)

    abs_err = (c_fused - c_ref).abs()
    max_abs = abs_err.max().item()

    normalization = torch.abs(a) @ torch.abs(b)
    rel_err = abs_err / (normalization + 1e-8)
    max_rel = rel_err.max().item()

    print(f"{name:6s}  max absolute error = {max_abs:.2e}")
    print(f"{name:6s}  max relative error = {max_rel:.2e}")

def bench_activation(act, name):
    M = N = K = 8192
    a = torch.randn((M, K), device="cuda", dtype=torch.float32)
    b = torch.randn((K, N), device="cuda", dtype=torch.float32)

    for _ in range(5):
        sgemm(a, b, activation=act); torch.cuda.synchronize()
        c = torch.mm(a, b)
        if act == Activation.GELU:
            torch.nn.functional.gelu(c)
        elif act == Activation.SILU:
            torch.nn.functional.silu(c)
        torch.cuda.synchronize()
        lin = torch.nn.Linear(K, N, bias=False).cuda()
        with torch.no_grad():
            lin.weight.copy_(b.T)
        out = lin(a)
        if act == Activation.GELU:
            torch.nn.functional.gelu(out)
        elif act == Activation.SILU:
            torch.nn.functional.silu(out)
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    sgemm(a, b, activation=act); torch.cuda.synchronize()
    t1 = time.perf_counter()
    fused_ms = (t1 - t0) * 1e3

    torch.cuda.synchronize()
    if act == Activation.GELU:
        t0 = time.perf_counter()
        torch.nn.functional.gelu(torch.mm(a, b))
    elif act == Activation.SILU:
        t0 = time.perf_counter()
        torch.nn.functional.silu(torch.mm(a, b))
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    mm_ms = (t1 - t0) * 1e3

    w = b.T.contiguous()
    if act == Activation.GELU:
        t0 = time.perf_counter()
        torch.nn.functional.gelu(torch.nn.functional.linear(a, w, bias=None))
    elif act == Activation.SILU:
        t0 = time.perf_counter()
        torch.nn.functional.silu(torch.nn.functional.linear(a, w, bias=None))
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    lin_ms = (t1 - t0) * 1e3

    print(f"{name:6s}  Fused       : {fused_ms:.2f} ms")
    print(f"{name:6s}  mm + act    : {mm_ms:.2f} ms")
    print(f"{name:6s}  Linear + act: {lin_ms:.2f} ms")
    print(f"{name:6s}  speed‑up vs mm  : {mm_ms / fused_ms:+.2f}×")
    print(f"{name:6s}  speed‑up vs lin : {lin_ms / fused_ms:+.2f}×\n")

if __name__ == "__main__":
    print("\n=== Correctness (abs & rel errors) ===\n")
    check_activation(Activation.GELU, "GELU")
    check_activation(Activation.SILU, "SiLU")

    print("\n=== Performance Benchmark ===\n")
    bench_activation(Activation.GELU, "GELU")
    bench_activation(Activation.SILU, "SiLU")
