import time
from typing import Optional

import torch
import triton
import triton.language as tl

def sgemm_get_configs():
    cfgs = []
    for BM in [16, 32, 64, 128, 256, 512]:
        for BN in [16, 32, 64, 128, 256, 512]:
            for BK in [16, 32, 64, 128, 256, 512]:
                for num_warps in ([1, 2, 4, 8, 16, 32]):
                    for stages in [1, 2, 3, 4, 5]:
                        for group_size_m in [1, 2, 4, 6, 8, 16, 32]:
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
    name = f"sgemm_{bm}x{bn}x{bk}  [M={M},N={N},K={K}]"
    flops = 2.0 * M * N * K
    bytes_accessed = (M * K + K * N + M * N) * 4  # FP32 bytes
    return {"name": name, "flops32": flops, "bytes": bytes_accessed}


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
    GROUP_SIZE_M: tl.constexpr, EVEN_K: tl.constexpr,
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
    
    tl.store(c_ptrs, result, mask=mask)

def sgemm(a, b, *, alpha = 1.0, beta = 0.0, out = None):
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
                alpha, beta,
    )
    return out


def _check():
    torch.manual_seed(0)
    a = torch.randn((16384, 16384), device="cuda", dtype=torch.float32)
    b = torch.randn((16384, 16384), device="cuda", dtype=torch.float32)
    c_triton = sgemm(a, b)
    c_ref = torch.mm(a, b)
    abs_err = (c_triton - c_ref).abs()
    normalization = torch.abs(a) @ torch.abs(b)
    relative_error = abs_err / (normalization + 1e-8)
    max_relative_error = relative_error.max().item()

    print(f"max relative error = {max_relative_error:.2e}")
    print(f"max absolute error = {abs_err.max().item():.2e}")


def _bench():
    M = N = K = 32768
    a = torch.randn((M, K), device="cuda", dtype=torch.float32)
    b = torch.randn((K, N), device="cuda", dtype=torch.float32)
    for _ in range(10):
        sgemm(a, b)
    torch.cuda.synchronize()
    t0 = time.perf_counter(); sgemm(a, b); torch.cuda.synchronize(); t1 = time.perf_counter()
    triton_ms = (t1 - t0) * 1e3
    for _ in range(10):
        torch.mm(a, b)
    t0 = time.perf_counter(); torch.mm(a, b); torch.cuda.synchronize(); t1 = time.perf_counter()
    blas_ms = (t1 - t0) * 1e3
    linear = torch.nn.Linear(K, N, bias=False).cuda().float()
    with torch.no_grad():
        linear.weight.copy_(b.T)
    for _ in range(10):
        linear(a)
    torch.cuda.synchronize()
    t0 = time.perf_counter(); linear(a); torch.cuda.synchronize(); t1 = time.perf_counter()
    linear_ms = (t1 - t0) * 1e3

    print(f"Triton  : {triton_ms:.2f} ms")
    print(f"cuBLAS  : {blas_ms:.2f} ms")
    print(f"nn.Linear: {linear_ms:.2f} ms")
    print(f"speed‑up (Triton vs cuBLAS): {blas_ms / triton_ms:+.2f}×")
    print(f"speed‑up (Triton vs Linear): {linear_ms / triton_ms:+.2f}×")


if __name__ == "__main__":
    _check()
    _bench()

