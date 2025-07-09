import argparse, itertools, json, math, os, re, statistics, subprocess, time
import pathlib, pandas as pd, torch, triton, triton.language as tl
from ir.device import get_device_info

DEV = get_device_info()
print(json.dumps(DEV, indent=2))

DTYPE_MAP    = {"fp16": torch.float16, "float16": torch.float16,
                "fp32": torch.float32, "float32": torch.float32}
BYTES_PER_ELT = 2                 # for fp16
HW_SMEM_BYTES = DEV["smem_kib"] * 1024
CACHE_DIR     = pathlib.Path(os.getenv("TRITON_CACHE_DIR",
                            pathlib.Path.home() / ".triton" / "cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

@triton.jit
def kernel_gemm(A, B, C, M, N, K,
                SA0, SA1, SB0, SB1, SC0, SC1,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                BLOCK_K: tl.constexpr):
    pid    = tl.program_id(axis=0)
    grid_m = tl.cdiv(M, BLOCK_M)
    m      = pid // grid_m
    n      = pid %  grid_m

    offs_m = m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_k = offs_k < K

    a = tl.load(A + (offs_m[:, None] * SA0 + offs_k[None, :] * SA1),
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0)
    b = tl.load(B + (offs_k[:, None] * SB0 + offs_n[None, :] * SB1),
                mask=mask_k[:, None] & mask_n[None, :],
                other=0.0)
    acc = tl.zeros((BLOCK_M, BLOCK_N), a.dtype)
    acc += tl.dot(a, b)
    tl.store(C + (offs_m[:, None] * SC0 + offs_n[None, :] * SC1),
             acc,
             mask=mask_m[:, None] & mask_n[None, :])

_ptx_cache = {}

def _force_compile(tile, nw, ns, dtype_t):
    bm, bn, bk = tile
    dummy = torch.empty(1, 1, device="cuda", dtype=dtype_t)
    kernel_gemm[(1,)](
        dummy, dummy, dummy, 1, 1, 1,
        1, 1, 1, 1, 1, 1,
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
        num_warps=nw, num_stages=ns,
    )

def _new_ptx_path(before):
    after = set(CACHE_DIR.rglob("*.ptx"))
    diff  = after.difference(before)
    return max(diff, key=lambda p: p.stat().st_mtime) if diff else \
           max(after, key=lambda p: p.stat().st_mtime)

def regs_per_thread(tile, nw, ns, dtype_str="fp16"):
    key = (tile, nw, ns, dtype_str)
    if key in _ptx_cache:
        return _ptx_cache[key]

    dtype_t = DTYPE_MAP[dtype_str]
    before  = set(CACHE_DIR.rglob("*.ptx"))
    _force_compile(tile, nw, ns, dtype_t)
    ptx_path = _new_ptx_path(before)

    cc_flag = "sm_" + "".join(map(str, torch.cuda.get_device_capability()))
    out = subprocess.check_output(
            ["ptxas", "-v", "-arch=" + cc_flag, str(ptx_path), "-o", os.devnull],
            stderr=subprocess.STDOUT).decode("utf-8", "ignore")
    regs = int(re.search(r"([0-9]+)\s+registers", out).group(1))
    _ptx_cache[key] = regs
    return regs

def bench_tile(M, N, K, tile, nw, ns, dtype="fp16", iters=10):
    dtype_t = DTYPE_MAP[dtype]
    a = torch.randn(M, K, device="cuda", dtype=dtype_t)
    b = torch.randn(K, N, device="cuda", dtype=dtype_t)
    c = torch.empty(M, N, device="cuda", dtype=dtype_t)
    SA0, SA1 = a.stride(); SB0, SB1 = b.stride(); SC0, SC1 = c.stride()
    grid = (triton.cdiv(M, tile[0]) * triton.cdiv(N, tile[1]),)
    times = []
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        kernel_gemm[grid](
            a, b, c, M, N, K,
            SA0, SA1, SB0, SB1, SC0, SC1,
            BLOCK_M=tile[0], BLOCK_N=tile[1], BLOCK_K=tile[2],
            num_warps=nw, num_stages=ns,
        )
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)

BM_CAND        = [16, 32, 64, 128, 256, 512, 1024]
BN_CAND        = [16, 32, 64, 128, 256, 512, 1024]
BK_CAND        = [16, 32, 64, 128, 256, 512, 1024]
NUM_WARPS_CAND = [1, 2, 4, 8, 16, 32]
NUM_STAGE_CAND = [1, 2, 3]

def sweep(max_b, dtype):
    rows = []
    for bm, bn, bk, nw, ns in itertools.product(
            BM_CAND, BN_CAND, BK_CAND, NUM_WARPS_CAND, NUM_STAGE_CAND):

        if bm > max_b or bn > max_b:   # -- user cutoff
            continue

        smem_bytes = (bm * bk + bn * bk) * BYTES_PER_ELT * ns
        workset_mb = bm * bn * BYTES_PER_ELT / 1_048_576
        warp_m  = (bm + 31) // 32      # ceil(bm / 32)
        warp_n  = (bn + 31) // 32      # ceil(bn / 32)
        threads = warp_m * warp_n * nw * 32   # never zero
        if smem_bytes > HW_SMEM_BYTES:
            rows.append(dict(bm=bm, bn=bn, bk=bk, nw=nw, ns=ns,
                             smem=smem_bytes//1024, regs=math.nan,
                             threads=threads, blocks_sm=0, warps_sm=0,
                             workset_mb=round(workset_mb,2), ms=math.nan))
            print("s", end="", flush=True)
            continue

        try:
            regs = regs_per_thread((bm,bn,bk), nw, ns, dtype)
        except triton.runtime.errors.OutOfResources:
            print("r", end="", flush=True)
            rows.append(dict(bm=bm, bn=bn, bk=bk, nw=nw, ns=ns,
                             smem=smem_bytes//1024, regs=math.nan,
                             threads=threads, blocks_sm=0, warps_sm=0,
                             workset_mb=round(workset_mb,2), ms=math.nan))
            continue

        blocks_reg  = DEV["reg_per_sm"] // (regs * threads)
        blocks_smem = HW_SMEM_BYTES     // smem_bytes
        blocks_sm   = min(blocks_reg, blocks_smem, 32)
        warps_sm    = (threads // 32) * blocks_sm

        try:
            ms = bench_tile(1024, 1024, 1024, (bm,bn,bk), nw, ns, dtype)
            print(".", end="", flush=True)
        except triton.runtime.errors.OutOfResources:
            ms = math.nan
            print("x", end="", flush=True)

        rows.append(dict(
            bm=bm, bn=bn, bk=bk, nw=nw, ns=ns,
            smem=smem_bytes//1024, regs=regs,
            threads=threads, blocks_sm=blocks_sm, warps_sm=warps_sm,
            workset_mb=round(workset_mb,2), ms=ms,
        ))

    print()  # newline after progress dots
    fname = f"sweep_{DEV['cc']}_{dtype}.csv"
    pd.DataFrame(rows).to_csv(fname, index=False)
    print("➡  wrote", len(rows), "rows →", fname)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-b", type=int, default=256)
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    args = ap.parse_args()
    sweep(args.max_b, args.dtype)

