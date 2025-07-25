#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <cuda_runtime.h>
#include <cute/algorithm/copy.hpp>
#include <cute/arch/copy_sm90_tma.cuh>
#include <cute/arch/mma_sm90.hpp>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>

namespace cg = cooperative_groups;
using half_t = cutlass::half_t;
using namespace cute;

#ifndef CUDA_CHECK
#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t e = (x);                                                       \
    if (e) {                                                                   \
      printf("CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e));    \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#endif

template <int TM = 128, int TN = 128, int TK = 32, int StageCount = 2>
struct DualGemmKernel {
  using Atom = SM90_16x16x16_F16F16F16F16_TN;
  using MMA = TiledMMA<Shape<Int<TM>, Int<TN>, Int<TK>>, Atom>;
  static constexpr int ThreadsCTA = 128;
  static constexpr int Vec = 8;

  template <class GTensor, class STensor>
  __device__ static void
  copy_tma_async(cuda::pipeline<cuda::thread_scope_threadblock> &pipe,
                 const GTensor &gSrc, const STensor &sDst) {
    cp_async_bulk_tensor_2d_global_to_shared(pipe, gSrc, sDst);
  }

  __global__ static void kernel(const half_t *__restrict__ A,
                                const half_t *__restrict__ B1,
                                const half_t *__restrict__ B2,
                                half_t *__restrict__ C1, half_t *__restrict__ C,
                                int M, int N, int K) {
    extern __shared__ half_t sm[];
    half_t *sA = sm;
    half_t *sB = sA + TM * TK * StageCount;

    auto SA = [&](int st) {
      return make_tensor(make_smem_ptr(sA + st * TM * TK),
                         Shape<Int<TM>, Int<TK>>{});
    };
    auto SB = [&](int st) {
      return make_tensor(make_smem_ptr(sB + st * TK * TN),
                         Shape<Int<TK>, Int<TN>>{});
    };

    cuda::pipeline<cuda::thread_scope_threadblock> pipe = cuda::make_pipeline();
    int warp_id = threadIdx.x >> 5;
    bool is_loader = warp_id == 0;

    int gemm_id = blockIdx.z; // 0 or 1
    const half_t *B = gemm_id ? B2 : B1;

    int bm0 = blockIdx.y * TM;
    int bn0 = blockIdx.x * TN;

    Tensor gA_full = make_tensor(make_gmem_ptr(A + bm0 * K), make_shape(TM, K),
                                 Layout<RowMajor>{});
    Tensor gB_full = make_tensor(make_gmem_ptr(B) + bn0, make_shape(K, TN),
                                 Layout<ColumnMajor>{});

    Tensor acc =
        partition_fragment<typename MMA::FragmentC>(Shape<Int<TM>, Int<TN>>{});
    fill(acc, half_t(0));

    int load_k = min(TK, K);
    if (is_loader) {
      copy_tma_async(pipe, gA_full(_, _(0, load_k)), SA(0)(_, _(0, load_k)));
      copy_tma_async(pipe, gB_full(_(0, load_k), _), SB(0)(_(0, load_k), _));
    }
    pipe.commit();
    pipe.wait_prior<1>();
    __syncthreads();

    int stage = 0;
    for (int k = TK; k < K; k += TK) {
      int nxt = stage ^ 1;
      int seg = min(TK, K - k);
      if (is_loader) {
        copy_tma_async(pipe, gA_full(_, _(k, seg)), SA(nxt)(_, _(0, seg)));
        copy_tma_async(pipe, gB_full(_(k, seg), _), SB(nxt)(_(0, seg), _));
      }
      pipe.commit();

      if (!is_loader) {
        MMA{}(acc, SA(stage), SB(stage), acc);
      }
      pipe.wait_prior<1>();
      __syncthreads();
      stage = nxt;
    }
    if (!is_loader) {
      MMA{}(acc, SA(stage), SB(stage), acc);
    }
    __syncthreads();

    cg::grid_group grid = cg::this_grid();
    grid.sync();

    auto vec_store = [&](half_t *dst, const half_t *src) {
      *reinterpret_cast<half8 *>(dst) = *reinterpret_cast<const half8 *>(src);
    };
    auto scalar_mul = [&](half_t &d, half_t x, half_t y) { d = x * y; };

    int rows = min(TM, M - bm0);
    int cols = min(TN, N - bn0);

    for (int idx = threadIdx.x * Vec; idx < rows * cols;
         idx += ThreadsCTA * Vec) {
      int r = idx / cols;
      int c = idx % cols;
      int gRow = bm0 + r;
      int gCol = bn0 + c;
      int rem = cols - c;
      int vec_elems = (rem >= Vec && (gCol % Vec == 0)) ? Vec : 1;

      half_t tmpC1[Vec];
      half_t tmpAcc[Vec];
      for (int i = 0; i < vec_elems; i++) {
        tmpAcc[i] = acc(r, c + i);
      }

      if (gemm_id == 0) { // write C1
        if (vec_elems == Vec) {
          vec_store(C1 + gRow * N + gCol, tmpAcc);
        } else {
          for (int i = 0; i < vec_elems; i++)
            C1[gRow * N + gCol + i] = tmpAcc[i];
        }
      } else { // GEMM‑1 fused multiply → C
        for (int i = 0; i < vec_elems; i++)
          tmpC1[i] = C1[gRow * N + gCol + i];
        for (int i = 0; i < vec_elems; i++)
          tmpAcc[i] *= tmpC1[i];
        if (vec_elems == Vec) {
          vec_store(C + gRow * N + gCol, tmpAcc);
        } else {
          for (int i = 0; i < vec_elems; i++)
            C[gRow * N + gCol + i] = tmpAcc[i];
        }
      }
    }
  }
};

int main(int argc, char **argv) {
  int M = 4096, N = 4096, K = 4096;
  if (argc == 4) {
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    K = atoi(argv[3]);
  }
  using Kernel = DualGemmKernel<>;
  size_t lenA = size_t(M) * K, lenB = size_t(N) * K, lenC = size_t(M) * N;
  half_t *dA, *dB1, *dB2, *dC1, *dC;
  CUDA_CHECK(cudaMalloc(&dA, lenA * sizeof(half_t)));
  CUDA_CHECK(cudaMalloc(&dB1, lenB * sizeof(half_t)));
  CUDA_CHECK(cudaMalloc(&dB2, lenB * sizeof(half_t)));
  CUDA_CHECK(cudaMalloc(&dC1, lenC * sizeof(half_t)));
  CUDA_CHECK(cudaMalloc(&dC, lenC * sizeof(half_t)));
  CUDA_CHECK(cudaMemset(dA, 0, lenA * sizeof(half_t)));
  CUDA_CHECK(cudaMemset(dB1, 0, lenB * sizeof(half_t)));
  CUDA_CHECK(cudaMemset(dB2, 0, lenB * sizeof(half_t)));

  dim3 grid((N + Kernel::TN - 1) / Kernel::TN,
            (M + Kernel::TM - 1) / Kernel::TM, 2);
  dim3 block(Kernel::ThreadsCTA);
  size_t smem = sizeof(half_t) *
                (Kernel::TM * Kernel::TK + Kernel::TK * Kernel::TN) *
                Kernel::StageCount;

  void *params[] = {(void *)&dA, (void *)&dB1, (void *)&dB2, (void *)&dC1,
                    (void *)&dC, &M,           &N,           &K};
  CUDA_CHECK(cudaFuncSetAttribute(
      Kernel::kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  CUDA_CHECK(cudaLaunchCooperativeKernel((void *)Kernel::kernel, grid, block,
                                         params, smem));
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("v7 dual GEMM finished\n");
  cudaFree(dA);
  cudaFree(dB1);
  cudaFree(dB2);
  cudaFree(dC1);
  cudaFree(dC);
  return 0;
}
