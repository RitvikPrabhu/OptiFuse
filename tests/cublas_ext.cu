#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>    
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <stdexcept>

#define CHECK_CUDA(call)                                                     \
  do {                                                                        \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess)                                                   \
      throw std::runtime_error(std::string("CUDA error: ") +                  \
        cudaGetErrorString(err));                                             \
  } while (0)

#define CHECK_CUBLAS(call)                                                   \
  do {                                                                        \
    cublasStatus_t s = (call);                                                \
    if (s != CUBLAS_STATUS_SUCCESS)                                           \
      throw std::runtime_error(std::string("cuBLAS error: ") + std::to_string(s)); \
  } while (0)

__global__ void act_cast_kernel(const float* __restrict__ C_fp32_in,
                                float*       __restrict__ C_fp32_out,
                                __half*      __restrict__ C_fp16_out,
                                size_t size, int act, bool write_half) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) return;
  float x = C_fp32_in ? C_fp32_in[idx] : C_fp32_out[idx];
  // Activation
  if      (act == 1) x = fmaxf(x, 0.0f);
  else if (act == 2) x = 0.5f * x * (1.0f + erff(x * 0.70710678f));
  else if (act == 3) x = x / (1.0f + expf(-x));  // SiLU
  if (write_half) {
    C_fp16_out[idx] = __float2half_rn(x);
  } else {
    C_fp32_out[idx] = x;
  }
}

double gemm_act(at::Tensor A, at::Tensor B, int act_id) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
  TORCH_CHECK(A.dim()==2 && B.dim()==2, "A and B must be 2‑D");
  TORCH_CHECK(A.scalar_type()==B.scalar_type(), "A/B dtypes must match");
  int64_t M = A.size(0), K = A.size(1), N = B.size(1);
  bool is_half = (A.scalar_type() == at::kHalf);

  auto opts32 = at::TensorOptions().dtype(at::kFloat).device(A.device());
  at::Tensor C32 = at::empty({M, N}, opts32);
  at::Tensor C16;
  if (is_half) {
    auto opts16 = at::TensorOptions().dtype(at::kHalf).device(A.device());
    C16 = at::empty({M, N}, opts16);
  }

  // cuBLAS setup
  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));
#if defined(CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)
  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
#endif
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  CHECK_CUBLAS(cublasSetStream(handle, stream));

  const float alpha = 1.0f, beta = 0.0f;
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  // Warm‑up GEMM
  CHECK_CUBLAS(cublasGemmEx(handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      int(N), int(M), int(K),
      &alpha,
      B.data_ptr(), is_half?CUDA_R_16F:CUDA_R_32F, int(N),
      A.data_ptr(), is_half?CUDA_R_16F:CUDA_R_32F, int(K),
      &beta,
      C32.data_ptr(), CUDA_R_32F, int(N),
      CUDA_R_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // Record start
  CHECK_CUDA(cudaEventRecord(start, stream));

  // 1) GEMM → C32
  CHECK_CUBLAS(cublasGemmEx(handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      int(N), int(M), int(K),
      &alpha,
      B.data_ptr(), is_half?CUDA_R_16F:CUDA_R_32F, int(N),
      A.data_ptr(), is_half?CUDA_R_16F:CUDA_R_32F, int(K),
      &beta,
      C32.data_ptr(), CUDA_R_32F, int(N),
      CUDA_R_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  // 2) Activation 
  size_t size = size_t(M)*size_t(N);
  int blockSize = 0, minGridSize = 0;
  CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(
    &minGridSize,           
    &blockSize,              
    act_cast_kernel,         
    0,                       
    0                        
  ));

   int gridSize = int((size + blockSize - 1) / blockSize);
  // Grab raw pointers
  const float* C32_in  = C32.data_ptr<float>();
  float*       C32_out = C32.data_ptr<float>();
  __half*      C16_out = is_half
                         ? reinterpret_cast<__half*>(C16.data_ptr())
                         : nullptr;
  act_cast_kernel<<<gridSize,blockSize,0,stream>>>(
      C32_in, C32_out, C16_out, size, act_id, is_half);

  // Record stop & sync
  CHECK_CUDA(cudaEventRecord(stop, stream));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float ms;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

  cublasDestroy(handle);
  return double(ms);
}

PYBIND11_MODULE(cublas_ext, m) {
  m.def("gemm_act", &gemm_act,
        "gemm_act(A: Tensor, B: Tensor, act_id: int) -> float ms");
}

