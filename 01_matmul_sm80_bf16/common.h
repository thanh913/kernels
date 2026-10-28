// common functions, 

#pragma once
#include <cstdint>
#include <iostream>
#include <cuda_bf16.h>

#define CUDA_CHECK(call)                                                                                               \
  do {                                                                                                                 \
    cudaError_t err = call;                                                                                            \
    if (err != cudaSuccess) {                                                                                          \
      std::cerr << "CUDA error " << cudaGetErrorString(err) << " at " << __FILE__ ":" << __LINE__ << std::endl;        \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (0)

__host__ __device__ inline
constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

template <typename T, typename... Args>
void launch_kernel(T *kernel, int num_blocks, int block_size, int shm_size, Args... args) {
  if (shm_size > 48'000)
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size));

  kernel<<<num_blocks, block_size, shm_size>>>(args...);
  CUDA_CHECK(cudaGetLastError());
}


//-------------------------------------------------------------------------------
constexpr int WARP_SIZE = 32;

// mma.sync.m16n8k16 tile shape
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;

__device__ inline void cp_async16(void *smem_dst, const void *gmem_src) {
  uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));  // 32-bit shared address
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(gmem_src));
}
__device__ inline void cp_async_commit() {
  asm volatile("cp.async.commit_group;");
}
template <int N>
__device__ inline void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" :: "n"(N));
}

template <int BR, int BC, int LDS, int NT>
__device__ void async_load_tile(nv_bfloat16 *dst, const nv_bfloat16 *src, int K) {
  #pragma unroll
  for (int it = 0; it*NT*8 < BR*BC; it++) { // do this to unroll; it*NT*8 + threadIdx.x*8 will be < BR*BC
    int idx = threadIdx.x*8 + it*NT*8;
    int r = idx/BC, c = idx%BC;
    cp_async16(&dst[r*LDS + c], &src[r*K + c]);
  }
}

__device__ inline void ldmatrix_x4(uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3, const void *smem) {
  uint32_t a = __cvta_generic_to_shared(smem);  // PTX wants a 32-bit shared address
  asm volatile(
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" 
    : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a)
  );
}