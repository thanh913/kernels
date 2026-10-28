#include "common.h"

// v7: replace padding with swizzling.

template <int BC>
__device__ inline int swz(int r, int col) {
  return r*BC + (((col/8) ^ (r%8)) * 8) + col%8;
}

template <int BR, int BC, int NT>
__device__ void async_load_tile_swz(nv_bfloat16 *dst, const nv_bfloat16 *src, int K) {
  #pragma unroll
  for (int it = 0; it*NT*8 < BR*BC; it++) {
    int idx = threadIdx.x*8 + it*NT*8;
    int r = idx/BC, c = idx%BC;
    cp_async16(&dst[swz<BC>(r, c)], &src[r*K + c]);
  }
}

// assume num warp = (BN/WM * BK/WK), num thread *= 32, *WK = BK*
template <int BM, int BN, int BK, int WM, int WN, int NT>
__global__ void __launch_bounds__(256, 1) matmul_v7_kernel(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *out, int M, int N, int K) {
  static_assert(BM % WM == 0);
  static_assert(BN % WN == 0);
  static_assert(NT == (BM/WM) * (BN/WN) * 32);


  auto load_bf16x8 = [](nv_bfloat16 *dst, const nv_bfloat16 *src) {
    *reinterpret_cast<uint4 *>(dst) = *reinterpret_cast<const uint4 *>(src);
  };
  auto store_bf16x2 = [](nv_bfloat16 *dst, float x, float y) {
    *reinterpret_cast<__nv_bfloat162 *>(dst) = __floats2bfloat162_rn(x, y);
  };

  A += blockIdx.x * BM * K;
  B += blockIdx.y * BN * K;
  out += blockIdx.x * BM * N + blockIdx.y * BN;

  int tid = threadIdx.x;
  int warpIdx = tid/32;
  int wtid = tid%32; 
  constexpr int TM = WM/MMA_M, TN = WN/MMA_N;
  // any access to shared mem / global mem must add warp offset;
  int warpR = (warpIdx / (BN/WN)) * WM;
  int warpC = (warpIdx % (BN/WN)) * WN;
  
  float acc_reg[TM][TN][4] = {};

  extern __shared__ __align__(16) uint8_t smem_raw[];

  static_assert(BK == 64, "swz assumes 8 chunks per row");
  auto As = reinterpret_cast<nv_bfloat16 (*)[BM*BK]>(smem_raw);
  auto Bs = reinterpret_cast<nv_bfloat16 (*)[BN*BK]>(As+3);
  
  // fetch first pair
  async_load_tile_swz<BM, BK, NT>(As[0], A, K);
  async_load_tile_swz<BN, BK, NT>(Bs[0], B, K);
  cp_async_commit();
  async_load_tile_swz<BM, BK, NT>(As[1], A+BK, K);
  async_load_tile_swz<BN, BK, NT>(Bs[1], B+BK, K);
  cp_async_commit();

  for (int kIdx = 0; kIdx*BK < K; kIdx++) { // each warp compute WM*WN subblocks in TM*TN fragments;
    cp_async_wait<1>();
    __syncthreads();
    if ((kIdx+2) * BK < K) {
      async_load_tile_swz<BM, BK, NT>(As[(kIdx+2)%3], A + (kIdx+2) * BK, K);
      async_load_tile_swz<BN, BK, NT>(Bs[(kIdx+2)%3], B + (kIdx+2) * BK, K);
    }
    cp_async_commit();

    for (int offK = 0; offK < BK; offK += MMA_K) { // WK = BK, so we use BK
      uint32_t a_reg[TM][4];
      uint32_t b_reg[TN][2];

      for (int i = 0; i < TM; i++) {
        int row = wtid%16, col = (wtid/16)*8;
        auto addr = &As[kIdx%3][swz<BK>(warpR + i*MMA_M + row, offK + col)];
        ldmatrix_x4(a_reg[i][0], a_reg[i][1], a_reg[i][2], a_reg[i][3], addr);
      }
      for (int i = 0; i < TN; i += 2) {
        int row = wtid%8 + (wtid/16)*8, col = (wtid/8 % 2) * 8;
        auto addr = &Bs[kIdx%3][swz<BK>(warpC + i*MMA_N + row, offK + col)];
        ldmatrix_x4(b_reg[i][0], b_reg[i][1], b_reg[i+1][0], b_reg[i+1][1], addr);
      }
      
      for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
          asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(acc_reg[i][j][0]), "+f"(acc_reg[i][j][1]), "+f"(acc_reg[i][j][2]), "+f"(acc_reg[i][j][3])
            : "r"(a_reg[i][0]), "r"(a_reg[i][1]), "r"(a_reg[i][2]), "r"(a_reg[i][3]), "r"(b_reg[j][0]), "r"(b_reg[j][1])
          );
        }
      }
    }
  }

  int g = wtid/4, t = wtid%4;
  for (int i = 0; i < TM; i++) {
    for (int j = 0; j < TN; j++) {
      int r = warpR + i*MMA_M + g;
      int c = warpC + j*MMA_N + t*2;
      store_bf16x2(&out[r*N + c], acc_reg[i][j][0], acc_reg[i][j][1]);
      store_bf16x2(&out[(r+8)*N + c], acc_reg[i][j][2], acc_reg[i][j][3]);
    }
  }
}

void matmul_v7(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
  constexpr int BM = 128, BN = 128, BK = 64, WM = 64, WN = 64;
  constexpr int NT = (BM / WM) * (BN / WN) * WARP_SIZE;

  if (M % BM != 0 || N % BN != 0 || K % BK != 0) {
    std::cerr << "matmul_v7: M, N, K must be multiples of " << BM << ", " << BN << ", " << BK << std::endl;
    exit(1);
  }

  constexpr int smem_bytes = 3 * (BM + BN) * BK * sizeof(nv_bfloat16);
  static_assert(smem_bytes <= 163 * 1024, "too much SMEM for A100");
  auto kernel = matmul_v7_kernel<BM, BN, BK, WM, WN, NT>;
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
  kernel<<<dim3(M / BM, N / BN), NT, smem_bytes>>>(A, B, C, M, N, K);
  CUDA_CHECK(cudaGetLastError());
}
