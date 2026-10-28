#include "common.h"

// v4: pipelining

// assume num warp = (BN/WM * BK/WK), num thread *= 32, *WK = BK*
template <int BM, int BN, int BK, int WM, int WN, int NT>
__global__ void __launch_bounds__(128, 2) matmul_v4_kernel(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *out, int M, int N, int K) {
  static_assert(BM % WM == 0);
  static_assert(BN % WN == 0);
  static_assert(NT == (BM/WM) * (BN/WN) * 32);

  auto pair = [](const nv_bfloat16 *p) {
    return *reinterpret_cast<const uint32_t *>(p);
  };
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
  int g = wtid/4, t = wtid%4;
  constexpr int TM = WM/MMA_M, TN = WN/MMA_N;
  // any access to shared mem / global mem must add warp offset;
  int warpR = (warpIdx / (BN/WN)) * WM;
  int warpC = (warpIdx % (BN/WN)) * WN;
  
  float acc_reg[TM][TN][4] = {};

  extern __shared__ __align__(16) uint8_t smem_raw[];

  constexpr int LDS = BK+8;
  auto As = reinterpret_cast<nv_bfloat16 (*)[BM][LDS]>(smem_raw);
  auto Bs = reinterpret_cast<nv_bfloat16 (*)[BN][LDS]>(As+2);
  
  // fetch first pair
  async_load_tile<BM, BK, LDS, NT>(&As[0][0][0], A, K);
  async_load_tile<BN, BK, LDS, NT>(&Bs[0][0][0], B, K);
  cp_async_commit();

  for (int kIdx = 0; kIdx*BK < K; kIdx++) { // each warp compute WM*WN subblocks in TM*TN fragments;
    __syncthreads();
    if ((kIdx+1) * BK < K) {
      async_load_tile<BM, BK, LDS, NT>(&As[(kIdx+1)%2][0][0], A + (kIdx+1) * BK, K);
      async_load_tile<BN, BK, LDS, NT>(&Bs[(kIdx+1)%2][0][0], B + (kIdx+1) * BK, K);
    }
    cp_async_commit();
    cp_async_wait<1>();
    __syncthreads();

    for (int offK = 0; offK < BK; offK += MMA_K) { // WK = BK, so we use BK
      uint32_t a_reg[TM][4];
      uint32_t b_reg[TN][2];

      for (int i = 0; i < TM; i++) {
        a_reg[i][0] = pair(&As[kIdx%2][warpR + i*MMA_M + g][2*t + offK]);
        a_reg[i][1] = pair(&As[kIdx%2][warpR + i*MMA_M + g+8][2*t + offK]);
        a_reg[i][2] = pair(&As[kIdx%2][warpR + i*MMA_M + g][2*t + 8 + offK]);
        a_reg[i][3] = pair(&As[kIdx%2][warpR + i*MMA_M + g+8][2*t + 8 + offK]);
      }
      for (int i = 0; i < TN; i++) {
        b_reg[i][0] = pair(&Bs[kIdx%2][warpC + i*MMA_N + g][2*t + offK]); 
        b_reg[i][1] = pair(&Bs[kIdx%2][warpC + i*MMA_N + g][2*t + 8 + offK]);
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

  for (int i = 0; i < TM; i++) {
    for (int j = 0; j < TN; j++) {
      int r = warpR + i*MMA_M + g;
      int c = warpC + j*MMA_N + t*2;
      store_bf16x2(&out[r*N + c], acc_reg[i][j][0], acc_reg[i][j][1]);
      store_bf16x2(&out[(r+8)*N + c], acc_reg[i][j][2], acc_reg[i][j][3]);
    }
  }
}

void matmul_v4(const nv_bfloat16 *A, const nv_bfloat16 *B, nv_bfloat16 *C, int M, int N, int K) {
  constexpr int BM = 128, BN = 128, BK = 64, WM = 64, WN = 64;
  constexpr int NT = (BM / WM) * (BN / WN) * WARP_SIZE;

  if (M % BM != 0 || N % BN != 0 || K % BK != 0) {
    std::cerr << "matmul_v4: M, N, K must be multiples of " << BM << ", " << BN << ", " << BK << std::endl;
    exit(1);
  }

  constexpr int smem_bytes = 2 * (BM + BN) * (BK + 8) * sizeof(nv_bfloat16);
  static_assert(smem_bytes <= 163 * 1024, "too much SMEM for A100");
  auto kernel = matmul_v4_kernel<BM, BN, BK, WM, WN, NT>;
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
  kernel<<<dim3(M / BM, N / BN), NT, smem_bytes>>>(A, B, C, M, N, K);
  CUDA_CHECK(cudaGetLastError());
}
