// pearl_turing_tc.cu — Turing (sm_75) int8 tensor-core fused Pearl kernel, v2.
//
// v2 = smem-staged k-slice pipeline (Ampere-style), register-prefetched
// double-buffering (Turing has no cp.async), same consensus transcript math
// as DP4A/sm80 paths (proven bit-exact: 4096^3 region gate on real T4).
//
// Layout of one inner step: load a 16-wide k window of the full
// BLOCK_M x BLOCK_N tile into smem ONCE (coalesced), then every warp reads
// its MMA fragments from smem (each fragment element reused ~ (BLOCK_N/16)*2
// times for A and (BLOCK_M/16)*2 for B — vs 4x duplicate global reads in v1).
//
// Fragment mapping (proven empirically bit-exact vs CPU ref on sm_86):
//   A row-major:  row=lane>>2, k=(lane&3)*4+byte
//   B col-major:  n=lane>>2, k=(lane&3)*4+byte
//   D:            row=lane>>2, col=(lane&3)*2+reg
#include <cuda_runtime.h>
#include <cstdint>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
__device__ __forceinline__ void mma_m8n8k16(int32_t& d0, int32_t& d1,
                                            uint32_t a, uint32_t b,
                                            int32_t c0, int32_t c1) {
  asm volatile(
      "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%4,%5};\n"
      : "=r"(d0), "=r"(d1) : "r"(a), "r"(b), "r"(c0), "r"(c1));
}

#endif  // __CUDA_ARCH__ >= 750

static constexpr int TUR_HT = 16;
static constexpr int TUR_HASH_ROT = 13;
static constexpr int TUR_TRANSCRIPT_LEN = 16;
static constexpr int KWIN = 16;  // k window width per mma step

// Fused kernel: one warp per 16x16 tile of a BLOCK_M x BLOCK_N block.
// blockDim: WARPS_M*WARPS_N*32; each warp = 1 tile (warp_m row, warp_n col).
//
// Pipeline: INNER_K k-windows per transcript step. Per window:
//   stage: BLOCK_M*KWIN + BLOCK_N*KWIN bytes -> smem (coalesced u32/x8/x16 copies)
//   compute: warps read frags from smem, 4 quadrant mmas each.
// Double-buffer smem (STAGES=2) with register prefetch of NEXT window while
// computing CURRENT: prefetch window w+1 to registers, __syncthreads, commit
// to smem, compute window w.
template <int BLOCK_M, int BLOCK_N, int WARPS_M, int WARPS_N, int MINB>
__global__ void __launch_bounds__(WARPS_M * WARPS_N * 32, MINB)
pearl_turing_fused_kernel_v2(const int8_t* __restrict__ A,
                             const int8_t* __restrict__ Bt,
                             int n, int k, int R,
                             uint32_t* __restrict__ transcript_buffer) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  static_assert(BLOCK_M == WARPS_M * 16, "BLOCK_M must equal WARPS_M*16");
  static_assert(BLOCK_N == WARPS_N * 32, "BLOCK_N must equal WARPS_N*32 (two 16-col tiles per warp)");

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int nthr = WARPS_M * WARPS_N * 32;
  const int lane = tid & 31;
  const int groupID = lane >> 2;
  const int tig = lane & 3;
  const int warp_m = warp / WARPS_N;
  const int warp_n = warp % WARPS_N;

  const int tiles_w = n / TUR_HT;
  const int blocks_n = n / BLOCK_N;  // grid columns
  const int block_row = blockIdx.x / blocks_n;
  const int block_col = blockIdx.x % blocks_n;
  const int row_base = block_row * BLOCK_M;
  const int col_base = block_col * BLOCK_N;
  const int warp_row0 = row_base + warp_m * TUR_HT;
  const int warp_col0 = col_base + warp_n * TUR_HT;

  constexpr int SMEM_A = BLOCK_M * KWIN;  // bytes per stage
  constexpr int SMEM_B = BLOCK_N * KWIN;
  constexpr int SMEM_STAGE = SMEM_A + SMEM_B;
  __shared__ __align__(16) int8_t smem_pipe[2 * SMEM_STAGE];
  __shared__ __align__(16) uint32_t sT[WARPS_M * WARPS_N * 2][TUR_TRANSCRIPT_LEN];

  if (lane == 0) {
    #pragma unroll
    for (int i = 0; i < TUR_TRANSCRIPT_LEN; ++i) {
      sT[warp * 2][i] = 0;
      sT[warp * 2 + 1][i] = 0;
    }
  }

  int32_t acc[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

  const int T = k / R;
  const int INNER_K = R / KWIN;  // windows per transcript step (256/16 = 16)

  // Register prefetch via strided u32 tiles: thread tid loads words
  // tid, tid+nthr, ... so consecutive threads read consecutive u32 (coalesced).
  constexpr int WORDS_A = SMEM_A / 4, WORDS_B = SMEM_B / 4;
  constexpr int CHUNKA = (WORDS_A + nthr - 1) / nthr;
  constexpr int CHUNKB = (WORDS_B + nthr - 1) / nthr;
  uint32_t preA[CHUNKA], preB[CHUNKB];

  auto prefetch = [&](int k_off) {
    #pragma unroll
    for (int ch = 0; ch < CHUNKA; ++ch) {
      const int w = tid + ch * nthr;          // word index into window slice A
      const int byte = w * 4;
      const int r = byte / KWIN, c = byte % KWIN;
      preA[ch] = (w < WORDS_A) ? __ldg((const uint32_t*)&A[(size_t)(row_base + r) * k + k_off + c]) : 0;
    }
    #pragma unroll
    for (int ch = 0; ch < CHUNKB; ++ch) {
      const int w = tid + ch * nthr;
      const int byte = w * 4;
      const int r = byte / KWIN, c = byte % KWIN;
      preB[ch] = (w < WORDS_B) ? __ldg((const uint32_t*)&Bt[(size_t)(col_base + r) * k + k_off + c]) : 0;
    }
  };

  auto commit_stage = [&](int8_t* dst) {
    uint32_t* dstA = (uint32_t*)dst;
    uint32_t* dstB = (uint32_t*)(dst + SMEM_A);
    #pragma unroll
    for (int ch = 0; ch < CHUNKA; ++ch) {
      const int w = tid + ch * nthr;
      if (w < WORDS_A) dstA[w] = preA[ch];
    }
    #pragma unroll
    for (int ch = 0; ch < CHUNKB; ++ch) {
      const int w = tid + ch * nthr;
      if (w < WORDS_B) dstB[w] = preB[ch];
    }
  };

  auto compute_window = [&](int stg) {
    const int8_t* smem_A = &smem_pipe[stg * SMEM_STAGE];
    const int8_t* smem_B = &smem_pipe[stg * SMEM_STAGE + SMEM_A];
    // A/B frags: lane=groupID rows of warp's tile at k-window columns tig*4.
    const int ar = (warp_m * TUR_HT) * KWIN;
    const uint32_t a_lo = *(const uint32_t*)&smem_A[(ar + groupID * KWIN) + tig * 4];
    const uint32_t a_hi = *(const uint32_t*)&smem_A[(ar + (8 + groupID) * KWIN) + tig * 4];
    #pragma unroll
    for (int wg = 0; wg < 2; ++wg) {          // two 16-col tile groups per warp
      const int tc = (warp_n * TUR_HT + wg * 128) * KWIN;
      const uint32_t b_lo = *(const uint32_t*)&smem_B[(tc + groupID * KWIN) + tig * 4];
      const uint32_t b_hi = *(const uint32_t*)&smem_B[(tc + (8 + groupID) * KWIN) + tig * 4];
      int32_t* ac = &acc[wg * 8];
#ifndef PEARL_NO_MMA
      mma_m8n8k16(ac[0], ac[1], a_lo, b_lo, ac[0], ac[1]);
      mma_m8n8k16(ac[2], ac[3], a_hi, b_lo, ac[2], ac[3]);
      mma_m8n8k16(ac[4], ac[5], a_lo, b_hi, ac[4], ac[5]);
      mma_m8n8k16(ac[6], ac[7], a_hi, b_hi, ac[6], ac[7]);
#else
      ac[0] += (int)(a_lo ^ b_lo);
      ac[2] += (int)(a_hi ^ b_lo);
      ac[4] += (int)(a_lo ^ b_hi);
      ac[6] += (int)(a_hi ^ b_hi);
#endif
    }
  };

  // Warmup: prefetch window 0 -> registers, commit to smem[0].
  prefetch(0);
  commit_stage(&smem_pipe[0]);
  __syncthreads();

  for (int t = 0; t < T; ++t) {
    for (int step = 0; step < INNER_K; ++step) {
      const int k_off = t * R + step * KWIN;
      const int cur = step % 2, nxt = 1 - cur;

      // Prefetch next window (global -> registers) while computing current.
      const bool have_next = (step + 1 < INNER_K) || (t + 1 < T);
      int next_k_off = k_off + KWIN;
      if (step + 1 >= INNER_K) next_k_off = (t + 1 < T) ? (t + 1) * R : 0;
      if (have_next) prefetch(next_k_off);

      compute_window(cur);

      if (have_next) {
        __syncthreads();
        commit_stage(&smem_pipe[nxt * SMEM_STAGE]);
        __syncthreads();
      }
    }

    // Transcript folds — one per walked tile column (8 accumulators each).
    uint32_t lx = 0, lx2 = 0;
    #pragma unroll
    for (int e = 0; e < 8; ++e) lx ^= (uint32_t)acc[e];
    #pragma unroll
    for (int e = 8; e < 16; ++e) lx2 ^= (uint32_t)acc[e];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      lx ^= __shfl_xor_sync(0xffffffffu, lx, off);
      lx2 ^= __shfl_xor_sync(0xffffffffu, lx2, off);
    }

    if (lane == 0) {
      const int idx = t % TUR_TRANSCRIPT_LEN;
      sT[warp * 2][idx] = ((sT[warp * 2][idx] << TUR_HASH_ROT) |
                       (sT[warp * 2][idx] >> (32 - TUR_HASH_ROT))) ^ lx;
      sT[warp * 2 + 1][idx] = ((sT[warp * 2 + 1][idx] << TUR_HASH_ROT) |
                       (sT[warp * 2 + 1][idx] >> (32 - TUR_HASH_ROT))) ^ lx2;
    }
    __syncthreads();
  }

  if (lane == 0) {
    #pragma unroll
    for (int wg = 0; wg < 2; ++wg) {
      const int tile_id = (warp_row0 / TUR_HT) * tiles_w + ((warp_col0 + wg * 128) / TUR_HT);
      uint32_t* tb = &transcript_buffer[(size_t)tile_id * TUR_TRANSCRIPT_LEN];
      #pragma unroll
      for (int i = 0; i < TUR_TRANSCRIPT_LEN; i += 4)
        *((int4*)&tb[i]) = *((int4*)&sT[warp * 2 + wg][i]);
    }
  }
#else
  (void)A; (void)Bt; (void)n; (void)k; (void)R; (void)transcript_buffer;
#endif  // __CUDA_ARCH__ >= 750
}

#ifndef PEARL_UNIT_TEST

// Host launcher: grid = (m/BLOCK_M) * ((n/BLOCK_N)) blocks of WARPS*32 threads.
// WARPS_M=8, WARPS_N=8 (64 warps=2048 threads is too big for 1 block? cap via
// template per arch. Default here 4x4 (16 warps, 512 threads).
cudaError_t launch_pearl_turing(const int8_t* A, const int8_t* Bt,
                                int m, int n, int k, int R,
                                uint32_t* transcript_buffer,
                                cudaStream_t stream) {
  static int s_major = -1;
  if (s_major < 0) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) return err;
    s_major = prop.major;
  }
  if (s_major < 7) return cudaErrorNotSupported;
  if (k % 16 != 0 || m % 16 != 0 || n % 256 != 0) return cudaErrorInvalidValue;

  constexpr int BM = 64, BN = 256, WM = 4, WN = 8;
  dim3 block(WM * WN * 32);
  dim3 grid((m / BM) * (n / BN));
  pearl_turing_fused_kernel_v2<BM, BN, WM, WN, 1>
      <<<grid, block, 0, stream>>>(A, Bt, n, k, R, transcript_buffer);
  return cudaGetLastError();
}

#endif  // !defined(PEARL_UNIT_TEST)
