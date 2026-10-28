
| kernel     |   latency (us) | Noise   |   TFLOPS | % SOL   |
|:-----------|---------------:|:--------|---------:|:--------|
| cublas     |         562.73 | 1.42%   |   244.24 | 78.28%  |
| matmul_v1  |        2674.28 | 1.53%   |    51.39 | 16.47%  |
| matmul_v2  |        1248.37 | 0.02%   |   110.09 | 35.29%  |
| matmul_v3  |         863.24 | 0.50%   |   159.21 | 51.03%  |
| matmul_v4  |         828.52 | 0.74%   |   165.89 | 53.17%  |
| matmul_v4b |         925.79 | 0.40%   |   148.46 | 47.58%  |
| matmul_v5  |         766.24 | 0.70%   |   179.37 | 57.49%  |

Todo:
- Clean up code + writing.

Notes:
- Tensor core usage via mma (HMMA):
  + sm80's BF16 one is m16n8k16. A thread handles the same fragment per 8x8 tile, think in terms of tile arrangement regardless of mXnXkX.
- ldmatrix (LDSM): non-blocking instruction, loads from SMEM -> Reg fragments of a 8x8 tile / thread.
  + Issues loads of up to 4 tiles (.x4) in 1 instruction, which makes this preferrable over normal LDS, as now the compiler can schedule eligible instructions faster, and weaving loads between HMMAs cost less.
- cp.async (LDGSTS): 
  + Does not consume registers.
  + LDGSTS.E.BYPASS.128 needs **128B** alignment for optimal perf! LDG.128 / STS.128 need only 16B.

