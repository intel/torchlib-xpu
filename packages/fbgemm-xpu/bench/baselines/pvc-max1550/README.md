# jagged_tensor.csv

Timings of the XPU jagged operators written by `fbgemm_gpu/bench/jagged_tensor_benchmark.py jagged-sweep --output jagged_tensor.csv`.
Columns: `family`, `direction`, `dtype`, `batch_size`, `max_len`, `embedding_dim`, `total_lengths`, `time_us`, `bytes`, `gb_per_s`.

| Key | Value |
| --- | --- |
| command | fbgemm_gpu/bench/jagged_tensor_benchmark.py jagged-sweep --output jagged_tensor.csv |
| date_utc | 2026-10-02T00:06:53+00:00 |
| device | Intel(R) Data Center GPU Max 1550 |
| driver | 1.6.33578+77 |
| ze_affinity_mask | 0 |
| torch | 2.14.1+xpu |
| fbgemm_gpu | 1.9.0 |
| fbgemm_xpu | 0.8.0 |
| fbgemm_checkout | fe29065ccd593aa06a17e6953614a3a791568c9c |
| timing | median of XPU event times, 2 warm-ups, 384 MB cache flush (2x last-level cache) per iteration |
| lengths | uniform integers in [0, max_len], one draw per shape |
| batch_sizes | 32,128,512,2048 |
| max_lens | 16,64,200,256,512,1024 |
| embedding_dim | 128 |
| dtypes | float32,bfloat16,float16 |
| families | jagged_to_padded_dense,jagged_2d_to_dense,dense_to_jagged,jagged_dense_elementwise_add_jagged_output |
| iters | 100 |
| seed | 0 |
