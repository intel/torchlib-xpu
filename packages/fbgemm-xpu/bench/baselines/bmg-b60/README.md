# jagged_tensor.csv

Timings of the XPU jagged operators written by `python -m fbgemm_xpu.bench.jagged_sweep --output jagged_tensor.csv`.
Columns: `family`, `direction`, `dtype`, `batch_size`, `max_len`, `embedding_dim`, `total_lengths`, `time_us`, `bytes`, `gb_per_s`.

| Key | Value |
| --- | --- |
| command | python -m fbgemm_xpu.bench.jagged_sweep --output jagged_tensor.csv |
| date_utc | 2026-10-03T00:10:21+00:00 |
| device | Intel(R) Arc(TM) Pro B60 Graphics |
| driver | 1.17.39395+13 |
| ze_affinity_mask | 0 |
| torch | 2.14.1+xpu |
| fbgemm_gpu | 1.9.0 |
| fbgemm_xpu | 0.8.0 |
| timing | median of XPU event times, 2 warm-ups, 36 MB cache flush (2x last-level cache) per iteration, then a torch.xpu._sleep spin so the GPU does not wait for the host in the timed window |
| lengths | uniform integers in [0, max_len], one draw per shape |
| batch_sizes | 32,128,512,2048 |
| max_lens | 16,64,200,256,512,1024 |
| embedding_dim | 128 |
| dtypes | float32,bfloat16,float16 |
| families | jagged_to_padded_dense,jagged_2d_to_dense,dense_to_jagged,jagged_dense_elementwise_add_jagged_output |
| iters | 100 |
| seed | 0 |
