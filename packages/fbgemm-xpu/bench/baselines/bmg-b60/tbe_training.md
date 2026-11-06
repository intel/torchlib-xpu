# tbe_training.csv

Timings of the XPU TBE training lookups written by `python -m fbgemm_xpu.bench.tbe_training packages/fbgemm-xpu/bench/tbe/configs/small.json packages/fbgemm-xpu/bench/tbe/configs/medium.json packages/fbgemm-xpu/bench/tbe/configs/large.json --output tbe_training.csv`.
Columns: `config`, `family`, `direction`, `dtype`, `batch_size`, `num_tables`, `num_embeddings`, `embedding_dim`, `bag_size`, `total_lengths`, `time_us`, `bytes`, `gb_per_s`.

| Key | Value |
| --- | --- |
| command | python -m fbgemm_xpu.bench.tbe_training packages/fbgemm-xpu/bench/tbe/configs/small.json packages/fbgemm-xpu/bench/tbe/configs/medium.json packages/fbgemm-xpu/bench/tbe/configs/large.json --output tbe_training.csv |
| date_utc | 2026-11-06T00:46:18+00:00 |
| device | Intel(R) Arc(TM) Pro B60 Graphics |
| driver | 1.17.39395+13 |
| ze_affinity_mask | 0 |
| torch | 2.14.1+xpu |
| fbgemm_gpu | 1.9.0 |
| fbgemm_xpu | 0.9.0 |
| timing | median of XPU event times, 2 warm-ups, 36 MB cache flush (2x last-level cache, written, then a second buffer read so the cache is left clean) per iteration, then a torch.xpu._sleep spin sized from the fastest warm-up so the GPU does not wait for the host in the timed window |
| configs | packages/fbgemm-xpu/bench/tbe/configs/small.json packages/fbgemm-xpu/bench/tbe/configs/medium.json packages/fbgemm-xpu/bench/tbe/configs/large.json |
| config_sha256 | small.json: 35cc8cdbd23c344c, medium.json: ce3afc0e864c4072, large.json: b231ffd4ef513171 |
| workloads | small: batch_size=512, num_tables=4, num_embeddings=100000, embedding_dim=128, bag_size=20, Zipf q=0.1 s=0.1, 0 heavy hitters, indices int64, offsets int64; medium: batch_size=2048, num_tables=8, num_embeddings=1000000, embedding_dim=128, bag_size=20, Zipf q=0.1 s=0.1, 0 heavy hitters, indices int64, offsets int64; large: batch_size=4096, num_tables=16, num_embeddings=1000000, embedding_dim=128, bag_size=50, Zipf q=0.1 s=0.1, 0 heavy hitters, indices int64, offsets int64 |
| smoke | False |
| cpu_check | dense: a plain torch gather and index_add on CPU (FBGEMM's CPU dense kernel sum-pools even for no-bag lookups), split: FBGEMM's CPU kernel for the same operator, host weights |
| pooling | none (no-bag), unweighted |
| optimizer | split: rowwise Adagrad, learning_rate=0.1, eps=0.1, no stochastic rounding; dense: none |
| weights_init | uniform in [-0.0003, 0.0003] |
| grad_output | standard normal |
| backward | the backward of a graph built once; dense differentiates the weights with torch.autograd.grad; split runs torch.autograd.backward, which applies the optimiser update in place |
| dtypes | float32,float16 |
| output_dtype | float32 |
| families | dense = dense_embedding_codegen_lookup_function, split = split_embedding_codegen_lookup_rowwise_adagrad_function_pt2 |
| iters | 100 |
| seed | 0 |
| peak_device_memory_gib | 18.7 |
