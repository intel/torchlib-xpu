# Benchmark baselines

Committed performance baselines for the fbgemm-xpu benchmarks:

- [Jagged operators](#jagged-operators), timed by `jagged-sweep`;
- [TBE training lookups](#tbe-training-lookups), timed by `tbe-training`.

Each CSV is a plain table, so GitHub renders it. The command, date, device,
driver, versions and benchmark definition of the run that produced it are in
the Markdown file of the same name beside it, as the benchmark writes them.
Compare two runs only if those match.

## Jagged operators

Committed performance baselines for the XPU jagged operators, forward and
backward:

- `jagged_to_padded_dense`
- `jagged_2d_to_dense`
- `dense_to_jagged`
- `jagged_dense_elementwise_add_jagged_output`

| Hardware | Timings | Run notes |
| --- | --- | --- |
| Intel® Arc™ Pro B60 Graphics (BMG) | [bmg-b60/jagged_tensor.csv](bmg-b60/jagged_tensor.csv) | [bmg-b60/jagged_tensor.md](bmg-b60/jagged_tensor.md) |

### Reproduce

```
uv pip install -e "$TORCHLIB_XPU_PATH/packages/fbgemm-xpu[test]" \
  --index https://download.pytorch.org/whl/xpu
ZE_AFFINITY_MASK=0 python -m fbgemm_xpu.bench.jagged_sweep \
  --output jagged_tensor.csv
```

Run on an idle machine. `ZE_AFFINITY_MASK=0` pins the run to one device. In a
container with a CPU limit, also set `OMP_NUM_THREADS` to that limit: PyTorch
otherwise starts a thread per host core, and the container is throttled.

The sweep writes two files: `jagged_tensor.csv` with the timings, and
`jagged_tensor.md` with the run's command, date, device, driver, versions and
sweep definition.

The BMG baseline was recorded on one B60 in a Kubernetes pod limited to 4
CPUs, with `OMP_NUM_THREADS=4`. A second run on the same pod agreed with it
within the noise given under Reading the numbers, and so did an earlier
recording taken with upstream's zero fill in place of the flush described
under Timing below: on a B60 the two flushes time alike.

### What the sweep measures

- **Shapes:** batch size B ∈ {32, 128, 512, 2048} × max sequence length
  max_len ∈ {16, 64, 200, 256, 512, 1024}. The embedding dimension D is fixed
  at 128, the DLRM v3 XPU `hstu_transducer_embedding_dim`.
- **Lengths:** each of the B lengths is drawn uniformly from [0, max_len]
  with seed 0, one draw per shape. The `total_lengths` column records the
  resulting number of jagged rows.
- **Dtypes:** float32, bfloat16 and float16.
- **Correctness first:** for each shape, the forward output and the input
  gradients are compared against the CPU implementation with
  `torch.testing.assert_close` before anything is timed. If any shape
  mismatches, the run fails and no CSV is written.
- **Timing:** the median of 100 iterations, timed with XPU events, after 2
  warm-ups. Before each iteration, outside the timed region, a buffer twice
  the size of the device's last-level cache, holding random data, is negated
  in place, and then a second buffer of the same size is read, so operands
  start in device memory rather than L2 and L2 holds clean lines. Without the
  read, the flush leaves L2 dirty and the timed call pays for the write-back;
  the docstring of `benchmark_torch_function` in
  [`bench_utils.py`](../../src/fbgemm_xpu/bench/bench_utils.py) explains the
  flush. Events time the GPU timeline, so the GPU must not wait for the host
  inside the timed window.
  GPU work lasting twice the host submission time is therefore queued ahead
  of each start event: a `torch.xpu._sleep` spin where the driver supports
  it, otherwise further passes of the flush. An iteration that may still have
  waited is repeated with more work, and a warning is logged if it persists.
  The forward times the operator call. The backward times
  `torch.autograd.grad` on a graph built once. All four backwards run XPU
  kernels:
  - `jagged_to_padded_dense` and `jagged_2d_to_dense` use
    `jagged_to_padded_dense_backward`;
  - `dense_to_jagged` and `jagged_dense_elementwise_add_jagged_output` use
    `jagged_to_padded_dense_forward`.

The notes beside each CSV record the command, date, device, driver, versions
and the sweep definition. Compare two runs only if the device, driver,
versions and sweep definition match.

#### Bandwidth

`gb_per_s` is effective bandwidth, the logical bytes read plus written by one
call divided by its time. A jagged tensor counts its values and offsets. A
dense tensor counts the full B × max_len × D, padding included.

| Family | Forward bytes | Backward bytes |
| --- | --- | --- |
| `jagged_to_padded_dense`, `jagged_2d_to_dense` | values + offsets + dense out | the same |
| `dense_to_jagged` | dense + values out + offsets | the same |
| `jagged_dense_elementwise_add_jagged_output` | offsets + 3 × values | values grad + offsets + dense grad |

The forward formulas follow the upstream `device` sub-benchmarks. Kernels that
read a dense input only touch its valid rows, which is about half of them with
these lengths. `dense_to_jagged` can therefore report more than a plain copy
achieves.

#### Reading the numbers

- Small shapes are latency-bound. Below a few MB of traffic the time floors,
  at about 5 µs on BMG, whatever the size, and their GB/s does not reflect
  bandwidth. Use them to catch kernel latency regressions.
- Shapes whose operands fit in the last-level cache (18 MB on B60) start from
  device memory, as the flush evicts them.
- Run to run, BMG times varied by a median of 0.1%, a 95th percentile of
  1.0% and at most 4.1%, on shapes under 60 µs; rows over 60 µs stayed
  within 3%. Treat changes smaller than that as noise.
- For reference, a large (2 GB) `copy_` streams at about 400 GB/s on a B60
  with this helper, against the 456 GB/s its memory is rated at.
- BMG compresses memory traffic: the same `copy_` of zeros reports about
  1.8 TB/s. The sweep's inputs are random, but the zero padding of dense
  outputs can compress, which is one reason some BMG rows exceed the rated
  bandwidth.

### Not benchmarked on XPU

The patched script runs only on XPU. The cases below log
`SKIPPED ... on xpu: <reason>` and return.

| Case | Reason |
| --- | --- |
| `batched_dense_vec_jagged_2d_mul` | `batched_dense_vec_jagged_2d_mul_forward` has no XPU implementation |
| `jagged_1d_to_truncated_values` | no XPU implementation; CPU-only case upstream |
| `masked_select_jagged_1d` | no XPU implementation; CPU-only case upstream |
| `keyed_jagged_index_select_dim1` | no XPU implementation, and its backward needs `jagged_index_add_2d_forward`, which has none |
| `jagged_slice_cpu` | CPU-only by design; `jagged_slice_forward` has no XPU implementation |
| `permute_pooled_embs_bench` | `permute_pooled_embs` and `permute_multi_embedding` have no XPU implementation |
| `jagged_acc_weights_and_counts_bench` | no XPU implementation |

The `device` case runs `jagged_2d_to_dense`, `dense_to_jagged` (2D) and
`jagged_dense_elementwise_add_jagged_output`, with their NestedTensor
references. It skips these sub-benchmarks:

- `dense_dense_add`: no XPU forward;
- `1d_to_dense`: not an XPU-registered operator, and the kernel it reaches is
  already timed by `jagged-sweep`;
- `dense_to_jagged_1d`: the D = 1 form of `dense_to_jagged`;
- the `jagged_dense_elementwise_mul` half of the elementwise sub-benchmark:
  no XPU implementation.

Trace export (`--export-trace`) is not supported on XPU.

## TBE training lookups

Committed performance baselines for the XPU table batched embedding (TBE)
training lookups, forward and backward, on the no-bag, unweighted path:

- `dense`: `dense_embedding_codegen_lookup_function`. Its backward returns the
  gradient of the whole embedding table.
- `split`: `split_embedding_codegen_lookup_rowwise_adagrad_function_pt2`. Its
  backward also applies the rowwise Adagrad update to the weights and the
  momentum in place.

| Hardware | Timings | Run notes |
| --- | --- | --- |
| Intel® Arc™ Pro B60 Graphics (BMG) | [bmg-b60/tbe_training.csv](bmg-b60/tbe_training.csv) | [bmg-b60/tbe_training.md](bmg-b60/tbe_training.md) |

### Reproduce

From the root of the repository:

```
uv pip install -e "packages/fbgemm-xpu[test]" \
  --index https://download.pytorch.org/whl/xpu
ZE_AFFINITY_MASK=0 python -m fbgemm_xpu.bench.tbe_training \
  packages/fbgemm-xpu/bench/tbe/configs/{small,medium,large,skewed}.json \
  --output tbe_training.csv
```

The advice for the jagged sweep applies: run on an idle machine, pin one
device with `ZE_AFFINITY_MASK=0`, and in a container with a CPU limit set
`OMP_NUM_THREADS` to that limit. Like the jagged sweep, the benchmark writes
`tbe_training.csv` with the timings and `tbe_training.md` with the run notes,
in the same format. The notes also record the workloads, a SHA-256 prefix of
each config file and the peak device memory. Select lookups with `--families`
(`dense`, `split`) and weight dtypes with `--dtypes` (`float32`, `float16`).

A run takes about 7 minutes on a B60 with 4 CPUs, most of it in the CPU
checks and in starting the timing processes. The `large` config needs
18.7 GiB of device memory, for the FP32
dense backward, and about 30 GiB of host memory for the CPU check. On a
device with less memory, pass only the configs that fit, and say so in the
notes you commit.

The weights and momentum are copied to the device in 128 MiB slices. On the
B60 nodes used for the baseline (compute-runtime 26.31, host kernel
7.0.0-38), a single pageable copy of the 8.2 GB `large` table failed
intermittently, with `UR_RESULT_ERROR_DEVICE_LOST`, or with
`UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` after which later allocations of the
table failed in the same process, also with a plain `tensor.to("xpu")` and
no benchmark code. Copies in 128 MiB and 1 GiB slices never failed.
[intel/torch-xpu-ops#5867](https://github.com/intel/torch-xpu-ops/issues/5867)
reports the same failure with the same driver, and a comment in
[intel/compute-runtime#948](https://github.com/intel/compute-runtime/issues/948)
traces it to copies of 512 MiB or more being mapped for the copy engine as
an external host pointer.

The BMG baseline was recorded on one B60 in a Kubernetes pod limited to 4
CPUs and 32 GiB of memory, with `OMP_NUM_THREADS=4`. Three more runs of
each config with the same timing code on the same pod agreed with it within
the noise given under Reading the numbers below.

### What the benchmark measures

- **Workloads:** each config in [`../tbe/configs`](../tbe/configs) is a
  `TBEDataConfig`, loaded with `TBEDataConfig.from_json`. Indices and offsets
  come from upstream's `generate_requests`, seeded with `--seed`, and are
  moved to XPU. All four use D = 128, the DLRM v3 XPU
  `hstu_transducer_embedding_dim`, a fixed bag length L, and int64 indices
  and offsets. `small`, `medium` and `large` use upstream's default
  near-uniform Zipf distribution, q = 0.1 and s = 0.1, under which a row is
  read at most 4, 6 and 11 times. Real indices are skewed: at GR's default
  MovieLens batch of 128, the sex feature puts about 64 samples on each of
  its two rows. `skewed` therefore uses q = 1 and s = 1.05. Its most read row
  is read 10,339 times, 964 rows 32 to 1,024 times and 31 rows more often,
  so it reaches the paths of the XPU backward for repeated rows: a CTA per
  row from 32 lookups, and several work-groups with atomics above 1,024.

  | Config | `num_tables` (T) | `num_embeddings` (E) | `batch_size` (B) | `bag_size` (L) | `total_lengths` (lookups) | FP32 weights |
  | --- | --- | --- | --- | --- | --- | --- |
  | `small` | 4 | 100,000 | 512 | 20 | 40,960 | 205 MB |
  | `medium` | 8 | 1,000,000 | 2,048 | 20 | 327,680 | 4.1 GB |
  | `large` | 16 | 1,000,000 | 4,096 | 50 | 3,276,800 | 8.2 GB |
  | `skewed` | 4 | 1,000,000 | 2,048 | 50 | 409,600 | 2.0 GB |

  Every config sets `use_cpu` to true: otherwise `generate_requests` places
  its tensors on CUDA or MTIA. The benchmark rejects configs the XPU kernels
  cannot run: weighted, mixed dimensions or variable batch sizes (VBE). It
  also rejects per-table `Es`, `embedding_specs` and `feature_table_map`,
  which this harness does not lay out. Variable bag lengths are accepted,
  either drawn around L with `sigma_L` or set per table with `Ls`, as in
  DLRM-v3's mix of a jagged history and single-ID features: the kernels read
  the length of each bag from the offsets. `bag_size` is then the mean bag
  length.
- **Precisions:** FP32 and FP16 weights, FP32 output.
- **Optimiser and initialisation:** the optimiser is upstream
  `tbe_training_benchmark.py`'s default: rowwise Adagrad with a learning rate
  of 0.1 and eps of 0.1, and no stochastic rounding. Weights are drawn
  uniformly from [-0.0003, 0.0003] for every precision, the range upstream
  uses for INT8 weights. Upstream, by contrast, keeps each FP32 and FP16
  module's own initialisation: zeros for the split module and standard
  normal for the dense one. The output gradient is standard normal.
- **Correctness first:** for each config, precision and lookup, one forward
  and one backward on XPU are compared against CPU with
  `torch.testing.assert_close` before anything is timed. Both start from the
  same initial state. The split lookup is compared with FBGEMM's own CPU
  kernel for the operator, on the output and the updated weights and
  momentum. The dense lookup is compared with a plain torch gather and a
  float64 `index_add_`, on the output and the gradient of the whole table,
  because FBGEMM's CPU dense kernel sum-pools whatever the pooling mode. A
  gradient element sums the output gradients of every lookup of its row,
  and the rounding of a float32 sum of n terms grows like √n, so the
  gradient's absolute tolerance is the default 1e-5 times the square root
  of the most read row's lookups: 1.0e-3 for `skewed`, where the XPU
  gradient is 4.8e-5 from float64. A float32 reference was further from
  float64 than the XPU kernel, 1.2e-3, and failed it. A lookup left out of
  the gradient moves its row by about 1 and still fails the check. If any
  case mismatches, the run fails and no CSV is written.
- **Timing:** the median of 100 iterations, with the same XPU event timing,
  cache flush and GPU lead as the jagged sweep, described under
  [What the sweep measures](#what-the-sweep-measures). The forward times one
  operator call, which also builds the autograd graph, as in training. The
  backward times the backward of a graph built once, as upstream's
  `bwd_only` mode does. The dense lookup differentiates its weights with
  `torch.autograd.grad`, so the gradient is not accumulated into `.grad`. The
  split lookup runs `torch.autograd.backward`, whose in-place update moves
  the weights at every iteration, as in training.
- **One process per case:** each case is timed in a newly spawned process,
  after its CPU check in the benchmark's own process. The process rebuilds
  the batch and the initial state from `--seed`, and the benchmark fails if
  the batch differs from the one it checked. In one shared process, the
  forward time depends on the device allocations of the earlier cases: on a
  B60, the small FP32 dense forward took 92.6 µs in a fresh process,
  140.8 µs after the dense CPU check and 190.6 µs after an earlier dense
  timing. With a process per case, the rows do not depend on the case order,
  so a run with `--families`, `--dtypes` or fewer configs reproduces the
  same rows of the full run.

The operators are called directly through `torch.ops.fbgemm`, not through
`SplitTableBatchedEmbeddingBagsCodegen` or
`DenseTableBatchedEmbeddingBagsCodegen`. The times are those of the operators
alone, and exclude the module's own work, such as index casts, bounds checks
and Python dispatch.

#### Bandwidth

`gb_per_s` follows upstream `tbe_training_benchmark.py` for no-bag lookups,
so it can be compared with upstream's figures:

- forward bytes = `total_lengths` × D × (weight bytes + output bytes);
- backward bytes = 2 × forward bytes.

The formula counts the embedding rows read and the output written. With a
fixed L, `total_lengths` is B × T × L, upstream's count; with variable bag
lengths it is the number of indices actually drawn, where upstream would
still use L.

#### Reading the numbers

- Every XPU backward synchronises with the host once. Run-length encoding of
  the sorted indices reads the number of runs back to the host
  (`run_positions[-1].item()` in
  [`backward_utils.cpp`](../../src/fbgemm_xpu/fbgemm_utils/backward_utils.cpp)),
  and the GPU idles for that round trip. The timing helper therefore logs, for
  each backward, that the GPU may have waited for the host. The backward
  times include that wait, which is part of the operator's cost.
- The dense backward also zero-fills and writes its gradient of the whole
  table, T × E × D values, whatever the batch. It is therefore slower than
  the split backward: in FP32, 7.9 ms against 3.8 ms at `medium` and 29.1 ms
  against 23.4 ms at `large` on a B60.
- FP16 weights cut the `medium` and `large` forward times by about 20%, close
  to the 25% fewer bytes the bandwidth formula counts.
- The `skewed` forward reports about 506 GB/s in FP32 on a B60, above the
  456 GB/s its memory is rated at. The formula counts every lookup as a
  row read from memory, but the most read rows, thousands of lookups each,
  stay in the cache.
- Over four runs of each config on one B60, forward times varied by at most
  0.5%, the `large` backwards by at most 0.8% and the `skewed` backwards by
  at most 4.7%. The `small` and `medium` backwards, where the host round trip
  weighs most and the pod had only 4 CPUs, varied by up to 15%. Treat changes
  smaller than that as noise.

### Not benchmarked on XPU

- Pooled lookups (`PoolingMode.SUM`, `PoolingMode.MEAN`), weighted lookups,
  VBE, mixed embedding dimensions and cache- or UVM-backed placements: they
  have no XPU kernels.
- Optimisers other than rowwise Adagrad: they have no XPU split lookup.
- The module path, `SplitTableBatchedEmbeddingBagsCodegen` and
  `DenseTableBatchedEmbeddingBagsCodegen`: pristine FBGEMM 1.9.0 has no
  `ComputeDevice.XPU`, and they hard-code CUDA, MTIA or CPU. Adding XPU there
  is a separate upstream change.
- Upstream's `tbe_training_benchmark.py` is not patched. It is not in the
  `fbgemm-gpu` wheel, its helpers are imported from the installed wheel, so
  patching the checkout would change nothing that runs, and it builds the
  modules above.

For the same reason, the upstream TBE training tests
(`fbgemm_gpu/test/tbe/training/`) do not run on XPU.
