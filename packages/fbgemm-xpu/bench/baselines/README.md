# Jagged operator baselines

Committed performance baselines for the XPU jagged operators, forward and
backward:

- `jagged_to_padded_dense`
- `jagged_2d_to_dense`
- `dense_to_jagged`
- `jagged_dense_elementwise_add_jagged_output`

| Hardware | Timings | Run notes |
| --- | --- | --- |
| Intel® Data Center GPU Max 1550 (PVC), one tile | [pvc-max1550/jagged_tensor.csv](pvc-max1550/jagged_tensor.csv) | [pvc-max1550/README.md](pvc-max1550/README.md) |
| Intel® Arc™ Pro B60 Graphics (BMG) | [bmg-b60/jagged_tensor.csv](bmg-b60/jagged_tensor.csv) | [bmg-b60/README.md](bmg-b60/README.md) |

Each CSV is a plain table, so GitHub renders it. The command, date, device,
driver, versions and sweep definition of the run that produced it are in the
`README.md` beside it, in the form the sweep writes them.

## Reproduce

```
uv pip install -e "$TORCHLIB_XPU_PATH/packages/fbgemm-xpu[test]" \
  --index https://download.pytorch.org/whl/xpu
ZE_AFFINITY_MASK=0 python -m fbgemm_xpu.bench.jagged_sweep \
  --output jagged_tensor.csv
```

Run on an idle machine. `ZE_AFFINITY_MASK=0` pins the run to one device: on
PVC that is a single tile. In a container with a CPU limit, also set
`OMP_NUM_THREADS` to that limit: PyTorch otherwise starts a thread per host
core, and the container is throttled.

The sweep writes two files: `jagged_tensor.csv` with the timings, and
`jagged_tensor.md` with the run's command, date, device, driver, versions and
sweep definition.

The PVC baseline was recorded when jagged-sweep was a case of the patched
upstream script, so its notes show that command and an `fbgemm_checkout`
line. It also predates the GPU lead described under Timing below, so its
`timing` line does not mention it. Rerunning PVC rows with the lead matched
the smallest and largest shapes within noise, and made some mid-size rows,
around 40 to 60 µs, up to 10% faster: the 384 MB flush only partly hid the
host there.

The BMG baseline was recorded on one B60 in a Kubernetes pod limited to 4
CPUs, with `OMP_NUM_THREADS=4`. Both baselines predate the clean flush
described under Timing below; they were taken with a zero fill. On PVC that
fill leaves L2 dirty just as an in-place negation does, so every row includes
write-back the clean flush keeps out of the timed window, and rows for shapes
that fit in the 192 MB L2 can be about twice as slow as a rerun: a copy of
half the L2 took 314 µs after a zero fill and 139 µs after a clean flush. On
a B60 a zero fill timed within 2% of a clean flush, so its rows should hold
until the baseline is rerun.

## What the sweep measures

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
  read, the flush leaves L2 dirty and the timed call pays for the write-back:
  on a PVC tile that added ~170 µs to a copy that fits in L2 and ~210 µs to
  one four times its size. Events time the GPU timeline, so the GPU must not
  wait for the host inside the timed window.
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

### Bandwidth

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

### Reading the numbers

- Small shapes are latency-bound. Below a few MB of traffic the time floors,
  at about 14 µs on PVC and 5 µs on BMG, whatever the size, and their GB/s
  does not reflect bandwidth. Use them to catch kernel latency regressions.
  Without the GPU lead, BMG reported 100 µs to 1 ms here: the 36 MB flush is
  too short to keep the GPU busy while the host submits, so the timed window
  measured the host instead.
- Shapes whose operands fit in the last-level cache (192 MB per PVC tile,
  18 MB on B60) start from device memory, as the flush evicts them. The
  alternative, upstream's fixed 40 MB flush, leaves PVC operands in L2 and
  reports up to twice the tile's HBM bandwidth.
- Run to run, PVC times varied by a median of 0.7%, a 95th percentile of
  2.8% and at most 5.5%, the last on shapes under 60 µs. BMG times varied by
  a median of 0.1%, a 95th percentile of 1.2% and at most 4.9%, again on
  shapes under 60 µs; rows over 60 µs stayed within 1%. Treat changes smaller
  than that as noise.
- For reference, a large (2 GB) `copy_` streams at about 1.05 TB/s on one
  PVC tile with this helper, and at about 400 GB/s on a B60, whose memory is
  rated at 456 GB/s.
- BMG compresses memory traffic: the same `copy_` of zeros reports about
  1.8 TB/s. The sweep's inputs are random, but the zero padding of dense
  outputs can compress, which is one reason some BMG rows exceed the rated
  bandwidth.

## Not benchmarked on XPU

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
