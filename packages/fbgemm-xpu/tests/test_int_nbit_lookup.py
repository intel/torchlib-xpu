# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import importlib.metadata
import importlib.util
import subprocess  # nosec B404
import sys
from pathlib import Path

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

LOOKUP = "fbgemm::int_nbit_split_embedding_codegen_lookup_function"
BOUNDS = "fbgemm::bounds_check_indices"


@pytest.fixture
def fbgemm():
    assert importlib.metadata.version("fbgemm-gpu-cpu") == "1.9.0"  # nosec B101
    importlib.import_module("fbgemm_gpu")
    return importlib.import_module("fbgemm_gpu.split_table_batched_embeddings_ops_inference")


@pytest.fixture
def xpu(fbgemm):
    importlib.import_module("fbgemm_xpu")
    assert torch.__version__.split("+")[0] == "2.14.0", torch.__version__  # nosec B101
    assert torch.xpu.is_available(), "XPU validation requires a real device; no skip/fallback"  # nosec B101
    for operator in (LOOKUP, BOUNDS):
        assert torch._C._dispatch_has_kernel_for_dispatch_key(operator, "XPU"), operator  # nosec B101
    return torch.device("xpu:0")


def quantized_tables(bit_rates, dimension, row_counts):
    from fbgemm_gpu.split_embedding_configs import SparseType
    from fbgemm_gpu.tbe.utils.quantize import quantize_embs

    tables = []
    for table, (bit_rate, rows) in enumerate(zip(bit_rates, row_counts)):
        values = torch.arange(rows * dimension, dtype=torch.float32).reshape(rows, dimension)
        values = torch.sin(values * 0.13 + table) * 0.75 + torch.arange(rows)[:, None] * 0.1
        sparse_type = SparseType.INT4 if bit_rate == 4 else SparseType.INT8
        packed, scale_bias = quantize_embs(values.contiguous(), sparse_type)
        assert packed.shape == (rows, dimension * bit_rate // 8)  # nosec B101
        assert scale_bias.shape == (rows, 4)  # nosec B101
        tables.append((packed.contiguous(), scale_bias.contiguous()))
    return tables


def unpack_table(packed, scale_bias, bit_rate):
    qparams = scale_bias.contiguous().view(torch.float16).float()
    if bit_rate == 4:
        values = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(1).float()
    else:
        values = packed.float()
    return values * qparams[:, :1] + qparams[:, 1:]


def lookup_args(bit_rates, dimension, row_counts, indices, offsets, alignment=16):
    tables = quantized_tables(bit_rates, dimension, row_counts)
    storage = []
    table_offsets = []
    size = 0
    for packed, scale_bias in tables:
        row_bytes = packed.shape[1] + 4
        stride = (row_bytes + alignment - 1) // alignment * alignment
        rows = torch.full((packed.shape[0], stride), 213, dtype=torch.uint8)
        rows[:, :4] = scale_bias
        rows[:, 4:row_bytes] = packed
        table_offsets.append(size)
        storage.append(rows.flatten())
        size += rows.numel()
    return dict(
        dev_weights=torch.cat(storage),
        uvm_weights=torch.empty(0, dtype=torch.uint8),
        weights_placements=torch.full((len(tables),), 3, dtype=torch.int32),
        weights_offsets=torch.tensor(table_offsets, dtype=torch.int64),
        weights_tys=torch.tensor([3 if bits == 4 else 2 for bits in bit_rates], dtype=torch.uint8),
        D_offsets=torch.arange(len(tables) + 1, dtype=torch.int32) * dimension,
        total_D=dimension * len(tables),
        max_int2_D=0,
        max_int4_D=dimension if 4 in bit_rates else 0,
        max_int8_D=dimension if 8 in bit_rates else 0,
        max_float16_D=0,
        max_float32_D=0,
        indices=indices,
        offsets=offsets,
        pooling_mode=2,
        indice_weights=None,
        output_dtype=0,
        row_alignment=alignment,
    ), tables


def to_xpu(arguments, device):
    result = {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in arguments.items()
    }
    result["weights_placements"] = torch.zeros_like(result["weights_placements"])
    return result


def lookup(arguments):
    return torch.ops.fbgemm.int_nbit_split_embedding_codegen_lookup_function(**arguments)


def assert_parity(arguments, tables, bit_rates, device):
    dimension = arguments["total_D"] // len(tables)
    batches = (arguments["offsets"].numel() - 1) // len(tables)
    reference = []
    for table, ((packed, scale_bias), bit_rate) in enumerate(zip(tables, bit_rates)):
        begin = arguments["offsets"][table * batches].item()
        end = arguments["offsets"][(table + 1) * batches].item()
        reference.append(unpack_table(packed, scale_bias, bit_rate)[arguments["indices"][begin:end].long()])
    expected = torch.cat(reference).reshape(-1, dimension)
    dtype = {0: torch.float32, 1: torch.float16, 5: torch.bfloat16}[arguments["output_dtype"]]
    expected = expected.to(dtype)
    cpu_result = lookup(arguments)
    torch.testing.assert_close(cpu_result, expected, rtol=1e-5 if dtype == torch.float32 else 1e-2, atol=1e-6)
    xpu_result = lookup(to_xpu(arguments, device))
    assert xpu_result.device.type == "xpu"  # nosec B101
    assert xpu_result.shape == expected.shape  # nosec B101
    assert torch.isfinite(xpu_result).all().item()  # nosec B101
    torch.testing.assert_close(xpu_result.cpu(), cpu_result, rtol=1e-5 if dtype == torch.float32 else 1e-2, atol=1e-6)


def test_codegen_contract(tmp_path):
    source = Path(__file__).resolve().parents[1] / "src"
    subprocess.run(  # nosec B603
        [sys.executable, str(source / "codegen/genscript/generate_forward_quantized.py"),
         "--opensource", "--install_dir", str(tmp_path)],
        check=True,
    )
    for bits in (4, 8):
        generated = (tmp_path / f"sycl_kernels/gen_embedding_forward_int{bits}_nobag.h").read_text()
        assert f"lookup_int{bits}_nobag" in generated  # nosec B101
        assert f"class LookupInt{bits}NobagKernel" in generated  # nosec B101
        assert "void operator()(const sycl::nd_item<1>& item) const" in generated  # nosec B101
        assert f"queue.parallel_for<LookupInt{bits}NobagKernel<index_t, output_t>>" in generated  # nosec B101
        assert "SPDX-License-Identifier: BSD-3-Clause" in generated  # nosec B101
        assert "SYCL PORT MAPPING TO FBGEMM CUDA SOURCE" in generated  # nosec B101
        assert "embedding_forward_quantized_split_nbit_kernel_template.cu" in generated  # nosec B101
        assert "{{" not in generated and "{%" not in generated  # nosec B101
        assert f"& {(1 << bits) - 1}" in generated  # nosec B101
        assert "element += work_items_" in generated  # nosec B101
        assert "error_ref.fetch_or(row_index < 0 ? 1 : 2)" in generated  # nosec B101
        assert generated.index("if (row_index < 0 || row_index >= storage_rows_)") < generated.index(  # nosec B101
            "const uint8_t* row = weights_ + row_index * row_stride_"
        )


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("dimension", [4, 512])
def test_cpu_packed_layout(fbgemm, bits, dimension):
    arguments, tables = lookup_args(
        [bits], dimension, [5], torch.tensor([0, 4, 2]), torch.tensor([0, 3]),
    )
    expected = unpack_table(*tables[0], bits)[arguments["indices"]]
    torch.testing.assert_close(lookup(arguments), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("bit_rates", [(4,), (8,), (4, 4), (8, 8), (4, 8)])
@pytest.mark.parametrize("dimension", [4, 512])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("output_dtype", [0, 1, 5])
def test_direct_parity(xpu, bit_rates, dimension, index_dtype, output_dtype):
    count = len(bit_rates)
    indices = torch.tensor([0, 4, 1] * count, dtype=index_dtype)
    offsets = torch.tensor([0, 0, 3] if count == 1 else [0, 0, 3, 4, 6], dtype=index_dtype)
    arguments, tables = lookup_args(bit_rates, dimension, [5] * count, indices, offsets)
    arguments["output_dtype"] = output_dtype
    assert_parity(arguments, tables, bit_rates, xpu)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("dimension", [2, 6])
def test_rejects_dimension_outside_frontend_contract(xpu, bits, dimension):
    arguments, _ = lookup_args(
        [bits, bits], 4, [5, 5], torch.tensor([0, 1]), torch.tensor([0, 1, 2]),
    )
    arguments = to_xpu(arguments, xpu)
    arguments["D_offsets"] = (
        torch.arange(3, dtype=torch.int32, device=xpu) * dimension
    )
    arguments["total_D"] = 2 * dimension
    arguments["max_int4_D"] = dimension if bits == 4 else 0
    arguments["max_int8_D"] = dimension if bits == 8 else 0
    with pytest.raises(RuntimeError, match="divisible by four"):
        lookup(arguments)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("offset_values", [[0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 2]])
def test_empty_inputs_and_tables(xpu, bits, offset_values):
    indices = torch.tensor([0, 4][:offset_values[-1]], dtype=torch.int32)
    arguments, tables = lookup_args(
        [bits, bits], 512, [5, 5], indices, torch.tensor(offset_values, dtype=torch.int64),
    )
    assert_parity(arguments, tables, [bits, bits], xpu)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("alignment", [1, 16, 128])
def test_large_grid_and_alignment(xpu, bits, alignment):
    count = 4096 * 256 // 4 + 17
    arguments, tables = lookup_args(
        [bits], 4, [7], torch.arange(count, dtype=torch.int64) % 7,
        torch.tensor([0, count]), alignment,
    )
    assert_parity(arguments, tables, [bits], xpu)


@pytest.mark.parametrize("bits", [4, 8])
def test_reordered_shared_tables_and_padding(xpu, bits):
    arguments, tables = lookup_args(
        [bits, bits], 4, [5, 5], torch.tensor([0, 4, 2]), torch.tensor([0, 1, 2, 3]),
    )
    boundary = arguments["weights_offsets"][1].item()
    arguments["dev_weights"] = torch.cat((
        arguments["dev_weights"][:boundary], torch.full((128,), 219, dtype=torch.uint8),
        arguments["dev_weights"][boundary:],
    ))
    arguments["weights_offsets"] = torch.tensor([boundary + 128, 0, boundary + 128])
    arguments["weights_placements"] = torch.full((3,), 3, dtype=torch.int32)
    arguments["weights_tys"] = torch.full((3,), 3 if bits == 4 else 2, dtype=torch.uint8)
    arguments["D_offsets"] = torch.tensor([0, 4, 8, 12], dtype=torch.int32)
    arguments["total_D"] = 12
    assert_parity(arguments, [tables[1], tables[0], tables[1]], [bits] * 3, xpu)


@pytest.mark.parametrize("case,message", [
    ("pooling", "PoolingMode.NONE"), ("weighted", "weighted lookup"),
    ("cache_weights", "cache weights"), ("cache_locations", "cache locations"),
    ("uvm", "UVM/cache"), ("placement", "DEVICE placement"),
    ("type", "only INT4 and INT8"), ("mixed", "mixed dimensions"),
    ("output", "output dtype"), ("alignment", "row_alignment"),
    ("pruning", "negative indices/pruning"), ("out_of_bounds", "index exceeds packed storage"),
    ("offsets", "offsets must be monotonic"), ("metadata", "inconsistent table metadata"),
    ("weight_dtype", "dev_weights has unsupported dtype"),
    ("indices_dtype", "indices must have int32 or int64"),
    ("noncontiguous", "contiguous and one-dimensional"),
    ("max_dimension", "disagrees with table dimensions"),
])
def test_unsupported_and_invalid_inputs(xpu, case, message):
    arguments, _ = lookup_args([4, 4], 4, [5, 5], torch.tensor([0, 1]), torch.tensor([0, 1, 2]))
    arguments = to_xpu(arguments, xpu)
    if case == "pooling":
        arguments["pooling_mode"] = 0
    elif case == "weighted":
        arguments["indice_weights"] = torch.ones(2, device=xpu)
    elif case == "cache_weights":
        arguments["lxu_cache_weights"] = torch.ones(4, dtype=torch.uint8, device=xpu)
    elif case == "cache_locations":
        arguments["lxu_cache_locations"] = torch.zeros(2, dtype=torch.int32, device=xpu)
    elif case == "uvm":
        arguments["uvm_weights"] = torch.ones(1, dtype=torch.uint8, device=xpu)
    elif case == "placement":
        arguments["weights_placements"].fill_(2)
    elif case == "type":
        arguments["weights_tys"].fill_(4)
    elif case == "mixed":
        arguments["D_offsets"][-1] = 12
        arguments["total_D"] = 12
    elif case == "output":
        arguments["output_dtype"] = 2
    elif case == "alignment":
        arguments["row_alignment"] = 3
    elif case == "pruning":
        arguments["indices"][0] = -1
    elif case == "out_of_bounds":
        arguments["indices"][0] = 5
    elif case == "offsets":
        arguments["offsets"][1] = -1
    elif case == "metadata":
        arguments["weights_tys"] = arguments["weights_tys"][:1]
    elif case == "weight_dtype":
        arguments["dev_weights"] = arguments["dev_weights"].long()
    elif case == "indices_dtype":
        arguments["indices"] = arguments["indices"].float()
    elif case == "noncontiguous":
        arguments["indices"] = torch.arange(4, device=xpu)[::2]
    elif case == "max_dimension":
        arguments["max_int4_D"] = 8
    with pytest.raises(RuntimeError, match=message):
        lookup(arguments)


@pytest.mark.parametrize("field,position,value,message", [
    ("weights_placements", 0, 2, "DEVICE placement"),
    ("weights_tys", 0, 4, "only INT4 and INT8"),
    ("D_offsets", 0, 4, "D must be positive"),
    ("D_offsets", 1, 8, "mixed dimensions"),
    ("weights_offsets", 0, -1, "weights_offsets outside packed storage"),
    ("weights_offsets", 1, 161, "weights_offsets outside packed storage"),
    ("weights_offsets", 1, 16, "index exceeds packed storage"),
    ("offsets", 0, 1, "offsets must span all indices"),
    ("offsets", 1, -1, "offsets must be monotonic"),
    ("offsets", 1, 3, "offsets must be monotonic"),
    ("indices", 0, -1, "negative indices/pruning"),
    ("indices", 1, 2**63 - 1, "index exceeds packed storage"),
])
def test_in_place_inputs_revalidated(xpu, field, position, value, message):
    arguments, _ = lookup_args(
        [4, 4], 4, [5, 5], torch.tensor([4, 4]), torch.tensor([0, 1, 2], dtype=torch.int32),
    )
    arguments = to_xpu(arguments, xpu)
    expected = lookup(arguments).cpu()
    original = arguments[field].clone()
    arguments[field][position] = value
    with pytest.raises(RuntimeError, match=message):
        lookup(arguments)
    arguments[field].copy_(original)
    torch.testing.assert_close(lookup(arguments).cpu(), expected)


@pytest.mark.parametrize("bits", [4, 8])
def test_valid_table_mapping_mutation(xpu, bits):
    arguments, tables = lookup_args(
        [bits, bits], 4, [5, 5], torch.tensor([0, 4]), torch.tensor([0, 1, 2]),
    )
    arguments = to_xpu(arguments, xpu)
    initial = lookup(arguments).cpu()
    arguments["weights_offsets"].copy_(arguments["weights_offsets"].flip(0))
    expected = torch.cat((unpack_table(*tables[1], bits)[[0]], unpack_table(*tables[0], bits)[[4]]))
    assert not torch.equal(initial, expected)  # nosec B101
    torch.testing.assert_close(lookup(arguments).cpu(), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("invalid,message", [
    (-1, "negative indices/pruning"), (7, "index exceeds packed storage"),
])
def test_invalid_index_after_grid_stride(xpu, bits, index_dtype, invalid, message):
    count = 4096 * 256 // 4 + 17
    arguments, _ = lookup_args(
        [bits], 4, [7], torch.arange(count, dtype=index_dtype) % 7,
        torch.tensor([0, count], dtype=torch.int64 if index_dtype == torch.int32 else torch.int32),
    )
    arguments = to_xpu(arguments, xpu)
    arguments["indices"][-1] = invalid
    with pytest.raises(RuntimeError, match=message):
        lookup(arguments)


class ObserveEmbeddingDispatch(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.calls = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        result = func(*args, **kwargs)
        if func._schema.name in (LOOKUP, BOUNDS):
            tensors = [value for value in (*args, *kwargs.values()) if isinstance(value, torch.Tensor)]
            self.calls.append((func._schema.name, tensors[0].device.type))
        return result

    def assert_xpu_calls(self):
        assert (BOUNDS, "xpu") in self.calls, self.calls  # nosec B101
        assert (LOOKUP, "xpu") in self.calls, self.calls  # nosec B101


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("dimension", [4, 512])
def test_high_level_int_nbit(xpu, fbgemm, bits, dimension):
    from fbgemm_gpu.split_embedding_configs import SparseType
    from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
        BoundsCheckMode,
        EmbeddingLocation,
        PoolingMode,
    )

    weight_type = SparseType.INT4 if bits == 4 else SparseType.INT8
    tables = quantized_tables([bits, bits], dimension, [5, 7])

    def make_module(device):
        module = fbgemm.IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (f"table_{table}", rows, dimension, weight_type,
                 EmbeddingLocation.HOST if device.type == "cpu" else EmbeddingLocation.DEVICE)
                for table, rows in enumerate([5, 7])
            ],
            device=device, weight_lists=[(packed.clone(), params.clone()) for packed, params in tables],
            pooling_mode=PoolingMode.NONE, output_dtype=SparseType.FP32,
            bounds_check_mode=BoundsCheckMode.WARNING, row_alignment=16,
        ).eval()
        module.bounds_check_version = 1
        return module

    cpu_module = make_module(torch.device("cpu"))
    xpu_module = make_module(xpu)
    indices = torch.tensor([0, -2, 99, 1, 0, 6], dtype=torch.int32)
    offsets = torch.tensor([0, 2, 3, 4, 6], dtype=torch.int32)
    with torch.no_grad():
        expected = cpu_module(indices.clone(), offsets.clone())
        with ObserveEmbeddingDispatch() as observed:
            actual = xpu_module(indices.to(xpu), offsets.to(xpu))
    observed.assert_xpu_calls()
    assert actual.device.type == "xpu"  # nosec B101
    assert xpu_module.bounds_check_warning.item() == cpu_module.bounds_check_warning.item() > 0  # nosec B101
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(
    importlib.util.find_spec("torchrec") is None,
    reason="TorchRec is not installed",
)
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("dimension", [4, 512])
def test_high_level_quant_embedding_collection(xpu, bits, dimension):
    from fbgemm_gpu.split_table_batched_embeddings_ops_common import BoundsCheckMode
    from torchrec.modules.embedding_configs import DataType, EmbeddingConfig
    from torchrec.quant.embedding_modules import (
        EmbeddingCollection as QuantEmbeddingCollection,
    )
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

    tables = quantized_tables([bits, bits], dimension, [5, 7])

    def make_module(device):
        module = QuantEmbeddingCollection(
            tables=[EmbeddingConfig(
                name=f"table_{table}", num_embeddings=rows, embedding_dim=dimension,
                data_type=DataType.INT4 if bits == 4 else DataType.INT8,
                feature_names=[f"feature_{table}"],
            ) for table, rows in enumerate([5, 7])],
            device=device, output_dtype=torch.float32, register_tbes=True, row_alignment=16,
            table_name_to_quantized_weights={
                f"table_{table}": (packed.clone(), params.clone())
                for table, (packed, params) in enumerate(tables)
            },
        ).eval()
        assert len(module.tbes) > 0  # nosec B101
        for tbe in module.tbes:
            tbe.bounds_check_mode_int = BoundsCheckMode.WARNING.value
            tbe.bounds_check_version = 1
        return module

    def features(device):
        return KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.tensor([0, -2, 99, 1, 0, 6], dtype=torch.int32, device=device),
            lengths=torch.tensor([2, 1, 1, 2], dtype=torch.int32, device=device),
        )

    cpu_module = make_module(torch.device("cpu"))
    xpu_module = make_module(xpu)
    with torch.no_grad():
        expected = cpu_module(features(torch.device("cpu")))
        with ObserveEmbeddingDispatch() as observed:
            actual = xpu_module(features(xpu))
    observed.assert_xpu_calls()
    assert actual.keys() == expected.keys()  # nosec B101
    assert sum(tbe.bounds_check_warning.item() for tbe in xpu_module.tbes) > 0  # nosec B101
    for name in actual:
        assert actual[name].values().device.type == "xpu"  # nosec B101
        torch.testing.assert_close(actual[name].lengths().cpu(), expected[name].lengths())
        torch.testing.assert_close(actual[name].values().cpu(), expected[name].values(), rtol=1e-5, atol=1e-6)
