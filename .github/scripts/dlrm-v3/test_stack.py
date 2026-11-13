# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests FBGEMM, TorchRec and fbgemm-xpu together on xpu:0.

Runs one training step of a fused EmbeddingCollection (exact row-wise
Adagrad), sharded table-wise through DistributedModelParallel, and compares
the lookups and the updated weights with plain PyTorch. Also checks that
TorchRec detects XPU and that the FBGEMM TBE modules keep their tables in
device memory.

XCCL needs CCL_ROOT and LD_LIBRARY_PATH to point to the virtual environment.
"""

import os
import sys

import fbgemm_xpu  # noqa: F401
import torch
import torch.distributed as dist
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from fbgemm_gpu.tbe.config import ComputeDevice
from torchrec import EmbeddingCollection, EmbeddingConfig, KeyedJaggedTensor
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.utils import get_device_type
from torchrec.test_utils import get_free_port

NUM_ROWS, DIM, BATCH, LR, EPS = 64, 16, 8, 0.1, 1e-8
DEVICE = torch.device("xpu:0")
TABLES = [
    EmbeddingConfig(num_embeddings=NUM_ROWS, embedding_dim=DIM, name=f"t{i}", feature_names=[f"f{i}"])
    for i in range(2)
]


def make_kjt():
    g = torch.Generator().manual_seed(1)
    lengths = torch.randint(0, 4, (2 * BATCH,), generator=g)
    values = torch.randint(0, NUM_ROWS, (int(lengths.sum()),), generator=g)
    return KeyedJaggedTensor.from_lengths_sync(keys=["f0", "f1"], values=values, lengths=lengths)


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        sys.exit(1)


def local(tensor):
    return tensor.local_tensor() if hasattr(tensor, "local_tensor") else tensor


def tbe_placement(model):
    # Sharded modules keep their lookups in a plain list, which modules()
    # does not traverse.
    tbes = [
        sub
        for m in model.modules()
        for lookup in getattr(m, "_lookups", [])
        for sub in lookup.modules()
        if hasattr(sub, "use_cpu")
    ]
    return [(m.use_cpu, m.weights_dev.device.type, m.weights_host.numel(), m.weights_uvm.numel()) for m in tbes]


def main():
    torch.manual_seed(0)
    check("torchrec.get_device_type", get_device_type() == "xpu", get_device_type())
    torch.xpu.set_device(DEVICE)
    check("fbgemm.ComputeDevice.XPU", hasattr(ComputeDevice, "XPU"), str(list(ComputeDevice)))

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(get_free_port()))
    backend = "xccl" if dist.is_xccl_available() else "gloo"
    dist.init_process_group(backend=backend, rank=0, world_size=1)
    print(f"process group: {backend}", flush=True)

    ec = EmbeddingCollection(tables=TABLES, device=torch.device("meta"))
    sharder = EmbeddingCollectionSharder(
        fused_params={"optimizer": EmbOptimType.EXACT_ROWWISE_ADAGRAD, "learning_rate": LR, "eps": EPS}
    )
    topology = Topology(world_size=1, compute_device="xpu")
    constraints = {
        t.name: ParameterConstraints(sharding_types=["table_wise"], compute_kernels=["fused"]) for t in TABLES
    }
    plan = EmbeddingShardingPlanner(topology=topology, constraints=constraints).plan(ec, [sharder])
    model = DistributedModelParallel(ec, device=DEVICE, plan=plan, sharders=[sharder], init_data_parallel=False)
    weights = {name: torch.rand(NUM_ROWS, DIM) for name in ("t0", "t1")}
    for name, tensor in model.state_dict().items():
        local(tensor).copy_(weights[name.split(".")[-2]])

    placement = tbe_placement(model)
    check(
        "training.tbe_on_xpu",
        bool(placement) and all(p == (False, "xpu", 0, 0) for p in placement),
        f"(use_cpu, weights_dev device, weights_host numel, weights_uvm numel) per TBE: {placement}",
    )

    kjt = make_kjt()
    out = model(kjt.to(DEVICE))
    sum((out[k].values() * (i + 1)).sum() for i, k in enumerate(("f0", "f1"))).backward()
    state = {name.split(".")[-2]: local(tensor).cpu() for name, tensor in model.state_dict().items()}
    for i, (key, table) in enumerate((("f0", "t0"), ("f1", "t1"))):
        ids, w = kjt[key].values(), weights[table]
        torch.testing.assert_close(out[key].values().detach().cpu(), w[ids], atol=1e-5, rtol=1e-5)
        grad = torch.zeros(NUM_ROWS, DIM).index_add_(0, ids, torch.full((ids.numel(), DIM), float(i + 1)))
        momentum = grad.square().mean(dim=1)
        expected = w - (LR / (momentum.sqrt() + EPS))[:, None] * grad
        torch.testing.assert_close(state[table], expected, atol=1e-5, rtol=1e-5)
        check(f"training.{table}", True, "lookup and row-wise Adagrad update == reference")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
