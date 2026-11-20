# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Checks the log of a DLRM-v3 train-eval run.

Fails when the log has a traceback, has no training or evaluation steps, has
a non-finite metric, or when the planner did not place the embedding tables in
device memory (HBM) on every rank.

Usage: check_training_log.py <train.log>
"""

import math
import re
import sys

STEP = re.compile(r"(train|eval) - Step (\d+) metrics: (\{.*\})")
METRIC = re.compile(r"'([^']+)':\s*(?:tensor\()?\s*([-+0-9.eE]+|nan|inf)")
RANK_HBM = re.compile(r"^\W*(\d+)\W+([\d.]+) \((\d+)%\)")


def main(path):
    text = open(path).read()
    steps = {"train": [], "eval": []}
    last_eval, not_finite = {}, []
    for mode, n, metrics in STEP.findall(text):
        steps[mode].append(int(n))
        values = {k: float(v) for k, v in METRIC.findall(metrics)}
        not_finite += [f"{mode} {n} {k}" for k, v in values.items() if not math.isfinite(v)]
        if mode == "eval":
            last_eval = values
    hbm = {}
    for line in text.splitlines():
        if "planner.stats:" not in line:
            continue
        line = line.split("planner.stats:", 1)[1]
        match = RANK_HBM.match(line)
        if match and "HBM" not in line:
            hbm.setdefault(int(match[1]), float(match[2]))

    failures = []
    if "Traceback (most recent call last)" in text:
        failures.append("traceback in the log")
    if not steps["train"] or not steps["eval"]:
        failures.append(f"train steps {steps['train']}, eval steps {steps['eval']}")
    if not hbm or not all(v > 0 for v in hbm.values()):
        failures.append(f"planner HBM per rank {hbm}")
    if not_finite:
        failures.append(f"non-finite metrics: {not_finite[:5]}")
    print(f"train steps {steps['train']}, eval steps {steps['eval']}, planner HBM GB {hbm}")
    print(f"last eval metrics {last_eval}")
    if failures:
        sys.exit("FAIL: " + "; ".join(failures))


if __name__ == "__main__":
    main(sys.argv[1])
