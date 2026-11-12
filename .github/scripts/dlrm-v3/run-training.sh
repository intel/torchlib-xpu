#!/usr/bin/env bash
# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Runs a bounded DLRM-v3 train-eval smoke on MovieLens-1M from
# $E2E/data. Writes the log to $E2E/logs/train.log.
#
# Usage: run-training.sh <generative-recommenders checkout>
# Environment: E2E; for XCCL, CCL_ROOT and LD_LIBRARY_PATH pointing to the
# virtual environment.

set -euo pipefail

: "${E2E:?}"
cd "${1:?usage: run-training.sh <generative-recommenders checkout>}"

# Shortened copy of movielens_1m_xpu.gin: the overrides come last, and gin
# keeps the last binding of a parameter.
gin=generative_recommenders/dlrm_v3/train/gin/movielens_1m_xpu.gin
cat >> "$gin" <<EOF

# torchlib-xpu CI (test-dlrm-v3): bounded train-eval smoke.
make_train_test_dataloaders.new_path_prefix = "$E2E/data"
train_eval_loop.num_epochs = 1
train_eval_loop.num_train_batches = 40
train_eval_loop.eval_frequency = 20
train_eval_loop.num_eval_batches = 5
MetricsLogger.tensorboard_log_path = "$E2E/logs/tensorboard"
EOF
cp "$gin" "$E2E/logs/"

export PYTHONPATH="$PWD" WORLD_SIZE=1
timeout --kill-after=30s 3600s python -m generative_recommenders.dlrm_v3.train.train_ranker \
  --dataset movielens-1m-xpu --mode train-eval 2>&1 | tee "$E2E/logs/train.log"
