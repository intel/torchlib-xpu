#!/usr/bin/env bash
# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Downloads MovieLens-1M, prepares it with the DLRM-v3 preprocessor into
# $E2E/data/data/ml-1m, and checks the SHA-256 of the archive and of the
# prepared sasrec_format.csv.
#
# Usage: prepare-movielens-1m.sh <generative-recommenders checkout>
# Environment: E2E, ML1M_URL, ML1M_SHA256, ML1M_PREPARED_SHA256

set -euo pipefail

: "${E2E:?}" "${ML1M_URL:?}" "${ML1M_SHA256:?}" "${ML1M_PREPARED_SHA256:?}"
gr=$(cd "${1:?usage: prepare-movielens-1m.sh <generative-recommenders checkout>}" && pwd)
here=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$E2E/prep/tmp" "$E2E/data/data"
curl -fsSL --retry 3 -o "$E2E/prep/tmp/movielens1m.zip" "$ML1M_URL"
echo "$ML1M_SHA256  $E2E/prep/tmp/movielens1m.zip" | sha256sum -c -

# The preprocessor finds the archive in tmp/ and does not download it again.
cd "$E2E/prep"
PYTHONPATH="$gr" python "$here/preprocess_movielens_1m.py"
mv tmp/ml-1m "$E2E/data/data/ml-1m"
echo "$ML1M_PREPARED_SHA256  $E2E/data/data/ml-1m/sasrec_format.csv" | sha256sum -c -
