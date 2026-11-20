#!/usr/bin/env bash
# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Applies the Python changes of an FBGEMM pull request (FBGEMM_PR_BASE to
# FBGEMM_PR_HEAD under fbgemm_gpu/fbgemm_gpu) to the installed fbgemm_gpu
# package, then checks that ComputeDevice.XPU is available.
#
# Usage: apply-fbgemm-pr.sh
# Environment: FBGEMM_PR_REPO, FBGEMM_PR_BASE, FBGEMM_PR_HEAD, E2E

set -euo pipefail

: "${FBGEMM_PR_REPO:?}" "${FBGEMM_PR_BASE:?}" "${FBGEMM_PR_HEAD:?}" "${E2E:?}"
here=$(cd "$(dirname "$0")" && pwd)
repo="$E2E/fbgemm-pr"
diff="$E2E/logs/fbgemm-pr.diff"

git init -q "$repo"
git -C "$repo" remote add origin "https://github.com/$FBGEMM_PR_REPO.git"
# Blob-less fetch: git diff downloads only the blobs it compares.
git -C "$repo" fetch -q --depth=1 --filter=blob:none origin "$FBGEMM_PR_BASE" "$FBGEMM_PR_HEAD"
git -C "$repo" diff "$FBGEMM_PR_BASE" "$FBGEMM_PR_HEAD" -- fbgemm_gpu/fbgemm_gpu > "$diff"

unexpected=$(grep '^+++ ' "$diff" | grep -vE '^\+\+\+ b/fbgemm_gpu/fbgemm_gpu/.*\.py$' || true)
if [ -n "$unexpected" ] || [ ! -s "$diff" ]; then
  echo "Expected a non-empty diff of Python files of the fbgemm_gpu package only:" >&2
  echo "$unexpected" >&2
  exit 1
fi

site=$(python -c "import fbgemm_gpu, os; print(os.path.dirname(os.path.dirname(fbgemm_gpu.__file__)))")
patch -d "$site" -p2 --forward --batch < "$diff"

python "$here/check_compute_device.py"
pip freeze > "$E2E/logs/pip-freeze.txt"
