# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Runs the DLRM-v3 MovieLens-1M preprocessor reproducibly.

Run from the directory whose tmp/ holds movielens1m.zip, with the
generative-recommenders checkout on PYTHONPATH; writes tmp/ml-1m.

The preprocessor shuffles with NumPy's global RNG, so it is seeded. It also
sorts the ratings by timestamp with pandas' default, unstable sort, and NumPy
orders equal timestamps differently depending on the CPU's SIMD level, so the
sort is made stable and the output does not depend on the machine.
"""

import numpy as np
import pandas as pd
from generative_recommenders.research.data.preprocessor import get_common_preprocessors


def main():
    sort_values = pd.DataFrame.sort_values
    pd.DataFrame.sort_values = lambda self, *args, **kwargs: sort_values(self, *args, **{**kwargs, "kind": "stable"})
    np.random.seed(0)
    get_common_preprocessors()["ml-1m"].preprocess_rating()


if __name__ == "__main__":
    main()
