#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

# pyre-strict
# flake8: noqa F401

import argparse
import itertools
import sys
from typing import List

try:
    from .common import CodeTemplate
except ImportError:
    # pyre-ignore[21]
    from common import CodeTemplate


class ForwardSplitGenerator:
    @staticmethod
    def render_forward_templates(
        template_filepath: str,
        filename_format: str,
        dense_options: List[bool],
        nobag_options: List[bool],
        vbe_options: List[bool],
        ssd_options: List[bool],
        is_gwd: bool = False,
    ) -> None:
        template = CodeTemplate.load(template_filepath)
        weighted_options = [True, False]

        for dense, weighted, nobag, vbe, ssd in itertools.product(
            dense_options, weighted_options, nobag_options, vbe_options, ssd_options
        ):
            if nobag and (weighted or vbe):
                continue
            if dense and ssd:
                continue
            if ssd and is_gwd:
                continue

            desc = "".join(
                [
                    f"{ 'dense' if dense else ('ssd' if ssd else 'split') }",
                    f"{ '_weighted' if weighted else '_unweighted' }",
                    f"{ '_nobag' if nobag else '' }",
                    f"{ '_vbe' if vbe else '' }",
                ]
            )
            fname = filename_format.format(desc)
            template.write(
                fname,
                dense=dense,
                weighted=weighted,
                nobag=nobag,
                vbe=vbe,
                ssd=ssd,
                is_index_select=False,
                is_gwd=is_gwd,
            )

    @staticmethod
    def generate_pt2_wrappers() -> None:
        pt2_impl_template = CodeTemplate.load(
            "training/pt2/embedding_forward_nobag_unweighted_pt2_wrapper_template.cpp",
        )
            
        # Generate implementation file
        pt2_impl_template.write(
            f"sycl_kernels/gen_embedding_forward_split_unweighted_nobag_pt2_wrapper.cpp",
            is_forward=True,
        )


    @staticmethod
    def generate_small_kernels() -> None:
        # Generate the SYCL small kernel headers (for nobag only).
        # The operator() implementation is now inlined in the header template,
        # so no separate .cpp file is generated.
        sycl_header_template = CodeTemplate.load(
            "training/forward/embedding_forward_split_kernel_nobag_small_template.h"
        )
        for dense in [True, False]:
            ddesc = f"{ 'dense' if dense else 'split' }"
            sycl_header_template.write(
                f"sycl_kernels/gen_embedding_forward_{ ddesc }_unweighted_nobag_kernel_small.h",
                dense=dense,
                is_index_select=False,
            )


    @staticmethod
    def generate_kernels() -> None:
        """
        Generate the SYCL general D kernels (for nobag unweighted, all embedding dimensions)
        These are the main kernel implementations that handle general embedding dimensions.
        Small kernels (D<=32) are generated separately by generate_small_kernels().
        The kernel implementation (operator()) is inlined in the .h header file.
        """
        # Load the unified header template (contains both class definition and operator() impl)
        sycl_header_template = CodeTemplate.load(
            "training/forward/embedding_forward_split_kernel_template.h"
        )
        
        # Generate both dense and split variants
        for dense in [True, False]:
            ddesc = f"{ 'dense' if dense else 'split' }"
            
            # Generate .h header file (contains class definition + inline operator() impl)
            sycl_header_template.write(
                f"sycl_kernels/gen_embedding_forward_{ ddesc }_unweighted_nobag_kernel.h",
                dense=dense,
                ssd=False,
                is_index_select=False,
            )

        # Load the unified templates for general D kernels
        sycl_template = CodeTemplate.load(
            "training/forward/embedding_forward_nobag_unweighted_host_template.cpp"
        )

        for dense in [True, False]:
            ddesc = f"{ 'dense' if dense else 'split' }"
            sycl_template.write(
                f"sycl_kernels/gen_embedding_forward_{ ddesc }_unweighted_nobag_host.cpp",
                dense=dense,
            )

    @staticmethod
    def generate() -> None:
        ForwardSplitGenerator.generate_kernels()
        ForwardSplitGenerator.generate_small_kernels()
        ForwardSplitGenerator.generate_pt2_wrappers()


def main() -> None:
    ForwardSplitGenerator.generate()


if __name__ == "__main__":
    print(f"[GENERATE FORWARD SPLIT]: {sys.argv}")
    main()
