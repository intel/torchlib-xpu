#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

# pyre-strict
# flake8: noqa F401

import itertools
import sys

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
        dense_options: list[bool],
        nobag_options: list[bool],
        vbe_options: list[bool],
        ssd_options: list[bool],
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
        CodeTemplate.load(
            "training/pt2/embedding_forward_nobag_unweighted_pt2_wrapper_template.cpp",
        ).write(
            "sycl_kernels/gen_embedding_forward_split_unweighted_nobag_pt2_wrapper.cpp",
            is_forward=True,
        )

    @staticmethod
    def generate_small_kernels() -> None:
        ForwardSplitGenerator.render_forward_templates(
            "training/forward/embedding_forward_split_kernel_nobag_small_template.h",
            "sycl_kernels/gen_embedding_forward_{}_kernel_small.h",
            dense_options=[True, False],
            nobag_options=[True],
            vbe_options=[False],
            ssd_options=[False],
        )

    @staticmethod
    def generate_kernels() -> None:
        ForwardSplitGenerator.render_forward_templates(
            "training/forward/embedding_forward_split_kernel_template.h",
            "sycl_kernels/gen_embedding_forward_{}_kernel.h",
            dense_options=[True, False],
            nobag_options=[True],
            vbe_options=[False],
            ssd_options=[False],
        )
        ForwardSplitGenerator.render_forward_templates(
            "training/forward/embedding_forward_nobag_unweighted_host_template.cpp",
            "sycl_kernels/gen_embedding_forward_{}_host.cpp",
            dense_options=[True, False],
            nobag_options=[True],
            vbe_options=[False],
            ssd_options=[False],
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
