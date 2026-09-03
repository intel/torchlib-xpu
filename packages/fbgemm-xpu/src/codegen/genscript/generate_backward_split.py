#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

# pyre-strict
# flake8: noqa F401

import sys

try:
    from .common import CodeTemplate
except ImportError:
    # pyre-ignore[21]
    from common import CodeTemplate


class BackwardSplitGenerator:
    @staticmethod
    def render_backward_templates(
        template_filepath: str,
        filename_format: str,
    ) -> None:
        template = CodeTemplate.load(template_filepath)
        variants = [
            (True, "dense"),
            (False, "rowwise_adagrad_split"),
        ]

        for dense, descriptor in variants:
            template.write(filename_format.format(descriptor), dense=dense)

    @staticmethod
    def generate_kernels() -> None:
        for template_filepath, filename_format in [
            (
                "training/backward/embedding_backward_split_kernel_templates.h",
                "sycl_kernels/gen_embedding_backward_{}_unweighted_nobag_kernels.h",
            ),
            (
                "training/backward/embedding_backward_nobag_unweighted_host_template.cpp",
                "sycl_kernels/gen_embedding_backward_{}_unweighted_nobag_host.cpp",
            ),
        ]:
            BackwardSplitGenerator.render_backward_templates(
                template_filepath,
                filename_format,
            )

    @staticmethod
    def generate_pt2_wrappers() -> None:
        CodeTemplate.load(
            "training/pt2/embedding_forward_nobag_unweighted_pt2_wrapper_template.cpp",
        ).write(
            "sycl_kernels/gen_embedding_backward_split_rowwise_adagrad_nobag_unweighted_pt2_xpu_wrapper.cpp",
            is_forward=False,
        )

    @staticmethod
    def generate() -> None:
        BackwardSplitGenerator.generate_kernels()
        BackwardSplitGenerator.generate_pt2_wrappers()


def main() -> None:
    BackwardSplitGenerator.generate()


if __name__ == "__main__":
    print(f"[GENERATE BACKWARD SPLIT]: {sys.argv}")
    main()
