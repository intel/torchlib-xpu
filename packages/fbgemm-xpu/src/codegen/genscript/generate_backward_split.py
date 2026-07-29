#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to generate SYCL backward kernels from Jinja2 templates.

Usage:
    python generate_backward_split.py [--template-dir <dir>] [--output-dir <dir>]

Example:
    python generate_backward_split.py \
        --template-dir codegen/training/backward \
        --output-dir src/fbgemm_xpu/sycl_kernels/
"""

import argparse
import sys
from pathlib import Path

try:
    from jinja2 import Environment, FileSystemLoader, Template, select_autoescape
except ImportError:
    print("Error: jinja2 not installed. Install with: pip install jinja2")
    sys.exit(1)


def generate_kernel(template: Template, dense: bool, output_path: Path, template_name: str = "") -> None:
    """
    Generate a kernel variant and write to file.
    
    Args:
        template: Jinja2 template object
        dense: If True, generate dense kernel; if False, generate split kernel
        output_path: Path to write generated code
        template_name: Name of the template source file (used in file header comment)
    """
    variant_name = "dense" if dense else "split_rowwise_adagrad"
    print(f"Generating {variant_name} kernel...")
    
    # Render template with parameters
    output = template.render(dense=dense)
    
    # Prepend generated file header comment
    if template_name:
        header = (
            "////////////////////////////////////////////////////////////////////////////////\n"
            "// GENERATED FILE INFO\n"
            "//\n"
            f"// Template Source: training/backward/{template_name}\n"
            "////////////////////////////////////////////////////////////////////////////////\n"
            "\n\n"
        )
        output = header + output
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(output)
    
    print(f"  ✓ Written to: {output_path}")
    print(f"  ✓ Lines: {len(output.splitlines())}")


def generate_pt2_wrapper(template: Template, is_forward: bool, output_path: Path, template_name: str = "") -> None:
    """
    Generate a pt2 wrapper variant (forward or backward) and write to file.

    Args:
        template: Jinja2 template object
        is_forward: If True, generate forward wrapper; if False, generate backward wrapper
        output_path: Path to write generated code
        template_name: Name of the template source file (used in file header comment)
    """
    direction = "forward" if is_forward else "backward"
    print(f"Generating {direction} pt2 wrapper...")

    output = template.render(is_forward=is_forward)

    if template_name:
        header = (
            "////////////////////////////////////////////////////////////////////////////////\n"
            "// GENERATED FILE INFO\n"
            "//\n"
            f"// Template Source: training/pt2/{template_name}\n"
            "////////////////////////////////////////////////////////////////////////////////\n"
            "\n\n"
        )
        output = header + output

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(output)

    print(f"  ✓ Written to: {output_path}")
    print(f"  ✓ Lines: {len(output.splitlines())}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate SYCL backward kernels from Jinja2 template'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./generated',
        help='Output directory for generated kernels (default: ./generated)'
    )
    parser.add_argument(
        '--template-dir',
        type=str,
        default='.',
        help='Directory containing template file (default: current directory)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print output to stdout instead of writing files'
    )
    
    args = parser.parse_args()
    
    # Setup Jinja2 environment
    template_dir = Path(args.template_dir)
    sycl_template_file = 'embedding_backward_split_kernel_templates.h'
    host_sycl_template_file = 'embedding_backward_nobag_unweighted_host_template.cpp'
    pt2_template_dir = template_dir.parent / "pt2"
    pt2_sycl_template_file = 'embedding_forward_nobag_unweighted_pt2_wrapper_template.cpp'

    for tf in (sycl_template_file, host_sycl_template_file):
        if not (template_dir / tf).exists():
            print(f"Error: Template not found at {template_dir / tf}")
            sys.exit(1)
    if not (pt2_template_dir / pt2_sycl_template_file).exists():
        print(f"Error: Template not found at {pt2_template_dir / pt2_sycl_template_file}")
        sys.exit(1)

    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(
            enabled_extensions=("html", "htm", "xml"),
            default_for_string=False,
            default=False,
        ),
        trim_blocks=True,
        lstrip_blocks=True
    )

    sycl_template = env.get_template(sycl_template_file)
    host_sycl_template = env.get_template(host_sycl_template_file)

    pt2_env = Environment(
        loader=FileSystemLoader(str(pt2_template_dir)),
        autoescape=select_autoescape(
            enabled_extensions=("html", "htm", "xml"),
            default_for_string=False,
            default=False,
        ),
        trim_blocks=True,
        lstrip_blocks=True
    )
    pt2_sycl_template = pt2_env.get_template(pt2_sycl_template_file)

    # Define output variants: each produces a .h header and a .cpp implementation
    output_dir = Path(args.output_dir)
    variants = [
        {
            'dense': True,
            'sycl_filename': 'gen_embedding_backward_dense_unweighted_nobag_kernels.h',
            'host_sycl_filename': 'gen_embedding_backward_dense_unweighted_nobag_host.cpp',
            'description': 'Dense backward kernel (gradient computation only)',
        },
        {
            'dense': False,
            'sycl_filename': 'gen_embedding_backward_rowwise_adagrad_split_unweighted_nobag_kernels.h',
            'host_sycl_filename': 'gen_embedding_backward_rowwise_adagrad_split_unweighted_nobag_host.cpp',
            'pt2_fwd_sycl_filename': 'gen_embedding_forward_split_nobag_unweighted_pt2_xpu_wrapper.cpp',
            'pt2_bwd_sycl_filename': 'gen_embedding_backward_split_rowwise_adagrad_nobag_unweighted_pt2_xpu_wrapper.cpp',
            'description': 'Split backward kernel with Rowwise Adagrad optimizer',
        },
    ]

    print("=" * 70)
    print("SYCL Backward Kernel Generator")
    print("=" * 70)
    print(f"SYCL template:        {sycl_template_file}")
    print(f"Host SYCL template:   {host_sycl_template_file}")
    print(f"PT2 SYCL template:    {pt2_sycl_template_file}")
    print(f"Output directory: {output_dir}")
    print(f"Dry run: {args.dry_run}")
    print()

    # Generate each variant
    for variant in variants:
        print(f"\n{variant['description']}")
        print("-" * 70)

        if args.dry_run:
            # Print first 500 chars of each file to stdout
            for tmpl, label in (
                (sycl_template, '.cpp'),
                (host_sycl_template, '_host.cpp'),
            ):
                output = tmpl.render(dense=variant['dense'])
                print(f"--- {label} ---")
                print(output[:500] + "\n... (truncated) ...\n")
            if 'pt2_fwd_sycl_filename' in variant:
                for is_fwd, label in ((True, 'pt2_fwd'), (False, 'pt2_bwd')):
                    output = pt2_sycl_template.render(is_forward=is_fwd)
                    print(f"--- {label} ---")
                    print(output[:500] + "\n... (truncated) ...\n")
        else:
            # Write sycl kernel implementation file
            generate_kernel(
                sycl_template,
                variant['dense'],
                output_dir / variant['sycl_filename'],
                template_name=sycl_template_file,
            )
            # Write host sycl implementation file
            generate_kernel(
                host_sycl_template,
                variant['dense'],
                output_dir / variant['host_sycl_filename'],
                template_name=host_sycl_template_file,
            )
            print("Here")
            # Write pt2 wrapper files (forward and backward) for split variants
            if 'pt2_fwd_sycl_filename' in variant:
                print("Generating PT2 wrapper files...")
                generate_pt2_wrapper(
                    pt2_sycl_template,
                    False,
                    output_dir / variant['pt2_bwd_sycl_filename'],
                    template_name=pt2_sycl_template_file,
                )

    print("\n" + "=" * 70)
    if not args.dry_run:
        print("✓ All kernels generated successfully!")
        print(f"\nGenerated files in: {output_dir}")
        print("\nGenerated files:")
        for variant in variants:
            print(f"  {variant['sycl_filename']}")
            print(f"  {variant['host_sycl_filename']}")
            if 'pt2_fwd_sycl_filename' in variant:
                print(f"  {variant['pt2_bwd_sycl_filename']}")
        print("\nNext steps:")
        print("  1. Include generated .h and .cpp files in your CMakeLists.txt")
        print("  2. Register kernels in operator dispatch")
    else:
        print("Dry run complete. No files were written.")
    print("=" * 70)


def validate_template():
    """
    Validate template syntax without generating output.
    Useful for CI/CD pipelines.
    """
    templates = [
        'embedding_backward_split_kernel_warp_template.h',
        'embedding_backward_split_kernel_warp_template.cpp',
    ]

    env = Environment(
        loader=FileSystemLoader('.'),
        autoescape=select_autoescape(
            enabled_extensions=("html", "htm", "xml"),
            default_for_string=False,
            default=False,
        ),
    )

    for template_file in templates:
        if not Path(template_file).exists():
            print(f"Error: Template not found: {template_file}")
            return False

        try:
            template = env.get_template(template_file)

            # Try rendering both variants
            for dense in [True, False]:
                output = template.render(dense=dense)
                if not output or len(output) < 100:
                    print(f"Error: Template rendered empty output for dense={dense}: {template_file}")
                    return False

            print(f"✓ Template validation passed: {template_file}")
        except Exception as e:
            print(f"Error: Template validation failed for {template_file}: {e}")
            return False

    return True


if __name__ == '__main__':
    # Check if running validation mode
    if len(sys.argv) > 1 and sys.argv[1] == '--validate':
        success = validate_template()
        sys.exit(0 if success else 1)
    
    main()
