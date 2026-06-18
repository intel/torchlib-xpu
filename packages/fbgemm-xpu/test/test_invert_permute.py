# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Test suite for invert_permute custom operator

This module tests the correctness of the invert_permute operator
on XPU (Intel GPU) devices.

Tests are designed to match TorchRec/DLRM sparse embedding scenarios.
"""

import torch
import unittest
import fbgemm_gpu
import fbgemm_xpu
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.optests import opcheck

# Reproducible seed for all random tests
SEED = 42


def generate_random_permutation(n, device="cpu", dtype=torch.int32, seed=None):
    """
    Generate a random permutation of [0, 1, ..., n-1]
    
    Args:
        n: Size of the permutation
        device: Device to place the tensor on
        dtype: Data type (torch.int32 or torch.int64)
        seed: Random seed for reproducibility
        
    Returns:
        Random permutation tensor
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    permute = torch.arange(n, dtype=dtype, device=device)
    # Generate random permutation using randperm
    indices = torch.randperm(n, device=device)
    return permute[indices]


def compute_inverse_reference(permute):
    """
    Reference CPU implementation of inverse permutation
    
    This is the ground truth implementation used for testing.
    
    Args:
        permute: Input permutation tensor
        
    Returns:
        Inverse permutation tensor
    """
    n = permute.size(0)
    inversed = torch.empty_like(permute)
    
    permute_cpu = permute.cpu()
    inversed_cpu = inversed.cpu()
    
    for i in range(n):
        inversed_cpu[permute_cpu[i]] = i
    
    return inversed_cpu.to(permute.device)


def verify_inverse(permute, inversed):
    """
    Verify that inversed is truly the inverse of permute
    
    Checks that: inversed[permute[i]] == i for all i
    
    Args:
        permute: Original permutation
        inversed: Claimed inverse permutation
        
    Returns:
        True if inversed is correct, False otherwise
    """
    n = permute.size(0)
    for i in range(n):
        if inversed[permute[i]] != i:
            return False
    return True


class TestInvertPermute(TestCase):
    """Test cases for invert_permute operator"""
    
    def _test_correctness(self, device, dtype):
        """
        Helper method to test correctness on a specific device and dtype
        
        Args:
            device: Device string ("cpu" or "xpu")
            dtype: torch dtype (torch.int32 or torch.int64)
        """
        # Test case 1: Empty tensor
        permute_empty = torch.tensor([], dtype=dtype, device=device)
        result_empty = torch.ops.fbgemm.invert_permute(permute_empty)
        self.assertEqual(result_empty.shape, permute_empty.shape)
        self.assertEqual(result_empty.dtype, dtype)
        
        # Test case 2: Single element [0]
        permute_single = torch.tensor([0], dtype=dtype, device=device)
        result_single = torch.ops.fbgemm.invert_permute(permute_single)
        expected_single = torch.tensor([0], dtype=dtype, device=device)
        torch.testing.assert_close(result_single, expected_single)
        self.assertTrue(verify_inverse(permute_single, result_single))
        
        # Test case 3: Small permutation [2, 0, 1]
        permute_small = torch.tensor([2, 0, 1], dtype=dtype, device=device)
        result_small = torch.ops.fbgemm.invert_permute(permute_small)
        expected_small = torch.tensor([1, 2, 0], dtype=dtype, device=device)
        torch.testing.assert_close(result_small, expected_small)
        self.assertTrue(verify_inverse(permute_small, result_small))
        
        # Test case 4: Identity permutation [0, 1, 2, ..., 99]
        n = 100
        permute_identity = torch.arange(n, dtype=dtype, device=device)
        result_identity = torch.ops.fbgemm.invert_permute(permute_identity)
        torch.testing.assert_close(result_identity, permute_identity)
        self.assertTrue(verify_inverse(permute_identity, result_identity))
        
        # Test case 5: Reverse permutation [99, 98, ..., 1, 0]
        permute_reverse = torch.arange(n - 1, -1, -1, dtype=dtype, device=device)
        result_reverse = torch.ops.fbgemm.invert_permute(permute_reverse)
        torch.testing.assert_close(result_reverse, permute_reverse)
        self.assertTrue(verify_inverse(permute_reverse, result_reverse))
        
        # Test case 6: Medium random permutation (N=1000)
        n = 1000
        permute_medium = generate_random_permutation(n, device=device, dtype=dtype)
        result_medium = torch.ops.fbgemm.invert_permute(permute_medium)
        expected_medium = compute_inverse_reference(permute_medium)
        torch.testing.assert_close(result_medium, expected_medium)
        self.assertTrue(verify_inverse(permute_medium, result_medium))
        
        # Test case 7: Large random permutation (N=100000)
        n = 100000
        permute_large = generate_random_permutation(n, device=device, dtype=dtype)
        result_large = torch.ops.fbgemm.invert_permute(permute_large)
        expected_large = compute_inverse_reference(permute_large)
        torch.testing.assert_close(result_large, expected_large)
        self.assertTrue(verify_inverse(permute_large, result_large))
    
    @unittest.skipIf(not torch.xpu.is_available(), "XPU not available")
    def test_correctness_xpu_int32(self):
        """Test correctness on XPU (Intel GPU) with int32 dtype"""
        self._test_correctness("xpu", torch.int32)
    
    @unittest.skipIf(not torch.xpu.is_available(), "XPU not available")
    def test_correctness_xpu_int64(self):
        """Test correctness on XPU (Intel GPU) with int64 dtype"""
        self._test_correctness("xpu", torch.int64)
    
    def test_invalid_input_2d(self):
        """Test that 2D input raises an error"""
        permute_2d = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device="xpu")
        with self.assertRaisesRegex(RuntimeError, "must be 1-dimensional"):
            torch.ops.fbgemm.invert_permute(permute_2d)
    
    def test_invalid_dtype_float(self):
        """Test that float dtype raises an error"""
        permute_float = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32, device="xpu")
        with self.assertRaisesRegex(RuntimeError, "must be int32 or int64"):
            torch.ops.fbgemm.invert_permute(permute_float)
    
    @unittest.skipIf(not torch.xpu.is_available(), "XPU not available")
    def test_meta_function(self):
        """Test that meta function works correctly for torch.compile"""
        device = "xpu"
        dtype = torch.int32
        n = 100
        
        permute = generate_random_permutation(n, device=device, dtype=dtype)
        
        # Create a FakeTensor to test meta function
        from torch._subclasses.fake_tensor import FakeTensorMode
        
        with FakeTensorMode():
            fake_permute = torch.empty(n, dtype=dtype, device=device)
            # This should use the meta function without actual computation
            fake_result = torch.ops.fbgemm.invert_permute(fake_permute)
            
            # Check output properties
            self.assertEqual(fake_result.shape, permute.shape)
            self.assertEqual(fake_result.dtype, dtype)
            self.assertEqual(fake_result.device.type, device)
    
    def _test_opcheck(self, device, dtype):
        """
        Test operator with PyTorch's opcheck utility
        
        opcheck validates that the operator follows PyTorch's conventions
        for autograd, vmap, and other advanced features.
        """
        n = 100
        permute = generate_random_permutation(n, device=device, dtype=dtype)
        
        # opcheck validates the operator implementation
        # It checks schema, device handling, and other PyTorch conventions
        opcheck(torch.ops.fbgemm.invert_permute, (permute,))
    
    @unittest.skipIf(not torch.xpu.is_available(), "XPU not available")
    def test_opcheck_xpu_int32(self):
        """Run opcheck validation on XPU with int32"""
        self._test_opcheck("xpu", torch.int32)
    
    @unittest.skipIf(not torch.xpu.is_available(), "XPU not available")
    def test_opcheck_xpu_int64(self):
        """Run opcheck validation on XPU with int64"""
        self._test_opcheck("xpu", torch.int64)


class TestInvertPermuteParametric(TestCase):
    """
    Parametric tests with varying sizes and edge cases
    
    These tests cover TorchRec/DLRM-like scenarios with reproducible seeds.
    """
    
    @unittest.skipIf(not torch.xpu.is_available(), "XPU not available")
    def test_varying_sizes_xpu(self):
        """Test various tensor sizes on XPU"""
        torch.manual_seed(SEED)
        
        test_sizes = [1, 10, 128, 1024, 8192, 65536, 100000]
        
        for n in test_sizes:
            with self.subTest(size=n):
                permute = generate_random_permutation(n, device="xpu", dtype=torch.int32, seed=SEED + n)
                result = torch.ops.fbgemm.invert_permute(permute)
                expected = compute_inverse_reference(permute)
                torch.testing.assert_close(result, expected)
                self.assertTrue(verify_inverse(permute, result))
    
    @unittest.skipIf(not torch.xpu.is_available(), "XPU not available")
    def test_torchrec_batch_shapes_xpu(self):
        """Test with TorchRec shapes on XPU"""
        torch.manual_seed(SEED)
        
        torchrec_shapes = [100, 1000, 10000, 100000]
        
        for num_embeddings in torchrec_shapes:
            with self.subTest(num_embeddings=num_embeddings):
                permute = generate_random_permutation(
                    num_embeddings, device="xpu", dtype=torch.int64, seed=SEED
                )
                result = torch.ops.fbgemm.invert_permute(permute)
                expected = compute_inverse_reference(permute)
                torch.testing.assert_close(result, expected)
    
    @unittest.skipIf(not torch.xpu.is_available(), "XPU not available")
    def test_power_of_two_sizes_xpu(self):
        """Test power-of-2 sizes on XPU"""
        torch.manual_seed(SEED)
        
        for power in [4, 8, 10, 12, 14, 16]:
            n = 2 ** power
            with self.subTest(size=n, power=power):
                permute = generate_random_permutation(n, device="xpu", dtype=torch.int32, seed=SEED)
                result = torch.ops.fbgemm.invert_permute(permute)
                self.assertTrue(verify_inverse(permute, result))


class TestCPUXPUParity(TestCase):
    """Test that CPU and XPU implementations produce identical results"""
    
    @unittest.skipIf(not torch.xpu.is_available(), "XPU not available")
    def test_parity_small(self):
        """Test CPU-XPU parity for small inputs"""
        torch.manual_seed(SEED)
        
        for n in [10, 100, 1000]:
            with self.subTest(size=n):
                permute_cpu = generate_random_permutation(n, device="cpu", dtype=torch.int32, seed=SEED)
                permute_xpu = permute_cpu.to("xpu")
                
                result_cpu = torch.ops.fbgemm.invert_permute(permute_cpu)
                result_xpu = torch.ops.fbgemm.invert_permute(permute_xpu)
                
                # CPU and XPU should produce identical results
                torch.testing.assert_close(result_cpu, result_xpu.cpu())
    
    @unittest.skipIf(not torch.xpu.is_available(), "XPU not available")
    def test_parity_large(self):
        """Test CPU-XPU parity for large inputs"""
        torch.manual_seed(SEED)
        
        for n in [10000, 100000]:
            with self.subTest(size=n):
                permute_cpu = generate_random_permutation(n, device="cpu", dtype=torch.int64, seed=SEED)
                permute_xpu = permute_cpu.to("xpu")
                
                result_cpu = torch.ops.fbgemm.invert_permute(permute_cpu)
                result_xpu = torch.ops.fbgemm.invert_permute(permute_xpu)
                
                torch.testing.assert_close(result_cpu, result_xpu.cpu())
    
    @unittest.skipIf(not torch.xpu.is_available(), "XPU not available")
    def test_parity_edge_cases(self):
        """Test CPU-XPU parity for edge cases"""
        torch.manual_seed(SEED)
        
        # Empty tensor
        permute_empty = torch.tensor([], dtype=torch.int32)
        result_cpu_empty = torch.ops.fbgemm.invert_permute(permute_empty)
        result_xpu_empty = torch.ops.fbgemm.invert_permute(permute_empty.to("xpu"))
        torch.testing.assert_close(result_cpu_empty, result_xpu_empty.cpu())
        
        # Single element
        permute_single = torch.tensor([0], dtype=torch.int32)
        result_cpu_single = torch.ops.fbgemm.invert_permute(permute_single)
        result_xpu_single = torch.ops.fbgemm.invert_permute(permute_single.to("xpu"))
        torch.testing.assert_close(result_cpu_single, result_xpu_single.cpu())
        
        # Identity
        permute_identity = torch.arange(100, dtype=torch.int32)
        result_cpu_identity = torch.ops.fbgemm.invert_permute(permute_identity)
        result_xpu_identity = torch.ops.fbgemm.invert_permute(permute_identity.to("xpu"))
        torch.testing.assert_close(result_cpu_identity, result_xpu_identity.cpu())


class TestInvertPermutePerformance(TestCase):
    """Performance benchmarking for invert_permute operator"""
    
    @unittest.skipIf(not torch.xpu.is_available(), "XPU not available")
    def test_benchmark_xpu(self):
        """
        Benchmark invert_permute on XPU
        
        This test measures execution time for different input sizes.
        It's marked as a test but primarily serves as a benchmarking tool.
        """
        import time
        
        print("\n=== XPU Performance Benchmark ===")
        sizes = [1000, 10000, 100000, 1000000]
        num_iterations = 100
        
        for n in sizes:
            for dtype in [torch.int32, torch.int64]:
                permute = generate_random_permutation(n, device="xpu", dtype=dtype, seed=SEED)
                
                # Warmup
                for _ in range(10):
                    _ = torch.ops.fbgemm.invert_permute(permute)
                
                # Benchmark
                torch.xpu.synchronize()
                start = time.perf_counter()
                
                for _ in range(num_iterations):
                    result = torch.ops.fbgemm.invert_permute(permute)
                
                torch.xpu.synchronize()
                end = time.perf_counter()
                
                avg_time_ms = (end - start) * 1000 / num_iterations
                bytes_transferred = 2 * n * (4 if dtype == torch.int32 else 8)
                bandwidth_gb_s = (bytes_transferred / (avg_time_ms / 1000)) / (1024**3)
                
                dtype_str = "int32" if dtype == torch.int32 else "int64"
                print(f"N={n:>7} ({dtype_str}): {avg_time_ms:.4f} ms, "
                      f"Bandwidth: {bandwidth_gb_s:.2f} GB/s")


if __name__ == "__main__":
    run_tests()
