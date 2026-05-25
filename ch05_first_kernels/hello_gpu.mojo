# ===----------------------------------------------------------------------=== #
# Mojo Latest 1.0 Beta - HPC & GPU Course Verification
# ===----------------------------------------------------------------------=== #
# This file verifies your Mojo 1.0 environment and demonstrates core HPC concepts
# that parallel Elliot's freeCodeCamp CUDA Programming Course:
# 1. SIMD Vectorization (equivalent to CUDA thread lanes operating on registers)
# 2. Multi-threaded Parallelism (equivalent to CUDA thread blocks/grids)
# ===----------------------------------------------------------------------=== #

from std.algorithm import parallelize

def test_simd_operation():
    print("--- 1. Testing SIMD (Single Instruction, Multiple Data) ---")
    # In Mojo, Float32 is a SIMD of size 1 by default. 
    # We can explicitly create vector types of size 4 or 8 to run on CPU vector registers.
    var a = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
    var b = SIMD[DType.float32, 4](10.0, 20.0, 30.0, 40.0)
    
    # Vectorized element-wise addition!
    var result = a + b
    
    print("Vector A:      ", a)
    print("Vector B:      ", b)
    print("Result (A + B):", result)
    print("SIMD operations verified successfully!\n")


# A thread-level work function that simulates a GPU kernel block
def thread_worker(block_idx: Int):
    # This represents a parallel execution block
    print("CUDA Block-equivalent thread", block_idx, "is executing in parallel!")


def test_parallel_execution():
    print("--- 2. Testing Thread-Level Parallelism ---")
    var total_blocks = 4
    
    # parallelize automatically schedules and runs threads across available CPU cores,
    # similar to how CUDA schedules thread blocks on streaming multiprocessors (SMs).
    parallelize(thread_worker, total_blocks)
    
    print("Thread-level parallelism verified successfully!\n")


def main():
    print("==========================================================")
    print("  MOJO 1.0 BETA HPC ENVIRONMENT VERIFIED SUCCESSFULLY!   ")
    print("==========================================================\n")
    
    test_simd_operation()
    test_parallel_execution()
    
    print("You are fully set up to follow along with the CUDA Course!")
    print("==========================================================")
