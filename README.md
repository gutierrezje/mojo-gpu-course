# Mojo GPU Programming Course

This repository contains Mojo ports and adaptations of the exercises, kernels, and concepts from the popular **[Infatoshi CUDA Course](https://github.com/Infatoshi/cuda-course)** (offered in partnership with FreeCodeCamp).

Rather than using traditional C++ CUDA or Python, this course explores highly parallel accelerator programming using **Mojo**, a systems programming language designed for AI and high-performance computing.

## Key Differences: CUDA vs. Mojo

Unlike raw CUDA C/C++ programming:
- **No Special Decorators**: Mojo GPU kernels are defined using plain `def` functions.
- **Explicit Host Orchestration**: Kernels are compiled and enqueued using standard host-side objects like `DeviceContext` and compiled via `ctx.enqueue_function[...]`.
- **Modern Package Management**: The entire workspace is fully managed by [Pixi](https://pixi.sh), providing isolated and reproducible builds.

## Project Structure

- [ch05_first_kernels/](file:///home/jesus/learn/fcc-cuda/ch05_first_kernels) — Initial GPU kernels, parallel index mapping, and coordinate math.
- [ch06_cuda_api/](file:///home/jesus/learn/fcc-cuda/ch06_cuda_api) — Mapping standard CUDA API patterns into Mojo.
- [ch07_faster_matmul/](file:///home/jesus/learn/fcc-cuda/ch07_faster_matmul) — Optimized matrix multiplications.
- [ch08_triton/](file:///home/jesus/learn/fcc-cuda/ch08_triton) — Triton ports and architectural comparisons.
- [ch09_pytorch_extensions/](file:///home/jesus/learn/fcc-cuda/ch09_pytorch_extensions) — Bridging custom Mojo GPU kernels with PyTorch.
- [ch10_mnist_mlp/](file:///home/jesus/learn/fcc-cuda/ch10_mnist_mlp) — Full Multi-Layer Perceptron trained on MNIST using Mojo GPU.

## Getting Started

### 1. Installation
This repository uses the Pixi package manager to automatically isolate all dependencies (including the Mojo runtime).

If you don't have Pixi installed, run:
```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

### 2. Initialize Environment & Install Dependencies
Run the install command inside the repository root to download and set up the Mojo compiler and required packages:
```bash
pixi install
```

### 3. Running a Kernel
You can run Mojo files in two ways:

```bash
pixi run mojo ch05_first_kernels/hello_gpu.mojo
```
or
```bash
pixi shell
mojo ch05_first_kernels/hello_gpu.mojo
```

---