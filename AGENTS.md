# AGENTS.md

## Mojo skills

This repo is a Mojo GPU programming course. Always load the relevant Mojo
skills before writing code. The skills correct pretrained model misconceptions
about Mojo syntax and GPU programming. Available skills:

- **mojo-syntax** — Load for any Mojo code. Core syntax, conventions, and
  corrections for common mistakes.
- **mojo-gpu-fundamentals** — Load for any GPU/accelerator code. Mojo has NO
  CUDA syntax (`__global__`, `<<<>>>`, `__shared__`, etc.). Always use this
  skill when writing kernels or using DeviceContext.
- **mojo-python-interop** — Load when bridging Mojo and Python code.
- **new-modular-project** — Load when creating new Mojo/MAX projects.

## Critical rules

1. **Never hallucinate CUDA syntax** — Mojo kernels are plain `def` functions.
   No decorators. Launch via `ctx.enqueue_function[...](...)`. See the GPU
   skill for the full concept mapping.

2. **Never hallucinate Mojo syntax** — When in doubt, load the mojo-syntax
   skill. Do not guess based on Python, Rust, or Swift influences.

3. **Check the actual API** — If code from a skill or external source doesn't
   compile, inspect the installed stdlib or mojo docs before assuming the
   skill is correct. Mojo is evolving rapidly and API surfaces change.

4. **Use `pixi run mojo` or `mojo`** — The project uses pixi. Mojo is the
   compiler/runtime. No separate build system.

5. **Pair Programming / Learning-First Approach** — This is a learning repository.
   Do not write full-file solutions or spoon-feed completed exercises. Instead, 
   act as an expert pair programming mentor: explain core concepts,
   point out syntax corrections (especially for Mojo 1.0 beta changes), and provide
   high-precision *minimal* code snippets so the user can write the code and learn.

## Project structure

- `ch05_first_kernels/` — First GPU kernels in Mojo
- `ch06_cuda_api/` — CUDA API equivalents in Mojo
- `ch07_faster_matmul/` — Optimized matrix multiplication
- `ch08_triton/` — Triton comparisons/ports
- `ch09_pytorch_extensions/` — PyTorch extension modules
- `ch10_mnist_mlp/` — MNIST MLP on GPU
- `pixi.toml` / `pixi.lock` — Project dependencies (mojo >=1.0.0b1)
