# CUDA Batch Image Processor (Shared-Memory 2D Tiling)

A high-throughput GPU batch image processing pipeline written in CUDA C++ (following the Google C++ Style Guide). It processes hundreds of images in a single execution using 2D `__shared__` memory tiling with halo padding to perform **Gaussian denoising** followed by **Sobel edge detection**.

## Project Structure

- `src/image_processor.h` — Data structures, filter mode enums, and public API declarations.
- `src/image_processor.cu` — CUDA kernels (`GaussianDenoiseSharedKernel`, `SobelEdgeDetectSharedKernel`), CUDA event timing, and CLI runner.
- `scripts/generate_dataset.py` — Generates 200 synthetic noisy `256x256` PGM images and exports viewable PNG comparisons.
- `Makefile` — Builds `bin/image_processor` with `nvcc`.
- `run.sh` — End-to-end script that builds, generates 200 images, runs GPU filtering, logs metrics, and creates `proof_of_execution.tar.gz`.

## Build & Run

### Quick Start (Automated)
```bash
chmod +x run.sh
./run.sh
