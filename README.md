# birdwing

GPU-accelerated large-integer multiplication using the Number Theoretic Transform (NTT) in a Residue Number System (RNS), with Chinese Remainder Theorem (CRT) reconstruction and segmented carry propagation.

For architecture, module layout, and design notes, see [ROADMAP.md](ROADMAP.md).

## Requirements

| Tool | Version / notes |
|------|-----------------|
| NVIDIA GPU + CUDA Toolkit | `nvcc` on `PATH` (or set `CUDA_PATH`) |
| CMake | >= 3.26 |
| C++ compiler | GCC or Clang with C++17 |
| GMP | Required for tests and GMP comparison benchmarks |

On Ubuntu/Debian:

```bash
sudo apt install cmake g++ libgmp-dev
# Install CUDA from https://developer.nvidia.com/cuda-downloads
```

On macOS (Homebrew):

```bash
brew install cmake gmp
# Install CUDA separately if not already present
```

## Clone

GPU-NTT is vendored as a git submodule under `GPU-NTT/`.

```bash
git clone --recurse-submodules <repo-url>
cd ntt_cgbn
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

## Build

The top-level `make` builds the vendored GPU-NTT library automatically, then links this project against it.

```bash
make
```

Set your GPU architecture if CMake cannot auto-detect it (example: RTX 3080 = `86`):

```bash
make CMAKE_CUDA_ARCHITECTURES=86
```

Other useful overrides:

```bash
make CUDA_PATH=/usr/local/cuda          # non-default CUDA install
make GMP_PREFIX=$HOME/gmp-local         # custom GMP prefix
make DEBUG=1                            # extra checks
make TIMING=1                           # per-stage CUDA timing
make PROFILE=1                          # debug symbols for profiling
```

GPU-NTT is built into `GPU-NTT/build/` as a static library (`libntt-1.0.a`). To rebuild only the dependency:

```bash
make gpu-ntt
```

## Run

```bash
make test              # unit and integration tests
make bench             # NTT round-trip benchmark
make bench_full        # end-to-end multiply benchmarks (32- and 64-bit)
./main                 # CLI driver
```

Run a single test:

```bash
make test TEST=test_zero_pad_32
```

## GPU-NTT submodule

This repo pins [GPU-NTT](https://github.com/Alisah-Ozcan/GPU-NTT) at the commit recorded in the parent repository. The submodule provides merge and 4-step NTT kernels used by `src/gpu_ntt.cu`.

To update the submodule to a newer upstream commit:

```bash
cd GPU-NTT
git fetch origin
git checkout <commit-or-tag>
cd ..
git add GPU-NTT
```

## Troubleshooting

**CMake cannot find `nvcc`**

```bash
export PATH=/usr/local/cuda/bin:$PATH
# or
make CUDA_PATH=/usr/local/cuda
```

**Link errors for GMP**

Install the development package (`libgmp-dev` on Debian/Ubuntu), or point to a custom install:

```bash
make GMP_PREFIX=/path/to/gmp
```

**Wrong GPU architecture / kernel launch failures**

Pass an explicit architecture matching your GPU:

```bash
make clean
make CMAKE_CUDA_ARCHITECTURES=89   # example: Ada Lovelace
```
