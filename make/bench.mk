# ================================================================
# Benchmark build
# ================================================================

FULL_MUL_BENCH_SRC := bench/gpu_full_multiply_benchmark.cpp
BENCH_FULL_SRCS    := $(FULL_MUL_BENCH_SRC) $(CPP_SRCS) $(CU_SRCS)

BENCH_FULL_32    := build/bench_full_multiply_32
BENCH_FULL_HYBRID := build/bench_full_multiply_hybrid
BENCH_FULL_64BIT := build/bench_full_multiply_64bit

$(BENCH_OBJ): $(BENCH_SRC) | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(BENCH_TARGET): $(BENCH_OBJ) $(CPP_OBJS) $(CU_OBJS) | $(GPU_NTT_LIB)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_PATHS) \
		-o $@ $^ $(LIBS)

$(BENCH_FULL_32): $(BENCH_FULL_SRCS) | $(OBJ_DIR) $(GPU_NTT_LIB)
	$(NVCC) $(NVCCFLAGS) -DLIMB_BITS=32 $(INCLUDES) $(LIB_PATHS) \
		$^ -o $@ $(LIBS)

$(BENCH_FULL_HYBRID): $(BENCH_FULL_SRCS) | $(OBJ_DIR) $(GPU_NTT_LIB)
	$(NVCC) $(NVCCFLAGS) -DLIMB_BITS=64 $(INCLUDES) $(LIB_PATHS) \
		$^ -o $@ $(LIBS)

$(BENCH_FULL_64BIT): $(BENCH_FULL_SRCS) | $(OBJ_DIR) $(GPU_NTT_LIB)
	$(NVCC) $(NVCCFLAGS) -DLIMB_BITS=64 -DNATIVE_HOST_LIMBS $(INCLUDES) $(LIB_PATHS) \
		$^ -o $@ $(LIBS)

# ================================================================
# GMP benchmark build (host-only, no CUDA)
# ================================================================

GMP_BENCH_SRC := bench/bench_gmp_multiply.cpp

BENCH_GMP_32     := build/bench_gmp_multiply_32
BENCH_GMP_64BIT  := build/bench_gmp_multiply_64bit

$(BENCH_GMP_32): $(GMP_BENCH_SRC) | $(OBJ_DIR)
	$(CXX) -O2 -std=c++17 $< -o $@ -lgmp

$(BENCH_GMP_64BIT): $(GMP_BENCH_SRC) | $(OBJ_DIR)
	$(CXX) -O2 -std=c++17 -DNATIVE_HOST_LIMBS $< -o $@ -lgmp

# ================================================================
# Convenience targets
# ================================================================

bench: gpu-ntt $(BENCH_TARGET)
bench_full_32: $(BENCH_FULL_32)
bench_full_hybrid: $(BENCH_FULL_HYBRID)
bench_full_64bit: $(BENCH_FULL_64BIT)
bench_full: bench_full_32 bench_full_hybrid bench_full_64bit
bench_gmp_32: $(BENCH_GMP_32)
bench_gmp_64bit: $(BENCH_GMP_64BIT)
bench_gmp: bench_gmp_32 bench_gmp_64bit

# Deprecated aliases
bench_full_64: bench_full_hybrid
bench_full_64native: bench_full_64bit

.PHONY: bench bench_full bench_full_32 bench_full_hybrid bench_full_64bit \
	bench_full_64 bench_full_64native bench_gmp bench_gmp_32 bench_gmp_64bit
