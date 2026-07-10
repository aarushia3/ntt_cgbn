# ================================================================
# Test build rules
# ================================================================

# Build C++ tests
$(TEST_BUILD)/%: $(TEST_DIR)/%.cpp $(CPP_OBJS) $(CU_OBJS) | $(TEST_BUILD) $(GPU_NTT_LIB)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) \
		$< $(CPP_OBJS) $(CU_OBJS) \
		-o $@ $(LIB_PATHS) $(LIBS)

# Build CUDA tests
$(TEST_BUILD)/%: $(TEST_DIR)/%.cu $(CPP_OBJS) $(CU_OBJS) | $(TEST_BUILD) $(GPU_NTT_LIB)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) \
		$< $(CPP_OBJS) $(CU_OBJS) \
		-o $@ $(LIB_PATHS) $(LIBS)

# ================================================================
# Run tests
# ================================================================

# Run all:
#   make test
#
# Run one:
#   make test TEST=test_name

test: gpu-ntt $(TEST_BINS)
ifdef TEST
	@echo "Running test: $(TEST)"
	@$(TEST_BUILD)/$(TEST)
else
	@echo "Running all tests..."
	@for t in $(TEST_BINS); do \
		echo "===== $$t ====="; \
		$$t || exit 1; \
	done
	@echo "All tests passed!"
endif

# ================================================================
# zero_pad dual-width tests
# ================================================================

ZP_SRC      := $(TEST_DIR)/test_zero_pad.cu
ZP_SRCS_DEP := $(SRC_DIR)/zero_pad.cu

$(TEST_BUILD)/test_zero_pad_32: $(ZP_SRC) $(ZP_SRCS_DEP) | $(TEST_BUILD) $(GPU_NTT_LIB)
	$(NVCC) $(NVCCFLAGS) -DLIMB_BITS=32 $(INCLUDES) \
		$^ -o $@ $(LIB_PATHS) $(LIBS)

NATIVE_FLAGS := -DLIMB_BITS=64 -DNATIVE_HOST_LIMBS
# Pipeline naming: 32-bit | hybrid (LIMB_BITS=64) | 64-bit (LIMB_BITS=64 + NATIVE_HOST_LIMBS)

$(TEST_BUILD)/test_zero_pad_hybrid: $(ZP_SRC) $(ZP_SRCS_DEP) | $(TEST_BUILD) $(GPU_NTT_LIB)
	$(NVCC) $(NVCCFLAGS) -DLIMB_BITS=64 $(INCLUDES) \
		$^ -o $@ $(LIB_PATHS) $(LIBS)

$(TEST_BUILD)/test_zero_pad_64bit: $(ZP_SRC) $(ZP_SRCS_DEP) | $(TEST_BUILD) $(GPU_NTT_LIB)
	$(NVCC) $(NVCCFLAGS) $(NATIVE_FLAGS) $(INCLUDES) \
		$^ -o $@ $(LIB_PATHS) $(LIBS)

test_zero_pad: \
	$(TEST_BUILD)/test_zero_pad_32 \
	$(TEST_BUILD)/test_zero_pad_hybrid \
	$(TEST_BUILD)/test_zero_pad_64bit
	@echo "===== zero_pad 32-bit ====="
	@$(TEST_BUILD)/test_zero_pad_32
	@echo "===== zero_pad hybrid ====="
	@$(TEST_BUILD)/test_zero_pad_hybrid
	@echo "===== zero_pad 64-bit ====="
	@$(TEST_BUILD)/test_zero_pad_64bit

# ================================================================
# crt_gpu dual-width tests
# ================================================================

CRT_SRC      := $(TEST_DIR)/test_crt_gpu.cu
CRT_SRCS_DEP := $(SRC_DIR)/crt_gpu.cu $(SRC_DIR)/crt_utils.cpp

$(TEST_BUILD)/test_crt_gpu_32: $(CRT_SRC) $(CRT_SRCS_DEP) | $(TEST_BUILD) $(GPU_NTT_LIB)
	$(NVCC) $(NVCCFLAGS) -DLIMB_BITS=32 $(INCLUDES) \
		$^ -o $@ $(LIB_PATHS) $(LIBS)

$(TEST_BUILD)/test_crt_gpu_hybrid: $(CRT_SRC) $(CRT_SRCS_DEP) | $(TEST_BUILD) $(GPU_NTT_LIB)
	$(NVCC) $(NVCCFLAGS) -DLIMB_BITS=64 $(INCLUDES) \
		$^ -o $@ $(LIB_PATHS) $(LIBS)

$(TEST_BUILD)/test_crt_gpu_64bit: $(CRT_SRC) $(CRT_SRCS_DEP) | $(TEST_BUILD) $(GPU_NTT_LIB)
	$(NVCC) $(NVCCFLAGS) $(NATIVE_FLAGS) $(INCLUDES) \
		$^ -o $@ $(LIB_PATHS) $(LIBS)

test_crt_gpu: \
	$(TEST_BUILD)/test_crt_gpu_32 \
	$(TEST_BUILD)/test_crt_gpu_hybrid \
	$(TEST_BUILD)/test_crt_gpu_64bit
	@echo "===== crt_gpu 32-bit ====="
	@$(TEST_BUILD)/test_crt_gpu_32
	@echo "===== crt_gpu hybrid ====="
	@$(TEST_BUILD)/test_crt_gpu_hybrid
	@echo "===== crt_gpu 64-bit ====="
	@$(TEST_BUILD)/test_crt_gpu_64bit

# ================================================================
# Carry Propagation dual-width tests
# ================================================================

CARRY_SRC      := $(TEST_DIR)/test_carry_prop.cu
CARRY_SRCS_DEP := $(SRC_DIR)/carry_prop.cu

$(TEST_BUILD)/test_carry_prop_32: $(CARRY_SRC) $(CARRY_SRCS_DEP) | $(TEST_BUILD) $(GPU_NTT_LIB)
	$(NVCC) $(NVCCFLAGS) -DLIMB_BITS=32 $(INCLUDES) \
		$^ -o $@ $(LIB_PATHS) $(LIBS)

$(TEST_BUILD)/test_carry_prop_hybrid: $(CARRY_SRC) $(CARRY_SRCS_DEP) | $(TEST_BUILD) $(GPU_NTT_LIB)
	$(NVCC) $(NVCCFLAGS) -DLIMB_BITS=64 $(INCLUDES) \
		$^ -o $@ $(LIB_PATHS) $(LIBS)

$(TEST_BUILD)/test_carry_prop_64bit: $(CARRY_SRC) $(CARRY_SRCS_DEP) | $(TEST_BUILD) $(GPU_NTT_LIB)
	$(NVCC) $(NVCCFLAGS) $(NATIVE_FLAGS) $(INCLUDES) \
		$^ -o $@ $(LIB_PATHS) $(LIBS)

test_carry_prop: \
	$(TEST_BUILD)/test_carry_prop_32 \
	$(TEST_BUILD)/test_carry_prop_hybrid \
	$(TEST_BUILD)/test_carry_prop_64bit
	@echo "===== carry_prop 32-bit ====="
	@$(TEST_BUILD)/test_carry_prop_32
	@echo "===== carry_prop hybrid ====="
	@$(TEST_BUILD)/test_carry_prop_hybrid
	@echo "===== carry_prop 64-bit ====="
	@$(TEST_BUILD)/test_carry_prop_64bit

test_unit_64bit: \
	$(TEST_BUILD)/test_crt_gpu_64bit \
	$(TEST_BUILD)/test_carry_prop_64bit
	@echo "===== crt_gpu 64-bit ====="
	@$(TEST_BUILD)/test_crt_gpu_64bit
	@echo "===== carry_prop 64-bit ====="
	@$(TEST_BUILD)/test_carry_prop_64bit

# ================================================================
# ntt_limits dual-width tests
# ================================================================

NTT_LIMITS_SRC     := $(TEST_DIR)/test_ntt_limits.cpp
NTT_LIMITS_OBJS    := $(OBJ_DIR)/ntt_limits_test.o $(OBJ_DIR)/ntt_limits.o

$(OBJ_DIR)/ntt_limits_test.o: $(NTT_LIMITS_SRC) | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -DLIMB_BITS=32 $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/ntt_limits_32.o: $(SRC_DIR)/ntt_limits.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -DLIMB_BITS=32 $(INCLUDES) -c $< -o $@

$(TEST_BUILD)/test_ntt_limits_32: $(OBJ_DIR)/ntt_limits_test.o $(OBJ_DIR)/ntt_limits_32.o | $(TEST_BUILD)
	$(CXX) $(CXXFLAGS) -DLIMB_BITS=32 $(INCLUDES) \
		$^ -o $@

$(OBJ_DIR)/ntt_limits_test_hybrid.o: $(NTT_LIMITS_SRC) | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -DLIMB_BITS=64 $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/ntt_limits_hybrid.o: $(SRC_DIR)/ntt_limits.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -DLIMB_BITS=64 $(INCLUDES) -c $< -o $@

$(TEST_BUILD)/test_ntt_limits_hybrid: $(OBJ_DIR)/ntt_limits_test_hybrid.o $(OBJ_DIR)/ntt_limits_hybrid.o | $(TEST_BUILD)
	$(CXX) $(CXXFLAGS) -DLIMB_BITS=64 $(INCLUDES) \
		$^ -o $@

$(OBJ_DIR)/ntt_limits_test_64bit.o: $(NTT_LIMITS_SRC) | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(NATIVE_FLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/ntt_limits_64bit.o: $(SRC_DIR)/ntt_limits.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(NATIVE_FLAGS) $(INCLUDES) -c $< -o $@

$(TEST_BUILD)/test_ntt_limits_64bit: $(OBJ_DIR)/ntt_limits_test_64bit.o $(OBJ_DIR)/ntt_limits_64bit.o | $(TEST_BUILD)
	$(CXX) $(CXXFLAGS) $(NATIVE_FLAGS) $(INCLUDES) \
		$^ -o $@

test_ntt_limits: \
	$(TEST_BUILD)/test_ntt_limits_32 \
	$(TEST_BUILD)/test_ntt_limits_hybrid \
	$(TEST_BUILD)/test_ntt_limits_64bit
	@echo "===== ntt_limits 32-bit ====="
	@$(TEST_BUILD)/test_ntt_limits_32
	@echo "===== ntt_limits hybrid ====="
	@$(TEST_BUILD)/test_ntt_limits_hybrid
	@echo "===== ntt_limits 64-bit ====="
	@$(TEST_BUILD)/test_ntt_limits_64bit

# ================================================================
# End-to-end dual-width builds
# ================================================================

E2E_SRCS := tests/test_full_multiply.cpp $(CPP_SRCS) $(CU_SRCS)
PIPELINE_CRT_SRCS := tests/test_pipeline_crt.cpp $(CPP_SRCS) $(CU_SRCS)

main_32: | $(OBJ_DIR) $(GPU_NTT_LIB)
	$(NVCC) $(NVCCFLAGS) -DLIMB_BITS=32 $(INCLUDES) $(LIB_PATHS) \
		$(E2E_SRCS) -o build/test_full_multiply_32 $(LIBS)

main_hybrid: | $(OBJ_DIR) $(GPU_NTT_LIB)
	$(NVCC) $(NVCCFLAGS) -DLIMB_BITS=64 $(INCLUDES) $(LIB_PATHS) \
		$(E2E_SRCS) -o build/test_full_multiply_hybrid $(LIBS)

$(TEST_BUILD)/test_pipeline_crt_hybrid: | $(TEST_BUILD) $(GPU_NTT_LIB)
	$(NVCC) $(NVCCFLAGS) -DLIMB_BITS=64 $(INCLUDES) $(LIB_PATHS) \
		$(PIPELINE_CRT_SRCS) -o $@ $(LIBS)

main_64bit: | $(OBJ_DIR) $(GPU_NTT_LIB)
	$(NVCC) $(NVCCFLAGS) $(NATIVE_FLAGS) $(INCLUDES) $(LIB_PATHS) \
		$(E2E_SRCS) -o build/test_full_multiply_64bit $(LIBS)

test_full_multiply_64bit: main_64bit
	@echo "===== full multiply 64-bit ====="
	@build/test_full_multiply_64bit

$(TEST_BUILD)/test_pipeline_crt_64bit: | $(TEST_BUILD) $(GPU_NTT_LIB)
	$(NVCC) $(NVCCFLAGS) $(NATIVE_FLAGS) $(INCLUDES) $(LIB_PATHS) \
		$(PIPELINE_CRT_SRCS) -o $@ $(LIBS)

test_pipeline_crt_hybrid: $(TEST_BUILD)/test_pipeline_crt_hybrid
	@echo "===== pipeline CRT hybrid ====="
	@$(TEST_BUILD)/test_pipeline_crt_hybrid

test_pipeline_crt_64bit: $(TEST_BUILD)/test_pipeline_crt_64bit
	@echo "===== pipeline CRT 64-bit ====="
	@$(TEST_BUILD)/test_pipeline_crt_64bit

# ================================================================
# Profile builds (single warmup + single timed run, for nsys/ncu)
# ================================================================

profile_32: | $(OBJ_DIR) $(GPU_NTT_LIB)
	$(NVCC) $(NVCCFLAGS) -DLIMB_BITS=32 -DPROFILE $(INCLUDES) $(LIB_PATHS) \
		$(E2E_SRCS) -o build/profile_full_multiply_32 $(LIBS)

# ================================================================
# Phony targets
# ================================================================

.PHONY: test test_zero_pad test_crt_gpu test_carry_prop test_ntt_limits \
	main_32 main_hybrid main_64bit test_pipeline_crt_hybrid test_pipeline_crt_64bit \
	test_unit_64bit test_full_multiply_64bit \
	main_64 main_64native test_pipeline_crt_64 test_pipeline_crt_64native \
	test_unit_64native test_full_multiply_64native

# Deprecated aliases
main_64: main_hybrid
main_64native: main_64bit
test_unit_64native: test_unit_64bit
test_full_multiply_64native: test_full_multiply_64bit
test_pipeline_crt_64: test_pipeline_crt_hybrid
test_pipeline_crt_64native: test_pipeline_crt_64bit