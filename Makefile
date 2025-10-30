# ================================================================
# CUDA + CGBN + GPU-NTT Hybrid Project Makefile
# ================================================================

# Compiler and paths
NVCC        := nvcc
CXX         := g++
CXXFLAGS    := -std=c++17 -O3 -Wall -Wextra
NVCCFLAGS   := -std=c++17 -O3 -Xcompiler -Wall -Xcompiler -Wextra

CUDA_PATH   ?= /usr/local/cuda
INCLUDES    := -Iinclude -I$(CUDA_PATH)/include \
               -I$(HOME)/.local/include \
               -I$(HOME)/CGBN/include \
               -I$(HOME)/GPU-NTT/src/include \
			   -I$(HOME)/GPU-NTT/src/include/common \
               -I$(HOME)/GPU-NTT/src/include/ntt_merge

LIB_PATHS   := -L$(HOME)/.local/lib
LIBS        := -lntt-1.0

# Source directories
SRC_DIR     := src
OBJ_DIR     := build

# Output binary
TARGET      := main

# Source files
CPP_SRCS    := $(wildcard $(SRC_DIR)/*.cpp)
CU_SRCS     := $(wildcard $(SRC_DIR)/*.cu)
MAIN_SRC    := main.cpp

# Object files
CPP_OBJS    := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(CPP_SRCS))
CU_OBJS     := $(patsubst $(SRC_DIR)/%.cu,  $(OBJ_DIR)/%.o, $(CU_SRCS))
MAIN_OBJ    := $(OBJ_DIR)/main.o

OBJS        := $(CPP_OBJS) $(CU_OBJS) $(MAIN_OBJ)

# make rules 

all: $(TARGET)

# Create build directory if missing
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# C++ compilation
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# CUDA compilation
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# Main file (pure C++)
$(OBJ_DIR)/main.o: main.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Linking everything together
$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIB_PATHS) -o $@ $^ $(LIBS)

clean:
	rm -rf $(OBJ_DIR) $(TARGET)

.PHONY: all clean compile_commands.json
