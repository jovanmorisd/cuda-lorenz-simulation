.PHONY: all openmp cuda clean help

# Project paths
PROJECT_DIR := .
SRC_DIR := $(PROJECT_DIR)/src
CUDA_DIR := $(PROJECT_DIR)/cuda

# Compiler settings
CXX := g++
NVCC := nvcc
CXXFLAGS := -O3 -Wall
OMPFLAGS := -fopenmp
NVCCFLAGS := -O3

# Targets
OPENMP_TARGET := $(SRC_DIR)/openmp_lorenz
CUDA_TARGET := $(CUDA_DIR)/cuda_lorenz

help:
	@echo "AI-HPC Lorenz Project - Build Targets"
	@echo "======================================"
	@echo "make openmp     - Compile OpenMP C++ solver"
	@echo "make cuda       - Compile CUDA GPU solver"
	@echo "make all        - Compile all targets"
	@echo "make clean      - Remove compiled binaries"
	@echo "make help       - Show this help message"

all: openmp cuda

openmp:
	@echo "Compiling OpenMP solver..."
	@$(CXX) $(CXXFLAGS) $(OMPFLAGS) $(SRC_DIR)/openmp_lorenz.cpp -o $(OPENMP_TARGET)
	@echo "✓ OpenMP executable created: $(OPENMP_TARGET)"

cuda:
	@command -v $(NVCC) >/dev/null 2>&1 || { echo "CUDA (nvcc) not found. Skipping CUDA compilation."; exit 0; }
	@echo "Compiling CUDA solver..."
	@$(NVCC) $(NVCCFLAGS) $(CUDA_DIR)/cuda_lorenz.cu -o $(CUDA_TARGET)
	@echo "✓ CUDA executable created: $(CUDA_TARGET)"

clean:
	@echo "Cleaning up..."
	@rm -f $(OPENMP_TARGET)
	@rm -f $(CUDA_TARGET)
	@echo "✓ Cleaned"

.DEFAULT_GOAL := help
