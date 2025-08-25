# Makefile for CUDA Sylvester Solver

# Compiler and flags
NVCC = nvcc
CXXFLAGS = -std=c++17 -O3
CUDAFLAGS = -arch=sm_70 -lcublas -lcusparse 
# Source files
SRC = gadi_solver.cu
TARGET = gadi_solver

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) $(CUDAFLAGS) -o $(TARGET) $(SRC)

# Clean
clean:
	rm -f $(TARGET)

.PHONY: all clean run