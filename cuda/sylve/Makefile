# Makefile for CUDA Sylvester Solver

# Compiler and flags
NVCC = nvcc
CXXFLAGS = -std=c++17 -O3
CUDAFLAGS = -arch=sm_70 -lcublas -lcusparse 
# Source files
SRC = sylvester_solver.cu
TARGET = sylvester_solver

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) $(CUDAFLAGS) -o $(TARGET) $(SRC)

# Clean
clean:
	rm -f $(TARGET)

# Run with example
run: $(TARGET)
	./$(TARGET) sylve256.mtx

.PHONY: all clean run