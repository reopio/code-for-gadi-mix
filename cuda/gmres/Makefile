# Makefile for CUDA GMRES Solver

# Compiler and flags
NVCC = nvcc
CXXFLAGS = -std=c++17 -O3
CUDAFLAGS = -arch=sm_70 -lcublas -lcusparse

# Source files
SRC = gmres_solver.cu
TARGET = gmres_solver

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) $(CUDAFLAGS) -o $(TARGET) $(SRC)

# Clean
clean:
	rm -f $(TARGET)

# Install (copy to bin directory if needed)
install: $(TARGET)
	cp $(TARGET) /usr/local/bin/ 2>/dev/null || echo "Could not install to /usr/local/bin, run 'sudo make install' if needed"

.PHONY: all clean install
