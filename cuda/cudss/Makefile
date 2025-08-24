# Makefile for cuDSS sparse linear solver

# 编译器和标志
NVCC = nvcc
CXX = g++

# 包含路径
INCLUDES = -I$(CUDA_PATH)/include -I$(CUDSS_PATH)/include

# 库路径
LIBDIRS = -L$(CUDA_PATH)/lib64 -L$(CUDSS_PATH)/lib

# 链接库
LIBS = -lcudss -lcusparse -lcublas -lcuda -lcudart

# 编译标志
NVCCFLAGS = -std=c++14 -O3 -arch=sm_70
CXXFLAGS = -std=c++14 -O3

# 目标文件
TARGET = cudss_solver
SOURCE = cudss_solver.cu

# 默认目标
all: $(TARGET)

# 编译规则
$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIBDIRS) $(SOURCE) -o $(TARGET) $(LIBS)

# 清理
clean:
	rm -f $(TARGET) *.o

# 安装依赖说明
help:
	@echo "cuDSS稀疏线性求解器编译说明:"
	@echo ""
	@echo "编译前请确保已安装以下依赖:"
	@echo "1. CUDA Toolkit (11.0+)"
	@echo "2. cuDSS库"
	@echo "3. cuSPARSE (通常包含在CUDA Toolkit中)"
	@echo ""
	@echo "使用方法:"
	@echo "  make           - 编译程序"
	@echo "  make clean     - 清理生成文件"
	@echo "  make help      - 显示帮助信息"
	@echo ""
	@echo "运行示例:"
	@echo "  ./cudss_solver matrix.mtx"
	
# 检查环境
check-env:
	@echo "检查编译环境..."
	@echo "CUDA路径: $(CUDA_PATH)"
	@echo "cuDSS路径: $(CUDSS_PATH)"
	@echo ""
	@if [ ! -d "$(CUDA_PATH)" ]; then \
		echo "错误: CUDA路径不存在: $(CUDA_PATH)"; \
		exit 1; \
	fi
	@if [ ! -d "$(CUDSS_PATH)" ]; then \
		echo "错误: cuDSS路径不存在: $(CUDSS_PATH)"; \
		exit 1; \
	fi
	@echo "环境检查通过"

.PHONY: all clean help check-env