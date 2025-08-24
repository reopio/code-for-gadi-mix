#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <cuda_runtime.h>
#include <cudss.h>
#include <cusparse.h>

// 显存监控结构
struct MemoryInfo {
    size_t free_bytes;
    size_t total_bytes;
    size_t used_bytes;
    
    void update() {
        cudaMemGetInfo(&free_bytes, &total_bytes);
        used_bytes = total_bytes - free_bytes;
    }
    
    void print(const std::string& stage) {
        std::cout << "[" << stage << "] GPU Memory: "
                  << "Used=" << used_bytes / 1024 / 1024 << "MB, "
                  << "Free=" << free_bytes / 1024 / 1024 << "MB, "
                  << "Total=" << total_bytes / 1024 / 1024 << "MB" << std::endl;
    }
};

// 检查CUDA错误
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

// 检查cuDSS错误
#define CHECK_CUDSS(call) do { \
    cudssStatus_t status = call; \
    if (status != CUDSS_STATUS_SUCCESS) { \
        std::cerr << "cuDSS error at " << __FILE__ << ":" << __LINE__ << " - Status: " << status << std::endl; \
        exit(1); \
    } \
} while(0)

// 检查cuSPARSE错误
#define CHECK_CUSPARSE(call) do { \
    cusparseStatus_t status = call; \
    if (status != CUSPARSE_STATUS_SUCCESS) { \
        std::cerr << "cuSPARSE error at " << __FILE__ << ":" << __LINE__ << " - Status: " << status << std::endl; \
        exit(1); \
    } \
} while(0)

class SparseMatrix {
public:
    int m, n, nnz;
    std::vector<int> row_ptr, col_idx;
    std::vector<double> values;
    
    bool readFromMTX(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "无法打开文件: " << filename << std::endl;
            return false;
        }
        
        std::string line;
        // 跳过注释行
        do {
            std::getline(file, line);
        } while (line[0] == '%');
        
        // 读取矩阵尺寸信息
        std::istringstream iss(line);
        iss >> m >> n >> nnz;
        
        std::cout << "矩阵尺寸: " << m << "x" << n << ", 非零元素: " << nnz << std::endl;
        
        // 临时存储坐标格式数据
        std::vector<int> row_coords(nnz), col_coords(nnz);
        values.resize(nnz);
        
        // 读取矩阵数据
        for (int i = 0; i < nnz; i++) {
            std::getline(file, line);
            std::istringstream data_iss(line);
            data_iss >> row_coords[i] >> col_coords[i] >> values[i];
            // MTX格式通常是1-based索引，转换为0-based
            row_coords[i]--;
            col_coords[i]--;
        }
        
        file.close();
        
        // 转换为CSR格式
        convertToCSR(row_coords, col_coords);
        return true;
    }
    
private:
    void convertToCSR(const std::vector<int>& row_coords, const std::vector<int>& col_coords) {
        row_ptr.resize(m + 1, 0);
        col_idx.resize(nnz);
        
        // 计算每行的非零元素数量
        for (int i = 0; i < nnz; i++) {
            row_ptr[row_coords[i] + 1]++;
        }
        
        // 累积求和得到行指针
        for (int i = 1; i <= m; i++) {
            row_ptr[i] += row_ptr[i - 1];
        }
        
        // 填充列索引数组
        std::vector<int> row_counters(m, 0);
        for (int i = 0; i < nnz; i++) {
            int row = row_coords[i];
            int pos = row_ptr[row] + row_counters[row];
            col_idx[pos] = col_coords[i];
            row_counters[row]++;
        }
    }
};

class CuDSSSolver {
private:
    cudssHandle_t handle;
    cudssConfig_t config;
    cudssData_t data;
    cudssMatrix_t A_matrix;
    cudssMatrix_t x_matrix, b_matrix;
    MemoryInfo mem_info;
    int matrix_size;
    
    // GPU内存指针
    int *d_row_ptr, *d_col_idx;
    double *d_values, *d_x, *d_b;
    
public:
    CuDSSSolver() : handle(nullptr), config(nullptr), data(nullptr), A_matrix(nullptr),
                    x_matrix(nullptr), b_matrix(nullptr), matrix_size(0),
                    d_row_ptr(nullptr), d_col_idx(nullptr), d_values(nullptr), 
                    d_x(nullptr), d_b(nullptr) {}
    
    ~CuDSSSolver() {
        cleanup();
    }
    
    bool initialize() {
        mem_info.update();
        mem_info.print("初始化前");
        
        // 初始化cuDSS
        CHECK_CUDSS(cudssCreate(&handle));
        CHECK_CUDSS(cudssConfigCreate(&config));
        CHECK_CUDSS(cudssDataCreate(handle, &data));
        
        mem_info.update();
        mem_info.print("cuDSS初始化后");
        
        return true;
    }
    
    bool setupMatrix(const SparseMatrix& matrix) {
        int m = matrix.m;
        int n = matrix.n;
        int nnz = matrix.nnz;
        matrix_size = n;  // 保存矩阵大小
        
        std::cout << "设置矩阵，尺寸: " << m << "x" << n << ", 非零元素: " << nnz << std::endl;
        
        // 分配GPU内存
        CHECK_CUDA(cudaMalloc(&d_row_ptr, (m + 1) * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_col_idx, nnz * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_values, nnz * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_x, n * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_b, m * sizeof(double)));
        
        mem_info.update();
        mem_info.print("GPU内存分配后");
        
        // 复制数据到GPU
        CHECK_CUDA(cudaMemcpy(d_row_ptr, matrix.row_ptr.data(), (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_col_idx, matrix.col_idx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_values, matrix.values.data(), nnz * sizeof(double), cudaMemcpyHostToDevice));
        
        // 创建稀疏矩阵描述符
        CHECK_CUDSS(cudssMatrixCreateCsr(&A_matrix, m, n, nnz, d_row_ptr, nullptr, d_col_idx, d_values,
                                        CUDA_R_32I, CUDA_R_64F, CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL,
                                        CUDSS_BASE_ZERO));
        
        mem_info.update();
        mem_info.print("矩阵设置完成后");
        
        return true;
    }
    
    bool factorize() {
        std::cout << "开始矩阵分解..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        
        mem_info.update();
        mem_info.print("分解前");
        
        // 创建临时的x和b矩阵描述符用于分析和分解阶段
        CHECK_CUDSS(cudssMatrixCreateDn(&x_matrix, matrix_size, 1, matrix_size, d_x, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));
        CHECK_CUDSS(cudssMatrixCreateDn(&b_matrix, matrix_size, 1, matrix_size, d_b, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));
        
        // 符号分析
        CHECK_CUDSS(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, config, data, A_matrix, x_matrix, b_matrix));
        
        mem_info.update();
        mem_info.print("符号分析后");
        
        // 数值分解
        CHECK_CUDSS(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, config, data, A_matrix, x_matrix, b_matrix));
        
        mem_info.update();
        mem_info.print("数值分解后");
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "矩阵分解完成，耗时: " << duration.count() << " ms" << std::endl;
        
        return true;
    }
    
    bool solve(std::vector<double>& x, const std::vector<double>& b) {
        if (b.size() != x.size()) {
            std::cerr << "向量尺寸不匹配" << std::endl;
            return false;
        }
        
        int n = b.size();
        std::cout << "开始求解线性系统，向量维度: " << n << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // 复制右端向量到GPU
        CHECK_CUDA(cudaMemcpy(d_b, b.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        
        mem_info.update();
        mem_info.print("求解前");
        
        // 求解 - 使用已创建的矩阵描述符
        CHECK_CUDSS(cudssExecute(handle, CUDSS_PHASE_SOLVE, config, data, A_matrix, x_matrix, b_matrix));
        
        // 复制结果回CPU
        CHECK_CUDA(cudaMemcpy(x.data(), d_x, n * sizeof(double), cudaMemcpyDeviceToHost));
        
        mem_info.update();
        mem_info.print("求解完成后");
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "线性系统求解完成，耗时: " << duration.count() << " ms" << std::endl;
        
        return true;
    }
    
    void cleanup() {
        if (d_row_ptr) { cudaFree(d_row_ptr); d_row_ptr = nullptr; }
        if (d_col_idx) { cudaFree(d_col_idx); d_col_idx = nullptr; }
        if (d_values) { cudaFree(d_values); d_values = nullptr; }
        if (d_x) { cudaFree(d_x); d_x = nullptr; }
        if (d_b) { cudaFree(d_b); d_b = nullptr; }
        
        if (x_matrix) { cudssMatrixDestroy(x_matrix); x_matrix = nullptr; }
        if (b_matrix) { cudssMatrixDestroy(b_matrix); b_matrix = nullptr; }
        if (A_matrix) { cudssMatrixDestroy(A_matrix); A_matrix = nullptr; }
        if (data) { cudssDataDestroy(handle, data); data = nullptr; }
        if (config) { cudssConfigDestroy(config); config = nullptr; }
        if (handle) { cudssDestroy(handle); handle = nullptr; }
        
        mem_info.update();
        mem_info.print("清理完成后");
    }
};

// 构造右端项：设置真解x全为1，通过Ax=b计算右端项b
std::vector<double> constructRightHandSide(const SparseMatrix& matrix) {
    int n = matrix.n;
    std::vector<double> x_true(n, 1.0);  // 真解：全为1
    std::vector<double> b(n, 0.0);       // 右端项：初始化为0
    
    std::cout << "构造右端项：设置真解x全为1，计算b=Ax..." << std::endl;
    
    // 计算 b = A * x，其中x全为1
    // 由于x全为1，所以b[i]就是矩阵A第i行所有元素的和
    for (int i = 0; i < matrix.m; i++) {
        double sum = 0.0;
        for (int j = matrix.row_ptr[i]; j < matrix.row_ptr[i + 1]; j++) {
            sum += matrix.values[j];  // x[col_idx[j]] = 1.0
        }
        b[i] = sum;
    }
    
    // 计算b向量的范数用于验证
    double b_norm = 0.0;
    for (double val : b) {
        b_norm += val * val;
    }
    b_norm = sqrt(b_norm);
    
    std::cout << "右端项构造完成，||b||_2 = " << b_norm << std::endl;
    
    return b;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "用法: " << argv[0] << " <matrix.mtx>" << std::endl;
        return 1;
    }
    
    std::string mtx_filename = argv[1];
    
    try {
        // 读取稀疏矩阵
        std::cout << "=== 读取MTX文件 ===" << std::endl;
        SparseMatrix matrix;
        if (!matrix.readFromMTX(mtx_filename)) {
            std::cerr << "读取MTX文件失败" << std::endl;
            return 1;
        }
        
        // 初始化求解器
        std::cout << "\n=== 初始化cuDSS求解器 ===" << std::endl;
        CuDSSSolver solver;
        if (!solver.initialize()) {
            std::cerr << "求解器初始化失败" << std::endl;
            return 1;
        }
        
        // 设置矩阵
        std::cout << "\n=== 设置系数矩阵 ===" << std::endl;
        if (!solver.setupMatrix(matrix)) {
            std::cerr << "矩阵设置失败" << std::endl;
            return 1;
        }
        
        // 分解矩阵
        std::cout << "\n=== 矩阵分解 ===" << std::endl;
        if (!solver.factorize()) {
            std::cerr << "矩阵分解失败" << std::endl;
            return 1;
        }
        
        // 创建右端向量（这里使用全1向量作为示例）
        std::cout << "\n=== 求解线性系统 ===" << std::endl;
        std::vector<double> b = constructRightHandSide(matrix);
        std::vector<double> x(matrix.n, 0.0);
        
        if (!solver.solve(x, b)) {
            std::cerr << "求解失败" << std::endl;
            return 1;
        }
        
        // 显示部分结果
        std::cout << "\n=== 求解结果 ===" << std::endl;
        std::cout << "解向量的前10个元素:" << std::endl;
        for (int i = 0; i < std::min(10, (int)x.size()); i++) {
            std::cout << "x[" << i << "] = " << x[i] << std::endl;
        }
        
        std::cout << "\n=== 求解完成 ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "异常: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}