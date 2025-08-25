#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_fp16.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t err = call; \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS error at %s %d: %d\n", __FILE__, __LINE__, err); \
        exit(1); \
    } \
} while(0)

#define CHECK_CUSPARSE(call) do { \
    cusparseStatus_t err = call; \
    if (err != CUSPARSE_STATUS_SUCCESS) { \
        printf("cuSPARSE error at %s %d: %d\n", __FILE__, __LINE__, err); \
        exit(1); \
    } \
} while(0)

// Structure to hold sparse matrix in CSR format
struct SparseMatrix {
    int rows, cols, nnz;
    int *d_csrRowPtr, *d_csrColInd;
    double *d_csrVal;
    __half *d_csrVal_half;
    float *d_csrVal_float;
};

// Structure to hold only double sparse matrix in CSR format
struct SparseMatrix_double {
    int rows, cols, nnz;
    int *d_csrRowPtr, *d_csrColInd;
    double *d_csrVal;
};

// Function to read MTX file
bool readMTXFile(const char* filename, SparseMatrix &matrix) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        printf("Error: Cannot open file %s\n", filename);
        return false;
    }
    
    std::string line;
    // Skip header and comments
    do {
        std::getline(file, line);
    } while (line[0] == '%');
    
    std::istringstream iss(line);
    iss >> matrix.rows >> matrix.cols >> matrix.nnz;
    
    std::vector<int> row_indices(matrix.nnz);
    std::vector<int> col_indices(matrix.nnz);
    std::vector<double> values(matrix.nnz);
    
    for (int i = 0; i < matrix.nnz; i++) {
        std::getline(file, line);
        std::istringstream iss(line);
        iss >> row_indices[i] >> col_indices[i] >> values[i];
        row_indices[i]--; // Convert to 0-based indexing
        col_indices[i]--;
    }
    
    file.close();
    
    // Convert to CSR format
    std::vector<int> csr_row_ptr(matrix.rows + 1, 0);
    std::vector<int> csr_col_ind(matrix.nnz);
    std::vector<double> csr_val(matrix.nnz);
    
    // Count entries in each row
    for (int i = 0; i < matrix.nnz; i++) {
        csr_row_ptr[row_indices[i] + 1]++;
    }
    
    // Convert counts to offsets
    for (int i = 1; i <= matrix.rows; i++) {
        csr_row_ptr[i] += csr_row_ptr[i - 1];
    }
    
    std::vector<int> temp_row_ptr = csr_row_ptr;
    
    // Fill CSR arrays
    for (int i = 0; i < matrix.nnz; i++) {
        int row = row_indices[i];
        if (row < 0 || row >= matrix.rows) {
            printf("Error: Invalid row index %d (should be 0-%d)\n", row, matrix.rows-1);
            return false;
        }
        int pos = temp_row_ptr[row]++;
        if (pos < 0 || pos >= matrix.nnz) {
            printf("Error: Invalid position %d (should be 0-%d)\n", pos, matrix.nnz-1);
            return false;
        }
        csr_col_ind[pos] = col_indices[i];
        csr_val[pos] = values[i];
    }
    
    // Allocate GPU memory
    CHECK_CUDA(cudaMalloc(&matrix.d_csrRowPtr, (matrix.rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&matrix.d_csrColInd, matrix.nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&matrix.d_csrVal, matrix.nnz * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&matrix.d_csrVal_half, matrix.nnz * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&matrix.d_csrVal_float, matrix.nnz * sizeof(float)));
    
    // Copy to GPU
    CHECK_CUDA(cudaMemcpy(matrix.d_csrRowPtr, csr_row_ptr.data(), (matrix.rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(matrix.d_csrColInd, csr_col_ind.data(), matrix.nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(matrix.d_csrVal, csr_val.data(), matrix.nnz * sizeof(double), cudaMemcpyHostToDevice));
    
    // Convert values to half precision
    std::vector<__half> csr_val_half(matrix.nnz);
    for (int i = 0; i < matrix.nnz; i++) {
        csr_val_half[i] = __double2half(csr_val[i]);
    }
    CHECK_CUDA(cudaMemcpy(matrix.d_csrVal_half, csr_val_half.data(), matrix.nnz * sizeof(__half), cudaMemcpyHostToDevice));

    // Convert values to float precision
    std::vector<float> csr_val_float(matrix.nnz);
    for (int i = 0; i < matrix.nnz; i++) {
        csr_val_float[i] = static_cast<float>(csr_val[i]);
    }
    CHECK_CUDA(cudaMemcpy(matrix.d_csrVal_float, csr_val_float.data(), matrix.nnz * sizeof(float), cudaMemcpyHostToDevice));
    
    return true;
}

// CUDA kernel to convert double to half precision
__global__ void double_to_half_kernel(double *d_double, __half *d_half, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_half[idx] = __float2half((float)d_double[idx]);
    }
}

// CUDA kernel to convert half to double precision
__global__ void half_to_double_kernel(__half *d_half, double *d_double, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_double[idx] = (double)__half2float(d_half[idx]);
    }
}

__global__ void double_to_float_kernel(double *d_double, float *d_float, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_float[idx] = (float)d_double[idx];
    }
}

__global__ void float_to_double_kernel(float *d_float, double *d_double, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_double[idx] = (double)d_float[idx];
    }
}

// CG solver in half precision (using manual implementations)
int cg_half_precision(cusparseHandle_t cusparseHandle, cublasHandle_t cublasHandle, 
                      SparseMatrix &A, __half *d_b, __half *d_x, 
                      double tol = 1e-4, int max_iter = 100) {
    
    int n = A.rows;
    __half *d_r, *d_p, *d_Ap;
    
    CHECK_CUDA(cudaMalloc(&d_r, n * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_p, n * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_Ap, n * sizeof(__half)));
    
    // Initialize x = 0
    CHECK_CUDA(cudaMemset(d_x, 0, n * sizeof(__half)));
    
    // r = b - A*x (since x = 0, r = b)
    CHECK_CUDA(cudaMemcpy(d_r, d_b, n * sizeof(__half), cudaMemcpyDeviceToDevice));
    
    // p = r
    CHECK_CUDA(cudaMemcpy(d_p, d_r, n * sizeof(__half), cudaMemcpyDeviceToDevice));
    
    float one_ht = 1.0f, zero_ht = 0.0f;

    // rsold = r^T * r
    __half rsold = 0.0f;
    float rsold_float = 0.0f;
    CHECK_CUBLAS(cublasDotEx(cublasHandle, 
                n, 
                d_r, 
                CUDA_R_16F, 
                1,
                d_r, 
                CUDA_R_16F,
                1, 
                &rsold, 
                CUDA_R_16F, CUDA_R_32F));
    rsold_float = __half2float(rsold);
    
    int iter, count = 0;

    __half pAp = 0.0f;
    __half dot_r = 0.0f;
    float pAp_float = 0.0f;
    float dot_r_float = 0.0f;

    float rsnew = 1.0f;
    float beta = 0.0f;
    float alpha1 = 0.0f;
    float neg_alpha = 0.0f;

    // Pre-allocate buffer for SpMV operations
    size_t bufferSize = 0;
    void *dBuffer = nullptr;
    
    // Create vector descriptors once before the loop
    cusparseDnVecDescr_t vecP, vecAp;
    cusparseSpMatDescr_t A_mat_half;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecP, n, d_p, CUDA_R_16F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecAp, n, d_Ap, CUDA_R_16F));

    CHECK_CUSPARSE(cusparseCreateCsr(&A_mat_half, A.rows, A.cols, A.nnz,
                                     A.d_csrRowPtr, A.d_csrColInd, A.d_csrVal_half,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F));

    // Get buffer size
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &one_ht, A_mat_half, vecP, &zero_ht, vecAp,
                                           CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    if (bufferSize > 0) {
        CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
    }

    // Main CG loop
    for (iter = 0; iter < max_iter; iter++) {
        // Ap = A * p
        CHECK_CUSPARSE(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &one_ht, A_mat_half, vecP, &zero_ht, vecAp,
                                    CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
        
        CHECK_CUBLAS(cublasDotEx(cublasHandle, 
                n, 
                d_p, 
                CUDA_R_16F, 
                1,
                d_Ap, 
                CUDA_R_16F,
                1, 
                &pAp, 
                CUDA_R_16F, CUDA_R_32F));
        pAp_float = __half2float(pAp);

        alpha1 = rsold_float / pAp_float;
        
        // x = x + alpha * p
        CHECK_CUBLAS(cublasAxpyEx(cublasHandle, 
                n, 
                &alpha1, 
                CUDA_R_32F, 
                d_p, 
                CUDA_R_16F, 
                1, 
                d_x, 
                CUDA_R_16F, 
                1, 
                CUDA_R_32F));
        
        // r = r - alpha * Ap
        neg_alpha = -alpha1;
        CHECK_CUBLAS(cublasAxpyEx(cublasHandle, 
                n, 
                &neg_alpha, 
                CUDA_R_32F, 
                d_Ap, 
                CUDA_R_16F, 
                1, 
                d_r, 
                CUDA_R_16F, 
                1, 
                CUDA_R_32F));
        
        CHECK_CUBLAS(cublasDotEx(cublasHandle, 
                n, 
                d_r, 
                CUDA_R_16F, 
                1,
                d_r, 
                CUDA_R_16F,
                1, 
                &dot_r, 
                CUDA_R_16F, CUDA_R_32F));
        dot_r_float = __half2float(dot_r);

        if (rsnew <= dot_r_float) {
           count++;
           rsnew = dot_r_float;
           if (count >= 2) {
                // printf("Early exit: rsnew <= dot_r, stopping CG iterations.\n");
                break;
            }
        }else {
            rsnew = dot_r_float;
            count = 0;
        }
        
        if (sqrt(rsnew) < tol) {
            break;
        }
        
        beta = rsnew / rsold_float;
        CHECK_CUBLAS(cublasScalEx(cublasHandle, 
                n, 
                &beta, 
                CUDA_R_32F, 
                d_p, 
                CUDA_R_16F, 
                1, 
                CUDA_R_32F));
        CHECK_CUBLAS(cublasAxpyEx(cublasHandle, 
                n, 
                &one_ht, 
                CUDA_R_32F, 
                d_r, 
                CUDA_R_16F, 
                1, 
                d_p, 
                CUDA_R_16F, 
                1, 
                CUDA_R_32F));
        
        rsold_float = rsnew;

    }
    
    // Final cleanup
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecP));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecAp));
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_Ap);
    if (dBuffer) cudaFree(dBuffer);
    CHECK_CUSPARSE(cusparseDestroySpMat(A_mat_half));

    
    return iter;
}

// CG solver in float precision (using manual implementations)
int cg_float_precision(cusparseHandle_t cusparseHandle, cublasHandle_t cublasHandle, 
                      SparseMatrix &A, float *d_b, float *d_x, 
                      double tol = 1e-4, int max_iter = 100) {
    
    int n = A.rows;
    float *d_r, *d_p, *d_Ap;
    
    CHECK_CUDA(cudaMalloc(&d_r, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_p, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Ap, n * sizeof(float)));
    
    // Initialize x = 0
    CHECK_CUDA(cudaMemset(d_x, 0, n * sizeof(float)));
    
    // r = b - A*x (since x = 0, r = b)
    CHECK_CUDA(cudaMemcpy(d_r, d_b, n * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // p = r
    CHECK_CUDA(cudaMemcpy(d_p, d_r, n * sizeof(float), cudaMemcpyDeviceToDevice));
    
    float one_ht = 1.0f, zero_ht = 0.0f;

    // rsold = r^T * r
    float rsold_float = 0.0f;
    CHECK_CUBLAS(cublasDotEx(cublasHandle, 
                n, 
                d_r, 
                CUDA_R_32F, 
                1,
                d_r, 
                CUDA_R_32F,
                1, 
                &rsold_float, 
                CUDA_R_32F, CUDA_R_32F));
    
    // Pre-allocate buffer for SpMV operations
    size_t bufferSize = 0;
    void *dBuffer = nullptr;
    
    // Create vector descriptors once before the loop
    cusparseDnVecDescr_t vecP, vecAp;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecP, n, d_p, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecAp, n, d_Ap, CUDA_R_32F));
    cusparseSpMatDescr_t A_mat_float;
    CHECK_CUSPARSE(cusparseCreateCsr(&A_mat_float, A.rows, A.cols, A.nnz,
                                     A.d_csrRowPtr, A.d_csrColInd, A.d_csrVal_float,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    
    int iter;

    float pAp_float = 0.0f;
    float dot_r_float = 0.0f;

    float rsnew = 1.0f;
    float beta = 0.0f;
    float alpha1 = 0.0f;
    float neg_alpha = 0.0f;

    // Get buffer size
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &one_ht, A_mat_float, vecP, &zero_ht, vecAp,
                                           CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    if (bufferSize > 0) {
        CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
    }

    // Main CG loop
    for (iter = 0; iter < max_iter; iter++) {
        // Ap = A * p
        CHECK_CUSPARSE(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &one_ht, A_mat_float, vecP, &zero_ht, vecAp,
                                    CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
        
        CHECK_CUBLAS(cublasDotEx(cublasHandle, 
                n, 
                d_p, 
                CUDA_R_32F, 
                1,
                d_Ap, 
                CUDA_R_32F,
                1, 
                &pAp_float, 
                CUDA_R_32F, CUDA_R_32F));
        alpha1 = rsold_float / pAp_float;
        
        // x = x + alpha * p
        CHECK_CUBLAS(cublasAxpyEx(cublasHandle, 
                n, 
                &alpha1, 
                CUDA_R_32F, 
                d_p, 
                CUDA_R_32F, 
                1, 
                d_x, 
                CUDA_R_32F, 
                1, 
                CUDA_R_32F));
        
        // r = r - alpha * Ap
        neg_alpha = -alpha1;
        CHECK_CUBLAS(cublasAxpyEx(cublasHandle, 
                n, 
                &neg_alpha, 
                CUDA_R_32F, 
                d_Ap, 
                CUDA_R_32F, 
                1, 
                d_r, 
                CUDA_R_32F, 
                1, 
                CUDA_R_32F));
        
        CHECK_CUBLAS(cublasDotEx(cublasHandle, 
                n, 
                d_r, 
                CUDA_R_32F, 
                1,
                d_r, 
                CUDA_R_32F,
                1, 
                &dot_r_float, 
                CUDA_R_32F, CUDA_R_32F));

        // if (rsnew <= dot_r_float) {
        //    count++;
        //    rsnew = dot_r_float;
        //    if (count >= 2) {
        //         // printf("Early exit: rsnew <= dot_r, stopping CG iterations.\n");
        //         break;
        //     }
        // }else {
        //     rsnew = dot_r_float;
        //     count = 0;
        // }

        rsnew = dot_r_float;
        
        if (sqrt(rsnew) < tol) {
            break;
        }
        
        beta = rsnew / rsold_float;
        CHECK_CUBLAS(cublasScalEx(cublasHandle, 
                n, 
                &beta, 
                CUDA_R_32F, 
                d_p, 
                CUDA_R_32F, 
                1, 
                CUDA_R_32F));
        CHECK_CUBLAS(cublasAxpyEx(cublasHandle, 
                n, 
                &one_ht, 
                CUDA_R_32F, 
                d_r, 
                CUDA_R_32F, 
                1, 
                d_p, 
                CUDA_R_32F, 
                1, 
                CUDA_R_32F));
        
        rsold_float = rsnew;

    }
    
    // Final cleanup
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecP));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecAp));
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_Ap);
    if (dBuffer) cudaFree(dBuffer);
    CHECK_CUSPARSE(cusparseDestroySpMat(A_mat_float));
    
    return iter;
}

// CG solver in double precision
int cg_double_precision(cusparseHandle_t cusparseHandle, cublasHandle_t cublasHandle,
                        SparseMatrix &A, double *d_b, double *d_x,
                        double tol = 1e-4, int max_iter = 100) {
    
    int n = A.rows;
    double *d_r, *d_p, *d_Ap;
    
    CHECK_CUDA(cudaMalloc(&d_r, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_p, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_Ap, n * sizeof(double)));
    
    // Initialize x = 0
    CHECK_CUDA(cudaMemset(d_x, 0, n * sizeof(double)));
    
    // r = b - A*x (since x = 0, r = b)
    CHECK_CUDA(cudaMemcpy(d_r, d_b, n * sizeof(double), cudaMemcpyDeviceToDevice));
    
    // p = r
    CHECK_CUDA(cudaMemcpy(d_p, d_r, n * sizeof(double), cudaMemcpyDeviceToDevice));
    
    double rsold, rsnew, alpha_cg, beta;
    double one = 1.0, zero = 0.0;
    
    // rsold = r^T * r
    CHECK_CUBLAS(cublasDdot(cublasHandle, n, d_r, 1, d_r, 1, &rsold));
    
    // Create vector descriptors once before the loop
    cusparseDnVecDescr_t vecP, vecAp;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecP, n, d_p, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecAp, n, d_Ap, CUDA_R_64F));

    cusparseSpMatDescr_t A_mat_double;
    CHECK_CUSPARSE(cusparseCreateCsr(&A_mat_double, A.rows, A.cols, A.nnz,
                                     A.d_csrRowPtr, A.d_csrColInd, A.d_csrVal,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    
    int iter;

    size_t bufferSize = 0;
    void *dBuffer = nullptr;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &one, A_mat_double, vecP, &zero, vecAp,
                                            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    if (bufferSize > 0) {
        CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
    }

    for (iter = 0; iter < max_iter; iter++) {
        // Ap = A * p
        
        CHECK_CUSPARSE(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &one, A_mat_double, vecP, &zero, vecAp,
                                    CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
        
        // alpha = rsold / (p^T * Ap)
        double pAp;
        CHECK_CUBLAS(cublasDdot(cublasHandle, n, d_p, 1, d_Ap, 1, &pAp));
        alpha_cg = rsold / pAp;
        
        // x = x + alpha * p
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, n, &alpha_cg, d_p, 1, d_x, 1));
        
        // r = r - alpha * Ap
        double neg_alpha = -alpha_cg;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, n, &neg_alpha, d_Ap, 1, d_r, 1));
        
        // rsnew = r^T * r
        CHECK_CUBLAS(cublasDdot(cublasHandle, n, d_r, 1, d_r, 1, &rsnew));
        
        if (sqrt(rsnew) < tol) {
            break;
        }
        
        // beta = rsnew / rsold
        beta = rsnew / rsold;
        
        // p = r + beta * p
        CHECK_CUBLAS(cublasDscal(cublasHandle, n, &beta, d_p, 1));
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, n, &one, d_r, 1, d_p, 1));
        
        rsold = rsnew;
    }
    
    // Cleanup
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecP));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecAp));
    cudaFree(d_r);
    cudaFree(d_p);
    cudaFree(d_Ap);
    if (dBuffer) cudaFree(dBuffer);
    CHECK_CUSPARSE(cusparseDestroySpMat(A_mat_double));
    
    return iter;
}

// Half-precision iterative solver
int half_precision_solver(cusparseHandle_t cusparseHandle, cublasHandle_t cublasHandle,
                          SparseMatrix &A, SparseMatrix &AH, SparseMatrix &ASS, SparseMatrix &S,
                          double *d_b, double *d_x0, double w, double alpha_param, double *res_out,
                          double tol = 1e-6, int max_iter = 1000) {
    
    int n = A.rows;
    double *d_r;
    __half *d_r_half, *d_r1_half, *d_temp_half;
    float alpha_param_f = (float)alpha_param;
    
    CHECK_CUDA(cudaMalloc(&d_r, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_r_half, n * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_r1_half, n * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_temp_half, n * sizeof(__half)));
    
    double nr0;
    CHECK_CUBLAS(cublasDnrm2(cublasHandle, n, d_b, 1, &nr0));
    
    double res = 1.0;
    int kk = 0;
    double one = 1.0, neg_one = -1.0;
    float one_f = 1.0f, neg_one_f = -1.0f, zero_f = 0.0f, factor;
    __half r3_norm;
    double r_norm;

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    cusparseDnVecDescr_t vecX0, vecR;
    cusparseDnVecDescr_t vecR1, vecTemp;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX0, n, d_x0, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecR, n, d_r, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecR1, n, d_r1_half, CUDA_R_16F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecTemp, n, d_temp_half, CUDA_R_16F));

    size_t bufferSize = 0, bufferSize1 = 0;
    void *dBuffer = nullptr, *dBuffer1 = nullptr;
    cusparseSpMatDescr_t matA, matS_half;
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, A.rows, A.cols, A.nnz,
                                     A.d_csrRowPtr, A.d_csrColInd, A.d_csrVal,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateCsr(&matS_half, S.rows, S.cols, S.nnz,
                                     S.d_csrRowPtr, S.d_csrColInd, S.d_csrVal_half,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F));
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &neg_one, matA, vecX0, &one, vecR,
                                            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    if (bufferSize > 0) {
        CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
    }

    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &one_f, matS_half, vecR1, &zero_f, vecTemp,
                                            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize1));
    if (bufferSize1 > 0) {
        CHECK_CUDA(cudaMalloc(&dBuffer1, bufferSize1));
    }
    
    while (res > tol && kk < max_iter) {

        // r = b - A * x0        
        CHECK_CUDA(cudaMemcpy(d_r, d_b, n * sizeof(double), cudaMemcpyDeviceToDevice));
        CHECK_CUSPARSE(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &neg_one, matA, vecX0, &one, vecR,
                                    CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

        CHECK_CUBLAS(cublasDnrm2(cublasHandle, n, d_r, 1, &r_norm));
        res = r_norm / nr0;
        
        // Convert r to half precision
        double_to_half_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_r, d_r_half, n);

        // r1 = CG_solve(AH, r) in half precision
        cg_half_precision(cusparseHandle, cublasHandle, AH, d_r_half, d_r1_half);

        // r2 = (2-w) * alpha * (alpha*I - S) * r1
        // First: d_r1_half = alpha * I * d_r1_half = alpha * d_r1_half
        // CHECK_CUBLAS(cublasDscal(cublasHandle, n, &alpha_param, d_eye_r1, 1));
        CHECK_CUBLAS(cublasScalEx(cublasHandle, 
            n, 
            &alpha_param_f, 
            CUDA_R_32F, 
            d_r1_half, 
            CUDA_R_16F, 
            1, 
            CUDA_R_32F));
        
        // Then: temp = S * r1
        CHECK_CUSPARSE(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &one_f, matS_half, vecR1, &zero_f, vecTemp,
                                            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer1));

        
        // r2 = alpha * r1 - S * r1 = eye_r1 - temp
        CHECK_CUBLAS(cublasAxpyEx(cublasHandle, 
            n, 
            &neg_one_f, 
            CUDA_R_32F, 
            d_temp_half, 
            CUDA_R_16F, 
            1, 
            d_r1_half, 
            CUDA_R_16F, 
            1,
            CUDA_R_32F));
        
        
        // r2 = (2-w) * alpha * (alpha*I - S) * r1
        factor = (2.0 - w) * alpha_param;
        CHECK_CUBLAS(cublasScalEx(cublasHandle, 
            n, 
            &factor, 
            CUDA_R_32F,
            d_r1_half, 
            CUDA_R_16F,
            1,
            CUDA_R_32F));
        
        // r3 = CG_solve(ASS, r2) in half precision
        cg_half_precision(cusparseHandle, cublasHandle, ASS, d_r1_half, d_r_half);
        
        // Check convergence
        CHECK_CUBLAS(cublasNrm2Ex(cublasHandle, 
            n, 
            d_r_half, 
            CUDA_R_16F, 
            1, 
            &r3_norm,
            CUDA_R_16F,
            CUDA_R_32F));
        if (__half2float(r3_norm) == 0) {
            break;
        }

        // Convert r3 back to double precision
        half_to_double_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_r_half, d_r, n);
        
        // x0 = x0 + r3
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, n, &one, d_r, 1, d_x0, 1));

        
        kk++;
    }

    // Cleanup
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX0));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecR));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecR1));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecTemp));
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroySpMat(matS_half));
    if (dBuffer) cudaFree(dBuffer);
    if (dBuffer1) cudaFree(dBuffer1);
    
    printf("Half-precision solver: res = %e, iter = %d\n", res, kk);
    
    // Cleanup
    cudaFree(d_r);
    cudaFree(d_r_half);
    cudaFree(d_r1_half);
    cudaFree(d_temp_half);

    *res_out = res;
    
    return kk;
}

// Half-precision iterative solver
int float_precision_solver(cusparseHandle_t cusparseHandle, cublasHandle_t cublasHandle,
                          SparseMatrix &A, SparseMatrix &AH, SparseMatrix &ASS, SparseMatrix &S,
                          double *d_b, double *d_x0, double w, double alpha_param,
                          double tol = 1e-6, int max_iter = 1000) {
    
    int n = A.rows;
    double *d_r;
    float *d_r_float, *d_r1_float, *d_temp_float;
    float alpha_param_f = (float)alpha_param;
    
    CHECK_CUDA(cudaMalloc(&d_r, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_r_float, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_r1_float, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_temp_float, n * sizeof(float)));
    
    double nr0;
    CHECK_CUBLAS(cublasDnrm2(cublasHandle, n, d_b, 1, &nr0));
    
    double res = 1.0;
    int kk = 0;
    double one = 1.0, neg_one = -1.0;
    float one_f = 1.0f, neg_one_f = -1.0f, zero_f = 0.0f, factor;
    float r3_norm;
    double r_norm;

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    cusparseDnVecDescr_t vecX0, vecR;
    cusparseDnVecDescr_t vecR1, vecTemp;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX0, n, d_x0, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecR, n, d_r, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecR1, n, d_r1_float, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecTemp, n, d_temp_float, CUDA_R_32F));

    size_t bufferSize = 0, bufferSize1 = 0;
    void *dBuffer = nullptr, *dBuffer1 = nullptr;
    cusparseSpMatDescr_t matA, matS_float;
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, A.rows, A.cols, A.nnz,
                                     A.d_csrRowPtr, A.d_csrColInd, A.d_csrVal,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateCsr(&matS_float, S.rows, S.cols, S.nnz,
                                     S.d_csrRowPtr, S.d_csrColInd, S.d_csrVal_float,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &neg_one, matA, vecX0, &one, vecR,
                                            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    if (bufferSize > 0) {
        CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
    }

    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &one_f, matS_float, vecR1, &zero_f, vecTemp,
                                            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize1));
    if (bufferSize1 > 0) {
        CHECK_CUDA(cudaMalloc(&dBuffer1, bufferSize1));
    }
    
    while (res > tol && kk < max_iter) {

        // r = b - A * x0        
        CHECK_CUDA(cudaMemcpy(d_r, d_b, n * sizeof(double), cudaMemcpyDeviceToDevice));
        CHECK_CUSPARSE(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &neg_one, matA, vecX0, &one, vecR,
                                    CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

        CHECK_CUBLAS(cublasDnrm2(cublasHandle, n, d_r, 1, &r_norm));
        res = r_norm / nr0;
        
        // Convert r to half precision
        double_to_float_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_r, d_r_float, n);

        // r1 = CG_solve(AH, r) in half precision
        cg_float_precision(cusparseHandle, cublasHandle, AH, d_r_float, d_r1_float);

        // r2 = (2-w) * alpha * (alpha*I - S) * r1
        // First: d_r1_half = alpha * I * d_r1_half = alpha * d_r1_half
        // CHECK_CUBLAS(cublasDscal(cublasHandle, n, &alpha_param, d_eye_r1, 1));
        CHECK_CUBLAS(cublasScalEx(cublasHandle, 
            n, 
            &alpha_param_f, 
            CUDA_R_32F, 
            d_r1_float, 
            CUDA_R_32F, 
            1, 
            CUDA_R_32F));
        
        // Then: temp = S * r1
        CHECK_CUSPARSE(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &one_f, matS_float, vecR1, &zero_f, vecTemp,
                                            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer1));

        
        // r2 = alpha * r1 - S * r1 = eye_r1 - temp
        CHECK_CUBLAS(cublasAxpyEx(cublasHandle, 
            n, 
            &neg_one_f, 
            CUDA_R_32F, 
            d_temp_float, 
            CUDA_R_32F, 
            1, 
            d_r1_float, 
            CUDA_R_32F, 
            1,
            CUDA_R_32F));
        
        
        // r2 = (2-w) * alpha * (alpha*I - S) * r1
        factor = (2.0 - w) * alpha_param;
        CHECK_CUBLAS(cublasScalEx(cublasHandle, 
            n, 
            &factor, 
            CUDA_R_32F,
            d_r1_float, 
            CUDA_R_32F,
            1,
            CUDA_R_32F));
        
        // r3 = CG_solve(ASS, r2) in half precision
        cg_float_precision(cusparseHandle, cublasHandle, ASS, d_r1_float, d_r_float);
        
        // Check convergence
        CHECK_CUBLAS(cublasNrm2Ex(cublasHandle, 
            n, 
            d_r_float, 
            CUDA_R_32F, 
            1, 
            &r3_norm,
            CUDA_R_32F,
            CUDA_R_32F));
        if (__half2float(r3_norm) == 0) {
            break;
        }

        // Convert r3 back to double precision
        float_to_double_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_r_float, d_r, n);
        
        // x0 = x0 + r3
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, n, &one, d_r, 1, d_x0, 1));

        kk++;
    }

    // Cleanup
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX0));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecR));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecR1));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecTemp));
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroySpMat(matS_float));
    if (dBuffer) cudaFree(dBuffer);
    if (dBuffer1) cudaFree(dBuffer1);
    
    printf("Half-precision solver: res = %e, iter = %d\n", res, kk);
    
    // Cleanup
    cudaFree(d_r);
    cudaFree(d_r_float);
    cudaFree(d_r1_float);
    cudaFree(d_temp_float);
    
    return kk;
}

// Double-precision iterative solver
int double_precision_solver(cusparseHandle_t cusparseHandle, cublasHandle_t cublasHandle,
                           SparseMatrix &A, SparseMatrix &AH, SparseMatrix &ASS, SparseMatrix &S,
                           double *d_b, double *d_x0, double w, double alpha_param,
                           double tol = 1e-6, int max_iter = 1000) {
    
    int n = A.rows;
    double *d_r, *d_r1, *d_r2, *d_r3;
    double *d_temp, *d_eye_r1;
    
    CHECK_CUDA(cudaMalloc(&d_r, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_r1, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_r2, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_r3, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_temp, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_eye_r1, n * sizeof(double)));
    
    double nr0;
    CHECK_CUBLAS(cublasDnrm2(cublasHandle, n, d_b, 1, &nr0));
    
    double res = 1.0;
    int kk = 0;

    cusparseDnVecDescr_t vecX0, vecR;
    cusparseDnVecDescr_t vecR1, vecTemp;
    
    // Create vector descriptors once before the loop
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX0, n, d_x0, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecR, n, d_r, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecR1, n, d_r1, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecTemp, n, d_temp, CUDA_R_64F));

    cusparseSpMatDescr_t matA, matS;
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, A.rows, A.cols, A.nnz,
                                     A.d_csrRowPtr, A.d_csrColInd, A.d_csrVal,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateCsr(&matS, S.rows, S.cols, S.nnz,
                                     S.d_csrRowPtr, S.d_csrColInd, S.d_csrVal,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    size_t bufferSize = 0, bufferSize1 = 0;
    void *dBuffer = nullptr, *dBuffer1 = nullptr;
    double one = 1.0, zero = 0.0, neg_one = -1.0;

    // spmv buffer size for A
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &neg_one, matA, vecX0, &one, vecR,
                                            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    if (bufferSize > 0) {
        CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
    }

    // spmv buffer size for S
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               &one, matS, vecR1, &zero, vecTemp,
                                               CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize1));
    if (bufferSize1 > 0) {
        CHECK_CUDA(cudaMalloc(&dBuffer1, bufferSize1));
    }
    
    while (res > tol && kk < max_iter) {

        CHECK_CUDA(cudaMemcpy(d_r, d_b, n * sizeof(double), cudaMemcpyDeviceToDevice));
        
        CHECK_CUSPARSE(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &neg_one, matA, vecX0, &one, vecR,
                                    CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
    
        // r1 = CG_solve(AH, r) in double precision
        cg_double_precision(cusparseHandle, cublasHandle, AH, d_r, d_r1);
        
        // r2 = (2-w) * alpha * (alpha*I - S) * r1
        // First: eye_r1 = alpha * I * r1 = alpha * r1
        CHECK_CUDA(cudaMemcpy(d_eye_r1, d_r1, n * sizeof(double), cudaMemcpyDeviceToDevice));
        CHECK_CUBLAS(cublasDscal(cublasHandle, n, &alpha_param, d_eye_r1, 1));
        
        // Then: temp = S * r1
        CHECK_CUSPARSE(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &one, matS, vecR1, &zero, vecTemp,
                                    CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer1));
        
        // r2 = alpha * r1 - S * r1 = eye_r1 - temp
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, n, &neg_one, d_temp, 1, d_eye_r1, 1));
        
        // r2 = (2-w) * alpha * (alpha*I - S) * r1
        double factor = (2.0 - w) * alpha_param;
        CHECK_CUBLAS(cublasDscal(cublasHandle, n, &factor, d_eye_r1, 1));
        CHECK_CUDA(cudaMemcpy(d_r2, d_eye_r1, n * sizeof(double), cudaMemcpyDeviceToDevice));
        
        // r3 = CG_solve(ASS, r2) in double precision
        cg_double_precision(cusparseHandle, cublasHandle, ASS, d_r2, d_r3);
        
        // x0 = x0 + r3
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, n, &one, d_r3, 1, d_x0, 1));
        
        // Check convergence
        double r_norm;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, n, d_r, 1, &r_norm));
        res = r_norm / nr0;
        
        kk++;
    }

    // Cleanup
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX0));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecR));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecR1));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecTemp));
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroySpMat(matS));
    if (dBuffer) cudaFree(dBuffer);
    if (dBuffer1) cudaFree(dBuffer1);
    
    printf("Double-precision solver: res = %e, iter = %d\n", res, kk);
    
    // Cleanup
    cudaFree(d_r);
    cudaFree(d_r1);
    cudaFree(d_r2);
    cudaFree(d_r3);
    cudaFree(d_temp);
    cudaFree(d_eye_r1);
    
    return kk;
}

// Function to create identity matrix in CSR format
void createIdentityMatrix(SparseMatrix &I, int n) {
    I.rows = n;
    I.cols = n;
    I.nnz = n;
    
    CHECK_CUDA(cudaMalloc(&I.d_csrRowPtr, (n + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&I.d_csrColInd, n * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&I.d_csrVal, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&I.d_csrVal_half, n * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&I.d_csrVal_float, n * sizeof(float)));
    
    std::vector<int> row_ptr(n + 1);
    std::vector<int> col_ind(n);
    std::vector<double> val(n, 1.0);
    std::vector<__half> val_half(n);
    std::vector<float> val_float(n, 1.0f);
    
    for (int i = 0; i <= n; i++) {
        row_ptr[i] = i;
    }
    for (int i = 0; i < n; i++) {
        col_ind[i] = i;
        val_half[i] = __float2half(1.0f);
        val_float[i] = 1.0f;
    }
    
    CHECK_CUDA(cudaMemcpy(I.d_csrRowPtr, row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(I.d_csrColInd, col_ind.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(I.d_csrVal, val.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(I.d_csrVal_half, val_half.data(), n * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(I.d_csrVal_float, val_float.data(), n * sizeof(float), cudaMemcpyHostToDevice));
}

// CUDA kernel to scale matrix values
__global__ void scale_matrix(double *d_csrVal, int nnz, double scalar) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nnz) {
        d_csrVal[idx] *= scalar;
    }
}

// Function to transpose a sparse matrix (proper implementation)
void transposeMatrix(cusparseHandle_t cusparseHandle, const SparseMatrix &A, SparseMatrix_double &AT) {
    AT.rows = A.cols;
    AT.cols = A.rows;
    AT.nnz = A.nnz;
    
    // Allocate memory for AT
    CHECK_CUDA(cudaMalloc(&AT.d_csrRowPtr, (AT.rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&AT.d_csrColInd, AT.nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&AT.d_csrVal, AT.nnz * sizeof(double)));
    
    // Use cuSPARSE transpose operation
    // First, create temporary buffers for the transpose operation
    size_t pBufferSizeInBytes = 0;
    void *pBuffer = nullptr;
    
    // Get buffer size for transpose operation
    CHECK_CUSPARSE(cusparseCsr2cscEx2_bufferSize(cusparseHandle, A.rows, A.cols, A.nnz,
                                                 A.d_csrVal, A.d_csrRowPtr, A.d_csrColInd,
                                                 AT.d_csrVal, AT.d_csrRowPtr, AT.d_csrColInd,
                                                 CUDA_R_64F, CUSPARSE_ACTION_NUMERIC,
                                                 CUSPARSE_INDEX_BASE_ZERO,
                                                 CUSPARSE_CSR2CSC_ALG1, &pBufferSizeInBytes));
    
    // Allocate buffer
    if (pBufferSizeInBytes > 0) {
        CHECK_CUDA(cudaMalloc(&pBuffer, pBufferSizeInBytes));
    }
    
    // Perform the transpose operation (CSR to CSC, which is equivalent to transpose)
    CHECK_CUSPARSE(cusparseCsr2cscEx2(cusparseHandle, A.rows, A.cols, A.nnz,
                                      A.d_csrVal, A.d_csrRowPtr, A.d_csrColInd,
                                      AT.d_csrVal, AT.d_csrRowPtr, AT.d_csrColInd,
                                      CUDA_R_64F, CUSPARSE_ACTION_NUMERIC,
                                      CUSPARSE_INDEX_BASE_ZERO,
                                      CUSPARSE_CSR2CSC_ALG1, pBuffer));
    
    // Clean up buffer
    if (pBuffer) {
        cudaFree(pBuffer);
        
    }
    printf("Matrix transpose completed: %d x %d -> %d x %d\n", A.rows, A.cols, AT.rows, AT.cols);
}

// 类型特征：检查是否需要half精度转换
template<typename T>
struct needs_convert_conversion : std::false_type {};

template<>
struct needs_convert_conversion<SparseMatrix> : std::true_type {};

template<typename Type1, typename Type2, typename Type3>
void matrixAdd(cusparseHandle_t cusparseHandle, cublasHandle_t cublasHandle,
               const Type1 &A, const Type2 &B, Type3 &C,
               double scaleA = 1.0, double scaleB = 1.0) {
    
    // Set pointer mode to host
    cusparseSetPointerMode(cusparseHandle, CUSPARSE_POINTER_MODE_HOST);
    
    // Initialize result matrix dimensions
    C.rows = A.rows;
    C.cols = A.cols;
    
    // Step 1: Allocate workspace buffer
    size_t bufferSizeInBytes = 0;
    void *pBuffer = nullptr;
    
    // Create matrix descriptors for legacy API
    cusparseMatDescr_t descrA, descrB, descrC;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrB));
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrC));
    
    CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL));
    
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ZERO));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO));
    
    // Allocate row pointer for result matrix
    CHECK_CUDA(cudaMalloc(&C.d_csrRowPtr, (C.rows + 1) * sizeof(int)));
    
    // Get buffer size for csrgeam2
    CHECK_CUSPARSE(cusparseDcsrgeam2_bufferSizeExt(cusparseHandle, C.rows, C.cols,
                                                   &scaleA, descrA, A.nnz, A.d_csrVal, A.d_csrRowPtr, A.d_csrColInd,
                                                   &scaleB, descrB, B.nnz, B.d_csrVal, B.d_csrRowPtr, B.d_csrColInd,
                                                   descrC, nullptr, C.d_csrRowPtr, nullptr,
                                                   &bufferSizeInBytes));
    
    // Allocate workspace buffer
    if (bufferSizeInBytes > 0) {
        CHECK_CUDA(cudaMalloc(&pBuffer, bufferSizeInBytes));
    }
    
    // Step 2: Determine number of non-zeros in result
    int nnzC;
    int *nnzTotalDevHostPtr = &nnzC;
    CHECK_CUSPARSE(cusparseXcsrgeam2Nnz(cusparseHandle, C.rows, C.cols,
                                        descrA, A.nnz, A.d_csrRowPtr, A.d_csrColInd,
                                        descrB, B.nnz, B.d_csrRowPtr, B.d_csrColInd,
                                        descrC, C.d_csrRowPtr, nnzTotalDevHostPtr, pBuffer));
    
    C.nnz = nnzC;
    
    // Step 3: Allocate memory for column indices and values
    CHECK_CUDA(cudaMalloc(&C.d_csrColInd, C.nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&C.d_csrVal, C.nnz * sizeof(double)));
    
    // Step 4: Perform the actual addition: C = scaleA * A + scaleB * B
    CHECK_CUSPARSE(cusparseDcsrgeam2(cusparseHandle, C.rows, C.cols,
                                     &scaleA, descrA, A.nnz, A.d_csrVal, A.d_csrRowPtr, A.d_csrColInd,
                                     &scaleB, descrB, B.nnz, B.d_csrVal, B.d_csrRowPtr, B.d_csrColInd,
                                     descrC, C.d_csrVal, C.d_csrRowPtr, C.d_csrColInd, pBuffer));
    
    // 如果输出类型需要half精度，分配额外的half精度存储
    if constexpr (needs_convert_conversion<Type3>::value) {
        CHECK_CUDA(cudaMalloc(&C.d_csrVal_half, C.nnz * sizeof(__half)));
        CHECK_CUDA(cudaMalloc(&C.d_csrVal_float, C.nnz * sizeof(float)));
        // Convert to half precision
        int threadsPerBlock = 256;
        int blocksPerGrid = (C.nnz + threadsPerBlock - 1) / threadsPerBlock;
        double_to_half_kernel<<<blocksPerGrid, threadsPerBlock>>>(C.d_csrVal, C.d_csrVal_half, C.nnz);
        
        // Convert to float precision
        double_to_float_kernel<<<blocksPerGrid, threadsPerBlock>>>(C.d_csrVal, C.d_csrVal_float, C.nnz);
    }
    
    // Cleanup
    if (pBuffer) cudaFree(pBuffer);
    cusparseDestroyMatDescr(descrA);
    cusparseDestroyMatDescr(descrB);
    cusparseDestroyMatDescr(descrC);
}

// Function to multiply two sparse matrices using cuSPARSE SpGEMM
void matrixMultiply(cusparseHandle_t cusparseHandle, const SparseMatrix &A, const SparseMatrix_double &B, SparseMatrix &C) {
    
    // Ensure host pointer mode for alpha/beta scalars
    CHECK_CUSPARSE(cusparseSetPointerMode(cusparseHandle, CUSPARSE_POINTER_MODE_HOST));

    // Dimensions
    C.rows = A.rows;
    C.cols = A.cols;
    CHECK_CUDA(cudaMalloc(&C.d_csrRowPtr, (C.rows + 1) * sizeof(int)));

    double scaleA = 1.0, scaleB = 0.0;

    // Create cuSPARSE sparse matrix descriptors for A, I (identity), and C (initialized with B for beta path)
    cusparseSpMatDescr_t matA = nullptr, matB = nullptr, matC = nullptr;

    CHECK_CUSPARSE(cusparseCreateCsr(&matA, A.rows, A.cols, A.nnz,
                                     A.d_csrRowPtr, A.d_csrColInd, A.d_csrVal,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    CHECK_CUSPARSE(cusparseCreateCsr(&matB, B.rows, B.cols, B.nnz,
                                     B.d_csrRowPtr, B.d_csrColInd, B.d_csrVal,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    // Initialize matC with B (as D in: C = alpha*A*I + beta*D), beta = scaleB
    CHECK_CUSPARSE(cusparseCreateCsr(&matC, C.rows, C.cols, 0,
                                     C.d_csrRowPtr, NULL, NULL,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    // SpGEMM descriptors and buffers
    cusparseSpGEMMDescr_t spgemmDesc;
    CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemmDesc));

    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType computeType = CUDA_R_64F;

    void *dBuffer1 = nullptr, *dBuffer2 = nullptr;
    size_t bufferSize1 = 0, bufferSize2 = 0;

    // Work estimation (beta = scaleB, matC initialized with B)
    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(cusparseHandle, opA, opB,
                                                 &scaleA, matA, matB, &scaleB, matC,
                                                 computeType, CUSPARSE_SPGEMM_DEFAULT,
                                                 spgemmDesc, &bufferSize1, nullptr));
    if (bufferSize1 > 0) {
        CHECK_CUDA(cudaMalloc(&dBuffer1, bufferSize1));
    }
    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(cusparseHandle, opA, opB,
                                                 &scaleA, matA, matB, &scaleB, matC,
                                                 computeType, CUSPARSE_SPGEMM_DEFAULT,
                                                 spgemmDesc, &bufferSize1, dBuffer1));

    // Compute phase
    CHECK_CUSPARSE(cusparseSpGEMM_compute(cusparseHandle, opA, opB,
                                          &scaleA, matA, matB, &scaleB, matC,
                                          computeType, CUSPARSE_SPGEMM_DEFAULT,
                                          spgemmDesc, &bufferSize2, nullptr));
    if (bufferSize2 > 0) {
        CHECK_CUDA(cudaMalloc(&dBuffer2, bufferSize2));
    }
    CHECK_CUSPARSE(cusparseSpGEMM_compute(cusparseHandle, opA, opB,
                                          &scaleA, matA, matB, &scaleB, matC,
                                          computeType, CUSPARSE_SPGEMM_DEFAULT,
                                          spgemmDesc, &bufferSize2, dBuffer2));

    // Get nnz of the resulting C
    int64_t c_rows64, c_cols64, c_nnz64;
    CHECK_CUSPARSE(cusparseSpMatGetSize(matC, &c_rows64, &c_cols64, &c_nnz64));
    C.nnz = static_cast<int>(c_nnz64);

    // Allocate output CSR arrays for C
    CHECK_CUDA(cudaMalloc(&C.d_csrColInd, C.nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&C.d_csrVal,    C.nnz * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&C.d_csrVal_half, C.nnz * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&C.d_csrVal_float, C.nnz * sizeof(float)));

    // Update matC pointers to final output buffers
    CHECK_CUSPARSE(cusparseCsrSetPointers(matC, C.d_csrRowPtr, C.d_csrColInd, C.d_csrVal));

    // Copy phase: finalize C = scaleA * (A * I) + scaleB * B
    CHECK_CUSPARSE(cusparseSpGEMM_copy(cusparseHandle, opA, opB,
                                       &scaleA, matA, matB, &scaleB, matC,
                                       computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc));

    // Cleanup SpGEMM resources
    CHECK_CUSPARSE(cusparseSpGEMM_destroyDescr(spgemmDesc));
    if (dBuffer1) cudaFree(dBuffer1);
    if (dBuffer2) cudaFree(dBuffer2);
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroySpMat(matB));
    CHECK_CUSPARSE(cusparseDestroySpMat(matC));

    // Convert values to half precision
    int threadsPerBlock = 256;
    int blocksPerGrid = (C.nnz + threadsPerBlock - 1) / threadsPerBlock;
    double_to_half_kernel<<<blocksPerGrid, threadsPerBlock>>>(C.d_csrVal, C.d_csrVal_half, C.nnz);

    // Convert values to float precision
    double_to_float_kernel<<<blocksPerGrid, threadsPerBlock>>>(C.d_csrVal, C.d_csrVal_float, C.nnz);
    
    printf("SpGEMM completed: %d x %d matrix with %d non-zeros\n", C.rows, C.cols, C.nnz);
}

// Function to print memory usage
void printMemoryUsage() {
    size_t free_bytes, total_bytes;
    CHECK_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));
    
    size_t used_bytes = total_bytes - free_bytes;
    printf("GPU Memory Usage:\n");
    printf("  Used: %.2f MB\n", used_bytes / 1024.0 / 1024.0);
    printf("  Free: %.2f MB\n", free_bytes / 1024.0 / 1024.0);
    printf("  Total: %.2f MB\n", total_bytes / 1024.0 / 1024.0);
}

int main(int argc, char **argv) {

    if (argc != 4) {
        printf("Usage: %s <mtx_file> <alpha> <w>\n", argv[0]);
        printf("  mtx_file: path to matrix market file\n");
        printf("  alpha: alpha parameter (e.g., 0.48)\n");
        printf("  w: w parameter (e.g., 0.1)\n");
        return 1;
    }

    // Initialize CUDA libraries
    cusparseHandle_t cusparseHandle;
    cublasHandle_t cublasHandle;
    
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));
    CHECK_CUBLAS(cublasCreate(&cublasHandle));
    
    // Parse command line arguments
    const char* mtx_file = argv[1];
    double alpha = atof(argv[2]);
    double w = atof(argv[3]);
    
    // Validate parameters
    if (alpha <= 0 || alpha >= 1) {
        printf("Warning: alpha = %f is outside typical range (0, 1)\n", alpha);
    }
    if (w <= 0 || w >= 2) {
        printf("Warning: w = %f is outside typical range (0, 2)\n", w);
    }
    
    printf("Parameters: alpha = %f, w = %f\n", alpha, w);
    
    // Read matrix from MTX file
    SparseMatrix A;
    if (!readMTXFile(mtx_file, A)) {
        printf("Failed to read MTX file\n");
        return 1;
    }
    
    printf("Matrix dimensions: %d x %d, nnz = %d\n", A.rows, A.cols, A.nnz);
    
    // Parameters (matching the Python code)
    // double alpha = 0.9;
    // double w = 0.6;
    int n = A.rows;
    
    // Create identity matrix
    SparseMatrix I;
    createIdentityMatrix(I, n);
    
    // Compute the required matrices according to the algorithm:
    // h = 0.5 * (A + A^T)  // Symmetric part (double)
    // s = 0.5 * (A - A^T)  // Skew-symmetric part (double)
    // ah = alpha * I + h   // AH matrix (two precision)
    // asx = alpha * I + s  // Auxiliary matrix (double)
    // ass = (alpha * I - s) @ asx  // ASS matrix (two precision)
    // (alpha * I - s) // (two precision)

    printf("Computing required matrices...\n");
    
    // Step 1: Compute A^T (transpose of A)
    SparseMatrix_double AT;
    transposeMatrix(cusparseHandle, A, AT);
    
    // Step 2: Compute h = 0.5 * (A + A^T) - symmetric part
    SparseMatrix_double H;
    matrixAdd(cusparseHandle, cublasHandle, A, AT, H, 0.5, 0.5);
    
    // Step 3: Compute s = 0.5 * (A - A^T) - skew-symmetric part  
    SparseMatrix S;
    matrixAdd(cusparseHandle, cublasHandle, A, AT, S, 0.5, -0.5);
    
    // Step 4: Compute AH = alpha * I + h
    SparseMatrix AH;
    matrixAdd(cusparseHandle, cublasHandle, I, H, AH, alpha, 1.0);
    
    // Step 5: Compute asx = alpha * I + s
    SparseMatrix_double ASX;
    matrixAdd(cusparseHandle, cublasHandle, I, S, ASX, alpha, 1.0);
    
    // Step 6: Compute alpha * I - s
    SparseMatrix ALPHA_I_MINUS_S;
    matrixAdd(cusparseHandle, cublasHandle, I, S, ALPHA_I_MINUS_S, alpha, -1.0);
    
    // Step 7: Compute ASS = (alpha * I - s) @ asx
    SparseMatrix ASS;
    matrixMultiply(cusparseHandle, ALPHA_I_MINUS_S, ASX, ASS);
    
    printf("Matrix computation completed.\n");
    printf("H: %d x %d, nnz = %d\n", H.rows, H.cols, H.nnz);
    printf("S: %d x %d, nnz = %d\n", S.rows, S.cols, S.nnz);  
    printf("AH: %d x %d, nnz = %d\n", AH.rows, AH.cols, AH.nnz);
    printf("ASS: %d x %d, nnz = %d\n", ASS.rows, ASS.cols, ASS.nnz);
    
    // Create right-hand side vector (b = A * ones)
    double *d_b, *d_x0_double, *d_x0_half_double, *d_x0_float_double;
    double *d_ones;
    
    CHECK_CUDA(cudaMalloc(&d_b, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_x0_double, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_x0_half_double, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_x0_float_double, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_ones, n * sizeof(double)));
    
    // Initialize ones vector
    std::vector<double> ones(n, 1.0);
    CHECK_CUDA(cudaMemcpy(d_ones, ones.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    
    // b = A @ ones
    cusparseDnVecDescr_t vecOnes, vecB;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecOnes, n, d_ones, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecB, n, d_b, CUDA_R_64F));
    
    double one = 1.0, zero = 0.0;
    size_t bufferSize;
    void *dBuffer = nullptr;

    cusparseSpMatDescr_t A_matDescr;
    CHECK_CUSPARSE(cusparseCreateCsr(&A_matDescr, A.rows, A.cols, A.nnz,
                                     A.d_csrRowPtr, A.d_csrColInd, A.d_csrVal,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &one, A_matDescr, vecOnes, &zero, vecB,
                                           CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    if (bufferSize > 0) {
        CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
    }
    
    CHECK_CUSPARSE(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &one, A_matDescr, vecOnes, &zero, vecB,
                                CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
    if (dBuffer) cudaFree(dBuffer);
    CHECK_CUSPARSE(cusparseDestroySpMat(A_matDescr));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecOnes));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecB));

    double res_half = 0.0;
    half_precision_solver(cusparseHandle, cublasHandle, A, AH, ASS, S,
                                         d_b, d_x0_half_double, w, alpha, &res_half, 1e-6, 5000);
    
    // Initialize solution vectors to zero
    CHECK_CUDA(cudaMemset(d_x0_double, 0, n * sizeof(double)));
    CHECK_CUDA(cudaMemset(d_x0_half_double, 0, n * sizeof(double)));
    CHECK_CUDA(cudaMemset(d_x0_float_double, 0, n * sizeof(double)));
    
    printMemoryUsage();

    // Run half-precision solver
    printf("\n=== Running Half-Precision Solver ===\n");
    
    cudaEvent_t start_half, stop_half;
    CHECK_CUDA(cudaEventCreate(&start_half));
    CHECK_CUDA(cudaEventCreate(&stop_half));
    
    CHECK_CUDA(cudaEventRecord(start_half));
    int iter_half = half_precision_solver(cusparseHandle, cublasHandle, A, AH, ASS, S,
                                         d_b, d_x0_half_double, w, alpha, &res_half, 1e-6, 1000);
    // int iter_half = 0;
    CHECK_CUDA(cudaEventRecord(stop_half));
    CHECK_CUDA(cudaEventSynchronize(stop_half));
    
    float time_half;
    CHECK_CUDA(cudaEventElapsedTime(&time_half, start_half, stop_half));
    
    printf("Half-precision iterations: %d\n", iter_half);
    printf("Half-precision time: %.4f seconds\n", time_half / 1000.0f);


    // Run half-precision solver
    printf("\n=== Running Single-Precision Solver ===\n");
    
    cudaEvent_t start_float, stop_float;
    CHECK_CUDA(cudaEventCreate(&start_float));
    CHECK_CUDA(cudaEventCreate(&stop_float));
    
    CHECK_CUDA(cudaEventRecord(start_float));
    int iter_float = float_precision_solver(cusparseHandle, cublasHandle, A, AH, ASS, S,
                                         d_b, d_x0_float_double, w, alpha, res_half, 1000);
    // int iter_half = 0;
    CHECK_CUDA(cudaEventRecord(stop_float));
    CHECK_CUDA(cudaEventSynchronize(stop_float));
    
    float time_float;
    CHECK_CUDA(cudaEventElapsedTime(&time_float, start_float, stop_float));
    
    printf("Single-precision iterations: %d\n", iter_float);
    printf("Single-precision time: %.4f seconds\n", time_float / 1000.0f);
    
    // Run double-precision solver
    printf("\n=== Running Double-Precision Solver ===\n");
    
    cudaEvent_t start_double, stop_double;
    CHECK_CUDA(cudaEventCreate(&start_double));
    CHECK_CUDA(cudaEventCreate(&stop_double));
    
    CHECK_CUDA(cudaEventRecord(start_double));
    int iter_double = double_precision_solver(cusparseHandle, cublasHandle, A, AH, ASS, S,
                                             d_b, d_x0_double, w, alpha, res_half, 1000);
    // int iter_double = 0;
    CHECK_CUDA(cudaEventRecord(stop_double));
    CHECK_CUDA(cudaEventSynchronize(stop_double));
    
    float time_double;
    CHECK_CUDA(cudaEventElapsedTime(&time_double, start_double, stop_double));
    
    printf("Double-precision iterations: %d\n", iter_double);
    printf("Double-precision time: %.4f seconds\n", time_double / 1000.0f);
    
    // Print speedup
    printf("\n=== Performance Comparison ===\n");
    printf("Speedup (double-precision time / half-precision time): %.2fx\n", 
           (time_double / time_half));

    printf("Speedup (double-precision time / single-precision time): %.2fx\n", 
           (time_double / time_float));

    printf("Speedup (single-precision time / halfe-precision time): %.2fx\n", 
           (time_float / time_half));
    
    printMemoryUsage();
    
    // Cleanup
    cudaFree(d_b);
    cudaFree(d_x0_double);
    cudaFree(d_x0_half_double);
    cudaFree(d_ones);
    
    // Cleanup original matrices
    cudaFree(A.d_csrRowPtr);
    cudaFree(A.d_csrColInd);
    cudaFree(A.d_csrVal);
    cudaFree(A.d_csrVal_half);
    
    cudaFree(I.d_csrRowPtr);
    cudaFree(I.d_csrColInd);
    cudaFree(I.d_csrVal);
    cudaFree(I.d_csrVal_half);
    
    // Cleanup computed matrices
    cudaFree(AT.d_csrRowPtr);
    cudaFree(AT.d_csrColInd);
    cudaFree(AT.d_csrVal);
    
    cudaFree(H.d_csrRowPtr);
    cudaFree(H.d_csrColInd);
    cudaFree(H.d_csrVal);
    
    cudaFree(S.d_csrRowPtr);
    cudaFree(S.d_csrColInd);
    cudaFree(S.d_csrVal);
    cudaFree(S.d_csrVal_half);
    
    cudaFree(AH.d_csrRowPtr);
    cudaFree(AH.d_csrColInd);
    cudaFree(AH.d_csrVal);
    cudaFree(AH.d_csrVal_half);
    
    cudaFree(ASX.d_csrRowPtr);
    cudaFree(ASX.d_csrColInd);
    cudaFree(ASX.d_csrVal);
    
    cudaFree(ASS.d_csrRowPtr);
    cudaFree(ASS.d_csrColInd);
    cudaFree(ASS.d_csrVal);
    cudaFree(ASS.d_csrVal_half);
    
    cudaFree(ALPHA_I_MINUS_S.d_csrRowPtr);
    cudaFree(ALPHA_I_MINUS_S.d_csrColInd);
    cudaFree(ALPHA_I_MINUS_S.d_csrVal);
    cudaFree(ALPHA_I_MINUS_S.d_csrVal_half);
    
    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);
    
    CHECK_CUDA(cudaEventDestroy(start_double));
    CHECK_CUDA(cudaEventDestroy(stop_double));
    CHECK_CUDA(cudaEventDestroy(start_half));
    CHECK_CUDA(cudaEventDestroy(stop_half));
    
    return 0;
}