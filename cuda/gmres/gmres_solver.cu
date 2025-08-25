// GMRES solver using CUDA APIs (cuSPARSE + cuBLAS), no custom kernels.
// Usage: gmres_solver <matrix.mtx> [tol=1e-8] [maxIter=1000] [restart=50]
// b is generated as b = A * x_true, where x_true is all ones.
// Timing (solver only) with CUDA events.

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cmath>

#define CHECK_CUDA(call) do { \
	cudaError_t _e = (call); \
	if(_e != cudaSuccess){ \
		fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
		exit(EXIT_FAILURE); \
	} \
} while(0)

#define CHECK_CUBLAS(call) do { \
	cublasStatus_t _s = (call); \
	if(_s != CUBLAS_STATUS_SUCCESS){ \
		fprintf(stderr, "CUBLAS Error %s:%d: status=%d\n", __FILE__, __LINE__, _s); \
		exit(EXIT_FAILURE); \
	} \
} while(0)

#define CHECK_CUSPARSE(call) do { \
	cusparseStatus_t _s = (call); \
	if(_s != CUSPARSE_STATUS_SUCCESS){ \
		fprintf(stderr, "CUSPARSE Error %s:%d: status=%d\n", __FILE__, __LINE__, _s); \
		exit(EXIT_FAILURE); \
	} \
} while(0)

struct CSRMatrix {
	int rows = 0, cols = 0, nnz = 0;
	std::vector<int> rowPtr; // size rows+1
	std::vector<int> colInd; // size nnz
	std::vector<double> values; // size nnz
};

static void trim(std::string &s){
	while(!s.empty() && (s.back()=='\r' || s.back()=='\n' || s.back()==' ' || s.back()=='\t')) s.pop_back();
}

CSRMatrix readMatrixMarket(const std::string &path){
	std::ifstream fin(path);
	if(!fin){
		throw std::runtime_error("Cannot open file: "+path);
	}
	std::string line;
	if(!std::getline(fin,line)) throw std::runtime_error("Empty file");
	trim(line);
	if(line.rfind("%%MatrixMarket",0)!=0) throw std::runtime_error("Invalid MatrixMarket header");
	bool isCoordinate = line.find("coordinate")!=std::string::npos;
	bool isReal = (line.find("real")!=std::string::npos) || (line.find("double")!=std::string::npos);
	bool isGeneral = line.find("general")!=std::string::npos;
	bool isSymmetric = line.find("symmetric")!=std::string::npos;
	if(!isCoordinate || !isReal){
		throw std::runtime_error("Only coordinate real format supported");
	}
	// Skip comments
	do {
		if(line.size()>0 && line[0] != '%') break;
		if(!std::getline(fin,line)) throw std::runtime_error("Unexpected EOF while reading size line");
		trim(line);
	} while(true);
	std::istringstream iss(line);
	int M,N,NNZ; iss>>M>>N>>NNZ;
	if(!iss) throw std::runtime_error("Failed to parse size line");
	std::vector<int> I; I.reserve(NNZ * (isSymmetric?2:1));
	std::vector<int> J; J.reserve(NNZ * (isSymmetric?2:1));
	std::vector<double> V; V.reserve(NNZ * (isSymmetric?2:1));
	int r,c; double v;
	int read=0;
	while(read<NNZ && (fin>>r>>c>>v)){
		--r; --c; // 0-based
		I.push_back(r); J.push_back(c); V.push_back(v);
		if(isSymmetric && r!=c){
			I.push_back(c); J.push_back(r); V.push_back(v);
		}
		read++;
	}
	if(read!=NNZ) throw std::runtime_error("Read fewer entries than expected");
	if(isSymmetric) NNZ = (int)I.size();
	CSRMatrix A; A.rows=M; A.cols=N; A.nnz=NNZ;
	A.rowPtr.assign(M+1,0);
	for(int k=0;k<NNZ;k++) A.rowPtr[I[k]+1]++;
	for(int i=0;i<M;i++) A.rowPtr[i+1]+=A.rowPtr[i];
	A.colInd.assign(NNZ,0);
	A.values.assign(NNZ,0.0);
	// Temporary copy of rowPtr to use as insertion offsets
	std::vector<int> offset = A.rowPtr;
	for(int k=0;k<NNZ;k++){
		int row = I[k];
		int dest = offset[row]++;
		A.colInd[dest] = J[k];
		A.values[dest] = V[k];
	}
	// Optionally sort columns within each row (simple insertion sort for small rows)
	for(int i=0;i<M;i++){
		int start = A.rowPtr[i];
		int end = A.rowPtr[i+1];
		int len = end-start;
		for(int a=start+1;a<end;a++){
			int cj = A.colInd[a];
			double vv = A.values[a];
			int b=a-1;
			while(b>=start && A.colInd[b]>cj){
				A.colInd[b+1]=A.colInd[b];
				A.values[b+1]=A.values[b];
				b--;
			}
			A.colInd[b+1]=cj; A.values[b+1]=vv;
		}
	}
	return A;
}

struct DeviceCSR {
	int rows=0, cols=0, nnz=0;
	int *d_rowPtr=nullptr; int *d_colInd=nullptr; double *d_vals=nullptr;
	~DeviceCSR(){
		if(d_rowPtr) cudaFree(d_rowPtr);
		if(d_colInd) cudaFree(d_colInd);
		if(d_vals) cudaFree(d_vals);
	}
};

DeviceCSR uploadToDevice(const CSRMatrix &A){
	DeviceCSR dA; dA.rows=A.rows; dA.cols=A.cols; dA.nnz=A.nnz;
	CHECK_CUDA(cudaMalloc((void**)&dA.d_rowPtr, sizeof(int)*(A.rows+1)));
	CHECK_CUDA(cudaMalloc((void**)&dA.d_colInd, sizeof(int)*A.nnz));
	CHECK_CUDA(cudaMalloc((void**)&dA.d_vals, sizeof(double)*A.nnz));
	CHECK_CUDA(cudaMemcpy(dA.d_rowPtr, A.rowPtr.data(), sizeof(int)*(A.rows+1), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(dA.d_colInd, A.colInd.data(), sizeof(int)*A.nnz, cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(dA.d_vals, A.values.data(), sizeof(double)*A.nnz, cudaMemcpyHostToDevice));
	return dA;
}

// Perform y = A * x (SpMV) using cuSPARSE (CSR double)
void spmv(cusparseHandle_t handle, const DeviceCSR &A, const double *d_x, double *d_y, void *dBuffer, size_t bufferSize){
	static cusparseSpMatDescr_t mat = nullptr; // reuse if possible
	static int lastNnz=-1, lastRows=-1, lastCols=-1;
	if(mat==nullptr || lastNnz!=A.nnz || lastRows!=A.rows || lastCols!=A.cols){
		if(mat) cusparseDestroySpMat(mat);
		CHECK_CUSPARSE(cusparseCreateCsr(&mat, A.rows, A.cols, A.nnz, (void*)A.d_rowPtr, (void*)A.d_colInd, (void*)A.d_vals,
										 CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
		lastNnz=A.nnz; lastRows=A.rows; lastCols=A.cols;
	}
	cusparseDnVecDescr_t vecX, vecY;
	CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, A.cols, (void*)d_x, CUDA_R_64F));
	CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, A.rows, (void*)d_y, CUDA_R_64F));
	const double alpha=1.0, beta=0.0;
	CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat, vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
	cusparseDestroyDnVec(vecX);
	cusparseDestroyDnVec(vecY);
}

// r = b - A x (using auxiliary temp for Ax)
void computeResidual(cusparseHandle_t handle, const DeviceCSR &A, cublasHandle_t blas,
					 const double *d_b, const double *d_x, double *d_r, double *d_tmp,
					 void *dBuffer, size_t bufferSize){
	spmv(handle, A, d_x, d_tmp, dBuffer, bufferSize); // d_tmp = A x
	// r = b - tmp
	CHECK_CUBLAS(cublasDcopy(blas, A.rows, d_b, 1, d_r, 1));
	const double negOne = -1.0;
	CHECK_CUBLAS(cublasDaxpy(blas, A.rows, &negOne, d_tmp, 1, d_r, 1));
}

int main(int argc, char** argv){
	if(argc < 2){
		std::cerr << "Usage: " << argv[0] << " <matrix.mtx> [tol=1e-8] [maxIter=1000] [restart=50]\n";
		return 1;
	}
	std::string matrixFile = argv[1];
	double tol = (argc>2)? std::atof(argv[2]) : 1e-6;
	int maxIter = (argc>3)? std::atoi(argv[3]) : 1000;
	int restart = (argc>4)? std::atoi(argv[4]) : 50;
	try {
		CSRMatrix A = readMatrixMarket(matrixFile);
		if(A.rows != A.cols){
			std::cerr << "Matrix must be square." << std::endl;
			return 1;
		}
		int n = A.rows;
		if(restart > n) restart = n;
		DeviceCSR dA = uploadToDevice(A);

		// cuSPARSE & cuBLAS handles
		cusparseHandle_t cusparseHandle; CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));
		cublasHandle_t cublasHandle; CHECK_CUBLAS(cublasCreate(&cublasHandle));

		// Allocate vectors
		double *d_x_true=nullptr, *d_b=nullptr, *d_x=nullptr, *d_r=nullptr, *d_tmp=nullptr; // size n
		CHECK_CUDA(cudaMalloc((void**)&d_x_true, sizeof(double)*n));
		CHECK_CUDA(cudaMalloc((void**)&d_b, sizeof(double)*n));
		CHECK_CUDA(cudaMalloc((void**)&d_x, sizeof(double)*n));
		CHECK_CUDA(cudaMalloc((void**)&d_r, sizeof(double)*n));
		CHECK_CUDA(cudaMalloc((void**)&d_tmp, sizeof(double)*n));
		// Initialize x_true = ones, x = 0
		std::vector<double> h_ones(n,1.0), h_zeros(n,0.0);
		CHECK_CUDA(cudaMemcpy(d_x_true, h_ones.data(), sizeof(double)*n, cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(d_x, h_zeros.data(), sizeof(double)*n, cudaMemcpyHostToDevice));

		// cuSPARSE SpMV buffer (query size)
		cusparseSpMatDescr_t matA;
		CHECK_CUSPARSE(cusparseCreateCsr(&matA, dA.rows, dA.cols, dA.nnz, (void*)dA.d_rowPtr, (void*)dA.d_colInd, (void*)dA.d_vals,
										CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
		cusparseDnVecDescr_t vecX, vecB;
		CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, n, (void*)d_x_true, CUDA_R_64F));
		CHECK_CUSPARSE(cusparseCreateDnVec(&vecB, n, (void*)d_b, CUDA_R_64F));
		size_t bufferSize=0; void *dBuffer=nullptr;
		const double alpha1=1.0, beta0=0.0;
		CHECK_CUSPARSE(cusparseSpMV_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha1, matA, vecX, &beta0, vecB, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
		CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
		// Compute b = A * ones
		CHECK_CUSPARSE(cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha1, matA, vecX, &beta0, vecB, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
		cusparseDestroyDnVec(vecX); cusparseDestroyDnVec(vecB); cusparseDestroySpMat(matA);

		// GMRES storage: V (n x (restart+1)) on device
		double *d_V=nullptr; CHECK_CUDA(cudaMalloc((void**)&d_V, sizeof(double)*n*(restart+1)));

		std::vector<double> H((restart+1)*restart,0.0); // (restart+1) x restart
		std::vector<double> cs(restart,0.0), sn(restart,0.0), e1(restart+1,0.0);
		e1[0]=1.0;

		// Residual history container
		std::vector<double> residualHistory; residualHistory.reserve(maxIter+1);

		// Prepare events for timing
		cudaEvent_t evStart, evStop; CHECK_CUDA(cudaEventCreate(&evStart)); CHECK_CUDA(cudaEventCreate(&evStop));
		CHECK_CUDA(cudaEventRecord(evStart));

		double normb=0.0;
		CHECK_CUBLAS(cublasDnrm2(cublasHandle, n, d_b, 1, &normb));
		if(normb == 0.0){
			std::cerr << "Zero RHS; solution is zero vector." << std::endl;
			return 0;
		}

		int iter=0; // total inner iterations done
		int restartCount=0; // number of outer restart cycles entered
		double relResidual=1.0; bool converged=false;
		// Temporary vector w (share d_tmp) for SpMV results and orthogonalization workspace
		// Outer loop (restarts)
		while(iter < maxIter && !converged){
			restartCount++;
			// Reset Hessenberg / rotations for this cycle
			std::fill(H.begin(), H.end(), 0.0);
			std::fill(cs.begin(), cs.end(), 0.0);
			std::fill(sn.begin(), sn.end(), 0.0);
			// r0 = b - A x
			computeResidual(cusparseHandle, dA, cublasHandle, d_b, d_x, d_r, d_tmp, dBuffer, bufferSize);
			double beta=0.0; CHECK_CUBLAS(cublasDnrm2(cublasHandle, n, d_r, 1, &beta));
			relResidual = beta / normb;
			residualHistory.push_back(relResidual); // residual at start of (re)start
			if(relResidual < tol){ converged=true; break; }
			// v1 = r0 / beta -> store as V_0
			CHECK_CUBLAS(cublasDcopy(cublasHandle, n, d_r, 1, d_V + 0, 1));
			double invBeta = 1.0 / beta;
			CHECK_CUBLAS(cublasDscal(cublasHandle, n, &invBeta, d_V + 0, 1));
			// g = beta * e1
			std::vector<double> g(restart+1,0.0); g[0]=beta;
			int k=0; // number of columns produced so far (Arnoldi steps)
			for(; k<restart; ++k){
				// w = A * v_k
				spmv(cusparseHandle, dA, d_V + k*n, d_tmp, dBuffer, bufferSize); // d_tmp = w
				// Modified Gram-Schmidt against existing basis vectors v_0..v_k
				for(int i=0;i<=k;i++){
					double hij=0.0; CHECK_CUBLAS(cublasDdot(cublasHandle, n, d_tmp, 1, d_V + i*n, 1, &hij));
					H[i + k*(restart+1)] = hij;
					double negHij = -hij;
					CHECK_CUBLAS(cublasDaxpy(cublasHandle, n, &negHij, d_V + i*n, 1, d_tmp, 1));
				}
				double hnext=0.0; CHECK_CUBLAS(cublasDnrm2(cublasHandle, n, d_tmp, 1, &hnext));
				H[(k+1) + k*(restart+1)] = hnext;
				if(hnext != 0.0 && k+1 < restart+1){
					double inv = 1.0 / hnext;
					CHECK_CUBLAS(cublasDcopy(cublasHandle, n, d_tmp, 1, d_V + (k+1)*n, 1));
					CHECK_CUBLAS(cublasDscal(cublasHandle, n, &inv, d_V + (k+1)*n, 1));
				}
				// Apply existing Givens rotations to the new column (rows 0..k+1)
				for(int i=0;i<k;i++){
					double temp = cs[i]*H[i + k*(restart+1)] + sn[i]*H[i+1 + k*(restart+1)];
					H[i+1 + k*(restart+1)] = -sn[i]*H[i + k*(restart+1)] + cs[i]*H[i+1 + k*(restart+1)];
					H[i + k*(restart+1)] = temp;
				}
				// Generate new Givens rotation to eliminate H[k+1,k]
				double h_kk = H[k + k*(restart+1)];
				double h_k1k = H[k+1 + k*(restart+1)];
				double denom = std::hypot(h_kk, h_k1k);
				if(denom == 0.0){
					cs[k]=1.0; sn[k]=0.0;
				} else {
					cs[k] = h_kk / denom;
					sn[k] = h_k1k / denom;
				}
				H[k + k*(restart+1)] = cs[k]*h_kk + sn[k]*h_k1k;
				H[k+1 + k*(restart+1)] = 0.0; // eliminated
				// Update g vector (apply rotation)
				double gtemp = cs[k]*g[k] + sn[k]*g[k+1];
				g[k+1] = -sn[k]*g[k] + cs[k]*g[k+1];
				g[k] = gtemp;
				relResidual = std::fabs(g[k+1]) / normb;
				residualHistory.push_back(relResidual);
				iter++; // count this Arnoldi / GMRES inner iteration
				// Early exit conditions
				if(relResidual < tol){
					k++; // include current column in solution phase
					break;
				}
				if(iter >= maxIter){
					k++; // we've reached global limit; solve with current basis
					break;
				}
			}
			int kCols = k; // number of columns in R / size of y
			// Solve upper triangular system R y = g (R is kCols x kCols in H)
			std::vector<double> y(kCols,0.0);
			for(int row=kCols-1; row>=0; --row){
				double sum = g[row];
				for(int col=row+1; col<kCols; ++col){
					sum -= H[row + col*(restart+1)] * y[col];
				}
				double diag = H[row + row*(restart+1)];
				if(std::fabs(diag) < 1e-30) diag = (diag >= 0 ? 1e-30 : -1e-30); // safeguard
				y[row] = sum / diag;
			}
			// x = x + V_k y
			for(int col=0; col<kCols; ++col){
				double alpha = y[col];
				CHECK_CUBLAS(cublasDaxpy(cublasHandle, n, &alpha, d_V + col*n, 1, d_x, 1));
			}
			if(relResidual < tol){ converged=true; break; }
		}

		CHECK_CUDA(cudaEventRecord(evStop));
		CHECK_CUDA(cudaEventSynchronize(evStop));
		float ms=0.0f; CHECK_CUDA(cudaEventElapsedTime(&ms, evStart, evStop));

		std::cout << "Matrix: " << matrixFile << " (" << A.rows << "x" << A.cols << ", nnz=" << A.nnz << ")\n";
		std::cout << "GMRES parameters: tol=" << tol << ", maxIter=" << maxIter << ", restart=" << restart << "\n";
		std::cout << "Total iterations: " << iter << "\n";
		std::cout << "Restart cycles: " << restartCount << "\n";
		std::cout << "Final relative residual: " << relResidual << "\n";
		std::cout << "Solve time (ms): " << ms << "\n";
		// Write residual history to file
		{
			std::ofstream fres("gmres_residuals.txt");
			if(fres){
				fres << "# iter residual\n";
				for(size_t i=0;i<residualHistory.size();++i){
					fres << i << " " << residualHistory[i] << "\n";
				}
				std::cout << "Residual history written to gmres_residuals.txt (" << residualHistory.size() << " entries)\n";
			} else {
				std::cerr << "Warning: cannot write gmres_residuals.txt" << std::endl;
			}
		}

		// Cleanup
		if(d_V) cudaFree(d_V);
		if(d_x_true) cudaFree(d_x_true);
		if(d_b) cudaFree(d_b);
		if(d_x) cudaFree(d_x);
		if(d_r) cudaFree(d_r);
		if(d_tmp) cudaFree(d_tmp);
		if(dBuffer) cudaFree(dBuffer);
		cublasDestroy(cublasHandle);
		cusparseDestroy(cusparseHandle);
		cudaEventDestroy(evStart); cudaEventDestroy(evStop);
	} catch(const std::exception &ex){
		std::cerr << "Error: " << ex.what() << std::endl;
		return 1;
	}
	return 0;
}

