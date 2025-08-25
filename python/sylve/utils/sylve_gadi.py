import torch
import time
import pynvml
import numpy as np
from scipy.sparse import spdiags, eye, kron, csr_matrix
from scipy import sparse
import scipy


def generate_tridiag(n, a, b, c):
    """生成三对角矩阵，a为下对角线，b为主对角线，c为上对角线"""
    diags = [a * np.ones(n-1), b * np.ones(n), c * np.ones(n-1)]
    return sparse.diags(diags, [-1, 0, 1], shape=(n, n), format='csr')

def generate_sylvester_matrix(n, r):
    """生成指定自由度n和参数r的Sylvester方程系数矩阵A和B"""
    # 生成三对角矩阵 M = Tridiag(-1, 2, -1)
    M = generate_tridiag(n, -1, 2, -1)
    
    # 生成三对角矩阵 N = Tridiag(0.5, 0, -0.5)
    N = generate_tridiag(n, 0.5, 0, -0.5)
    
    # 生成单位矩阵 I
    I = sparse.eye(n, format='csr')
    
    # 计算附加项 100 / (n+1)^2
    scalar = 100 / (n + 1) ** 2
    
    # 构造 A = B = M + 2rN + (100 / (n+1)^2) I
    A = M + 2 * r * N + scalar * I
    B = A.copy()  # 由于 A = B
    
    return A, B


def generate_sylvester_csr(n, r):
    """
    Generate the coefficient matrix of a Sylvester equation in CSR format.
    Parameters:
        m (int): Dimension of matrix A (m x m)
        n (int): Dimension of matrix B (n x n)
        seed (int): Random seed for reproducibility
    Returns:
        scipy.sparse.csr_matrix: Coefficient matrix I_n ⊗ A + B^T ⊗ I_m
    """
    A, B = generate_sylvester_matrix(n, r)
    
    # Compute Kronecker products
    In = sparse.identity(n, format='csr')
    Im = sparse.identity(n, format='csr')
    A_kron = sparse.kron(In, A, format='csr')
    B_transpose = B.T
    B_kron = sparse.kron(B_transpose, Im, format='csr')
    
    # Coefficient matrix: I_n ⊗ A + B^T ⊗ I_m
    coeff_matrix = A_kron + B_kron
    
    return coeff_matrix

# Initialize NVIDIA Management Library
def get_gpu_info():
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            compute_capability = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            # Tensor core support is available for compute capability >= 7.0
            tensor_core_support = compute_capability[0] >= 7
            print(f"GPU {i}: {name}")
            print(f"Compute Capability: {compute_capability[0]}.{compute_capability[1]}")
            print(f"Tensor Core Support: {'Yes' if tensor_core_support else 'No'}")
        pynvml.nvmlShutdown()
    except pynvml.NVMLError as e:
        print(f"NVML Error: {e}")
        return None, None, False
    return name, compute_capability, tensor_core_support

# Parameters
n = 256
r = 1.0

k = n**2

a = generate_sylvester_csr(n, r)

scipy.io.mmwrite('sylve256.mtx', a)

# Right-hand side
x = np.ones(k)
b = a @ x

# Initial guess
x0 = np.zeros(k)

# Preconditioner parameters
alpha = 0.9
w = 0.6
h = 0.5 * (a + a.T)  # Symmetric part
s = 0.5 * (a - a.T)  # Skew-symmetric part
ah = alpha * eye(k) + h
asx = alpha * eye(k) + s
ass = (alpha * eye(k) - s) @ asx

# Convert sparse matrices to PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
a_torch = torch.sparse_csr_tensor(
    torch.tensor(a.indptr, dtype=torch.int64),
    torch.tensor(a.indices, dtype=torch.int64),
    torch.tensor(a.data, dtype=torch.float64),
    size=a.shape
).to(device)
b_torch = torch.tensor(b, dtype=torch.float64, device=device)
x0_torch = torch.tensor(x0, dtype=torch.float64, device=device)

crow_indices = torch.arange(0, k + 1, dtype=torch.int64, device=device)
col_indices = torch.arange(0, k, dtype=torch.int64, device=device)
values = torch.ones(k, dtype=torch.float32, device=device)

eye_csr = torch.sparse_csr_tensor(
    crow_indices=crow_indices,
    col_indices=col_indices,
    values=values,
    size=(k, k)
).to(device)

# Half-precision CG solver with PyTorch
def cg_half_precision(A_half, b, tol=1e-4, max_iter=100):
    b_half = b.to(dtype=torch.float16)
    x = torch.zeros_like(b_half)
    r = b_half - torch.sparse.mm(A_half, x.unsqueeze(1)).squeeze()
    p = r.clone()
    rsold = torch.dot(r, r)

    rsnew = 1.0
    count = 0
    
    for ty in range(max_iter):
        Ap = torch.sparse.mm(A_half, p.unsqueeze(1)).squeeze()
        max_norm_tmp = [torch.max(torch.abs(r)), torch.max(torch.abs(p)), torch.max(torch.abs(Ap))]
        max_norm = torch.max(torch.tensor(max_norm_tmp))
        if max_norm < 1:
            max_norm = 1.0
        if (torch.dot(p/max_norm, Ap/max_norm) == 0).any():
            # print("Warning: Division by zero in CG solver.")
            break
        alpha1 = rsold/(max_norm**2) / torch.dot(p/max_norm, Ap/max_norm)
        x = x + alpha1 * p
        r = r - alpha1 * Ap
        # if rsnew < torch.dot(r, r):
        #     print("Warning: Negative rsnew in CG solver.")
        #     break
        # if rsnew == torch.dot(r, r):
        #     # print("Warning: rsnew equals rsold in CG solver.")
        #     break
        # rsnew = torch.dot(r, r)
        if rsnew <= torch.dot(r, r):
            count += 1
            rsnew = torch.dot(r, r)
            if count > 2:
            # print("Warning: rsnew equals rsold in CG solver.")
                break
        else:
            rsnew = torch.dot(r, r)
            count = 0
        if torch.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    # print("final rsold:", rsold)
    if torch.isnan(rsold).any():
        print("Warning: NaN in rsold.",ty)
        return x.to(dtype=torch.float64)  
    return x.to(dtype=torch.float64)

def cg_double_precision0(A, b, tol=1e-4, max_iter=100):
    A_half = A.to(dtype=torch.float64)
    b_half = b.to(dtype=torch.float64)
    x = torch.zeros_like(b_half)
    r = b_half - torch.sparse.mm(A_half, x.unsqueeze(1)).squeeze()
    p = r.clone()
    rsold = torch.dot(r, r)

    rsnew = 1.0
    count = 0
    
    for ty in range(max_iter):
        Ap = torch.sparse.mm(A_half, p.unsqueeze(1)).squeeze()
        max_norm_tmp = [torch.max(torch.abs(r)), torch.max(torch.abs(p)), torch.max(torch.abs(Ap))]
        max_norm = torch.max(torch.tensor(max_norm_tmp))
        if max_norm < 1:
            max_norm = 1.0
        if (torch.dot(p/max_norm, Ap/max_norm) == 0).any():
            # print("Warning: Division by zero in CG solver.")
            break
        alpha1 = rsold/(max_norm**2) / torch.dot(p/max_norm, Ap/max_norm)
        x = x + alpha1 * p
        r = r - alpha1 * Ap
        # if rsnew < torch.dot(r, r):
        #     print("Warning: Negative rsnew in CG solver.")
        #     break
        # if rsnew == torch.dot(r, r):
        #     # print("Warning: rsnew equals rsold in CG solver.")
        #     break
        # rsnew = torch.dot(r, r)
        if rsnew <= torch.dot(r, r):
            count += 1
            rsnew = torch.dot(r, r)
            if count > 2:
            # print("Warning: rsnew equals rsold in CG solver.")
                break
        else:
            rsnew = torch.dot(r, r)
            count = 0
        if torch.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    # print("final rsold:", rsold)
    if torch.isnan(rsold).any():
        print("Warning: NaN in rsold.",ty)
        return x.to(dtype=torch.float64)  
    return x.to(dtype=torch.float64)

# Double-precision CG solver with PyTorch
def cg_double_precision(A, b, tol=1e-4, max_iter=100):
    x = torch.zeros_like(b)
    r = b - torch.sparse.mm(A, x.unsqueeze(1)).squeeze()
    p = r.clone()
    rsold = torch.dot(r, r)
    
    for _ in range(max_iter):
        Ap = torch.sparse.mm(A, p.unsqueeze(1)).squeeze()
        alpha = rsold / torch.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.dot(r, r)
        if torch.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x

# Half-precision iterative solver
def half_precision_solver(a, b, x0, ah, asx, ass, w, alpha, tol=1e-6, max_iter=1000):
    a_torch = torch.sparse_csr_tensor(
        torch.tensor(a.indptr, dtype=torch.int64),
        torch.tensor(a.indices, dtype=torch.int64),
        torch.tensor(a.data, dtype=torch.float64),
        size=a.shape
    ).to(device)
    b_torch = torch.tensor(b, dtype=torch.float64, device=device)
    x0_torch = torch.tensor(x0, dtype=torch.float64, device=device)
    
    ah_torch = torch.sparse_csr_tensor(
        torch.tensor(ah.indptr, dtype=torch.int64),
        torch.tensor(ah.indices, dtype=torch.int64),
        torch.tensor(ah.data, dtype=torch.float64),
        size=ah.shape
    ).to(device)
    ah_torch_half = torch.sparse_csr_tensor(
        torch.tensor(ah.indptr, dtype=torch.int64),
        torch.tensor(ah.indices, dtype=torch.int64),
        torch.tensor(ah.data, dtype=torch.float16),
        size=ah.shape
    ).to(device)
    as_torch = torch.sparse_csr_tensor(
        torch.tensor(asx.indptr, dtype=torch.int64),
        torch.tensor(asx.indices, dtype=torch.int64),
        torch.tensor(asx.data, dtype=torch.float64),
        size=asx.shape
    ).to(device)
    ass_torch = torch.sparse_csr_tensor(
        torch.tensor(ass.indptr, dtype=torch.int64),
        torch.tensor(ass.indices, dtype=torch.int64),
        torch.tensor(ass.data, dtype=torch.float64),
        size=ass.shape
    ).to(device)
    ass_torch_half = torch.sparse_csr_tensor(
        torch.tensor(ass.indptr, dtype=torch.int64),
        torch.tensor(ass.indices, dtype=torch.int64),
        torch.tensor(ass.data, dtype=torch.float16),
        size=ass.shape
    ).to(device)
    
    s_torch = torch.sparse_csr_tensor(
        torch.tensor(s.indptr, dtype=torch.int64),
        torch.tensor(s.indices, dtype=torch.int64),
        torch.tensor(s.data, dtype=torch.float64),
        size=s.shape
    ).to(device)
    
    nr0 = torch.norm(b_torch)
    res = 1.0
    kk = 0

    start_time = time.time()
    
    while res > tol and kk < max_iter:
        r = b_torch - torch.sparse.mm(a_torch, x0_torch.unsqueeze(1)).squeeze()
        r1 = cg_half_precision(ah_torch_half, r)
        r2 = (2 - w) * alpha * torch.sparse.mm(
            (alpha * eye_csr.to(dtype=torch.float64) + (-1.0) * s_torch).to(dtype=torch.float16), 
            r1.to(dtype=torch.float16).unsqueeze(1)
        ).to(dtype=torch.float64).squeeze()
        r3 = cg_half_precision(ass_torch_half, r2)
        x0_torch = x0_torch + r3
        if (torch.norm(r3) == 0):
            # print("res: ", res, "iter:", kk)
            break
        res = torch.norm(r) / nr0
        # print(res)
        if torch.isnan(r3[0]):
            print("nan",kk)
        # print("res: ", res, "iter:", kk)
        kk += 1

    print("res: ", res, "iter:", kk)
    end_time = time.time()
    return kk, end_time - start_time, res

# Double-precision iterative solver
def double_precision_solver(a, b, x0, ah, asx, ass, w, alpha, tol=1e-6, max_iter=1000):
    a_torch = torch.sparse_csr_tensor(
        torch.tensor(a.indptr, dtype=torch.int64),
        torch.tensor(a.indices, dtype=torch.int64),
        torch.tensor(a.data, dtype=torch.float64),
        size=a.shape
    ).to(device)
    b_torch = torch.tensor(b, dtype=torch.float64, device=device)
    x0_torch = torch.tensor(x0, dtype=torch.float64, device=device)
    
    ah_torch = torch.sparse_csr_tensor(
        torch.tensor(ah.indptr, dtype=torch.int64),
        torch.tensor(ah.indices, dtype=torch.int64),
        torch.tensor(ah.data, dtype=torch.float64),
        size=ah.shape
    ).to(device)
    as_torch = torch.sparse_csr_tensor(
        torch.tensor(asx.indptr, dtype=torch.int64),
        torch.tensor(asx.indices, dtype=torch.int64),
        torch.tensor(asx.data, dtype=torch.float64),
        size=asx.shape
    ).to(device)
    ass_torch = torch.sparse_csr_tensor(
        torch.tensor(ass.indptr, dtype=torch.int64),
        torch.tensor(ass.indices, dtype=torch.int64),
        torch.tensor(ass.data, dtype=torch.float64),
        size=ass.shape
    ).to(device)
    
    s_torch = torch.sparse_csr_tensor(
        torch.tensor(s.indptr, dtype=torch.int64),
        torch.tensor(s.indices, dtype=torch.int64),
        torch.tensor(s.data, dtype=torch.float64),
        size=s.shape
    ).to(device)
    
    nr0 = torch.norm(b_torch)
    res = 1.0
    kk = 0

    start_time = time.time()
    
    while res > tol and kk < max_iter:
        r = b_torch - torch.sparse.mm(a_torch, x0_torch.unsqueeze(1)).squeeze()
        r1 = cg_double_precision0(ah_torch, r)
        r2 = (2 - w) * alpha * torch.sparse.mm(
            (alpha * eye_csr.to(dtype=torch.float64) + (-1.0) * s_torch), 
            r1.unsqueeze(1)
        ).squeeze()
        r3 = cg_double_precision0(ass_torch, r2)
        x0_torch = x0_torch + r3
        res = torch.norm(r) / nr0
        # print(res)
        kk += 1
    
    print("res: ", res, "iter:", kk)
    end_time = time.time()
    return kk, end_time - start_time

# Get GPU information
get_gpu_info()

# Run solvers and compare
print("\nRunning double-precision solver...")
kk_double, time_double = double_precision_solver(a, b, x0, ah, asx, ass, w, alpha, tol=1e-6, max_iter=5000)
print(f"Double-precision solver iterations: {kk_double}")
print(f"Double-precision solver time: {time_double:.4f} seconds")

print("\nRunning half-precision solver...")
kk_half, time_half, tol_half = half_precision_solver(a, b, x0, ah, asx, ass, w, alpha, tol=1e-6, max_iter=5000)
print(f"Half-precision solver iterations: {kk_half}")
print(f"Half-precision solver time: {time_half:.4f} seconds")

print("\nRunning double-precision solver...")
kk_double, time_double = double_precision_solver(a, b, x0, ah, asx, ass, w, alpha, tol=tol_half, max_iter=5000)
print(f"Double-precision solver iterations: {kk_double}")
print(f"Double-precision solver time: {time_double:.4f} seconds")


print(f"\nSpeedup (double-precision time / half-precision time): {time_double/time_half:.2f}x")