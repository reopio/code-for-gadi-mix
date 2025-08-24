import numpy as np
from scipy.sparse import spdiags, eye, kron, csr_matrix
from scipy import sparse
from scipy.io import mmwrite

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
    
    # Save matrix to MTX format
    filename = f"sylve{n}.mtx"
    mmwrite(filename, coeff_matrix)
    print(f"Matrix saved to {filename}")
    
    return coeff_matrix

def main():
    """主函数：生成Sylvester方程系数矩阵并保存
    """
    n = 2560  # 矩阵维度
    r = 1.0  # 参数r
    coeff_matrix = generate_sylvester_csr(n, r)
    print(f"Sylvester coefficient matrix of size {n} generated and saved.")


if __name__ == "__main__":
    main()