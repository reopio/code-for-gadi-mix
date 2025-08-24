import numpy as np
from scipy import sparse
import scipy.io
from generate_mat import generate_sylvester_csr, generate_sylvester_matrix

def check_csr_structure_symmetry(matrix):
    """
    检查CSR稀疏矩阵的结构是否对称
    返回: (is_symmetric, details)
    """
    if not sparse.issparse(matrix):
        raise ValueError("输入必须是稀疏矩阵")
    
    # 确保是CSR格式
    if not sparse.isspmatrix_csr(matrix):
        matrix = matrix.tocsr()
    
    # 计算转置
    matrix_T = matrix.T.tocsr()
    
    # 检查形状
    if matrix.shape != matrix_T.shape:
        return False, "矩阵不是方阵"
    
    # 检查indptr数组是否相同
    indptr_same = np.array_equal(matrix.indptr, matrix_T.indptr)
    
    # 检查每行的非零元素数量是否相同
    nnz_per_row_original = np.diff(matrix.indptr)
    nnz_per_row_transpose = np.diff(matrix_T.indptr)
    nnz_same = np.array_equal(nnz_per_row_original, nnz_per_row_transpose)
    
    # 检查indices数组结构
    # 对于结构对称的矩阵，每行的列索引排序后应该对应转置矩阵的相应行
    indices_structure_same = True
    for i in range(matrix.shape[0]):
        row_start, row_end = matrix.indptr[i], matrix.indptr[i+1]
        row_indices = np.sort(matrix.indices[row_start:row_end])
        
        row_start_T, row_end_T = matrix_T.indptr[i], matrix_T.indptr[i+1]
        row_indices_T = np.sort(matrix_T.indices[row_start_T:row_end_T])
        
        if not np.array_equal(row_indices, row_indices_T):
            indices_structure_same = False
            break
    
    # 总体结构对称性
    is_structure_symmetric = indptr_same and nnz_same and indices_structure_same
    
    details = {
        'indptr_same': indptr_same,
        'nnz_per_row_same': nnz_same,
        'indices_structure_same': indices_structure_same,
        'matrix_shape': matrix.shape,
        'total_nnz': matrix.nnz,
        'nnz_original': matrix.nnz,
        'nnz_transpose': matrix_T.nnz
    }
    
    return is_structure_symmetric, details

def analyze_matrix_structure(matrix, matrix_name="Matrix"):
    """
    分析矩阵结构的详细信息
    """
    print(f"\n=== {matrix_name} 结构分析 ===")
    print(f"矩阵形状: {matrix.shape}")
    print(f"非零元素数量: {matrix.nnz}")
    print(f"稀疏度: {1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.6f}")
    
    # 检查结构对称性
    is_symmetric, details = check_csr_structure_symmetry(matrix)
    
    print(f"结构对称性: {'是' if is_symmetric else '否'}")
    print(f"  - indptr数组相同: {'是' if details['indptr_same'] else '否'}")
    print(f"  - 每行非零元素数量相同: {'是' if details['nnz_per_row_same'] else '否'}")
    print(f"  - 列索引结构相同: {'是' if details['indices_structure_same'] else '否'}")
    
    # 检查数值对称性
    matrix_T = matrix.T.tocsr()
    diff_matrix = matrix - matrix_T
    is_numerically_symmetric = np.allclose(diff_matrix.data, 0, atol=1e-14)
    max_diff = np.max(np.abs(diff_matrix.data)) if diff_matrix.nnz > 0 else 0
    
    print(f"数值对称性: {'是' if is_numerically_symmetric else '否'}")
    print(f"最大对称性偏差: {max_diff:.2e}")
    
    return is_symmetric, is_numerically_symmetric

def main():
    """
    主函数：测试sylve_gadi.py中生成的矩阵
    """
    print("检测 sylve_gadi.py 中生成的矩阵结构对称性")
    
    # 测试参数
    n_values = [128, 256]
    r_values = [2.0]
    
    for n in n_values:
        for r in r_values:
            print(f"\n" + "="*60)
            print(f"测试参数: n={n}, r={r}")
            
            # # 生成Sylvester矩阵 A 和 B
            # A, B = generate_sylvester_matrix(n, r)
            # print(f"\n--- 矩阵 A (n={n}, r={r}) ---")
            # analyze_matrix_structure(A, f"Matrix A (n={n}, r={r})")
            
            # print(f"\n--- 矩阵 B (n={n}, r={r}) ---")
            # analyze_matrix_structure(B, f"Matrix B (n={n}, r={r})")
            
            # 生成Sylvester方程系数矩阵
            coeff_matrix = generate_sylvester_csr(n, r)
            print(f"\n--- Sylvester系数矩阵 (n={n}, r={r}) ---")
            analyze_matrix_structure(coeff_matrix, f"Sylvester Coefficient Matrix (n={n}, r={r})")
    
    # 测试当前脚本中的矩阵 (n=256, r=1.0)
    print(f"\n" + "="*80)
    print("测试当前脚本中的矩阵 (n=256, r=1.0)")
    
    # 读取保存的矩阵文件（如果存在）
    try:
        saved_matrix = scipy.io.mmread('sylve256.mtx').tocsr()
        print(f"\n--- 从文件读取的矩阵 sylve256.mtx ---")
        analyze_matrix_structure(saved_matrix, "Saved Matrix from sylve256.mtx")
    except FileNotFoundError:
        print("文件 sylve256.mtx 不存在，跳过文件测试")
    
    # 重新生成进行对比
    n, r = 256, 1.0
    coeff_matrix = generate_sylvester_csr(n, r)
    print(f"\n--- 重新生成的系数矩阵 (n=256, r=1.0) ---")
    analyze_matrix_structure(coeff_matrix, "Regenerated Coefficient Matrix")

if __name__ == "__main__":
    main()