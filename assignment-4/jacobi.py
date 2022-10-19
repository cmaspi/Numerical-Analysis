import numpy as np
from power import power


def jacobi(A, b, tolerance):
    n = A.shape[0]
    LU, D_inv = np.zeros_like(A), np.zeros_like(A)
    for i in range(n):
        for j in range(n):
            if i == j:
                D_inv[i, j] = 1/A[i, j]
            else:
                LU[i, j] = A[i, j]
    x = np.zeros(n)
    error = np.inf
    assert power(D_inv@LU, 0.01)[0] < 1, '''
The spectral radius should be less than 1'''
    itr = 0
    while error > tolerance and itr < 1e4:
        itr += 1
        prev = x
        x = -D_inv@LU@x+D_inv@b
        error = np.linalg.norm(prev-x)
    return x


A = np.array([[2, 1], [3, 4]], dtype=float)
b = np.array([5, 10], dtype=float)
print(jacobi(A, b, 0.001))
