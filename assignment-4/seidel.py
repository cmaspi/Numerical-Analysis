import numpy as np
from MatrixInverse_gauss import Inverse
from power import power


def seidel(A, b, tolerance):
    n = A.shape[0]
    U, LD_inv = np.zeros_like(A), np.zeros_like(A)
    for i in range(n):
        for j in range(n):
            if i <= j:
                LD_inv[i, j] = A[i, j]
            else:
                U[i, j] = A[i, j]
    LD_inv = Inverse(LD_inv)
    assert power(LD_inv@U, 0.001)[0] < 1, '''
The spectral radius should be less than 1'''
    x = np.zeros(n)
    error = np.inf
    itr = 0
    while error > tolerance and itr < 1e4:
        itr += 1
        prev = x
        x = -LD_inv@U@x+LD_inv@b
        error = np.linalg.norm(x-prev)
    return x


A = np.array([[2, 1], [3, 4]], dtype=float)
b = np.array([5, 10], dtype=float)
print(seidel(A, b, 0.001))
