import numpy as np


class Unsolvable(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


def gaussian_elimination(A: np.ndarray,
                         b: np.ndarray
                         ) -> np.ndarray:
    m, n = A.shape
    assert m == n, "The matrix should be square"
    assert b.size == m, f'{b.size} and {m} do not match'
    for i in range(n):
        if A[i, i] == 0:
            for j in range(i+1, n):
                if A[j, i] != 0:
                    A[i], A[j] = A[j], A[i]
                    b[i], b[j] = b[j], b[i]
                    break
        if A[i, i] == 0:
            raise Unsolvable("The system of equations is not solvable")
        for j in range(i+1, n):
            factor = A[j, i]/A[i, i]
            A[j] -= A[i] * factor
            b[j] -= b[i] * factor

        x = np.zeros(n)

    for i in range(n-1, -1, -1):
        x[i] = b[i]
        for j in range(i+1, n):
            x[i] -= x[j]*A[i, j]
        x[i] /= A[i, i]

    return x


A = np.array([[1, 2], [3, 4]], dtype=float)
b = np.array([4, 10], dtype=float)
print(gaussian_elimination(A, b))

A = np.array([
              [-4, 7, -2],
              [1, -2, 3],
              [2, -3, 1]],
             dtype=float)
b = np.array([2, 3, -4], dtype=float)
print(gaussian_elimination(A, b))
