import numpy as np
from typing import Tuple


class Unsolvable(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return message


def LU(A: np.ndarray
      ) -> Tuple[np.ndarray, np.ndarray]:
    m, n = A.shape
    assert m == n, "The matrix should be square"
    L = np.identity(n, dtype=float)

    for i in range(n):
        if A[i, i] == 0:
            raise Unsolvable("found zero diagonal element")
        for j in range(i+1, n):
            factor = A[j, i]/A[i, i]
            A[j] -= A[i] * factor
            L[j, i] = factor
    return L, A


A = np.array([[2, 4, -4],[1, -4, 3],[-6, -9, 5]], dtype=float)
L, U = LU(A)
print(L)
print(U)
print(L@U)

