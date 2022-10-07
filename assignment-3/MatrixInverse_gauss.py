import numpy as np


class Unsolvable(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


def Inverse(A: np.ndarray) -> np.ndarray:
    m, n = A.shape
    assert m == n, "The matrix should be square"
    I = np.eye(n, dtype=float)
    for i in range(n):
        if A[i, i] == 0:
            for j in range(i+1, n):
                if A[j, i] != 0:
                    A[i], A[j] = A[j], A[i]
                    I[i], I[j] = I[j], I[i]
                    break
        if A[i, i] == 0:
            raise Unsolvable("The matrix is not invertible")
        for j in range(i+1, n):
            factor = A[j, i]/A[i, i]
            A[j] -= A[i] * factor
            I[j] -= I[i] * factor

    for i in range(n-1, -1, -1):
        for j in range(i-1, -1, -1):
            factor = A[j, i]/A[i, i]
            A[j] -= A[i] * factor
            I[j] -= I[i] * factor

    for i in range(n):
        I[i] /= A[i, i]
    return I


if __name__ == '__main__':
    A = np.array([[1, 2], [3, 4]], dtype=float)
    print(Inverse(A))
