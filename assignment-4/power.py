import numpy as np


def power(A, tolerance):
    n, m = A.shape
    assert n == m, "The matrix should be square"
    error = np.inf
    x = np.ones(n)
    itr = 0
    while error > tolerance and itr < 1e4:
        itr += 1
        prev = x
        x = A@x
        eig = abs(x).max()
        assert eig, "The matrix should be full rank"
        x = x/eig
        error = np.linalg.norm(x-prev)
    return eig, x


if __name__ == '__main__':
    a = np.array([[0, 2],
                  [2, 3]])
    print(power(a, 0.01))
