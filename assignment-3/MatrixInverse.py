import numpy as np
from copy import deepcopy


def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    res = np.zeros((A.shape[0], B.shape[1]))
    for i, row in enumerate(A):
        for j, column in enumerate(zip(*B)):
            res[i, j] = sum(row * column)
    return res


def MatInv(A: np.ndarray, num_epochs: int):
    """
    AB = I - e
    A^- = B(I-e)^-
    A^- ~= B(I+e)
    2I - AB = I + e
    A^- = B(2I-AB)
    B_(k+1) = B_(k)(2I - AB_(k))
    """
    B = np.eye(A.shape[0])/(A.sum(axis=0).max()*3)
    I = np.eye(A.shape[0])
    for i in range(num_epochs):
        new_B = 2*I - matmul(A, B)
        new_B = matmul(B, new_B)
        print(new_B)
        B = new_B
    return B


if __name__ == '__main__':
    A = np.array([[1,2],[3,4]])
    print(MatInv(A, 10))


        

    
