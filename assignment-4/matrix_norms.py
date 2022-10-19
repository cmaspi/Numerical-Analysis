import numpy as np


def l_inf(A: np.ndarray):
    m, n = A.shape
    rows = np.array([np.abs(x).sum() for x in A])
    return rows.max()


def l1(A: np.ndarray):
    m, n = A.shape
    cols = np.array([np.abs(x).sum() for x in A.T])
    return cols.max()


def frobenius(A: np.ndarray):
    return np.sqrt((A*A).sum())

