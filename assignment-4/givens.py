import numpy as np
from typing import Callable, List
from functools import lru_cache


class MaxIterationError(Exception):
    """
    raised when the solver isn't able to
    find a solution in given number of
    max iterations.
    """
    def __init__(self, message):
        self.message = message

    def __str__(self) -> str:
        return self.message


# class LipschitzError(Exception):
#     """
#     raised when Lipschitz condition
#     isn't satisfied
#     """
#     def __str__(self) -> str:
#         return "Lipscitz condition not satisfied"


def derivative(f: Callable[[float], float]) -> Callable:
    """
    Returns the derivative of the given function

    Args:
        f (Callable[[float], float]): the function

    Returns:
        Callable: f'(x)
    """
    def inner(*args):
        epsilon = 0.001
        df = f(args[0]+epsilon) - f(*args)
        return df / epsilon
    return inner


def cache_function(func: Callable, size: int | None = None) -> Callable:
    """
    Returns the cached function

    Args:
        func (Callable): a callable object
        size (int, optional): size of cache

    Returns:
        Callable: the cached function
    """
    @lru_cache(maxsize=size)
    def inner(*args):
        return func(*args)
    return inner


def secant(func: Callable[[float], float],
           x1: float = 0,
           x0: float = 0.1,
           epsilon: float = 1e-3,
           MAX_ITER: int = 100
           ) -> float:
    """
    returns the root of the function

    Args:
        f (Callable[[float], float]): the callable function
        x1 (float): initial(x1) guess. Defaults to 0
        x0 (float): initial(x0) guess. Defaults to 0
        epsilon (float): the tolerance. Default to 1e-3
        MAX_ITER (int, optional): the maximum number of iterations.
        Defaults to 100.

    Raises:
        MaxIterationError: When iteration count exceeds set max

    Returns:
        float: the root of the given function

    >>> def func(x): return x**2-3
    >>> l, r = 0, 3
    >>> secant(func, r, l)
    1.731958762886598
    """
    cached_f = cache_function(func, 2)
    prev, curr = x0, x1
    iter = 0
    while abs(cached_f(curr)) > epsilon and iter < MAX_ITER:
        iter += 1
        curr, prev = curr-cached_f(curr)*(curr-prev)\
            / (cached_f(curr)-cached_f(prev)), curr
    if iter == MAX_ITER:
        raise MaxIterationError("Max Number of iterations exceeded!")
    return curr


def tri_diag(A):
    n, _ = A.shape
    assert (A.T == A).all(), 'The matrix should be square and symmetric'
    for i in range(n):
        for j in range(2+i, n):
            Q = np.eye(n)
            tan = A[i, j]/A[i, i+1]
            sin = tan/np.sqrt(1+tan**2)
            cos = 1/np.sqrt(1+tan**2)
            Q[j, j] = Q[i+1, i+1] = cos
            Q[j, i+1] = sin
            Q[i+1, j] = -sin
            A = Q.T@A@Q
    return A


def sturm_seq(A, lambda_, ret_only_fn=False):
    n = A.shape[0]
    fs = np.zeros(n+1)
    fs[0] = 1
    for i in range(n):
        fs[i+1] = (lambda_ - A[i, i])*fs[i] - fs[i-1]*A[i-1, i]**2
    return fs if not ret_only_fn else fs[-1]


def solve_sturm(A, start: int, end: int, step: int):
    signs = np.zeros((end-start)//step)
    eval_over = []
    evals: List[float] = []
    for lam in range(start, end, step):
        temp = sturm_seq(A, lam)
        if temp[-1] == 0:
            evals.append(lam)
        temp[1:] += temp[:-1]*1e-6
        asign = np.sign(temp)
        signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
        signs[lam] = signchange.sum()
    for lam in range(start, end-step, step):
        if signs[lam] != signs[lam+1]:
            eval_over.append((lam, lam+step))

    def f(lam): return sturm_seq(A, lam, ret_only_fn=True)

    for a, b in eval_over:
        evals.append(secant(f, a, b))
    return sorted(evals)


if __name__ == '__main__':
    A = np.array([[2, -1, 0],
                  [-1, 2, -1],
                  [0, -1, 2]])
    A = tri_diag(A)
    print(solve_sturm(A, -1, 5, 1))
