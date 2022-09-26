"""
This module contains functions for
1. Bisection Method
2. Regula-Falsi Method
3. Secant Method
4. Newton-Rapson Method
5. General Iteration Scheme
"""

from functools import lru_cache
import math
from typing import Callable


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


class LipschitzError(Exception):
    """
    raised when Lipschitz condition
    isn't satisfied
    """
    def __str__(self) -> str:
        return "Lipscitz condition not satisfied"


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


def bisection(left: float,
              right: float,
              func: Callable[[float], float],
              epsilon: float = 0.01,
              MAX_ITER: int = 100
              ) -> float:
    """
    Bisection method.
    Given a closed continuous real set, with end points
    lying on different side of x-axis, it returns the root


    Args:
        left (float): the left end point
        right (float): the right end point
        func Callable[[float], float]: the function
        epsilon (float, optional): tolerance of error. Defaults to 0.01.
        MAX_ITER (int, optional): the maximum number of iterations.
        Defaults to 100.

    Raises:
        MaxIterationError: When iteration count exceeds set max

    Returns:
        float: the root of the given function

    >>> def func(x): return x**2-3
    >>> l, r = 0, 3
    >>> bisection(l, r, func)
    1.734375
    """
    cached_f = cache_function(func, 3)

    assert cached_f(left)*cached_f(right) < 0, \
           "Points do not satisfy the " + \
           "intermediate value theorem"
    mid = (left+right)/2
    iter = 0
    while right-left > epsilon and iter < MAX_ITER:
        iter += 1
        mid = (left+right)/2
        if cached_f(mid) * cached_f(right) > 0:
            right = mid
        else:
            left = mid
    if iter == MAX_ITER:
        raise MaxIterationError("Max Number of iterations exceeded!")
    return mid


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


def newtonRaphson(func: Callable[[float], float],
                  x0: float = 0,
                  epsilon: float = 0.001,
                  MAX_ITER: int = 100
                  ) -> float:
    """
    Returns the root of the given function using
    Newton Raphson method

    Args:
        f (Callable[[float],float]): the callable function
        x0 (float, optional): initial guess. Defaults to 0.
        epsilon (float, optional): tolerance. Defaults to 0.001.
        MAX_ITER (int, optional): the maximum number of iterations.
        Defaults to 100.

    Raises:
        MaxIterationError: When iteration count exceeds set max

    Returns:
        float: the root of the given function

    >>> def func(x): return x**2-3
    >>> l, r = 0, 3
    >>> newtonRaphson(func)
    1.732331901591351
    """
    curr = x0
    f = cache_function(func, 1)
    f_dash = derivative(f)
    f_doubledash = derivative(f_dash)
    iter = 0
    while abs(f(curr)) > epsilon and iter < MAX_ITER:
        iter += 1
        if abs(f(curr)*f_doubledash(curr)) > f_dash(curr)**2:
            raise LipschitzError()
        curr -= f(curr) / f_dash(curr)

    if iter == MAX_ITER:
        raise MaxIterationError("Max Number of iterations exceeded!")

    return curr


def fixedPoint(phi: Callable[[float], float],
               x0: float = 0,
               epsilon: float = 0.001,
               MAX_ITER: int = 100
               ) -> float:
    """fixed point iteration scheme

    Args:
        phi (Callable[[float],float]): Callable update function
        x0 (float, optional): initial guess. Defaults to 0.
        epsilon (float, optional): the tolerance
        MAX_ITER (int, optional): the maximum number of iterations.
        Defaults to 100.

    Raises:
        MaxIterationError: When iteration count exceeds set max

    Returns:
        float: root of the function

    >>> def func(x): return x**2-2
    >>> fixedPoint(func)
    2
    """
    prev = -math.inf
    phi_dash = derivative(phi)
    curr = x0
    iter = 0
    while abs(curr-prev) > epsilon and iter < MAX_ITER:
        iter += 1
        if abs(phi_dash(curr)) >= 1:
            raise LipschitzError()
        curr, prev = phi(curr), curr

    if iter == MAX_ITER:
        raise MaxIterationError("Max Number of iterations exceeded!")

    return curr


def regulaFalsi(
                func: Callable[[float], float],
                a: float,
                b: float,
                epsilon: float = 0.001,
                MAX_ITER: int = 100
                ) -> float:
    """
    Regular Falsi method

    Args:
        func (Callable[[float], float]): Callable function
        a (float): left point
        b (float): right point
        epsilon (float, optional): tolerance. Defaults to 0.001.
        MAX_ITER (int, optional): max number of iterations. Defaults to 100.

    Raises:
        MaxIterationError: When iteration count exceeds set max

    Returns:
        float: root of the function
    """
    left, right = a, b
    cached_f = cache_function(func, 3)
    assert cached_f(left)*cached_f(right) < 0, \
           "Points do not satisfy the " + \
           "intermediate value theorem"
    iter = 0
    mid = left
    while abs(right-left) > epsilon and iter < MAX_ITER:
        iter += 1
        mid = (left*cached_f(right)
               - right*cached_f(left))/(cached_f(right)-cached_f(left))
        if cached_f(mid) == 0:
            break
        if cached_f(mid)*cached_f(left) < 0:
            right = mid
        else:
            left = mid
    if iter == MAX_ITER:
        raise MaxIterationError("Max Number of iterations exceeded!")
    return mid


class IO:
    methods = {
            1: bisection,
            2: secant,
            3: regulaFalsi,
            4: newtonRaphson,
            5: fixedPoint
        }

    def __init__(self, func):
        self.func = func

    def ask(self):
        print("Which function do you need?")
        for key, method in self.methods.items():
            print(f'press {key} for {method.__name__}')
        func = self.methods[int(input())]
        parameters = func.__annotations__
        args = []
        for i in parameters:
            if i in ['func', 'phi']:
                args.append(self.func)
                continue
            if i == 'return':
                continue
            print(f'Enter the value for {i}')
            temp = input()
            temp = float(temp) if i != 'MAX_ITER' else int(temp)
            args.append(temp)
        return func(*args)


bisection.__name__ = 'Bisection method'
secant.__name__ = 'Secant method'
regulaFalsi.__name__ = 'Regula Falsi method'
newtonRaphson.__name__ = 'Newton Raphson Method'
fixedPoint.__name__ = 'Fixed point iteration method'


if __name__ == "__main__":
    def func(x): return x**2-2
    print("I've chosen a particular function, the user may change the function"
          + "in the code")
    terminal_exp = IO(func)
    ans = terminal_exp.ask()
    print(ans)
