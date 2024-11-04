"""
Collection of the core mathematical operators used throughout the code base.
"""

import math

from typing import Callable, Iterable, Any


def mul(x: float, y: float) -> float:
    return x * y


def id(x: float) -> float:
    return x


def add(x: float, y: float) -> float:
    return x + y


def neg(x: float) -> float:
    return -1. * x


def lt(x: float, y: float) -> float:
    return 1. if x < y else 0.


def eq(x: float, y: float) -> float:
    return 1. if x == y else 0.


def max(x: float, y: float) -> float:
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    return 1. if abs(x - y) < 1e-2 else 0.


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)


def relu(x: float) -> float:
    return max(0., x)


def log(x: float) -> float:
    return math.log(x)


def exp(x: float) -> float:
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    return d / x


def inv(x: float) -> float:
    return 1. / x


def inv_back(x: float, d: float) -> float:
    return -d / (x * x)


def relu_back(x: float, d: float) -> float:
    return d if x > 0 else 0.


def map(fn: Callable[[Any], Any], lst: Iterable[Any]) -> Iterable[Any]:
    return [fn(x) for x in lst]


def zipWith(fn: Callable[[Any, Any], Any], lst1: Iterable[Any], lst2: Iterable[Any]) -> Iterable[Any]:
    return [fn(x, y) for x, y in zip(lst1, lst2)]


def reduce(fn: Callable[[Any, Any], Any], lst: Iterable[Any], start: Any) -> Any:
    result = start
    for x in lst:
        result = fn(result, x)
    return result


def negList(lst: Iterable[float]) -> Iterable[float]:
    return map(neg, lst)


def addLists(lst1: Iterable[float], lst2: Iterable[float]) -> Iterable[float]:
    return zipWith(add, lst1, lst2)


def sum(lst: Iterable[float]) -> float:
    return reduce(add, lst, 0.0)


def prod(lst: Iterable[float]) -> float:
    return reduce(mul, lst, 1.0)
