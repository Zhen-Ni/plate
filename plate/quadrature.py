#!/usr/bin/env python3


import numpy as np
from scipy.special import roots_legendre


class Quadrature:
    def __init__(self, x, w):
        """x is the sampling points in interval [-1,1] and w is the weights."""
        assert len(x) == len(w)
        self._x = np.array(x)
        self._w = np.array(w)
        self._call = self._call_limit_not_setted

    def set_limit(self, a, b):
        self._x = self._x * ((b - a) / 2) + (a + b) / 2
        self._w = self._w * ((b - a) / 2)
        self._call = self._call_limit_setted
        return self

    @property
    def x(self):
        return self._x

    @property
    def w(self):
        return self._w

    def __call__(self, *args, **kwargs):
        return self._call(*args, **kwargs)

    def _call_limit_not_setted(self, func, a, b, weights=None, vectorize=True):
        x = self._x * ((b - a) / 2) + (a + b) / 2
        w = self._w * ((b - a) / 2)
        if weights is not None:
            w = w * weights
        if vectorize:
            return (func(x) * w).sum(-1)
        else:
            res = 0.0
            for i in range(len(x)):
                res += w[i] * func(x[i])
            return res

    def _call_limit_setted(self, func=None, weights=None, vectorize=True):
        if weights is not None:
            w = self._w * weights
        else:
            w = self._w
        if func is None:
            return w.sum()
        if vectorize:
            return (func(self._x) * w).sum(-1)
        else:
            res = 0.0
            for i in range(len(self._x)):
                res += w[i] * func(self._x[i])
            return res


def quad_linear(n):
    x = np.linspace(-1, 1, n)
    w = np.zeros_like(x)
    w[:] = 2 / (n-1)
    w[0] /= 2
    w[-1] /= 2
    return Quadrature(x, w)


def quad_gauss(n):
    return Quadrature(*roots_legendre(n))


if __name__ == '__main__':
    qg = quad_gauss(10)
