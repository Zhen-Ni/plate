#!/usr/bin/env python3

import functools
import sympy as sym
import numpy as np


class Basis:
    def __init__(self, funcs, x=None, name=None):
        # Copy constructor
        if type(funcs) == type(self):
            other = funcs
            self._x = other._x
            self._name = other._name
            self._sfuncs = other._sfuncs.copy()
            self._nfuncs = other._nfuncs
            return

        self._sfuncs = list(funcs)
        if x is None:
            for f in funcs:
                if f.free_symbols:
                    x = f.free_symbols.pop()
                    break
        self._x = x
        self._nfuncs = np.array(
            [sym.lambdify((x,), f, "numpy") for f in funcs])
        self._name = name

    def __getstate__(self):
        funcs_str = [str(i) for i in self._sfuncs]
        name = self._name
        return {'expression': funcs_str, 'name': name}

    def __setstate__(self, state):
        funcs_str = state['expression']
        name = state['name']
        funcs = [sym.S(i) for i in funcs_str]
        x = set()
        [x.update(i.free_symbols) for i in funcs]
        obj = Basis(funcs, x.pop(), name)
        self.__dict__.update(obj.__dict__)

    def __repr__(self):
        mid = "'{name}'".format(name=self._name) if self._name else ''
        return ('Basis object ' + mid +
                ' with {n} basis functions'.format(n=len(self)))

    def __len__(self):
        return len(self._sfuncs)

    def __getitem__(self, idx):
        return self._nfuncs[idx]

    def __call__(self, x, *args):
        res = np.zeros([len(self)])
        for i in range(len(res)):
            res[i] = self._nfuncs[i](x, *args)
        return res

    @property
    def functions(self):
        return self._nfuncs

    @property
    def symbolic_functions(self):
        return self._sfuncs

    @property
    def symbol(self):
        return self._x

    @symbol.setter
    def symbol(self, x):
        for i in range(len(self._sfuncs)):
            self._sfuncs[i] = self._sfuncs[i].subs(self._x, x)
        self._x = x

    def diff(self):
        func = [f.diff(self._x) for f in self._sfuncs]
        return Basis(func, self._x,
                     name='differential-of-{name}'.format(name=self._name))

    def rename(self, name):
        self._name = name
        return self


def add_basis(left, right, name=None):
    left = Basis(left)
    right = Basis(right)
    if not left._x == right._x:
        raise ValueError('left and right have different variables')
    if name is None:
        name = "{left} + {right}".format(left=left._name, right=right._name)
    func = left._sfuncs + right._sfuncs
    return Basis(func, left._x, name)


class BasisND:
    def __init__(self, *basis, **kwargs):
        self._basis = [Basis(b) for b in basis]
        self._x = (b._x for b in self._basis)
        self._name = kwargs.get('name')
        self._dimension = len(self._basis)
        self._N_dimension = [len(i) for i in self._basis]
        self._N = functools.reduce(lambda x, y: x * y, self._N_dimension)
        self._nfuncs = np.zeros(self._N_dimension, dtype=object)
        self._sfuncs = np.zeros(self._N_dimension, dtype=object)
        self._init_func()
        self._is_sfunc_initialized = False

    def __repr__(self):
        mid = "'{name}'".format(name=self._name) if self._name else ''
        return ('BasisND object ' + mid +
                ' with {n} dimensions'.format(n=self._dimension))

    def __len__(self):
        return self._N

    def __getitem__(self, idx):
        return self._nfuncs[idx]

    def encode(self, *coordinate):
        if len(coordinate) != self._dimension:
            raise AttributeError('Dimension of coordinate not correct.')
        for i in range(self._dimension):
            if not coordinate[i] < self._N_dimension[i]:
                raise IndexError('Index out of range')
        return self._encode(*coordinate)

    def decode(self, idx):
        if not idx < self._N:
            raise IndexError('Index out of range')
        return self._decode(idx)

    def _encode(self, *coordinate):
        idx = 0
        bash_size = 1
        for i in range(self._dimension)[::-1]:
            idx += coordinate[i] * bash_size
            bash_size *= self._N_dimension[i]
        return idx

    def _decode(self, idx):
        coordinate = np.zeros([self._dimension], dtype=int)
        for i in range(self._dimension)[::-1]:
            coordinate[i] = idx % self._N_dimension[i]
            idx = idx // self._N_dimension[i]
        return tuple(coordinate)

    def _multiply_nfunc(self, *funcs):
        def func(*args):
            assert len(args) == len(funcs), 'Dimensiones do not match'
            res = funcs[0](args[0])
            for i in range(1, len(funcs)):
                res *= funcs[i](args[i])
            return res
        return func

    def _multiply_sfunc(self, *exprs):
        res = exprs[0]
        for e in exprs[1:]:
            res *= e
        return res

    def _init_func(self):
        for i in range(self._N):
            coor = self._decode(i)
            nfunc = self._multiply_nfunc(*[self._basis[j][coor[j]]
                                           for j in range(len(self._basis))])
            self._nfuncs[coor] = nfunc

    def _init_func_symbolic(self):
        for i in range(self._N):
            coor = self._decode(i)
            sfunc = self._multiply_sfunc(*[self._basis[j].symbolic_functions
                                           [coor[j]]
                                           for j in range(len(self._basis))])
            self._sfuncs[coor] = sfunc
        self._is_sfunc_initialized = True

    def __call__(self, *x):
        if len(x) != self._dimension:
            raise ValueError('Dimensiones do not match')
        res_i = []
        res = np.zeros(self._N)
        for i in range(self._dimension):
            res_i.append(self._basis[i](x[i]))
        idx = [0]*self._dimension
        for i in range(self._N):
            residx = 1
            for j in range(self._dimension):
                residx *= res_i[j][idx[j]]
            res[i] = residx
            idx[-1] += 1
            for j in range(self._dimension)[::-1]:
                if idx[j] == self._N_dimension[j]:
                    idx[j] = 0
                    idx[j-1] += 1
        return res.reshape(*self._N_dimension)

    @property
    def functions(self):
        return self._nfuncs

    @property
    def symbolic_functions(self):
        if not self._is_sfunc_initialized:
            self._init_func_symbolic()
        return self._sfuncs

    def diff(self, idx):
        if idx in self._x:
            idx = self._x.index(idx)
        basis_new = self._basis.copy()
        basis_new[idx] = basis_new[idx].diff()
        return BasisND(*basis_new,
                       name='differential-of-{name}'.format(name=self._name))

    def rename(self, name):
        self._name = name
        return self


def trigonometric_basis(order, interval=[0, 1], x=sym.Symbol('x'),
                        name='trigonmetric'):
    func = []
    start = interval[0]
    length = interval[1] - interval[0]
    for i in range(order):
        if i == 0:
            func.append(sym.core.numbers.One())
        elif i % 2:
            k = i // 2 + 1
            k = k * sym.pi / length
            func.append(sym.sin(k * x - start))
        else:
            k = i // 2
            k = k * sym.pi / length
            func.append(sym.cos(k * x - start))
    return Basis(func, x, name)


def cosine_basis(order, interval=[0, 1], x=sym.Symbol('x'),
                 name='cosine'):
    func = []
    start = interval[0]
    length = interval[1] - interval[0]
    for i in range(order):
        if i == 0:
            func.append(sym.core.numbers.One())
        else:
            k = i * sym.pi / length
            func.append(sym.cos(k * x - start))
    return Basis(func, x, name)


def legendre_basis(order, interval=[0, 1], x=sym.Symbol('x'),
                   name='legendre'):
    func = []
    length = interval[1] - interval[0]
    a = 2 / length
    b = -(interval[1] + interval[0]) / length
    for i in range(order):
        func.append(sym.legendre(i, a * x + b))
    return Basis(func, x, name)


def polynominal_basis(order, interval=[0, 1], x=sym.Symbol('x'),
                      name='polynominal'):
    func = []
    length = interval[1] - interval[0]
    for i in range(order):
        func.append(((x - interval[0]) / length)**i)
    return Basis(func, x, name)


if __name__ == '__main__':
    x = sym.Symbol('x')
    k = sym.Symbol('k')
    func = sym.cos(k * x)
    bx = trigonometric_basis(50)
    by = legendre_basis(50, x=sym.Symbol('y', real=True))
    b = BasisND(bx, by)
