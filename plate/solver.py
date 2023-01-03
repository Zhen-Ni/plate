#!/usr/bin/env python3

import abc
import numpy as np
import scipy as sp
import scipy.sparse.linalg as slinalg

from .plate import _get_T


__all__ = 'Solver', 'ModalSolver', 'ComplexModalSolver', 'HarmonicSolver'


class Solver(abc.ABC):
    def __init__(self, plate):
        self._plate = plate
        self._dtype = plate.dtype

    @property
    def dtype(self):
        return self._dtype

    @abc.abstractmethod
    def solve(self):
        u = np.zeros([self._plate._MN * 5], dtype=self.dtype)
        step = Step()
        step.append(Frame(self._plate, u))
        return step


class ModalSolver(Solver):
    def solve(self, n=20, frequency_shift=0.1, auto_filter=0.0,
              ncv=None, maxiter=None):
        M, K = self._plate.get_M(), self._plate.get_K()
        freq, mode_shape = slinalg.eigs(K.real, n, M.real, frequency_shift,
                                        v0=np.ones(M.shape[0]),
                                        ncv=ncv, maxiter=maxiter)
        freq = np.sqrt(freq) / 2 / np.pi

        if not freq.imag.any():
            freq = freq.real
            mode_shape = mode_shape.real

        index_array = np.argsort(freq.real)
        step = Step()
        for idx in index_array:
            f = freq[idx]
            x = mode_shape.T[idx]
            if auto_filter is not None:
                if abs(np.real(f) * auto_filter) < abs(np.imag(f)):
                    continue
            step.append(ModalFrame(self._plate, x, f.real))
        return step


class ComplexModalSolver(Solver):
    def _solver_helper_dense(self, A):
        eigv, mode_shape = np.linalg.eig(A)
        return eigv, mode_shape

    def _solver_helper_sparse(self, A, n, return_onesided,
                              frequency_shift, ncv, maxiter):
        if return_onesided:
            n *= 2
        eigv, mode_shape = slinalg.eigs(A, n, sigma=frequency_shift,
                                        ncv=ncv, maxiter=maxiter)
        return eigv, mode_shape

    def solve(self, n=20, return_onesided=True, solver='sparse',
              frequency_shift=0.1, ncv=None, maxiter=None):
        M, C, K = self._plate.get_M(), self._plate.get_C(), self._plate.get_K()
        # invM = np.linalg.inv(M)  # M is almost singular!!!
        # Do not use general eigen value decomposition (use argument M in
        # slinalg.eigs) to avoid errorous results. (maybe because of high
        # condition number)
        size = M.shape[0]
        A = np.zeros([size*2, size*2], dtype=self.dtype)
        A[:size, size:] = np.eye(size)
        # Use this method to get the inverse matrixes to ensure accuracy
        invMK, invMC = np.linalg.solve(M, [K, C])
        A[size:, :size] = -invMK
        A[size:, size:] = -invMC
        if solver == 'sparse':
            eigv, mode_shape = self._solver_helper_sparse(A, n,
                                                          return_onesided,
                                                          frequency_shift,
                                                          ncv, maxiter)
        elif solver == 'dense':
            eigv, mode_shape = self._solver_helper_dense(A)
        else:
            raise AttributeError('unknown solver')
        mode_shape = mode_shape[size:, :]
        if return_onesided:
            idx = eigv.imag >= 0
            eigv = eigv[idx]
            mode_shape = mode_shape[:, idx]

        index_array = np.argsort(abs(eigv.imag))
        step = Step()
        for idx in index_array[:n]:
            ev = eigv[idx]
            x = mode_shape.T[idx]
            step.append(ComplexModalFrame(self._plate, x, ev))
        return step


class HarmonicSolver(Solver):
    def _solver_helper_dense(self, A, f):
        return np.linalg.solve(A, f)

    def _solver_helper_sparse(self, A, f):
        As = sp.sparse.csc_matrix(A)
        x = slinalg.spsolve(As, f, use_umfpack=False)
        return x

    def _solver_helper_umfpack(self, A, f):
        As = sp.sparse.csc_matrix(A)
        x = slinalg.spsolve(As, f, use_umfpack=True)
        return x

    def solve(self, frequency, solver='umfpack'):
        frequency = np.asarray(frequency)
        omega = 2 * np.pi * frequency
        # Sparse solver saves computational time.
        if solver == 'dense':
            solve = self._solver_helper_dense
        elif solver == 'sparse':
            solve = self._solver_helper_sparse
        elif solver == 'umfpack':
            solve = self._solver_helper_umfpack
        else:
            raise AttributeError('unknown solver')
        M, C, K = self._plate.get_M(), self._plate.get_C(), self._plate.get_K()
        F = self._plate.get_F()
        flag_c = True if C.any() else False
        step = Step()
        for i, omegai in enumerate(omega):
            A = - omegai ** 2 * M + K
            if flag_c:
                A = A + 1j * omegai * C
            x = solve(A, F)
            step.append(HarmonicFrame(self._plate, x, frequency[i]))
        return step


class Step():
    def __init__(self, frames=[]):
        self._frames = list(frames)

    def append(self, frame):
        self._frames.append(frame)

    def __repr__(self):
        return "Step object with {n} frames".format(n=len(self._frames))

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, *args, **kwargs):
        return self._frames.__getitem__(*args, **kwargs)

    def __getattr__(self, name):
        return np.array([getattr(i, name) for i in self._frames])

    def response(self, dof, x, y, dx=False, dy=False):
        return np.array([i.response(dof, x, y, dx, dy)
                         for i in self._frames])

    def u(self, x, y, dx=False, dy=False):
        return np.array([i.u(x, y, dx, dy) for i in self._frames])

    def v(self, x, y, dx=False, dy=False):
        return np.array([i.v(x, y, dx, dy) for i in self._frames])

    def w(self, x, y, dx=False, dy=False):
        return np.array([i.w(x, y, dx, dy) for i in self._frames])

    def rx(self, x, y, dx=False, dy=False):
        return np.array([i.rx(x, y, dx, dy) for i in self._frames])

    def ry(self, x, y, dx=False, dy=False):
        return np.array([i.ry(x, y, dx, dy) for i in self._frames])

    def displacement(self, x, y):
        return np.array([i.displacement(x, y) for i in self._frames])

    def strain(self, x, y):
        return np.array([i.strain(x, y) for i in self._frames])


class Frame:
    def __init__(self, plate, u):
        self._plate = plate
        # Make a copy of basis in case plate._set_basis is called.
        self._basis_x = plate._basis_x
        self._basis_y = plate._basis_y

        self._u = np.asarray(u)

    def __repr__(self):
        return "frame object"

    def _response_helper(self, x, y, ui, dx, dy):
        plate = self._plate
        vx = self._basis_x[dx](x)
        vy = self._basis_y[dy](y)
        i = np.arange(plate._MN)
        ix = i // plate._N
        iy = i % plate._N
        res = np.sum(vx[ix] * vy[iy] * ui)
        return res

    def response(self, dof, x, y, dx=False, dy=False):
        MN = self._plate._MN
        res = self._response_helper(x, y, self._u[dof * MN: (dof + 1) * MN],
                                    dx, dy)
        return res

    def u(self, x, y, dx=False, dy=False):
        return self.response(0, x, y, dx, dy)

    def v(self, x, y, dx=False, dy=False):
        return self.response(1, x, y, dx, dy)

    def w(self, x, y, dx=False, dy=False):
        return self.response(2, x, y, dx, dy)

    def rx(self, x, y, dx=False, dy=False):
        return self.response(3, x, y, dx, dy)

    def ry(self, x, y, dx=False, dy=False):
        return self.response(4, x, y, dx, dy)

    def displacement(self, x, y):
        plate = self._plate
        MN = plate._MN
        vx = self._basis_x[0](x)
        vy = self._basis_y[0](y)
        i = np.arange(MN)
        ix = i // plate._N
        iy = i % plate._N
        vxy = vx[ix] * vy[iy]
        u = np.sum(vxy * self._u[0 * MN: 1 * MN])
        v = np.sum(vxy * self._u[1 * MN: 2 * MN])
        w = np.sum(vxy * self._u[2 * MN: 3 * MN])
        return np.abs(u*u+v*v+w*w)**.5

    def strain(self, x, y):
        plate = self._plate
        MN = plate._MN
        i = np.arange(MN)
        ix = i // plate._N
        iy = i % plate._N
        vx = self._basis_x[0](x)
        vy = self._basis_y[0](y)
        vxy = vx[ix] * vy[iy]
        u0 = np.sum(vxy * self._u.reshape(5, -1), axis=1)
        vx = self._basis_x[1](x)
        vy = self._basis_y[0](y)
        vxy = vx[ix] * vy[iy]
        u1 = np.sum(vxy * self._u.reshape(5, -1), axis=1)
        vx = self._basis_x[0](x)
        vy = self._basis_y[1](y)
        vxy = vx[ix] * vy[iy]
        u2 = np.sum(vxy * self._u.reshape(5, -1), axis=1)
        u = np.concatenate([u0, u1, u2])
        strain = _get_T().dot(u)
        return strain


class ModalFrame(Frame):
    def __init__(self, plate, u, frequency):
        super().__init__(plate, u)
        self._frequency = frequency

    @property
    def frequency(self):
        return self._frequency

    def __repr__(self):
        return "Frame object at {freq}Hz".format(freq=self._frequency)


class ComplexModalFrame(Frame):
    def __init__(self, plate, u, eigen_value):
        super().__init__(plate, u)
        self._eigen_value = eigen_value
        self._frequency = eigen_value / (2j * np.pi)
        self._stable = True if eigen_value.real < 0 else False

    @property
    def stable(self):
        return self._stable
        
    @property
    def frequency(self):
        return self._frequency

    @property
    def eigen_value(self):
        return self._eigen_value

    def __repr__(self):
        return "Frame object at {freq}Hz".format(freq=self._frequency)

    
class HarmonicFrame(Frame):
    def __init__(self, plate, u, frequency):
        super().__init__(plate, u)
        self._frequency = frequency

    @property
    def frequency(self):
        return self._frequency

    def __repr__(self):
        return "Frame object at {freq}Hz".format(freq=self._frequency)
