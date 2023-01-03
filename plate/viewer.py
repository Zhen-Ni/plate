#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

__all__ = 'plt', 'plot_frame', 'plot_frame_3d'

# plt.rc('font', family='times new roman', weight='normal', size=9)
# plt.rc('text', usetex=True)
# plt.rcParams['axes.unicode_minus'] = False
# DPI = 300
NPOINTS = 20


def plot_frame(frame, direction='displacement', nx=NPOINTS, ny=NPOINTS):
    plate = frame._plate
    a, b = plate._a, plate._b
    scalar = 3 / max(a, b)
    fig = plt.figure(figsize=(a*scalar, b*scalar))
    ax = fig.add_axes([0, 0, 1, 1])
    xs = np.linspace(0, a, nx)
    ys = np.linspace(0, b, ny)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            Z[j, i] = getattr(frame, direction)(x, y)
    ax.contourf(X, Y, Z, levels=512, cmap='jet')
    return fig


def plot_frame_3d(frame, direction='displacement', nx=NPOINTS, ny=NPOINTS,
                  normalize=True):
    plate = frame._plate
    a, b = plate._a, plate._b
    scalar = 3 / max(a, b)
    fig = plt.figure(figsize=(a*scalar, b*scalar))
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    xs = np.linspace(0, a, nx)
    ys = np.linspace(0, b, ny)
    X0, Y0 = np.meshgrid(xs, ys)
    X = np.zeros([ny, nx])
    Y = np.zeros([ny, nx])
    Z = np.zeros([ny, nx])
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            X[j, i] = frame.u(x, y)
            Y[j, i] = frame.v(x, y)
            Z[j, i] = frame.w(x, y)
    if normalize:
        D = X ** 2 + Y ** 2 + Z ** 2
        D = abs(D).max() ** .5
        X /= D
        Y /= D
        Z /= D
    ax.contourf3D(X0+X, Y0+Y, Z, levels=512, cmap='jet')
    ax.set_axis_off()
    ax.set_xlim((X+X0).min(), (X+X0).max())
    ax.set_ylim((Y+Y0).min(), (Y+Y0).max())
    ax.set_zlim(Z.min(), Z.max())
    return fig
