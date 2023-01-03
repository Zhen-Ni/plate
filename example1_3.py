#!/usr/bin/env python3

import numpy as np
import plate
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import matplotlib as mpl
mpl_fontpath = mpl.get_data_path() + '/fonts/ttf/STIXGeneral.ttf'
mpl_fontprop = mpl.font_manager.FontProperties(fname=mpl_fontpath)
plt.rc('font', family='STIXGeneral', weight='normal', size=11)
plt.rc('mathtext', fontset='stix')

FIG_PREFIX = 'figures/example1_3_'
REF = 61


class cv1:
    L1 = 0.4
    L2 = 0.2
    h = 0.002
    # damping
    xi = 0.05
    E = 70e9 * (1 + 2 * xi * 1j)
    rho = 2700
    nu = .33
    G = E / (2 * (1 + nu))
    M = N = 15
    material = plate.Material(E, E, G, G, G, nu, rho)
    profile = plate.Profile(material, 0, [-h/2, h/2])
    p = plate.Plate(L1, L2, M, N, profile)
    basis_x = plate.legendre_basis(50, interval=[0, L1])
    basis_y = plate.legendre_basis(50, interval=[0, L2])
    p.set_basis(basis_x, basis_y)
    plate = p
    # load
    Sxx_g = 0.5
    f = (Sxx_g) ** .5 * rho * h * 9.8


class cv2:
    ymajorlocator = LogLocator(base=10, numticks=10)
    yminorlocator = LogLocator(base=10, subs=np.arange(0, 1, 0.1),
                               numticks=10)


def read_data(filename):
    results = []
    with open(filename) as f:
        for line in f:
            words = line.split()
            if len(words) != 2:
                continue
            try:
                x, y = [float(i) for i in words]
                results.append([x, y])
            except Exception:
                pass
    return np.array(results)


def get_solver_SSSS():
    p = cv1.plate
    p.reset_spring()
    p.reset_force()
    p.add_spring(1e12, plate.BC_S, x=0)
    p.add_spring(1e12, plate.BC_S, x=cv1.L1)
    p.add_spring(1e12, plate.BC_S, y=0)
    p.add_spring(1e12, plate.BC_S, y=cv1.L2)
    p.add_force(cv1.f)
    solver = plate.HarmonicSolver(p)
    return solver


def get_solver_SCSC():
    p = cv1.plate
    p.reset_spring()
    p.reset_force()
    p.add_spring(1e12, plate.BC_S, x=0)
    p.add_spring(1e12, plate.BC_S, x=cv1.L1)
    p.add_spring(1e12, plate.BC_C, y=0)
    p.add_spring(1e12, plate.BC_C, y=cv1.L2)
    p.add_force(cv1.f)
    solver = plate.HarmonicSolver(p)
    return solver


def get_solver_SFSF():
    p = cv1.plate
    p.reset_spring()
    p.reset_force()
    p.add_spring(1e12, plate.BC_S, x=0)
    p.add_spring(1e12, plate.BC_S, x=cv1.L1)
    p.add_spring(1e12, plate.BC_F, y=0)
    p.add_spring(1e12, plate.BC_F, y=cv1.L2)
    p.add_force(cv1.f)
    solver = plate.HarmonicSolver(p)
    return solver


def get_solver_SCSF():
    p = cv1.plate
    p.reset_spring()
    p.reset_force()
    p.add_spring(1e12, plate.BC_S, x=0)
    p.add_spring(1e12, plate.BC_S, x=cv1.L1)
    p.add_spring(1e12, plate.BC_C, y=0)
    p.add_spring(1e12, plate.BC_F, y=cv1.L2)
    p.add_force(cv1.f)
    solver = plate.HarmonicSolver(p)
    return solver


def example1():
    freqs = np.linspace(20, 2000, 397)
    solver = get_solver_SSSS()
    step = solver.solve(freqs)
    results = step.w(cv1.L1/2, cv1.L2/2)
    results *= (freqs * 2 * np.pi) ** 2
    results = abs(results) ** 2
    # plot
    fig = plt.figure(figsize=(3, 2.25))
    ax = fig.add_subplot(111)
    ax.semilogy(freqs, results, 'k-', label='Present method')
    ax.semilogy(*read_data('../getdata/Chen, Zhou, Yang, 2017, fig2a.txt').T,
                'r--', label='Ref. [{}]'.format(REF))
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel(r'Acceleration PSD ($\mathrm{m^2/s^3}$)')
    ax.set_xlim(0, 2000)
    ax.set_ylim(1e-2, 1e5)
    ax.yaxis.set_major_locator(cv2.ymajorlocator)
    ax.yaxis.set_minor_locator(cv2.yminorlocator)
    ax.legend(fontsize='small', loc='lower right')
    fig.tight_layout(pad=0.15)
    fig.savefig(FIG_PREFIX + '1.svg')
    return freqs, results


def example2():
    freqs = np.linspace(20, 2000, 397)
    solver = get_solver_SCSC()
    step = solver.solve(freqs)
    results = step.w(cv1.L1/2, cv1.L2/2)
    results *= (freqs * 2 * np.pi) ** 2
    results = abs(results) ** 2
    # plot
    fig = plt.figure(figsize=(3, 2.25))
    ax = fig.add_subplot(111)
    ax.semilogy(freqs, results, 'k-', label='Present method')
    ax.semilogy(*read_data('../getdata/Chen, Zhou, Yang, 2017, fig2c.txt').T,
                'r--', label='Ref. [{}]'.format(REF))
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel(r'Acceleration PSD ($\mathrm{m^2/s^3}$)')
    ax.set_xlim(0, 2000)
    ax.set_ylim(1e-2, 1e5)
    ax.yaxis.set_major_locator(cv2.ymajorlocator)
    ax.yaxis.set_minor_locator(cv2.yminorlocator)
    ax.legend(fontsize='small', loc='lower right')
    fig.tight_layout(pad=0.15)
    fig.savefig(FIG_PREFIX + '2.svg')
    return freqs, results


def example3():
    freqs = np.linspace(20, 2000, 397)
    solver = get_solver_SFSF()
    step = solver.solve(freqs)
    results = step.w(cv1.L1/2, cv1.L2/2)
    results *= (freqs * 2 * np.pi) ** 2
    results = abs(results) ** 2
    # plot
    fig = plt.figure(figsize=(3, 2.25))
    ax = fig.add_subplot(111)
    ax.semilogy(freqs, results, 'k-', label='Present method')
    ax.semilogy(*read_data('../getdata/Chen, Zhou, Yang, 2017, fig2d.txt').T,
                'r--', label='Ref. [{}]'.format(REF))
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel(r'Acceleration PSD ($\mathrm{m^2/s^3}$)')
    ax.set_xlim(0, 2000)
    ax.set_ylim(1e-2, 1e5)
    ax.yaxis.set_major_locator(cv2.ymajorlocator)
    ax.yaxis.set_minor_locator(cv2.yminorlocator)
    ax.legend(fontsize='small', loc='lower right')
    fig.tight_layout(pad=0.15)
    fig.savefig(FIG_PREFIX + '3.svg')
    return freqs, results


def example4():
    freqs = np.linspace(20, 2000, 397)
    solver = get_solver_SCSF()
    step = solver.solve(freqs)
    results = step.w(cv1.L1/2, cv1.L2/2)
    results *= (freqs * 2 * np.pi) ** 2
    results = abs(results) ** 2
    # plot
    fig = plt.figure(figsize=(3, 2.25))
    ax = fig.add_subplot(111)
    ax.semilogy(freqs, results, 'k-', label='Present method')
    ax.semilogy(*read_data('../getdata/Chen, Zhou, Yang, 2017, fig2f.txt').T,
                'r--', label='Ref. [{}]'.format(REF))
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel(r'Acceleration PSD ($\mathrm{m^2/s^3}$)')
    ax.set_xlim(0, 2000)
    ax.set_ylim(1e-2, 1e5)
    ax.yaxis.set_major_locator(cv2.ymajorlocator)
    ax.yaxis.set_minor_locator(cv2.yminorlocator)
    ax.legend(fontsize='small', loc='lower right')
    fig.tight_layout(pad=0.15)
    fig.savefig(FIG_PREFIX + '4.svg')
    return freqs, results


if __name__ == '__main__':
    res1 = example1()
    res2 = example2()
    res3 = example3()
    res4 = example4()
    plt.show()
    plt.close('all')
