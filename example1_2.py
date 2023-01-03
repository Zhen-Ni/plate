#!/usr/bin/env python3

import numpy as np
import functools
import plate
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl_fontpath = mpl.get_data_path() + '/fonts/ttf/STIXGeneral.ttf'
mpl_fontprop = mpl.font_manager.FontProperties(fname=mpl_fontpath)
plt.rc('font', family='STIXGeneral', weight='normal', size=11)
plt.rc('mathtext', fontset='stix')

FIG_PREFIX = 'figures/example1_2_'
REF = 10


class cv1:
    L1 = L2 = 0.1
    h = 0.001
    E = 210e9
    rho = 7930
    nu = .33
    G = E / (2 * (1 + nu))
    M = N = 15
    material = plate.Material(E, E, G, G, G, nu, rho)
    profile = plate.Profile(material, 0, [-h/2, h/2])
    p = plate.SupersonicPlate(L1, L2, M, N, profile)
    basis_x = plate.legendre_basis(50, interval=[0, L1])
    basis_y = plate.legendre_basis(50, interval=[0, L2])
    p.set_basis(basis_x, basis_y)
    plate = p
    # A non-dimensional variable
    D = E * h ** 3 / (12 * (1 - nu ** 2))


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


def get_solver_SSSS(lambda_: float):
    p = cv1.plate
    p.reset_spring()
    p.reset_airflow()
    p.add_spring(1e12, plate.BC_S, x=0)
    p.add_spring(1e12, plate.BC_S, x=cv1.L1)
    p.add_spring(1e12, plate.BC_S, y=0)
    p.add_spring(1e12, plate.BC_S, y=cv1.L2)
    p.add_airflow_lambda(lambda_, 0)
    solver = plate.ComplexModalSolver(p)
    return solver


def get_solver_SCSC(lambda_: float):
    p = cv1.plate
    p.reset_spring()
    p.reset_airflow()
    p.add_spring(1e12, plate.BC_S, x=0)
    p.add_spring(1e12, plate.BC_S, x=cv1.L1)
    p.add_spring(1e12, plate.BC_C, y=0)
    p.add_spring(1e12, plate.BC_C, y=cv1.L2)
    p.add_airflow_lambda(lambda_, 0)
    solver = plate.ComplexModalSolver(p)
    return solver


def get_solver_SFSF(lambda_: float):
    p = cv1.plate
    p.reset_spring()
    p.reset_airflow()
    p.add_spring(1e12, plate.BC_S, x=0)
    p.add_spring(1e12, plate.BC_S, x=cv1.L1)
    p.add_spring(1e12, plate.BC_F, y=0)
    p.add_spring(1e12, plate.BC_F, y=cv1.L2)
    p.add_airflow_lambda(lambda_, 0)
    solver = plate.ComplexModalSolver(p)
    return solver


def get_solver_SSeSSe(lambda_: float, kw: float):
    """`e` stands for elastic BC"""
    p = cv1.plate
    p.reset_spring()
    p.reset_airflow()
    p.add_spring(1e12, plate.BC_S, x=0)
    p.add_spring(1e12, plate.BC_S, x=cv1.L1)
    p.add_spring(kw, [2], y=0)
    p.add_spring(kw, [2], y=cv1.L2)
    p.add_airflow_lambda(lambda_, 0)
    solver = plate.ComplexModalSolver(p)
    return solver


@functools.lru_cache()
def get_flutter_data_SSSS():
    mode_order = 2
    lambdas = np.linspace(100, 600, 51)
    lambdas = np.concatenate([lambdas, np.linspace(480, 510, 31)])
    lambdas = np.concatenate([lambdas, np.linspace(510, 515, 51)])
    lambdas = sorted(set(lambdas))
    results = []
    for lbd in lambdas:
        print(lbd)
        solver = get_solver_SSSS(lbd)
        res = solver.solve(mode_order * 2, solver='sparse')
        freq = res.frequency[:mode_order]
        results.append(freq)
    results = np.array(results)
    return lambdas, results


@functools.lru_cache()
def get_flutter_data_SCSC():
    mode_order = 2
    lambdas = np.linspace(100, 600, 51)
    lambdas = np.concatenate([lambdas, np.linspace(520, 550, 31)])
    lambdas = np.concatenate([lambdas, np.linspace(542, 547, 51)])
    lambdas = sorted(set(lambdas))
    results = []
    for lbd in lambdas:
        print(lbd)
        solver = get_solver_SCSC(lbd)
        res = solver.solve(mode_order * 2, solver='sparse')
        freq = res.frequency[:mode_order]
        results.append(freq)
    results = np.array(results)
    return lambdas, results


@functools.lru_cache()
def get_flutter_data_SFSF():
    mode_order = 3
    lambdas = np.linspace(50, 400, 36)
    lambdas = np.concatenate([lambdas, np.linspace(320, 340, 21)])
    lambdas = np.concatenate([lambdas, np.linspace(332, 337, 51)])
    lambdas = sorted(set(lambdas))
    results = []
    for lbd in lambdas:
        print(lbd)
        solver = get_solver_SFSF(lbd)
        res = solver.solve(mode_order * 2, solver='sparse')
        freq = res.frequency[:mode_order]
        results.append(freq)
    results = np.array(results)
    return lambdas, results


@functools.lru_cache()
def get_flutter_data_SSeSSe():
    kw = 200000 * cv1.D / cv1.L2
    mode_order = 3
    lambdas = np.linspace(100, 600, 51)
    # lambdas = np.concatenate([lambdas, np.linspace(320, 340, 21)])
    lambdas = np.concatenate([lambdas, np.linspace(485, 490, 51)])
    lambdas = sorted(set(lambdas))
    results = []
    for lbd in lambdas:
        print(lbd)
        solver = get_solver_SSeSSe(lbd, kw)
        res = solver.solve(mode_order * 2, solver='sparse')
        freq = res.frequency[:mode_order]
        results.append(freq)
    results = np.array(results)
    return lambdas, results


def example1():
    """Flutter curve for SSSS boundary conditions."""
    filenames = ('../getdata/Song and Li, 2014, fig3 - 1.txt',
                 '../getdata/Song and Li, 2014, fig3 - 2.txt')
    lambda_, frequency = get_flutter_data_SSSS()
    fig = plt.figure(figsize=(3, 2.25))
    ax = fig.add_subplot(111)
    line1, _ = ax.plot(lambda_, frequency, 'k-')
    for filename in filenames:
        line2, = ax.plot(*(read_data(filename).T), 'r--')
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'Frequency (Hz)')
    ax.legend([line1, line2], ['Present method', 'Ref. [{}]'.format(REF)],
              fontsize='small')
    ax.set_xlim(200, 600)
    ax.set_ylim(400, 1400)
    fig.tight_layout(pad=0)
    fig.savefig(FIG_PREFIX + '1.svg')
    return lambda_, frequency


def example2():
    """Flutter curve for SCSC boundary conditions."""
    filenames = ('../getdata/Song and Li, 2014, fig4 - 1.txt',
                 '../getdata/Song and Li, 2014, fig4 - 2.txt')
    lambda_, frequency = get_flutter_data_SCSC()
    fig = plt.figure(figsize=(3, 2.25))
    ax = fig.add_subplot(111)
    line1, _ = ax.plot(lambda_, frequency, 'k-')
    for filename in filenames:
        line2, = ax.plot(*(read_data(filename).T), 'r--')
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'Frequency (Hz)')
    ax.legend([line1, line2], ['Present method', 'Ref. [{}]'.format(REF)],
              fontsize='small')
    ax.set_xlim(200, 600)
    ax.set_ylim(600, 1400)
    fig.tight_layout(pad=0)
    fig.savefig(FIG_PREFIX + '2.svg')
    return lambda_, frequency


def example3():
    """Flutter curve for SFSF boundary conditions."""
    filenames = ('../getdata/Song and Li, 2014, fig6 - 1.txt',
                 '../getdata/Song and Li, 2014, fig6 - 2.txt',
                 '../getdata/Song and Li, 2014, fig6 - 3.txt')
    lambda_, frequency = get_flutter_data_SFSF()
    fig = plt.figure(figsize=(3, 2.25))
    ax = fig.add_subplot(111)
    line1, *_ = ax.plot(lambda_, frequency, 'k-')
    for filename in filenames:
        line2, = ax.plot(*(read_data(filename).T), 'r--')
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'Frequency (Hz)')
    ax.legend([line1, line2], ['Present method', 'Ref. [{}]'.format(REF)],
              fontsize='small')
    ax.set_xlim(50, 400)
    ax.set_ylim(200, 1000)
    fig.tight_layout(pad=0)
    fig.savefig(FIG_PREFIX + '3.svg')
    return lambda_, frequency


def example4():
    """Flutter curve for SFSF boundary conditions."""
    filenames = ('../getdata/Song and Li, 2014, fig8 - 1.txt',
                 '../getdata/Song and Li, 2014, fig8 - 2.txt',
                 '../getdata/Song and Li, 2014, fig8 - 3.txt')
    lambda_, frequency = get_flutter_data_SSeSSe()
    fig = plt.figure(figsize=(3, 2.25))
    ax = fig.add_subplot(111)
    line1, *_ = ax.plot(lambda_, frequency, 'k-')
    for filename in filenames:
        line2, = ax.plot(*(read_data(filename).T), 'r--')
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'Frequency (Hz)')
    ax.legend([line1, line2], ['Present method', 'Ref. [{}]'.format(REF)],
              fontsize='small', loc='lower right')
    ax.set_xlim(100, 600)
    ax.set_ylim(300, 1500)
    ax.text(150, 900, r'$k_3L_2/D=2\times 10^5$')
    fig.tight_layout(pad=0)
    fig.savefig(FIG_PREFIX + '4.svg')
    return lambda_, frequency


if __name__ == '__main__':
    res1 = example1()
    res2 = example2()
    res3 = example3()
    res4 = example4()
    plt.show()
    plt.close('all')
