#!/usr/bin/env python3

import numpy as np
import plate
import honeycomb

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
mpl_fontpath = mpl.get_data_path() + '/fonts/ttf/STIXGeneral.ttf'
mpl_fontprop = mpl.font_manager.FontProperties(fname=mpl_fontpath)
plt.rc('font', family='STIXGeneral', weight='normal', size=11)
plt.rc('mathtext', fontset='stix')


FIG_PREFIX = 'figures/example2_2_'
DPI = 600


class cv1:
    """Constants for example 2_1."""
    L1 = 0.3
    L2 = 0.3

    E = 70e9 * (1 + 0.01j)
    rho = 2710
    nu = 0.33
    G = E / (2 * (1 + nu))
    material_face = plate.Material(E, E, G, G, G, nu, rho)

    l1 = 3e-3
    l2 = 3e-3
    t = 0.2e-3
    theta = 30 / 180 * np.pi
    core_para = honeycomb.material_property(E, G, rho, t, l1, l2, theta)
    material_core = plate.Material(*core_para)

    angle = 0

    h = 0.01
    M = N = 15
    basis_x = plate.legendre_basis(50, interval=[0, L1])
    basis_y = plate.legendre_basis(50, interval=[0, L2])
    z = -0.005, -0.004, 0.004, 0.005
    profile = plate.Profile([material_face, material_core,
                             material_face],
                            [0, angle / 180 * np.pi, 0],
                            z)
    p = plate.SupersonicPlate(L1, L2, M, N, profile,
                              basis_x=basis_x, basis_y=basis_y)
    p.add_spring(1e12, plate.BC_C, x=0)
    p.add_spring(1e12, plate.BC_C, x=L1)
    p.add_spring(1e12, plate.BC_C, y=0)
    p.add_spring(1e12, plate.BC_C, y=L2)
    plate = p


def get_plate(alpha, dtype=None):
    """`alpha` is the thickness ratio."""
    h2 = cv1.h * alpha
    h1 = (cv1.h - h2) / 2
    h3 = h1
    z = [0, h1, h2, h3]
    zi = 0
    z = np.array([zi := zi + i for i in z])
    z -= cv1.h / 2
    profile = plate.Profile([cv1.material_face, cv1.material_core,
                             cv1.material_face],
                            [0, cv1.angle / 180 * np.pi, 0],
                            z)
    cv1.plate.set_dtype(dtype)
    cv1.plate.set_profile(profile)
    return cv1.plate


def example1():
    nmodes = 4
    alpha = np.linspace(0, 1, 51)
    freqs = []
    freqs_nc = []     # results with shear correction factor = 5/6
    kappas = []
    # Calculate the natural frequencies versus alpha
    for a in alpha:
        p = get_plate(a, dtype=float)
        kappas.append(np.diag(p.profile.kappa()))
        solver = plate.ModalSolver(p)
        res = solver.solve(nmodes)
        freqs.append(res.frequency)
        p.uninitialize()
        # Dangerous operation!
        # cv1.plate become invalid until get_plate is called again!
        p._profile._kappa = 5 / 6
        res = solver.solve(nmodes)
        freqs_nc.append(res.frequency)

    # plot shear correction factor
    fig = plt.figure(figsize=(3, 2.25))
    ax = fig.add_subplot(111)
    ax.plot(alpha, [k[0] for k in kappas], 'k-', label=r'Present method, $\kappa_1$')
    ax.plot(alpha, [k[1] for k in kappas], 'r--', label=r'Present method, $\kappa_2$')
    ax.plot(alpha, 5/6*np.ones_like(alpha), 'b:', label=r'$\kappa=5/6$')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'Shear correction factor')
    ax.legend(fontsize='small')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout(pad=0)
    fig.savefig(FIG_PREFIX+'1_kappa.svg')
    
    for i in range(nmodes):
        fig = plt.figure(figsize=(3, 2.25))
        ax = fig.add_subplot(111)
        ax.plot(alpha, [f[i] for f in freqs], 'k-', label='Present method')
        ax.plot(alpha, [f[i] for f in freqs_nc], 'r--', label=r'$\kappa$=5/6')
        ax.legend(fontsize='small')
        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel(r'Frequency (Hz)')
        ax.set_xlim(0, 1)
        fig.tight_layout(pad=0)
        fig.savefig(FIG_PREFIX+'1_{}.svg'.format(i+1))


def get_flutter_curve(p, detailed_range=[]):
    # Side effect: airflow of p is changed.
    nmodes = 2
    lambdas = np.linspace(100, 400, 31)
    lambdas = np.concatenate([lambdas, detailed_range])
    lambdas = np.array(sorted(set(lambdas)))
    results = []
    for lbd in lambdas:
        print(lbd)
        p.reset_airflow()
        p.add_airflow_lambda(lbd)
        solver = plate.ComplexModalSolver(p)
        res = solver.solve(nmodes, solver='sparse')
        freq = res.frequency
        results.append(freq)
    return lambdas, np.array(results)


def example2():
    # The flutter boundaries are: ~382+, 308.5, ~199+
    alphas = [0.7, 0.8, 0.9]
    lambdas = []
    freqs = []
    detailed_ranges = (np.linspace(375, 385, 11),
                       np.linspace(300, 310, 11),
                       np.linspace(191, 201, 11))
    for alpha, detailed_range in zip(alphas, detailed_ranges):
        p = get_plate(alpha, float)
        lambda_, freq = get_flutter_curve(p, detailed_range)
        lambdas.append(lambda_)
        freqs.append(freq)
    fig = plt.figure(figsize=(3, 2.25))
    ax = fig.add_subplot(111)
    ls = 'k-o', 'r-s', 'b-^', 'g-'
    lines = []
    for i, (alpha, lbd) in enumerate(zip(alphas, lambdas)):
        markevery = [i for i, v in enumerate(lbd) if round(v/3, -1) == v/3]
        for j, freq in enumerate(freqs[i].T):
            lsi = ls[i]
            if j > 1:
                break
            line, = ax.plot(lbd, freq, lsi,
                            label=r'$\alpha ={}$'.format(alpha),
                            markevery=markevery, markerfacecolor="None")
            if j == 0:
                lines.append(line)
    ax.legend(lines, [r'$\alpha={}$'.format(alpha) for alpha in alphas],
              fontsize='small')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlim(100, 400)
    ax.set_ylim(1000, 2500)
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel('Frequency (Hz)')
    fig.tight_layout(pad=0)
    fig.savefig(FIG_PREFIX + '2-flutter curve.svg')
    return freqs


def get_rms(step, nx=51, ny=51):
    p = step[0]._plate
    w = np.zeros([nx, ny, len(step.frequency)], dtype=complex)
    for i, x in enumerate(np.linspace(0, p._a, nx)):
        for j, y in enumerate(np.linspace(0, p._b, ny)):
            w[i, j, :] = step.w(x, y)
    rms = np.linalg.norm(w, axis=(0, 1))
    return rms


def example3():
    """Forced response."""
    freqs = np.linspace(0, 5000, 201)
    alphas = [0.7, 0.8, 0.9]
    lambdas = [382, 308.5, 199]
    results = []
    for i, alpha in enumerate(alphas):
        p = get_plate(alpha, dtype=complex)
        p.add_force(1)
        p.reset_airflow()
        solver = plate.HarmonicSolver(p)
        step = solver.solve(freqs)
        rms0 = get_rms(step)
        p.add_airflow_lambda(lambdas[i])
        step = solver.solve(freqs)
        rms1 = get_rms(step)
        results.append([rms0, rms1])

    ls = 'k-', 'r--', 'b:'
    # Plot curves for cases without airflow.
    fig = plt.figure(figsize=(3, 2.25))
    ax = fig.add_subplot(111)
    for i, alpha in enumerate(alphas):
        ax.semilogy(freqs, results[i][0], ls[i],
                    label=r'$\alpha={}$'.format(alphas[i]))
    ax.legend(fontsize='small')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Displacement (m)')
    ax.xaxis.set_ticks([0, 1000, 2000, 3000, 4000, 5000])
    ax.set_xlim(0, 5000)
    fig.tight_layout(pad=0)
    fig.savefig(FIG_PREFIX+'3-noflow.svg')
    # Plot curves for cases at critial aerodynamic pressure.
    fig = plt.figure(figsize=(3, 2.25))
    ax = fig.add_subplot(111)
    for i, alpha in enumerate(alphas):
        ax.semilogy(freqs, results[i][1], ls[i],
                    label=r'$\alpha={}$'.format(alphas[i]))
    ax.legend(fontsize='small')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Displacement (m)')
    ax.xaxis.set_ticks([0, 1000, 2000, 3000, 4000, 5000])
    ax.set_xlim(0, 5000)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(0, 1, 0.1),
                               numticks=10))
    ax.set_ylim(1e-7, 5e-3)
    fig.tight_layout(pad=0)
    fig.savefig(FIG_PREFIX+'3-critical.svg')
    return results

        
if __name__ == '__main__':
    res1 = example1()
    res2 = example2()
    res3 = example3()
    plt.show()
    plt.close('all')
