#!/usr/bin/env python3

import numpy as np
import functools
import plate
from flutter_bounds import find_critical_lambda
import honeycomb
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl_fontpath = mpl.get_data_path() + '/fonts/ttf/STIXGeneral.ttf'
mpl_fontprop = mpl.font_manager.FontProperties(fname=mpl_fontpath)
plt.rc('font', family='STIXGeneral', weight='normal', size=11)
plt.rc('mathtext', fontset='stix')

DPI = 300
FIG_PREFIX = 'figures/exampleR1_1_'


class cv1:
    L1 = 0.3
    L2 = 0.3

    E = 70e9
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
    z = -0.005, -0.004, 0.004, 0.005
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


def get_plate(stiffness, dof, dtype=None):
    """A CSeCSe plate with given elastic boundary."""
    cv1.plate.set_dtype(dtype)
    cv1.plate.reset_airflow()
    cv1.plate.reset_spring()
    cv1.plate.add_spring(stiffness, dof, x=0)
    cv1.plate.add_spring(stiffness, dof, x=cv1.L1)
    cv1.plate.add_spring(1e12, plate.BC_C, y=0)
    cv1.plate.add_spring(1e12, plate.BC_C, y=cv1.L2)
    return cv1.plate


def get_solver_plate(stiffness, dof, dtype=None):
    """Solver of a CSeCSe plate, where Se is elastic boundary."""
    p = get_plate(stiffness, dof, dtype)
    solver = plate.ModalSolver(p)
    return solver


def example1():
    """Natural frequecies of CSeCSe plate with different stiffness."""
    order = 4

    for dof in 0, 1, 2:
        ks = np.logspace(4, 13, 51)
        freqs = []
        for k in ks:
            solver = get_solver_plate(k, dof)
            freq = solver.solve(order).frequency
            freqs.append(freq)
        freqs = np.array(freqs)
        example1_helper(ks, freqs, dof)
    for dof in 3, 4:
        ks = np.logspace(2, 8, 51)
        freqs = []
        for k in ks:
            solver = get_solver_plate(k, dof)
            freq = solver.solve(order).frequency
            freqs.append(freq)
        freqs = np.array(freqs)
        example1_helper(ks, freqs, dof)


def example1_helper(ks, freqs, dof):
    fig = plt.figure(figsize=(3, 2.25))
    ax = fig.add_subplot(111)
    ls = 'ko-', 'rs-', 'b^-', 'gD-'
    markevery = 5
    labels = '1st mode', '2nd mode', '3rd mode', '4th mode'
    for i, freq in enumerate(freqs.T):
        ax.semilogx(ks, freq, ls[i],
                    markevery=markevery, label=labels[i],
                    markerfacecolor="None")
    unit = "N/m^2" if dof < 3 else "N/rad"
    ax.set_xlabel(r'$k_{}\ \mathrm{{({})}}$'.format(dof + 1, unit))
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlim(ks[0], ks[-1])
    ax.set_ylim(0, 3000)
    ax.legend(fontsize='small', ncol=2)
    fig.tight_layout(pad=0)

    # zoom in the intersection for dof=2 at k3=2.6e7
    if dof == 2:
        ax2 = fig.add_axes([0.22, 0.77, 0.18, 0.18])
        for i, freq in enumerate(freqs.T):
            ax2.semilogx(ks, freq, ls[i], markevery=markevery*10,
                         markerfacecolor=None)
        ax2.set_xlim(5e6, 5e7)
        ax2.set_ylim(1800, 2200)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax.annotate("", xy=(1.8e6, 2400), xytext=(2.5e7, 2100),
                    arrowprops=dict(arrowstyle="->", facecolor='k',
                                    edgecolor='k'))
        ax.legend(fontsize='x-small', ncol=2, loc='lower right')

    fig.savefig(FIG_PREFIX + f'1_{dof+1}.svg')


def example2():
    solver = get_solver_plate(0, 0)
    results = solver.solve(4)
    for i, r in enumerate(results):
        fig = plate.plot_frame_3d(r)
        fig.set_size_inches(2, 2)
        ax = fig.axes[0]
        ax.set_xlim(0, cv1.L1)
        ax.set_ylim(0, cv1.L2)
        ax.set_zlim(-1, 1)
        ax.text(cv1.L1, 0, -1,
                r'$f_{}={:.5g}\mathrm{{Hz}}$'.format(i+1, r.frequency),
                horizontalalignment='right', verticalalignment='bottom')
        bbox = fig.bbox_inches.from_bounds(0.1, 0.35, 1.8, 1.4)
        plt.savefig(FIG_PREFIX+'2_{}-{:.5g}Hz.png'.format(i+1, r.frequency),
                    bbox_inches=bbox, dpi=DPI)


def example3():
    """CFAP with different BCs."""
    """Natural frequecies of CSeCSe plate with different stiffness.

    Sudden changes in CFAP are observed. A more detailed study can
    be performed by running the code in example4(). It is too complicated
    to explain, so not presented in revised manuscript.
    """
    for dof in 0, 1, 2, 3, 4:
        ks, lambdas = example3_helper(dof)
        fig = plt.figure(figsize=(2, 1.5))
        ax = fig.add_subplot(111)
        # A sudden drop can be seen at about k2=4e6.
        ax.semilogx(ks, lambdas, 'k-')
        match dof:
            case 0 | 1 | 2:
                unit = "N/m^2"
            case _:
                unit = "N/rad"
        ax.set_xlabel(r'$k_{} \mathrm{{({})}}$'.format(dof+1, unit))
        ax.set_ylabel(r'$\lambda_{cr}$')
        ax.set_xlim(ks[0], ks[-1])
        ax.set_ylim(0, 500)
        fig.tight_layout(pad=0)
        fig.savefig(FIG_PREFIX+'3_dof{}.svg'.format(dof+1))


@functools.lru_cache(5)
def example3_helper(dof):
    lbd = 184
    lambdas = []
    match dof:
        case 0 | 1 | 2:
            ks = np.logspace(4, 13, 51)
        case 3 | 4:
            ks = np.logspace(2, 8, 51)
        case other:
            assert 0, f"dof should be one of 0, 1, 2, 3, 4, get {other}"
    for k in ks:
        p = get_plate(k, dof)
        lbd = find_critical_lambda(p, lbd-25, lbd+25, order=4)
        lambdas.append(lbd)
    return ks, np.array(lambdas)


def example4():
    """Show the natural frequencies versus lambda.

    Not included in paper.
    """
    nmodes = 4
    ks = 1e6, 3e6, 10e6
    lambdas = np.linspace(10, 600, 51)
    for k in ks:
        freqs = []
        p = get_plate(k, 2)
        for lbd in lambdas:
            p.add_airflow_lambda(lbd)
            solver = plate.ComplexModalSolver(p)
            res = solver.solve(nmodes, solver='sparse')
            freq = res.frequency
            freqs.append(freq)
            p.reset_airflow()
        freqs = np.array(freqs)
        fig = plt.figure(figsize=(3, 2.25))
        ax = fig.add_subplot(111)
        for i in range(nmodes):
            ax.plot(lambdas, freqs.T[i])
    plt.show()


if __name__ == '__main__':
    example1()
    example2()
    plt.show()
