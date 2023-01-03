#!/usr/bin/env python3

import time
import numpy as np
import plate
import honeycomb
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl_fontpath = mpl.get_data_path() + '/fonts/ttf/STIXGeneral.ttf'
mpl_fontprop = mpl.font_manager.FontProperties(fname=mpl_fontpath)
plt.rc('font', family='STIXGeneral', weight='normal', size=11)
plt.rc('mathtext', fontset='stix')

DPI = 300
FIG_PREFIX = 'figures/example1_1_'


class cv1:
    """Constants for example 1_1."""
    # Geometry of the plate
    L1 = 300e-3
    L2 = 300e-3
    h0 = 1e-3
    h1 = 8e-3
    h2 = 1e-3
    _z = np.array([0, h0, h1, h2])
    # workaround, see https://stackoverflow.com/questions/13905741
    def z(_z):
        h = 0
        return [h := h + i for i in _z] - h / 2
    z = z(_z)

    # material properties
    nu = 0.33                   # Poisson's ratio
    E = 70e9                    # Young's modulus
    G = E / (2 * (1 + nu))      # Shear modulus
    rho = 2710                  # density
    
    # geometry properties for the honeycomb core
    t = 0.2e-3                  # Thickness
    # DO NOT CONFUSE WITH `L1` and `L2`!
    l1 = 3e-3
    l2 = 3e-3
    theta = 30 / 180 * np.pi


aluminium = plate.Material(cv1.E, cv1.E, cv1.G, cv1.G, cv1.G, cv1.nu, cv1.rho)
honeycomb_property = honeycomb.material_property(cv1.E, cv1.G, cv1.rho,
                                                 cv1.t, cv1.l1, cv1.l2,
                                                 cv1.theta)
honeycomb_material = plate.Material(*honeycomb_property)
profile = plate.Profile([aluminium, honeycomb_material, aluminium],
                        [0, 0, 0], cv1.z)


def get_solver1(M: int, N: int, assemble: bool = True):
    p = plate.Plate(cv1.L1, cv1.L2, M, N, profile)
    # It is shown that using legendre basis can avoid the convergence
    # problem. When using the default trigonometric_basis, No convergence
    # error occures before M=N=14. Using legendre solves the problem.
    basis_x = plate.legendre_basis(50, interval=[0, cv1.L1])
    basis_y = plate.legendre_basis(50, interval=[0, cv1.L2])
    p.set_basis(basis_x, basis_y)
    p.add_spring(1e12, plate.BC_C, x=0)
    p.add_spring(1e12, plate.BC_C, x=cv1.L1)
    p.add_spring(1e12, plate.BC_C, y=0)
    p.add_spring(1e12, plate.BC_C, y=cv1.L2)
    if assemble:
        p.assemble()
    solver = plate.ModalSolver(p)
    return solver


def example1():
    mode_order = 10
    basis_order = 10, 11, 12, 13, 14, 15
    results = {'frequency': [], 'time': []}
    for order in basis_order:
        solver = get_solver1(order, order, assemble=False)
        t = time.time()
        res = solver.solve(mode_order*2)
        t = time.time() - t
        f = res.frequency
        results['frequency'].append(f)
        results['time'].append(t)
    print('mode', end='\t')
    print('\t'.join(['M=N={}'.format(i) for i in basis_order]))
    for i in range(mode_order):
        print(i+1, end='\t')
        print('\t'.join(['{:.5g}'.format(j[i]) for j in results['frequency']]))
    print('time', end='\t')
    print('\t'.join(['{:.4f}'.format(i) for i in results['time']]))
    return results


def get_solver2(k: float, p: plate.Plate):
    p.reset_spring()
    p.add_spring(k, plate.BC_C, x=0)
    p.add_spring(k, plate.BC_C, x=cv1.L1)
    p.add_spring(k, plate.BC_C, y=0)
    p.add_spring(k, plate.BC_C, y=cv1.L2)
    solver = plate.ModalSolver(p)
    return solver


def get_plate2(basis_order: int):
    p = plate.Plate(cv1.L1, cv1.L2, basis_order, basis_order, profile)
    # It is shown that using legendre basis can avoid the convergence
    # problem. When using the default trigonometric_basis, No convergence
    # error occures before M=N=14. Using legendre solves the problem.
    basis_x = plate.legendre_basis(50, interval=[0, cv1.L1])
    basis_y = plate.legendre_basis(50, interval=[0, cv1.L2])
    p.set_basis(basis_x, basis_y)
    return p


def example2():
    mode_order = 2
    basis_order = 15
    results = []
    stiffness = np.logspace(0, 21, 106)
    p = get_plate2(basis_order)
    for k in stiffness:
        solver = get_solver2(k, p)
        res = solver.solve(20)
        results.append(res)
    # Plot
    fig = plt.figure(figsize=(3, 2.25))
    ax = fig.add_subplot(111)
    # ax = fig.add_axes([0.21, 0.21, 0.73, 0.78])
    labels = '1st mode', '2nd mode'
    styles = 'k-', 'r--'
    for i in range(mode_order):
        ax.semilogx(stiffness, [j[i].frequency for j in results],
                    styles[i], label=labels[i])
    ax.legend(fontsize='small', loc='lower right')
    ax.set_xlabel('Stiffness of boundary springs')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xticks(10.**np.array([0, 3, 6, 9, 12, 15, 18, 21]))
    ax.set_xlim(10. ** 0, 10. ** 21)
    ax.set_yticks([0, 1000, 2000, 3000, 4000, 5000])
    ax.set_ylim(0, 3000)
    fig.tight_layout(pad=0)
    fig.savefig(FIG_PREFIX + '2.svg')


if __name__ == '__main__':
    res1 = example1()
    res2 = example2()
    plt.show()
    plt.close('all')
