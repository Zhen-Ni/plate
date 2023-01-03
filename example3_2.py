#!/usr/bin/env python3

import numpy as np
import plate
from flutter_bounds import find_critical_lambda
import honeycomb

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl_fontpath = mpl.get_data_path() + '/fonts/ttf/STIXGeneral.ttf'
mpl_fontprop = mpl.font_manager.FontProperties(fname=mpl_fontpath)
plt.rc('font', family='STIXGeneral', weight='normal', size=11)
plt.rc('mathtext', fontset='stix')

FIG_PREFIX = 'figures/example3_2_'


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

    h = 0.01
    h2 = 0.008
    M = N = 15
    basis_x = plate.legendre_basis(50, interval=[0, L1])
    basis_y = plate.legendre_basis(50, interval=[0, L2])
    z = -0.005, -0.004, 0.004, 0.005
    _profile = plate.Profile([material_face, material_core,
                              material_face],
                             [0, angle / 180 * np.pi, 0],
                             z)
    _p = plate.SupersonicPlate(L1, L2, M, N, _profile,
                               basis_x=basis_x, basis_y=basis_y)
    _p.add_spring(1e12, plate.BC_C, x=0)
    _p.add_spring(1e12, plate.BC_C, x=L1)
    _p.add_spring(1e12, plate.BC_C, y=0)
    _p.add_spring(1e12, plate.BC_C, y=L2)
    _plate = _p

    @classmethod
    def get_plate(cls, t=t, l1=l1, l2=l2, theta=theta, angle=angle):
        core_para = honeycomb.material_property(cls.E, cls.G, cls.rho,
                                                t, l1, l2, theta)
        material_core = plate.Material(*core_para)
        profile = plate.Profile([cls.material_face, material_core,
                                 cls.material_face],
                                [0, angle, 0],
                                cls.z)
        p = cls._plate
        p.reset_airflow()
        p.set_profile(profile)
        return p


def example1():
    """CFAP with honeycomb edge length."""
    eta1s = np.linspace(0.1, 4, 40)
    eta1s = np.concatenate([eta1s, []])
    eta1s = np.array(sorted(set(eta1s)))
    lambdas = []
    lbd = 386                   # initial guess
    for eta1 in eta1s:
        p = cv1.get_plate(l1=cv1.l2*eta1)
        lbd = find_critical_lambda(p, lbd-15, lbd+15)
        lambdas.append(lbd)
        print(eta1, lbd)

    fig = plt.figure(figsize=(2, 1.5))
    ax = fig.add_subplot(111)
    ax.plot(eta1s, lambdas, 'k-')
    ax.set_xlabel(r'$\eta_1$')
    ax.set_ylabel(r'$\lambda_{cr}$')
    ax.set_xlim([0, 4])
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_yticks([200, 300, 400])

    fig.tight_layout(pad=0)
    
    fig.savefig(FIG_PREFIX + '1.svg')
    return eta1s, lambdas


def example2():
    """CFAP with honeycomb wall thickness."""
    eta2s = np.linspace(0.01, 0.2, 20)
    eta2s = np.concatenate([eta2s, []])
    eta2s = np.array(sorted(set(eta2s)))
    lambdas = []
    lbd = 128                   # initial guess
    for eta2 in eta2s:
        p = cv1.get_plate(t=cv1.l1*eta2)
        lbd = find_critical_lambda(p, lbd-8, lbd+8)
        lambdas.append(lbd)
        print(eta2, lbd)

    fig = plt.figure(figsize=(2, 1.5))
    ax = fig.add_subplot(111)
    ax.plot(eta2s, lambdas, 'k-')
    ax.set_xlabel(r'$\eta_2$')
    ax.set_ylabel(r'$\lambda_{cr}$')
    ax.set_xlim([0, 0.2])
    ax.set_xticks([0, 0.05, 0.1, 0.15, 0.2])

    fig.tight_layout(pad=0)
    
    fig.savefig(FIG_PREFIX + '2.svg')
    return eta2s, lambdas


def example3():
    """CFAP with honeycomb cell angle."""
    angles = np.linspace(0, 85, 18)
    lambdas = []
    lbd = 307                   # initial guess
    for angle in angles:
        p = cv1.get_plate(theta=angle / 180 * np.pi)
        lbd = find_critical_lambda(p, lbd-8, lbd+8)
        lambdas.append(lbd)
        print(angle, lbd)

    fig = plt.figure(figsize=(2, 1.5))
    ax = fig.add_subplot(111)
    ax.plot(angles, lambdas, 'k-')
    ax.set_xlabel(r'$\theta_c$ ($\degree$)')
    ax.set_ylabel(r'$\lambda_{cr}$')
    ax.set_xlim([0, 90])
    ax.set_xticks([0, 30, 60, 90])
    ax.set_yticks([300, 310, 320, 330])
    
    fig.tight_layout(pad=0)
    
    fig.savefig(FIG_PREFIX + '3.svg')
    return angles, lambdas


if __name__ == '__main__':
    res1 = example1()
    res2 = example2()
    res3 = example3()
    plt.show()
    plt.close('all')
