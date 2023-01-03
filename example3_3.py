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

FIG_PREFIX = 'figures/example3_3_'


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
    """Natural frequencies with different honeycomb cell angle and 
    ply-angle."""
    nmodes = 4
    theta_cs = [0, 30, 60]
    theta_2s = np.linspace(0, 90, 10)
    frequencies = np.zeros([len(theta_cs), len(theta_2s), nmodes])
    for i, theta_c in enumerate(theta_cs):
        for j, theta_2 in enumerate(theta_2s):
            p = cv1.get_plate(theta=theta_c/180*np.pi,
                              angle=theta_2/180*np.pi)
            solver = plate.ComplexModalSolver(p)
            freq = solver.solve(nmodes)
            frequencies[i, j] = freq.frequency
            
    ls = 'k-o', 'r-s', 'b-^', 'g-D', 'c->', 'm-*'
    for i in range(4):
        fig = plt.figure(figsize=(3, 2.25))
        ax = fig.add_subplot(111)
        for j in range(len(theta_cs)):
            ax.plot(theta_2s, frequencies[j, :, i], ls[j],
                    label=r'$\theta_c={}\degree$'.format(theta_cs[j]))
        ax.set_xlabel(r'$\theta_2$ ($\degree$)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlim(0, 90)
        ax.legend(fontsize='small')
        fig.tight_layout(pad=0)
        fig.savefig(FIG_PREFIX+'1_{}.svg'.format(i+1))
    return frequencies
    

def example2():
    """Natural frequencies with different honeycomb cell angle and 
    ply-angle with airflow."""
    nmodes = 4
    theta_cs = [0, 30, 60]
    theta_2s = np.linspace(0, 90, 10)
    frequencies = np.zeros([len(theta_cs), len(theta_2s), nmodes])
    for i, theta_c in enumerate(theta_cs):
        for j, theta_2 in enumerate(theta_2s):
            p = cv1.get_plate(theta=theta_c/180*np.pi,
                              angle=theta_2/180*np.pi)
            p.add_airflow_lambda(200)
            solver = plate.ComplexModalSolver(p)
            freq = solver.solve(nmodes)
            frequencies[i, j] = freq.frequency
            
    ls = 'k-o', 'r-s', 'b-^', 'g-D', 'c->', 'm-*'
    for i in range(4):
        fig = plt.figure(figsize=(3, 2.25))
        ax = fig.add_subplot(111)
        for j in range(len(theta_cs)):
            ax.plot(theta_2s, frequencies[j, :, i], ls[j],
                    label=r'$\theta_c={}\degree$'.format(theta_cs[j]))
        ax.set_xlabel(r'$\theta_2$ ($\degree$)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlim(0, 90)
        if i == 0:
            ax.set_ylim(1350, 1510)
        ax.legend(fontsize='small', loc='lower center')
        fig.tight_layout(pad=0)
        fig.savefig(FIG_PREFIX+'2_{}.svg'.format(i+1))
    return frequencies


def example3():
    """CFAP of the plate with different honeycomb cell angle and 
    ply-angle with airflow."""
    theta_cs = [0, 30, 60]
    theta_2s = np.linspace(0, 90, 10)
    CFAPs = np.zeros([len(theta_cs), len(theta_2s)])
    lbd = 300                   # initial guess
    for i, theta_c in enumerate(theta_cs):
        for j, theta_2 in enumerate(theta_2s):
            p = cv1.get_plate(theta=theta_c/180*np.pi,
                              angle=theta_2/180*np.pi)
            lbd = find_critical_lambda(p, lbd-10, lbd+10)
            print(theta_c, theta_2, lbd)
            CFAPs[i, j] = lbd
            
    ls = 'k-o', 'r-s', 'b-^', 'g-D', 'c->', 'm-*'
    fig = plt.figure(figsize=(3, 2.25))
    ax = fig.add_subplot(111)
    for i, theta_c in enumerate(theta_cs):
        ax.plot(theta_2s, CFAPs[i], ls[i],
                label=r'$\theta_c={}\degree$'.format(theta_c),
                markerfacecolor="None")
    ax.set_xlim(0, 90)
    ax.set_xlabel(r'$\theta_2$ ($\degree$)')
    ax.set_ylabel(r'$\lambda_{cr}$')
    ax.set_ylim(280, 340)
    ax.legend(fontsize='small')

    fig.tight_layout(pad=0)
    fig.savefig(FIG_PREFIX+'3-CFAP.svg')
    return CFAPs



if __name__ == '__main__':
    res1 = example1()
    res2 = example2()
    res3 = example3()
    plt.show()
    plt.close('all')
