#!/usr/bin/env python3

import numpy as np
import plate
import honeycomb

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl_fontpath = mpl.get_data_path() + '/fonts/ttf/STIXGeneral.ttf'
mpl_fontprop = mpl.font_manager.FontProperties(fname=mpl_fontpath)
plt.rc('font', family='STIXGeneral', weight='normal', size=11)
plt.rc('mathtext', fontset='stix')

FIG_PREFIX = 'figures/example3_1_'


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


    
def _example1_find_intersection():
    from scipy.optimize import minimize

    def func(x0):
        eta1, = x0
        p = cv1.get_plate(l1=cv1.l2 * eta1)
        solver = plate.ModalSolver(p)
        res = solver.solve(3)
        y1 = res.frequency[1]
        y2 = res.frequency[2]
        return y2 - y1

    res = minimize(func, x0=[0.3])
    print(res)

    
def example1():
    """Effect of honeycomb edge length."""
    nmodes = 4
    eta1s = np.linspace(0.1, 4, 79)
    eta1s = np.concatenate([eta1s, np.linspace(0.95, 1.05, 11)])
    eta1s = np.concatenate([eta1s, np.linspace(0.25, 0.35, 11)])
    # The intersection points are calculated by _example1_find_intersection
    eta1s = np.concatenate([eta1s, [0.99882391]])
    eta1s = np.concatenate([eta1s, [0.3045047]])
    eta1s = np.array(sorted(set(eta1s)))
    freqs = []
    for eta1 in eta1s:
        p = cv1.get_plate(l1=cv1.l2*eta1)
        solver = plate.ModalSolver(p)
        res = solver.solve(nmodes)
        freqs.append(res.frequency)
        
    fig = plt.figure(figsize=(3, 2.5))
    ax = fig.add_subplot(111)
    ls = 'k-o', 'r-s', 'b-^', 'g-D'
    labels = '1st', '2nd', '3rd', '4th'
    markevery = [i for i, eta in enumerate(eta1s) if
                 abs(round(eta / 0.4) - (eta / 0.4)) < 1e-6]
    for i in range(nmodes):
        ax.plot(eta1s, [freq[i] for freq in freqs], ls[i],
                label=labels[i], markevery=markevery, markerfacecolor="None")
    ax.set_xlabel(r'$\eta_1$')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlim([0, 4])
    ax.set_ylim([400, 3200])
    ax.legend(fontsize='small', ncol=2, loc='lower right')
    # Zoom in the intersection at eta1=1
    ax2 = fig.add_axes([0.36, 0.8, 0.12, 0.12])
    for i in range(nmodes):
        ax2.plot(eta1s, [freq[i] for freq in freqs], ls[i],
                 label=labels[i], markevery=markevery, markerfacecolor="None")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlim(0.95, 1.05)
    ax2.set_ylim(2220, 2230)
    ax.annotate("", xy=(1.15, 2525), xytext=(1, 2230),
                arrowprops=dict(arrowstyle="->", facecolor='k',
                                edgecolor='k'))
    # Zoom in the intersection near eta1=0.3
    ax3 = fig.add_axes([0.34, 0.45, 0.18, 0.18])
    for i in range(nmodes):
        ax3.plot(eta1s, [freq[i] for freq in freqs], ls[i],
                 label=labels[i], markevery=markevery, markerfacecolor="None")
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_xlim(0.25, 0.35)
    ax3.set_ylim(1920, 2050)
    ax.annotate("", xy=(0.7, 1750), xytext=(0.31, 1950),
                arrowprops=dict(arrowstyle="->", facecolor='k',
                                edgecolor='k'))

    fig.tight_layout(pad=0)
    
    fig.savefig(FIG_PREFIX + '1.svg')
    return eta1s, freqs


def example2():
    """Effect of honeycomb thickness."""
    nmodes = 4
    eta2s = np.linspace(0.01, 0.2, 19*5+1)
    eta2s = np.concatenate([eta2s, [0.005]])
    eta2s = np.array(sorted(set(eta2s)))
    freqs = []
    for eta2 in eta2s:
        p = cv1.get_plate(t=eta2*cv1.l1)
        solver = plate.ModalSolver(p)
        res = solver.solve(nmodes)
        freqs.append(res.frequency)
    fig = plt.figure(figsize=(3, 2.5))
    ax = fig.add_subplot(111)
    ls = 'k-o', 'r-s', 'b-^', 'g-D'
    labels = '1st', '2nd', '3rd', '4th'
    markevery = [i for i, eta in enumerate(eta2s) if
                 abs(round(eta / 0.02) - (eta / 0.02)) < 1e-6]
    for i in range(nmodes):
        ax.plot(eta2s, [freq[i] for freq in freqs], ls[i],
                label=labels[i], markevery=markevery, markerfacecolor="None")
    ax.set_xlabel(r'$\eta_2$')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlim(0, 0.2)
    ax.set_ylim(200, 3200)
    ax.legend(fontsize='small', loc='lower right', ncol=2)
    fig.tight_layout(pad=0)
    fig.savefig(FIG_PREFIX + '2.svg')
    return eta2s, freqs


def example3():
    """Effect of honeycomb cell angle."""
    nmodes = 4
    angles = np.linspace(0, 85, 86)
    freqs = []
    for angle in angles:
        p = cv1.get_plate(theta=angle/180 * np.pi)
        solver = plate.ModalSolver(p)
        res = solver.solve(nmodes)
        freqs.append(res.frequency)
    fig = plt.figure(figsize=(3, 2.5))
    ax = fig.add_subplot(111)
    ls = 'k-o', 'r-s', 'b-^', 'g-D'
    labels = '1st', '2nd', '3rd', '4th'
    markevery = [i for i, eta in enumerate(angles) if
                 abs(round(eta / 10) - (eta / 10)) < 1e-6]
    for i in range(nmodes):
        ax.plot(angles, [freq[i] for freq in freqs], ls[i],
                label=labels[i], markevery=markevery, markerfacecolor="None")
    ax.set_xlabel(r'$\theta_c$ ($\degree$)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlim(0, 90)
    ax.set_ylim(200, 3200)
    ax.legend(fontsize='small', loc='lower left', ncol=2)

    # Zoom the interaction
    ax2 = fig.add_axes([0.45, 0.5, 0.15, 0.12])
    ax2.set_xticks([])
    ax2.set_yticks([])
    for i in range(nmodes):
        ax2.plot(angles, [freq[i] for freq in freqs], ls[i],
                 label=labels[i], markevery=markevery, markerfacecolor="None")
    ax2.set_xlim(29, 31)
    ax2.set_ylim(2222, 2227)
    ax.annotate("", xy=(35, 1820), xytext=(30, 2150),
                arrowprops=dict(arrowstyle="->", facecolor='k',
                                edgecolor='k'))
    
    fig.tight_layout(pad=0)  
    fig.savefig(FIG_PREFIX + '3.svg')
    return angles, freqs


if __name__ == '__main__':
    res1 = example1()
    res2 = example2()
    res3 = example3()
    plt.show()
    plt.close('all')
