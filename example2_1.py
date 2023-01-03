#!/usr/bin/env python3

import numpy as np
import plate
import honeycomb

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Rectangle
mpl_fontpath = mpl.get_data_path() + '/fonts/ttf/STIXGeneral.ttf'
mpl_fontprop = mpl.font_manager.FontProperties(fname=mpl_fontpath)
plt.rc('font', family='STIXGeneral', weight='normal', size=11)
plt.rc('mathtext', fontset='stix')


FIG_PREFIX = 'figures/example2_1_'
DPI = 600


class cv1:
    """Constants for example 2_1."""
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
    profile = plate.Profile([material_face, material_core, material_face],
                            [0, angle / 180 * np.pi, 0],
                            z)
    h = z[-1] - z[0]
    M = N = 15
    p = plate.SupersonicPlate(L1, L2, M, N, profile)
    basis_x = plate.legendre_basis(50, interval=[0, L1])
    basis_y = plate.legendre_basis(50, interval=[0, L2])
    p.set_basis(basis_x, basis_y)
    p.add_spring(1e12, plate.BC_C, x=0)
    p.add_spring(1e12, plate.BC_C, x=L1)
    p.add_spring(1e12, plate.BC_C, y=0)
    p.add_spring(1e12, plate.BC_C, y=L2)
    plate = p


class cv2:
    """Plate with structural damping."""
    L1 = 0.3
    L2 = 0.3
    
    E = 70e9 * (1+0.01j)
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
    profile = plate.Profile([material_face, material_core, material_face],
                            [0, angle / 180 * np.pi, 0],
                            z)
    h = z[-1] - z[0]
    M = N = 15
    p = plate.SupersonicPlate(L1, L2, M, N, profile)
    basis_x = plate.legendre_basis(50, interval=[0, L1])
    basis_y = plate.legendre_basis(50, interval=[0, L2])
    p.set_basis(basis_x, basis_y)
    p.add_spring(1e12, plate.BC_C, x=0)
    p.add_spring(1e12, plate.BC_C, x=L1)
    p.add_spring(1e12, plate.BC_C, y=0)
    p.add_spring(1e12, plate.BC_C, y=L2)
    plate = p


def example1():
    """Modals of the plate."""
    cv1.plate.reset_airflow()
    solver = plate.ModalSolver(cv1.plate)
    results = solver.solve(4)
    for i, r in enumerate(results):
        r._u = -r._u
        fig = plate.plot_frame_3d(r)
        fig.set_size_inches(2, 2)
        ax = fig.axes[0]
        ax.set_xlim(0, cv1.L1)
        ax.set_ylim(0, cv1.L2)
        ax.set_zlim(-1, 1)
        ax.text(cv1.L1, 0, -1,
                r'$f_{}={:.5g}\mathrm{{Hz}}$'.format(i+1, r.frequency),
                horizontalalignment='right', verticalalignment='bottom')
        bbox = fig.bbox_inches.from_bounds(0.1, 0.35, 1.8, 1.3)
        plt.savefig(FIG_PREFIX+'1_{}-{:.5g}Hz.png'.format(i+1, r.frequency),
                    bbox_inches=bbox, dpi=DPI)
    return results


def example2():
    # CFAP: 308.5(+0.1), 503.0(+0.1)
    mode_order = 4
    # lambdas  = []
    lambdas = np.linspace(100, 600, 51)
    lambdas = np.concatenate([lambdas, np.linspace(300, 310, 11)])
    lambdas = np.concatenate([lambdas, np.linspace(495, 505, 11)])
    lambdas = np.concatenate([lambdas, np.linspace(307, 309, 21)])
    lambdas = np.concatenate([lambdas, np.linspace(501.5, 503.5, 21)])
    lambdas = sorted(set(lambdas))
    lambdas = np.array(lambdas)
    results = []
    eigenvalues = []
    for lbd in lambdas:
        print(lbd)
        cv1.plate.reset_airflow()
        cv1.plate.add_airflow_lambda(lbd, 0)
        solver = plate.ComplexModalSolver(cv1.plate)
        res = solver.solve(mode_order * 2, solver='sparse')
        results.append(res)
        eigenvalues.append(res.eigen_value[:mode_order])
    eigenvalues = np.array(eigenvalues)
    
    ls = 'ko-', 'kx-', 'rD-', 'r+-'
    markevery = [i for i, v in enumerate(lambdas) if round(v/4, -1) == v/4]
    labels = '1st mode', '2nd mode', '3rd mode', '4th mode'
    # Real parts
    fig = plt.figure(figsize=(3, 2.25))
    ax = fig.add_subplot(111) 
    for i in range(mode_order):
        ax.plot(lambdas, eigenvalues.real[:, i], ls[i],
                markevery=markevery, markerfacecolor='None',
                label=labels[i])
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'Real part')
    ax.set_xlim(lambdas[0], lambdas[-1])
    ax.xaxis.set_major_locator(MultipleLocator(100.))
    # Zoom the CFAP point.
    zoom_xlim = 306, 312
    zoom_ylim = -80, 40
    p = Rectangle([270, -800], 80, 1600)
    p.set_facecolor('w')
    p.set_edgecolor('brown')
    p.set_ls('--')
    p.set_lw(0.8)
    ax.add_patch(p)
    ax.annotate("", xy=(220, -2000), xytext=(290, -1000),
                arrowprops=dict(arrowstyle="->", facecolor='k',
                                edgecolor='k'))
    ax2 = fig.add_axes([0.3, 0.25, 0.16, 0.16])
    for i in range(mode_order):
        ax2.plot(lambdas, eigenvalues[:, i].real, ls[i][0]+ls[i][2],
                 label=labels[i])
    ax2.set_xticks([])
    ax2.set_yticks([0])
    ax2.set_yticklabels([0], fontsize='small')
    ax2.set_xlim(zoom_xlim[0], zoom_xlim[1])
    ax2.set_ylim(zoom_ylim[0], zoom_ylim[1])
    fig.tight_layout(pad=0)
    fig.savefig(FIG_PREFIX+'2_real.svg')

    # Imaginary parts
    fig = plt.figure(figsize=(3, 2.25))
    ax = fig.add_subplot(111)
    for i in range(mode_order):
        ax.plot(lambdas, eigenvalues.imag[:, i]/np.pi/2, ls[i],
                markevery=markevery, markerfacecolor='None',
                label=labels[i])
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'Imaginary part ($\times 2\pi$, Hz)')
    ax.legend(loc='lower right', fontsize='small')
    ax.set_xlim(lambdas[0], lambdas[-1])
    ax.xaxis.set_major_locator(MultipleLocator(100.))
    fig.tight_layout(pad=0)
    fig.savefig(FIG_PREFIX+'2_imag.svg')

    return results
    

def example3():
    """Modals of the plate at 1st critial aerodynamic pressure."""
    lambda_ = 308.5
    cv1.plate.reset_airflow()
    cv1.plate.add_airflow_lambda(lambda_, 0)
    solver = plate.ModalSolver(cv1.plate)
    results = solver.solve(4)
    for i, r in enumerate(results):
        r._u = -r._u
        fig = plate.plot_frame_3d(r)
        fig.set_size_inches(2, 2)
        ax = fig.axes[0]
        ax.set_xlim(0, cv1.L1)
        ax.set_ylim(0, cv1.L2)
        ax.set_zlim(-1, 1)
        ax.text(cv1.L1, 0, -1,
                r'$f_{}={:.5g}\mathrm{{Hz}}$'.format(i+1, r.frequency),
                horizontalalignment='right', verticalalignment='bottom')
        bbox = fig.bbox_inches.from_bounds(0.1, 0.35, 1.8, 1.3)
        plt.savefig(FIG_PREFIX+'3_{}-{:.5g}Hz.png'.format(i+1, r.frequency),
                    bbox_inches=bbox, dpi=DPI)
    return results


def get_rms(step, nx=51, ny=51):
    p = step[0]._plate
    w = np.zeros([nx, ny, len(step.frequency)], dtype=complex)
    for i, x in enumerate(np.linspace(0, p._a, nx)):
        for j, y in enumerate(np.linspace(0, p._b, ny)):
            w[i, j, :] = step.w(x, y)
    rms = np.linalg.norm(w, axis=(0, 1))
    return rms


def example4():
    """Forced response under acoustic excitation."""
    lambda1, lambda2 = 100, 308.5
    freqs = np.linspace(0, 5000, 201)
    cv2.plate.add_force(1)
    cv2.plate.reset_airflow()
    solver = plate.HarmonicSolver(cv2.plate)
    step0 = solver.solve(freqs)
    cv2.plate.add_airflow_lambda(lambda1)
    step1 = solver.solve(freqs)
    cv2.plate.reset_airflow()
    cv2.plate.add_airflow_lambda(lambda2)
    step2 = solver.solve(freqs)
    rms0 = get_rms(step0)
    rms1 = get_rms(step1)
    rms2 = get_rms(step2)
    fig = plt.figure(figsize=(3, 2.25))
    ax = fig.add_subplot(111)
    ax.semilogy(freqs, rms0, 'k-', label='w/o airflow')
    ax.semilogy(freqs, rms1, 'r--', label=r'$\lambda$={}'.format(lambda1))
    ax.semilogy(freqs, rms2, 'b:', label=r'$\lambda=\lambda_{cr}$')
    ax.legend(fontsize='small')
    ax.set_xlim(0, 5000)
    ax.xaxis.set_ticks([0, 1000, 2000, 3000, 4000, 5000])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Displacement (m)')
    fig.tight_layout(pad=0)
    fig.savefig(FIG_PREFIX+'3-forced response.svg')


if __name__ == '__main__':
    res1 = example1()
    res2 = example2()
    res3 = example3()
    plt.show()
    plt.close('all')
