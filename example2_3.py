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


FIG_PREFIX = 'figures/example2_3_'


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
    M = N = 11   # use 11 instead of 15 for faster computation
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


def get_plate(beta, dtype=None):
    """`beta` is h1/h3."""
    h = cv1.h
    h2 = cv1.h2
    h1 = (h - h2) / (1 + beta) * beta
    h3 = h - h1 - h2
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
    betas = np.linspace(0, 1, 41)
    freqs = []
    freqs_nc = []     # results with shear correction factor = 5/6
    kappas = []
    # Calculate the natural frequencies versus alpha
    for a in betas:
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
    ax.plot(betas, [k[0] for k in kappas], 'k-', label=r'Present method, $\kappa_1$')
    ax.plot(betas, [k[1] for k in kappas], 'r--', label=r'Present method, $\kappa_2$')
    ax.plot(betas, 5/6*np.ones_like(betas), 'b:', label=r'$\kappa=5/6$')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'Shear correction factor')
    ax.legend(fontsize='small')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout(pad=0)
    fig.savefig(FIG_PREFIX+'1_kappa.svg')

    # Compare natural frequencies.
    for i in range(nmodes):
        fig = plt.figure(figsize=(3, 2.25))
        ax = fig.add_subplot(111)
        ax.plot(betas, [f[i] for f in freqs], 'k-', label='Present method')
        ax.plot(betas, [f[i] for f in freqs_nc], 'r--', label=r'$\kappa$=5/6')
        ax.legend(fontsize='small')
        ax.set_xlabel(r'$\beta$')
        ax.set_ylabel(r'Frequency (Hz)')
        ax.set_xlim(0, 1)
        fig.tight_layout(pad=0)
        fig.savefig(FIG_PREFIX+'1_{}.svg'.format(i+1))

    # Plot all natural frequencies
    fig = plt.figure(figsize=(3, 2.25))
    ax = fig.add_subplot(111)
    ls = 'k-.s', 'r-o', 'b--x', 'g:^'
    # ls = 'k--', 'r-', 'b:', 'g-.'
    labels = '1st', '2nd', '3rd', '4th'
    for i in range(nmodes):
        ax.plot(betas, [f[i] for f in freqs], ls[i],
                label=labels[i], markerfacecolor="None",
                markevery=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 3300)
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel('Frequency (Hz)')
    ax.legend(fontsize='small', ncol=2)
    fig.tight_layout(pad=0)
    fig.savefig(FIG_PREFIX+'1-natural frequencies.svg')
    return freqs


def get_flutter_curve(p, detailed_range=[]):
    # Side effect: airflow of p is changed.
    nmodes = 2
    lambdas = np.linspace(0, 400, 41)
    lambdas = np.concatenate([lambdas, detailed_range])
    lambdas = np.array(sorted(set(lambdas)))
    results = []
    eigensolver = 'sparse'
    for lbd in lambdas:
        print(lbd)
        p.reset_airflow()
        if lbd:
            p.add_airflow_lambda(lbd)
        solver = plate.ComplexModalSolver(p)
        res = solver.solve(nmodes, solver=eigensolver)
        freq = res.frequency
        results.append(freq)
    return lambdas, np.array(results)


def example2():
    # The flutter boundaries are: ~154-, ~250-, ~297-, 308.5
    cfap = [154, 250, 297, 309]
    betas = [0.1, 0.3, 0.6, 1.0]
    lambdas = []
    freqs = []
    detailed_ranges = (np.linspace(145, 155, 11),
                       np.linspace(240, 350, 11),
                       np.linspace(290, 300, 11),
                       np.linspace(300, 310, 11))
    for beta, detailed_range in zip(betas, detailed_ranges):
        p = get_plate(beta, float)
        lambda_, freq = get_flutter_curve(p, detailed_range)
        lambdas.append(lambda_)
        freqs.append(freq)

    fig = plt.figure(figsize=(3, 2.25))
    ax = fig.add_subplot(111)
    # ls = 'k-o', 'r-s', 'b-^', 'g-s'
    ls = 'k-', 'r--', 'b-.', 'g:'
    lines = []
    for i, (beta, lbd) in enumerate(zip(betas, lambdas)):
        markevery = [i for i, v in enumerate(lbd) if round(v/3, -1) == v/3]
        for j, freq in enumerate(freqs[i].T):
            lsi = ls[i]
            lbdi = lbd
            freqi = freq
            if j == 1:
                lbdi = lbd[lbd <= cfap[i]]
                freqi = freq[lbd <= cfap[i]]
            line, = ax.plot(lbdi, freqi, lsi,
                            label=r'$\beta ={}$'.format(beta),
                            markevery=markevery, markerfacecolor="None")
            if j == 0:
                lines.append(line)
    ax.legend(lines, [r'$\beta={}$'.format(beta) for beta in betas],
              fontsize='small')
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlim(0, 400)
    ax.set_ylim(500, 2500)
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel('Frequency (Hz)')
    fig.tight_layout(pad=0)
    fig.savefig(FIG_PREFIX + '2-flutter curve.svg')
    return freqs


if __name__ == '__main__':
    res1 = example1()
    res2 = example2()
    plt.show()
    plt.close('all')
