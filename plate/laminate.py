#!/usr/bin/env python3


"""Ref[1]: Mechanics of Laminated composite plates and shells. J.N. Reddy.
Ref[2]: [1] VLACHOUTSIS S. Shear correction factors for plates and shells[J/OL]. International Journal for Numerical Methods in Engineering, 1992, 33(7): 1537–1552."""


import numpy as np
from .quadrature import quad_gauss
from .misc import ClassConstant


NQUAD = 6


class Material:
    """Material property.

    Parameters
    ----------
    E1, E2, G23, G13, G12: scalar value
        Elastic parameters of the material.
    nu12: scalar value
        Poisson's ratio.
    rho: scalar value
        Density of the material.
    a1, a2, a12: scalar value
        Thermal coefficients of the material.
    name: string
        Name of the material.
    dtype: data-type
        The desired data-type used for storage.
    """
    def __init__(self, E1, E2, G23, G13, G12, nu12,
                 rho=None,
                 a1=0, a2=0, a12=0,
                 name='Material',
                 dtype=None):
        self._E1 = E1
        self._E2 = E2
        self._G23 = G23
        self._G13 = G13
        self._G12 = G12
        self._nu12 = nu12
        self._nu21 = E2 * nu12 / E1
        self._rho = rho
        self._a1 = a1
        self._a2 = a2
        self._a12 = a12
        self._name = name
        self._dtype = (np.result_type(E1, E2, G23, G13, G12, float)
                       if dtype is None else dtype)

    def __repr__(self):
        return 'Material: {name}'.format(name=self._name)

    @property
    def dtype(self):
        return self._dtype

    @ClassConstant()
    def Q(self):
        """Calculate the plane stress-reduced stiffness matrix.
        The denifition of the matrix can be found on p33 in Ref[1]."""
        E1 = self._E1
        E2 = self._E2
        G23 = self._G23
        G13 = self._G13
        G12 = self._G12
        nu12 = self._nu12
        nu21 = self._nu21
        den = 1 - nu12 * nu21
        Q11 = E1 / den
        Q12 = nu12 * E2 / den
        Q22 = E2 / den
        Q66 = G12
        Q44 = G23
        Q55 = G13
        Q = np.zeros([6, 6], dtype=self._dtype)
        Q[0, 0] = Q11
        Q[0, 1] = Q12
        Q[1, 0] = Q12
        Q[1, 1] = Q22
        Q[5, 5] = Q66
        Q[3, 3] = Q44
        Q[4, 4] = Q55
        return Q

    @ClassConstant()
    def a(self):
        """Get the thermal coefficients."""
        a = np.array([self._a1, self._a2, self._a12], dtype=self._dtype)
        return a

    def Qb(self, angle):
        """Calculate the transformed plane stress-reduced stiffness matrix.
        The denifition of the matrix can be found on p101 in Ref[1]."""
        Q = self.Q()
        Q11 = Q[0, 0]
        Q12 = Q[0, 1]
        Q22 = Q[1, 1]
        Q44 = Q[3, 3]
        Q55 = Q[4, 4]
        Q66 = Q[5, 5]
        c = np.cos(angle)
        s = np.sin(angle)
        Qb11 = Q11 * c**4 + 2 * (Q12 + 2 * Q66) * s**2 * c**2 + Q22 * s**4
        Qb12 = (Q11 + Q22 - 4 * Q66) * s**2 * c**2 + Q12 * (s**4 + c**4)
        Qb22 = Q11 * s**4 + 2 * (Q12 + 2 * Q66) * s**2 * c**2 + Q22 * c**4
        Qb16 = (Q11 - Q12 - 2 * Q66) * s * c**3 + \
            (Q12 - Q22 + 2 * Q66) * s**3 * c
        Qb26 = (Q11 - Q12 - 2 * Q66) * s**3 * c + \
            (Q12 - Q22 + 2 * Q66) * s * c**3
        Qb66 = (Q11 + Q22 - 2 * Q12 - 2 * Q66) * \
            s**2 * c**2 + Q66 * (s**4 + c**4)
        Qb44 = Q44 * c**2 + Q55 * s**2
        Qb45 = (Q55 - Q44) * c * s
        Qb55 = Q44 * s**2 + Q55 * c**2
        Qb = np.zeros([6, 6], dtype = self._dtype)
        Qb[0, 0] = Qb11
        Qb[0, 1] = Qb12
        Qb[0, 5] = Qb16
        Qb[1, 0] = Qb12
        Qb[1, 1] = Qb22
        Qb[1, 5] = Qb26
        Qb[5, 0] = Qb16
        Qb[5, 1] = Qb26
        Qb[5, 5] = Qb66
        Qb[3, 3] = Qb44
        Qb[3, 4] = Qb45
        Qb[4, 3] = Qb45
        Qb[4, 4] = Qb55
        return Qb

    def ab(self, angle):
        """Calculate the transformed thermal coefficients.
        The denifition of the can be found on p101 in Ref[1]."""
        a1, a2 = self._a1, self._a2
        c = np.cos(angle)
        s = np.sin(angle)
        a11 = a1 * c * c + a2 * s * s
        a22 = a1 * s * s + a2 * c * c
        a12 = (a1 - a2) * s * c
        a = np.array([a11, a22, a12], dtype=self._dtype)
        return a


class Profile:
    def __init__(self, material, angle, z, quad=quad_gauss(NQUAD),
                 name='Profile', dtype=None):
        self._material = np.array(material).reshape(-1)
        self._angle = np.array(angle).reshape(-1)
        self._z = np.array(z).reshape(-1)
        self._quad = quad
        if not len(self._material) == len(self._angle) == len(self._z) - 1:
            raise AttributeError("sizes of input variables don't match")
        self._Qb = [self._material[i].Qb(self._angle[i])
                    for i in range(len(self._material))]
        self._ab = [self._material[i].ab(self._angle[i])
                    for i in range(len(self._material))]
        self._name = name
        self._dtype = (np.result_type(*[i.dtype for i in self._material])
                       if dtype is None else dtype)

    @property
    def dtype(self):
        return self._dtype

    def __repr__(self):
        return 'Profile: {name}'.format(name=self._name)

    def __getattr__(self, name):
        name = name.upper()
        if name == 'K':
            return self.kappa()
        if name in ['I0', 'I1', 'I2']:
            i = int(name[1])
            return self.I()[i]
        return getattr(super(), name)

    @ClassConstant()
    def kappa(self):
        """Calculate the shear correction factor of the laminate.
        This method is from Ref[2]."""
        z = self._z
        moment = np.zeros([3, 2], dtype=self._dtype)
        d = np.zeros([2], dtype=self._dtype)
        for i, Qbi in enumerate(self._Qb):
            Q = Qbi[[0, 1], [0, 1]]
            moment[0] += self._quad(lambda z: Q,
                                    z[i], z[i+1], vectorize=False)
            moment[1] += self._quad(lambda z: Q * z,
                                    z[i], z[i+1], vectorize=False)
            moment[2] += self._quad(lambda z: Q * z * z,
                                    z[i], z[i+1], vectorize=False)
            Q = Qbi[[4, 3], [4, 3]]
            d += self._quad(lambda z: Q,
                            z[i], z[i+1], vectorize=False)
        zn = moment[1] / moment[0]
        R = moment[2] - 2 * zn * moment[1] + zn ** 2 * moment[0]
        I_ = np.zeros([2], dtype=self._dtype)
        for i, Qbi in enumerate(self._Qb):
            Q = Qbi[[4, 3], [4, 3]]
            I_ += self._quad(lambda z: self._kappa_helper_g(z, zn) ** 2 / Q,
                             z[i], z[i+1], vectorize=False)
        K = R ** 2 / (d * I_)
        return np.diag(K)

    def _kappa_helper_g(self, zx, zn):
        g = np.zeros([2], dtype=self._dtype)
        z = self._z
        for i, Qbi in enumerate(self._Qb):
            Q = Qbi[[0, 1], [0, 1]]
            if z[i+1] < zx:
                g -= self._quad(lambda x: Q * (x - zn),
                                z[i], z[i+1], vectorize=False)
            else:
                g -= self._quad(lambda x: Q * (x - zn),
                                z[i], zx, vectorize=False)
                break
        return g

    @ClassConstant()
    def A(self):
        """Calculate the extensional stiffness of the laminate.
        The definition of A can be found in Ref[1]."""
        Qbs = self._Qb
        z = self._z
        A = np.zeros([3, 3], dtype=self._dtype)
        for i in range(len(Qbs)):
            idx = np.array([0, 1, 5])
            Q = Qbs[i][idx].T[idx].T
            dz = z[i+1] - z[i]
            A += Q * dz
        return A

    @ClassConstant()
    def B(self):
        """Calculate the bending-extensional coupling stiffness of the laminate.
        The definition of B can be found in Ref[1]."""
        Qbs = self._Qb
        z = self._z
        B = np.zeros([3, 3], dtype=self._dtype)
        for i in range(len(Qbs)):
            idx = np.array([0, 1, 5])
            Q = Qbs[i][idx].T[idx].T
            ddz = (z[i+1]**2 - z[i]**2) / 2
            B += Q * ddz
        return B

    @ClassConstant()
    def D(self):
        """Calculate the bending stiffness of the laminate.
        The definition of D can be found in Ref[1]."""
        Qbs = self._Qb
        z = self._z
        D = np.zeros([3, 3], dtype=self._dtype)
        for i in range(len(Qbs)):
            idx = np.array([0, 1, 5])
            Q = Qbs[i][idx].T[idx].T
            dddz = (z[i+1]**3 - z[i]**3) / 3
            D += Q * dddz
        return D

    @ClassConstant()
    def As(self):
        """Calculate the bending stiffness of the laminate.
        The definition of As can be found in Ref[1]."""
        Qbs = self._Qb
        z = self._z
        As = np.zeros([2, 2], dtype=self._dtype)
        for i in range(len(Qbs)):
            idx = np.array([3, 4])
            Q = Qbs[i][idx].T[idx].T
            dz = z[i+1] - z[i]
            As += Q * dz
        return As

    def I(self):
        """Calculate the mass momemts of interia.
        The definition of I can be found on p122 in Ref[1]."""
        rhos = [i._rho for i in self._material]
        if None in rhos:
            raise ValueError('Containing undefined mass density')
        z = self._z
        I = np.zeros([3], dtype=self._dtype)
        for i in range(len(rhos)):
            rho = rhos[i]
            dz = z[i+1] - z[i]
            ddz = (z[i+1]**2 - z[i]**2) / 2
            dddz = (z[i+1]**3 - z[i]**3) / 3
            I[0] += rho * dz
            I[1] += rho * ddz
            I[2] += rho * dddz
        return I


STEEL = Material(210e9, 210e9, 210e9 / 2.6, 210e9 / 2.6, 210e9 / 2.6, 0.3,
                 rho=7800, name='Steel')
AS3501 = Material(144.84e9, 9.65e9, 3.45e9, 4.14e9, 4.14e9, 0.3,
                  rho=1389.79, name='AS/3501')


if __name__ == '__main__':
    boron_epoxy_psi = Material(
        30e6, 3e6, 0.6e6, 1.5e6, 1.5e6, 0.25, 1010101, name='boron-epoxy-psi')
    p = Profile([boron_epoxy_psi] * 8,
                [np.pi / 6, 0, np.pi / 2, -np.pi / 4,
                    np.pi / 4, 0, np.pi / 2, -np.pi / 6],
                np.linspace(-0.5, 0.5, 9, endpoint=True))

