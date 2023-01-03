#!/usr/bin/env python3

import math


def material_property(E, G, rho, t, l1, l2, theta):
    """Get the equivalent material properties for hexagonal honeycombs.

    Parameters
    ----------
    E: float
        The Young's modulus of the honeycomb material.
    G: float
        The shear modulus of the honeycomb material.
    rho: float
        The mass density of the honeycomb material.
    t: float
        Thickness of the honeycomb wall.
    l1: float
        Length of the inclined wall of the honeycomb.
    l2: float
        Length of the vertical wall of the honeycomb.
    theta: float
        Degree of the hexagonal. (in rad)
    """
    s = math.sin(theta)
    c = math.cos(theta)
    e1 = l2 / l1
    e2 = t / l1
    t = s / c
    E1 = E * e2 ** 3 / ((t ** 2 + e2 ** 2) * (e1+s) * c)
    E2 = E * e2 ** 3 * (e1 + s) / ((c**2 + (e1 + s**2)*e2**2)*c)
    v12 = (1-e2**2)*s/((t**2+e2**2)*(e1+s))
    G12 = E * e2**3 * (e1 + s) / (e1**2*(1+2*e1)*c)
    G13 = G * (e2*c)/(e1+s)
    G23 = G * e2 / (2*c) * (((e1+s)/(1+2*e1)) + (e1+2*s**2)/(2*(e1+s)))
    rhoc = rho * e2 * (1+e1)/((e1+s)*c)
    return E1, E2, G23, G13, G12, v12, rhoc
