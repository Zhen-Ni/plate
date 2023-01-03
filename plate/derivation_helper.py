#!/usr/bin/env python3

"""This is a script to symbolically derive some of the variables used in the
dynamics of a plate.

Note that this script has nothing to do with the numerical calculation and the
package, but only provides some convenience for equation derivation and paper
writting.
"""

import sympy as sym


def get_matrix(name, m, n):
    """Get a symbolic matrix with size m x n."""
    mat = sym.zeros(m, n)
    for i in range(m):
        for j in range(n):
            mat[i, j] = sym.Symbol('{}_{}{}'.format(name, i+1, j+1))
    return mat


def get_symmetrical_matrix(name, m, n):
    """Get a symbolic matrix with size m x n."""
    mat = sym.zeros(m, n)
    for i in range(m):
        for j in range(n):
            if i <= j:
                mat[i, j] = sym.Symbol('{}_{}{}'.format(name, i+1, j+1))
    for i in range(m):
        for j in range(n):
            if i > j:
                mat[i, j] = mat[j, i]
    return mat


# Coefficients for the relation between force resultant and strain.
A = get_symmetrical_matrix('A', 6, 6)[[0, 1, 5], [0, 1, 5]]
B = get_symmetrical_matrix('B', 6, 6)[[0, 1, 5], [0, 1, 5]]
D = get_symmetrical_matrix('D', 6, 6)[[0, 1, 5], [0, 1, 5]]
As = get_symmetrical_matrix('A', 6, 6)[[3, 4], [3, 4]]
ABD = sym.Matrix([[A, B], [B, D]])
ABDAs = sym.diag(ABD, As)
# Matrix for the relation between strain and displacement.
T = sym.Matrix([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
# Intermediate matrix used for derivation of the stiffness matrix.
R = T.T @ ABDAs @ T


def get_Rij(i, j):
    i -= 1
    j -= 1
    Rk = sym.Matrix([[R[i, j], R[i, j+5], R[i, j+10]],
                     [R[i+5, j], R[i+5, j+5], R[i+5, j+10]],
                     [R[i+10, j], R[i+10, j+5], R[i+10, j+10]]])
    globals()['Rk{}{}'.format(i+1, j+1)] = Rk
    return Rk


def print_Rk():
    for i in range(1, 6):
        for j in range(1, 6):
            Rk = get_Rij(i, j)
            print('Rk_{{{i},{j}}}'.format(i=i, j=j))
            sym.pprint(Rk)
            print()
    return


print_Rk()
