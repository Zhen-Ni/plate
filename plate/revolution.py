#!/usr/bin/env python3

import numpy as np
from scipy.integrate import dblquad, quadrature
from scipy.linalg import eig, eigvals
from scipy.sparse import lil_matrix, csc_matrix
import numpy.linalg as linalg
import scipy.sparse.linalg as slinalg

from .laminate import Material, Profile, Profile_beam, STEEL, AS3501
from .trail_function import Basis, BasisND, add_basis, trigonometric_basis, \
    polynominal_basis,  \
    legendre_basis
from .geometry import Curve, Shell_of_revolution_geometry
from .quadrature import quad_gauss, quad_linear
from .misc import get_part, round_array, Symbol_generator
#from quadrature import _quadrature_helper


EPS = 1e-4


class ShellOfRevolution():
    def __init__(self, curve, x1, x2, M, N, profile,
                 nquads=[None] * 2, initialize=False):
        self._x1 = x1
        if np.isscalar(x2) and abs(x2-np.pi*2) <= EPS:
            self._x2 = 0, np.pi * 2
            self._full_cycle = True
        else:
            self._x2 = x2
            self._full_cycle = False
        self._geometry = Shell_of_revolution_geometry(curve, self._x1,
                                                        self._x2)
        self._M = M
        self._N = N
        self._MN = self._M * self._N
        self._profile = profile

        # setup basis functions
        # self._basis_x1 = [\varphi(x1),d\varphi(x1)/{dx1}]
        # each number of basis functions = M
        self._basis_x1 = [None] * 2
        # self._basis_x2 = [\varphi(x2),d\varphi(x2)/{dx2}]
        # each number of basis functions = N
        self._basis_x2 = [None] * 2
        # self._basis = [\Varphi, d\Varphi/{dx1}, d\Varphi/{dx2}],
        # each number of basis functions = MN x MN
        self._basis = [None] * 3
        self.set_basis()

        # setup self._quad_x1 and self._quad_x2
        # number of quadrature points is the same as number of
        # basis functions by default
        self._quad_x1 = None
        self._quad_x2 = None
        self.set_quadrature([quad_gauss, quad_linear], nquads)

        # these arrays are used for performance
        self._basis_x1_values = [None, None]
        self._basis_x2_values = [None, None]

        # quadrature of basis functions over the whole plate
        # note that actually, self._integrated_basis can be a sparse matrix
#        self._integrated_basis = np.zeros([self._MN * 3] * 2)
        # quadrature of basis functions along single direction
#        self._integrated_basis_x1 = np.zeros([self._M] * 2)
        self._integrated_basis_x2 = np.zeros([self._N*2] * 2)

        # stiffness matrix K and mass matrix M
        self._stiffness_matrix = lil_matrix(tuple([self._MN * 5] * 2))
        self._mass_matrix = lil_matrix(tuple([self._MN * 5] * 2))
        self._stiffness_matrix_dense = None
        self._mass_matrix_dense = None

        # boundary conditons
        # [(direction, x1, k), ...]
        self._boundary_condition_x2_constant = []
        # [(direction, x2, k), ...]
        self._boundary_condition_x1_constant = []

        # elastic foundation
        # [(direction, k)]
        self._elastic_foundation = []
        
        # springs
        # [(direction, x1, x2, k), ...]
        self._spring = []

        # stringers
        # stringers along the x1 axis
        # [(x2, profile, range_of_x1), ...]
        self._stringer_x1 = []
        # stringers along the x2 axis
        # (x1, profile, range_of_x2), ...
        self._stringer_x2 = []

        # loads
        self._force_vector = np.zeros([self._MN * 5])
        # [(direction, x1, x2, force, h), ...]
        self._force = []
        # [(direction, p or fp, h), ...]
        self._pressure_constant = []
        self._pressure_function = []

        self._is_initialized = False
        if initialize:
            self._do_initialize()

    @property
    def geometry(self):
        return self._geometry

    def initialize(self):
        if self._is_initialized:
            return
        self._do_initialize()

    def _do_initialize(self):
        self._init_basis()

        self._init_integrated_basis()

        self._init_stiffness_matrix()
        self._init_mass_matrix()

        self._init_force_vector()

        self._is_initialized = True

    def uninitialize(self):
        self._is_initialized = False
        self._stiffness_matrix_dense = None
        self._mass_matrix_dense = None

    def set_quadrature(self, quad_objs=[quad_gauss, quad_linear], nquads=[None] * 2):
        if nquads[0] is None:
            nquads[0] = self._M
        if nquads[1] is None:
            nquads[1] = self._N
        quad_x1 = quad_objs[0](nquads[0]).set_limit(*self._x1)
        quad_x2 = quad_objs[1](nquads[1]).set_limit(*self._x2)
        self.uninitialize()
        self._quad_x1 = quad_x1
        self._quad_x2 = quad_x2

    def set_basis(self, bx1=None, bx2=None):
        if bx1 is None:
            bx1 = trigonometric_basis(self._M, self._x1)
        if bx2 is None:
            if self._full_cycle:
                bx2 = trigonometric_basis(self._N, [0, np.pi])
            else:
                bx2 = trigonometric_basis(self._N, self._x2)
        x1 = Basis(bx1.symbolic_functions[:self._M], name='x1')
        x2 = Basis(bx2.symbolic_functions[:self._N], name='x2')
        g = Symbol_generator('x', 1)
        x1.symbol = g.generate()
        x2.symbol = g.generate()
        dx1 = x1.diff()
        dx2 = x2.diff()
        if not (len(x1) == self._M and len(x2) == self._N):
            raise ValueError('Number of basis given is less than defined')
        self.uninitialize()
        self._basis_x1 = [x1, dx1]
        self._basis_x2 = [x2, dx2]

    def add_boundary_condition_x1(self, bc, x2, direction=2):
        "Location \in interval(self._x2)"
        if np.iterable(x2):
            for l in x2:
                self.add_boundary_condition_x1(bc, l, direction)
            return
        if np.iterable(direction):
            for d in direction:
                self.add_boundary_condition_x1(bc, x2, d)
            return

        return self._add_boundary_condition_x1_constant(bc, x2, direction)
        
    def add_boundary_condition_x2(self, bc, x1, direction=2):
        "Location \in interval(self._x1)"
        if np.iterable(x1):
            for l in x1:
                self.add_boundary_condition_x2(bc, l, direction)
            return
        if np.iterable(direction):
            for d in direction:
                self.add_boundary_condition_x2(bc, x1, d)
            return

        return self._add_boundary_condition_x2_constant(bc, x1, direction)

    def add_elastic_foundation(self, k, direction):
        if np.iterable(direction):
            for d in direction:
                self.add_elastic_foundation(k, d)
            return
        return self._add_elastic_foundation(k, direction)
    
    def add_spring(self, k, x1, x2, direction=2):
        "Location \in interval(self._x1)"
        if np.iterable(x1):
            if np.iterable(x2):
                if len(x1) != len(x2):
                    raise ValueError('shape of x1 and x2 do not match')
                for i in range(len(x1)):
                    self.add_boundary_condition(k, x1[i], x2[i], direction)
                return
            else:
                for i in range(len(x1)):
                    self.add_boundary_condition(k, x1[i], x2, direction)
                return
        else:
            if np.iterable(x2):
                for i in range(len(x2)):
                    self.add_boundary_condition(k, x1, x2[i], direction)
                return

        if np.iterable(direction):
            for d in direction:
                self.add_boundary_condition(k, x1, x2, d)
            return

        return self._add_spring(k, x1, x2, direction)

    def add_stringer(self, profile, x1=None, x2=None, range_x=None):
        if np.iterable(x1):
            for i in x1:
                self.add_stringer(profile, i, x2, range_x)
            return
        if np.iterable(x2):
            for i in x2:
                self.add_stringer(profile, x1, i, range_x)
            return
        if x1 is not None:
            self._add_stringer_along_x2(profile, x1=x1, range_x2=range_x)
        if x2 is not None:
            self._add_stringer_along_x1(profile, x2=x2, range_x1=range_x)

    def add_force(self, f, x1, x2, direction=2, h=0):
        self.uninitialize()
        self._force.append((direction, x1, x2, f, h))

    def add_pressure(self, p, direction=2, h=0):
        self.uninitialize()
        if callable(p):
            self._pressure_function.append((direction, p, h))
        else:
            self._pressure_constant.append((direction, p, h))

    def clear_spring(self):
        self.uninitialize()
        self._spring = []

    def clear_boundary_condition(self):
        self.uninitialize()
        self._boundary_condition_x1_constant = []
        self._boundary_condition_x2_constant = []
        self._boundary_condition_function = []

    def clear_elastic_foundation(self):
        self.uninitialize()
        self._elastic_foundation = []

    def clear_stringer(self):
        self.uninitialize()
        self._stringer_x1 = []
        self._stringer_x2 = []

    def clear_force(self):
        self.uninitialize()
        self._force = []

    def clear_pressure(self):
        self.uninitialize()
        self._pressure_constant = []
        self._pressure_function = []

    def _add_boundary_condition_x1_constant(self, k, x2, direction):
        self.uninitialize()
        self._boundary_condition_x1_constant.append((direction, x2, k))
        
    def _add_boundary_condition_x2_constant(self, k, x1, direction):
        self.uninitialize()
        self._boundary_condition_x2_constant.append((direction, x1, k))

    def _add_elastic_foundation(self, k, direction):
        self.uninitialize()
        self._elastic_foundation.append((direction, k))

    def _add_spring(self, k, x1, x2, direction):
        self.uninitialize()
        self._spring.append((direction, x1, x2, k))

    def _add_stringer_along_x1(self, profile, x2, range_x1):
        self.uninitialize()
        self._stringer_x1.append((x2, profile, range_x1))

    def _add_stringer_along_x2(self, profile, x1, range_x2):
        self.uninitialize()
        self._stringer_x2.append((x1, profile, range_x2))

    def _init_basis(self):
        bx1, bdx1 = self._basis_x1
        bx2, bdx2 = self._basis_x2
        self._basis[0] = BasisND(bx1, bx2, name='x1,x2')
        self._basis[1] = BasisND(bdx1, bx2, name='dx1,x2')
        self._basis[2] = BasisND(bx1, bdx2, name='x1,dx2')

        # calculate the values of the basis functions at the quadrature points
        x1 = self._quad_x1.x
        self._basis_x1_values[0] = np.zeros([self._M, len(x1)])
        self._basis_x1_values[1] = np.zeros([self._M, len(x1)])
        for i in range(self._M):
            self._basis_x1_values[0][i] = bx1[i](x1)
            self._basis_x1_values[1][i] = bdx1[i](x1)
        x2 = self._quad_x2.x
        self._basis_x2_values[0] = np.zeros([self._N, len(x2)])
        self._basis_x2_values[1] = np.zeros([self._N, len(x2)])
        for i in range(self._N):
            self._basis_x2_values[0][i] = bx2[i](x2)
            self._basis_x2_values[1][i] = bdx2[i](x2)

    def _init_integrated_basis(self):
        N = self._N

        res = self._get_integrated_v2v2(0,0)
        round_array(res)
        self._integrated_basis_x2[:N, :N] = res
        res = self._get_integrated_v2v2(0,1)
        round_array(res)
        self._integrated_basis_x2[:N, N:] = res
        self._integrated_basis_x2[N:, :N] = res.T
        res = self._get_integrated_v2v2(1,1)
        round_array(res)
        self._integrated_basis_x2[N:, N:] = res

    def _get_R(self, x1):
        a1 = self._geometry.a1(x1)
        a2 = self._geometry.a2(x1)
        k1 = self._geometry.k1(x1)
        k2 = self._geometry.k2(x1)
        T = self._geometry.T(x1)
        self._profile.set_curvature(k1, k2)
        ABD = np.zeros([10, 10])
        A, B, D = self._profile.A(), self._profile.B(), self._profile.D()
        As = self._profile.As()
        ABD[:4, :4] = A
        ABD[4:8, :4] = B
        ABD[:4, 4:8] = B
        ABD[4:8, 4:8] = D
        ABD[8:, 8:] = As * self._profile.K
        R = a1 * a2 * T.T.dot(ABD).dot(T)
        return R

    def _get_R_stringer_x1(self, x1, profile_beam):
        a1 = self._geometry.a1(x1)
        kn = self._geometry.kn1(x1)
        kg = self._geometry.kg1(x1)
        T = self._geometry.T_stringer_x1(x1)
        profile_beam.set_curvature(kn,kg)
        A = profile_beam.A()
        R = T.T.dot(a1*A).dot(T)
        return R

    def _get_R_stringer_x2(self, x1, profile_beam):
        a2 = self._geometry.a2(x1)
        kn = self._geometry.kn2(x1)
        kg = self._geometry.kg2(x1)
        T = self._geometry.T_stringer_x2(x1)
        profile_beam.set_curvature(kn,kg)
        A = profile_beam.A()
        R = T.T.dot(a2*A).dot(T)
        return R

    def _init_stiffness_matrix(self):
        MN = self._MN
        K = np.zeros([5 * MN, 5 * MN])
#        K = lil_matrix(tuple([5*MN]*2))
        self._init_stiffness_matrix_strain_energy(K)
        self._init_stiffness_matrix_boundary_condition_x1(K)
        self._init_stiffness_matrix_boundary_condition_x2(K)
        self._init_stiffness_matrix_elastic_foundation(K)
        self._init_stiffness_matrix_spring(K)
        self._init_stiffness_matrix_stringer_x1(K)
        self._init_stiffness_matrix_stringer_x2(K)
        self._stiffness_matrix = csc_matrix(K)

    def _init_stiffness_matrix_strain_energy(self, K):
        Rs = np.zeros([15,15,len(self._quad_x1.x)])
        for i,x in enumerate(self._quad_x1.x):
            Rs[:,:,i] = self._get_R(x)

        for i in range(5):
            for j in range(i+1):
                block = np.zeros([self._MN, self._MN])
                for ii in range(3):
                    for jj in range(3):
                        R = Rs[i+5*ii,j+5*jj]
                        block += self._init_stiffness_matrix_strain_energy_helper(ii,jj,R)
                round_array(block)
                self._block_add(K,i,j, block)
                if i != j:
                    self._block_add(K,j,i, block.T)

    def _init_stiffness_matrix_strain_energy_helper(self, ii, jj, R):
        MN = self._MN
        N = self._N
        vv1 = self._get_integrated_v1v1(ii==1,jj==1,weights=R)
        vv2 = self._integrated_basis_x2[(ii==2)*N:((ii==2)+1)*N,
                                        (jj==2)*N:((jj==2)+1)*N]

#        VV = np.zeros([MN,MN])
#        for i in range(MN):
#            i1 = i // N
#            i2 = i % N
#            for j in range(MN):
#                j1 = j // N
#                j2 = j % N
#                # Very slow code here due to large numbers of loops
#                VV[i,j] = vv1[i1,j1]*vv2[i2,j2]
#        return VV

        i = np.arange(MN)
        i1 = i // N
        i2 = i % N
        return vv1[i1][:,i1] * vv2[i2][:,i2]

    def _init_stiffness_matrix_boundary_condition_x1(self, K):
        if not self._boundary_condition_x1_constant:
            return
        MN = self._MN
        N = self._N
        quad_points_x1 = self._quad_x1.x
        a1 = np.zeros([len(quad_points_x1)])
        for i, x in enumerate(quad_points_x1):
            a1[i] = self._geometry.a1(x)
        ib1 = self._get_integrated_v1v1(0, 0, a1)
        for bc in self._boundary_condition_x1_constant:
            direction, x2, k = bc
            b2 = self._basis_x2[0](x2)
            i = np.arange(MN)
            i1 = i // N
            i2 = i % N
            VV = b2[i2] * ib1[i1][:,i1]
            VV = b2[i2] * VV.T
            VV *= k
            self._block_add(K, direction, direction, VV)
    
    def _init_stiffness_matrix_boundary_condition_x2(self, K):
        MN = self._MN
        N = self._N
        ib2 = self._integrated_basis_x2
        for bc in self._boundary_condition_x2_constant:
            direction, x1, k = bc
            b1 = self._basis_x1[0](x1)
#            # slow code
#            VV = np.zeros([MN, MN])
#            for i in range(MN):
#                i1 = i // N
#                i2 = i % N
#                for j in range(i+1):
#                    j1 = j // N
#                    j2 = j % N
#                    VV[i, j] = b1[i1] * b1[j1] * ib2[i2, j2]
#                    VV[j,i] = VV[i,j]
            # profiled code
            i = np.arange(MN)
            i1 = i // N
            i2 = i % N
            VV = b1[i1]*ib2[i2][:,i2]
            VV = b1[i1] * VV.T

            a2 = self._geometry.a2(x1)
            VV *= (a2*k)
            self._block_add(K, direction, direction, VV)

    def _init_stiffness_matrix_elastic_foundation(self, K):
        MN = self._MN
        N = self._N
        quad_points_x1 = self._quad_x1.x
        weights = np.zeros([len(quad_points_x1)])
        for i, x in enumerate(quad_points_x1):
            a1 = self._geometry.a1(x)
            a2 = self._geometry.a2(x)
            weights[i] = a1 * a2
        b1 = self._get_integrated_v1v1(weights=weights)
        round_array(b1)
        b2 = self._integrated_basis_x2[:N, :N]
        i = np.arange(MN)
        i1 = i // N
        i2 = i % N
        VV = b1[i1][:,i1] * b2[i2][:,i2]
        for direction, k in self._elastic_foundation:
            self._block_add(K, direction, direction, k * VV)

    def _init_stiffness_matrix_spring(self, K):
        MN = self._MN
        N = self._N
        for spring in self._spring:
            direction, x1, x2, k = spring
            b1 = self._basis_x1[0](x1)
            b2 = self._basis_x2[0](x2)
#            # slow code
#            VV = np.zeros([MN, MN])
#            for i in range(MN):
#                i1 = i // N
#                i2 = i % N
#                for j in range(i+1):
#                    j1 = j // N
#                    j2 = j % N
#                    VV[i,j] = b1[i1]*b2[i2]*b1[j1]*b2[j2]
#                    VV[j,i] = VV[i,j]
            # profiled code
            B1, B2 = np.meshgrid(b1,b2)
            i = np.arange(MN)
            i1 = i // N
            i2 = i % N
            VV = (B1[i1][:,i1]*B1.T[i1][:,i1]) * (B2[i2][:,i2]*B2.T[i2][:,i2])

            VV *= k
            self._block_add(K,direction,direction,VV)

    def _init_stiffness_matrix_stringer_x1(self,K):
        for x2, profile, range_x1 in self._stringer_x1:
            if range_x1 is None:
                quad = self._quad_x1
                get_integrated = self._get_integrated_v1v1
            else:
                quad = quad_gauss(len(self._quad_x1.x)).set_limit(*range_x1)
                get_integrated = lambda *args,**kwargs: \
                    self._get_integrated_v1v1_self_defined(quad,*args,**kwargs)

            v2 = self._basis_x2[0](x2)
            dv2 = self._basis_x2[1](x2)
            i = np.arange(self._MN)
            i1 = i // self._N
            i2 = i % self._N

            Rs = np.zeros([20,20,len(quad.x)])
            for i,x in enumerate(quad.x):
                Rs[:,:,i] = self._get_R_stringer_x1(x, profile)
            for i in range(5):
                for j in range(i+1):
                    block = np.zeros([self._MN,self._MN])
                    for  ii in range(4):
                        for jj in range(4):
                            R = Rs[i+5*ii,j+5*jj]
                            vv1 = get_integrated(ii%2,jj%2,weights=R)
                            if ii//2 == 0:
                                vv1 = (vv1[i1][:,i1].T * v2[i2]).T
                            else:
                                vv1 = (vv1[i1][:,i1].T * dv2[i2]).T
                            if jj // 2 == 0:
                                block += vv1 *v2[i2]
                            else:
                                block += vv1 *dv2[i2]
                    round_array(block)
                    self._block_add(K,i,j, block)
                    if i != j:
                        self._block_add(K,j,i, block.T)

    def _init_stiffness_matrix_stringer_x2(self,K):
        for x1, profile, range_x2 in self._stringer_x2:
            if range_x2 is None:
                quad = self._quad_x2
                get_integrated = self._get_integrated_v2v2
            else:
                quad = quad_gauss(len(self._quad_x2.x)).set_limit(*range_x2)
                get_integrated = lambda *args,**kwargs: \
                    self._get_integrated_v2v2_self_defined(quad,*args,**kwargs)

            v1 = self._basis_x1[0](x1)
            dv1 = self._basis_x1[1](x1)
            i = np.arange(self._MN)
            i1 = i // self._N
            i2 = i % self._N

            R = self._get_R_stringer_x2(x1, profile)

            for i in range(5):
                for j in range(i+1):
                    block = np.zeros([self._MN,self._MN])
                    for  ii in range(4):
                        for jj in range(4):
                            vv2 = get_integrated(ii//2,jj//2) * R[i+5*ii,j+5*jj]
                            if ii%2 == 0:
                                vv2 = (vv2[i2][:,i2].T * v1[i1]).T
                            else:
                                vv2 = (vv2[i2][:,i2].T * dv1[i1]).T
                            if jj % 2 == 0:
                                block += vv2 *v1[i1]
                            else:
                                block += vv2 *dv1[i1]
                    round_array(block)
                    self._block_add(K,i,j, block)
                    if i != j:
                        self._block_add(K,j,i, block.T)

    def _get_integrated_v1v1(self, basis_i=0, basis_j=0, weights=None):
        assert weights is not None
        M = self._M
        result = np.zeros([M, M])
        bi = self._basis_x1_values[basis_i]
        bj = self._basis_x1_values[basis_j]
        if basis_i == basis_j:
            for i in range(M):
                vi = bi[i]
                for j in range(i+1):
                    vj = bj[j]
                    resij = self._quad_x1(weights=vi*vj*weights)
                    result[i, j] = resij
                    if i != j:
                        result[j, i] = resij
        else:
            for i in range(M):
                vi = bi[i]
                for j in range(M):
                    vj = bj[j]
                    resij = self._quad_x1(weights=vi*vj*weights)
                    result[i, j] = resij
        return result

    def _get_integrated_v1v1_self_defined(self, quad, basis_i=0, basis_j=0,
                                          weights=None):
        M = self._M
        result = np.zeros([M, M])
        bi = self._basis_x1[basis_i]
        bj = self._basis_x1[basis_j]
        if basis_i == basis_j:
            bj = bi
            for i in range(M):
                vi = bi[i](quad.x)
                for j in range(i+1):
                    vj = bj[j](quad.x)
                    resij = quad(weights=vi*vj*weights)
                    result[i, j] = resij
                    if i != j:
                        result[j, i] = resij
        else:
            for i in range(M):
                vi = bi[i](quad.x)
                for j in range(M):
                    vj = bj[j](quad.x)
                    resij = quad(weights=vi*vj*weights)
                    result[i, j] = resij
        return result

    def _get_integrated_v2v2(self, basis_i=0, basis_j=0):
        N = self._N
        result = np.zeros([N, N])
        bi = self._basis_x2_values[basis_i]
        bj = self._basis_x2_values[basis_j]
        if basis_i == basis_j:
            for i in range(N):
                vi = bi[i]
                for j in range(i+1):
                    vj = bj[j]
                    resij = self._quad_x2(weights=vi*vj)
                    result[i, j] = resij
                    if i != j:
                        result[j, i] = resij
        else:
            for i in range(N):
                vi = bi[i]
                for j in range(N):
                    vj = bj[j]
                    resij = self._quad_x2(weights=vi*vj)
                    result[i, j] = resij
        return result

    def _get_integrated_v2v2_self_defined(self, quad, basis_i=0, basis_j=0):
        N = self._N
        result = np.zeros([N, N])
        bi = self._basis_x2[basis_i]
        bj = self._basis_x2[basis_j]
        if basis_i == basis_j:
            for i in range(N):
                vi = bi[i](quad.x)
                for j in range(i+1):
                    vj = bj[j](quad.x)
                    resij = quad(weights=vi*vj)
                    result[i, j] = resij
                    if i != j:
                        result[j, i] = resij
        else:
            for i in range(N):
                vi = bi[i](quad.x)
                for j in range(N):
                    vj = bj[j](quad.x)
                    resij = quad(weights=vi*vj)
                    result[i, j] = resij
        return result

    def _block_access(self, matrix, i, j):
        MN = self._MN
        i *= MN
        j *= MN
        return matrix[i: i + MN, j: j + MN]

    def _block_add(self, matrix, i, j, val):
        MN = self._MN
        i *= MN
        j *= MN
        matrix[i: i + MN, j: j + MN] += val

    def _init_mass_matrix(self):
        #        M = lil_matrix(tuple([self._MN * 5] * 2))
        M = np.zeros([self._MN * 5] * 2)
        self._init_mass_matrix_shell(M)
        self._init_mass_matrix_stringer_x1(M)
        self._init_mass_matrix_stringer_x2(M)
        #        self._mass_matrix = M.tocsc()
        self._mass_matrix = csc_matrix(M)

    def _init_mass_matrix_shell(self, M):
        quad_points_x1 = self._quad_x1.x
        weights = np.zeros([3, len(quad_points_x1)])
        for i, x in enumerate(quad_points_x1):
            a1 = self._geometry.a1(x)
            a2 = self._geometry.a2(x)
            k1 = self._geometry.k1(x)
            k2 = self._geometry.k2(x)
            self._profile.set_curvature(k1, k2)
            I = self._profile.I()
            weights[:, i] = a1 * a2 * I
        sigma0_part_1 = self._get_integrated_v1v1(weights=weights[0])
        sigma1_part_1 = self._get_integrated_v1v1(weights=weights[1])
        sigma2_part_1 = self._get_integrated_v1v1(weights=weights[2])
        round_array(sigma0_part_1)
        round_array(sigma1_part_1)
        round_array(sigma2_part_1)

        sigma_part_2 = self._integrated_basis_x2[:self._N,:self._N]

        i = np.arange(self._MN)
        i1 = i // self._N
        i2 = i % self._N
        sigma0 = sigma0_part_1[i1][:,i1] * sigma_part_2[i2][:,i2]
        sigma1 = sigma1_part_1[i1][:,i1] * sigma_part_2[i2][:,i2]
        sigma2 = sigma2_part_1[i1][:,i1] * sigma_part_2[i2][:,i2]

        add = self._block_add
        add(M, 0, 0, sigma0)
        add(M, 1, 1, sigma0)
        add(M, 2, 2, sigma0)
        add(M, 0, 3, sigma1)
        add(M, 1, 4, sigma1)
        add(M, 3, 0, sigma1)
        add(M, 4, 1, sigma1)
        add(M, 3, 3, sigma2)
        add(M, 4, 4, sigma2)

    def _init_mass_matrix_stringer_x1(self, M):
        for x2, profile, range_x1 in self._stringer_x1:
            if range_x1 is None:
                quad = self._quad_x1
                get_integrated = self._get_integrated_v1v1
            else:
                quad = quad_gauss(len(self._quad_x1.x)).set_limit(*range_x1)
                get_integrated = lambda *args,**kwargs: \
                    self._get_integrated_v1v1_self_defined(quad,*args,**kwargs)

            weight = np.zeros([len(quad.x)])
            weight2 = np.zeros([len(quad.x)])
            weight3 = np.zeros([len(quad.x)])
            weight22 = np.zeros([len(quad.x)])
            weight23 = np.zeros([len(quad.x)])
            weight33 = np.zeros([len(quad.x)])
            weight3_bar = np.zeros([len(quad.x)])
            weight22_bar = np.zeros([len(quad.x)])
            weight23_bar = np.zeros([len(quad.x)])
            for i, x in enumerate(quad.x):
                a1 = self._geometry.a1(x)
                a2 = self._geometry.a2(x)
                kn = self._geometry.kn1(x)
                kg = self._geometry.kg1(x)
                profile.set_curvature(kn,kg)
                I = profile.I(0,0)
                I2 = profile.I(1,0)
                I3 = profile.I(0,1)
                I22 = profile.I(2,0)
                I23 = profile.I(1,1)
                I33 = profile.I(0,2)
                weight[i] = a1*I
                weight2[i] = a1*I2
                weight3[i] = a1*I3
                weight22[i] = a1*I22
                weight23[i] = a1*I23
                weight33[i] = a1*I33
                weight3_bar[i] = I3 * a1 / a2
                weight22_bar[i] = I22 * a1 / a2**2
                weight23_bar[i] = I23 * a1 / a2
            sigma_part_1 = get_integrated(weights=weight)
            sigma2_part_1 = get_integrated(weights=weight2)
            sigma3_part_1 = get_integrated(weights=weight3)
            sigma22_part_1 = get_integrated(weights=weight22)
            sigma23_part_1 = get_integrated(weights=weight23)
            sigma33_part_1 = get_integrated(weights=weight33)
            sigma3_bar_part_1 = get_integrated(weights=weight3_bar)
            sigma22_bar_part_1 = get_integrated(weights=weight22_bar)
            sigma23_bar_part_1 = get_integrated(weights=weight23_bar)

            round_array(sigma_part_1)
            round_array(sigma2_part_1)
            round_array(sigma3_part_1)
            round_array(sigma22_part_1)
            round_array(sigma23_part_1)
            round_array(sigma33_part_1)
            round_array(sigma3_bar_part_1)
            round_array(sigma22_bar_part_1)
            round_array(sigma23_bar_part_1)

            x2_values = self._basis_x2[0](x2)
            dx2_values = self._basis_x2[1](x2)

            i = np.arange(self._MN)
            i1 = i // self._N
            i2 = i % self._N

            sigma = sigma_part_1[i1][:,i1].T * x2_values[i2]
            sigma = sigma.T *  x2_values[i2]
            sigma2 = sigma2_part_1[i1][:,i1].T * x2_values[i2]
            sigma2 = sigma2.T *  x2_values[i2]
            sigma3 = sigma3_part_1[i1][:,i1].T * x2_values[i2]
            sigma3 = sigma3.T *  x2_values[i2]
            sigma22 = sigma22_part_1[i1][:,i1].T * x2_values[i2]
            sigma22 = sigma22.T *  x2_values[i2]
#            sigma23 = sigma23_part_1[i1][:,i1].T * x2_values[i2]
#            sigma23 = sigma23.T *  x2_values[i2]
            sigma33 = sigma33_part_1[i1][:,i1].T * x2_values[i2]
            sigma33 = sigma33.T *  x2_values[i2]
            sigma3_bar = sigma3_bar_part_1[i1][:,i1].T * x2_values[i2]
            sigma3_bar = sigma3_bar.T *  dx2_values[i2]
            sigma22_bar = sigma22_bar_part_1[i1][:,i1].T * dx2_values[i2]
            sigma22_bar = sigma22_bar.T *  dx2_values[i2]
            sigma23_bar = sigma23_bar_part_1[i1][:,i1].T * x2_values[i2]
            sigma23_bar = sigma23_bar.T *  dx2_values[i2]

            add = self._block_add
            add(M, 0, 0, sigma+sigma3_bar+sigma3_bar.T+sigma22_bar)
            add(M, 0, 3, sigma3+sigma23_bar.T)
            add(M, 1, 1, sigma)
            add(M, 1, 4, sigma3)
            add(M, 2, 2, sigma)
            add(M, 2, 4, -sigma2)
            add(M, 3, 0, sigma3+sigma23_bar)
            add(M, 3, 3, sigma33)
            add(M, 4, 1, sigma3)
            add(M, 4, 2, -sigma2)
            add(M, 4, 4, sigma33+sigma22)

    def _init_mass_matrix_stringer_x2(self, M):
        for x1, profile, range_x2 in self._stringer_x2:
            if range_x2 is None:
                quad = self._quad_x2
                sigma_part_2 = self._integrated_basis_x2[:self._MN,:self._MN]
            else:
                quad = quad_gauss(len(self._quad_x2.x)).set_limit(*range_x2)
                sigma_part_2 = self._get_integrated_v2v2_self_defined(quad)
                round_array(sigma_part_2)

            a1 = self._geometry.a1(x1)
            a2 = self._geometry.a2(x1)
            kn = self._geometry.kn2(x1)
            kg = self._geometry.kg2(x1)
            profile.set_curvature(kn,kg)
            I = profile.I(0,0)
            I2 = profile.I(1,0)
            I3 = profile.I(0,1)
            I22 = profile.I(2,0)
            I23 = profile.I(1,1)
            I33 = profile.I(0,2)

            x1_values = self._basis_x1[0](x1)
            dx1_values = self._basis_x1[1](x1)

            i = np.arange(self._MN)
            i1 = i // self._N
            i2 = i % self._N

            sigma_base = x1_values[i1] * sigma_part_2[i2][:,i2].T
            sigma_base = sigma_base.T *  x1_values[i1]
            sigma_base1 = x1_values[i1] * sigma_part_2[i2][:,i2].T
            sigma_base1 = sigma_base1.T *  dx1_values[i1]
            sigma_base11 = dx1_values[i1] * sigma_part_2[i2][:,i2].T
            sigma_base11 = sigma_base11.T *  dx1_values[i1]
            sigma = sigma_base * I * a2
            sigma2 = sigma_base * I2 * a2
            sigma3 = sigma_base * I3 * a2
            sigma22 = sigma_base * I22 * a2
#            sigma23 = sigma_base * I23 * a2
            sigma33 = sigma_base * I33 * a2
            sigma2_bar = sigma_base1 * I2 * a2/a1
            sigma22_bar = sigma_base11 * I22 * a2/a1**2
            sigma23_bar = sigma_base1 * I23 * a2/a1

            add = self._block_add
            add(M, 0, 0, sigma)
            add(M, 0, 3, sigma3)
            add(M, 1, 1, sigma-sigma2_bar-sigma2_bar.T+sigma22_bar)
            add(M, 1, 4, sigma3-sigma23_bar.T)
            add(M, 2, 2, sigma)
            add(M, 2, 3, sigma2)
            add(M, 3, 0, sigma3)
            add(M, 3, 2, sigma2)
            add(M, 3, 3, sigma33+sigma22)
            add(M, 4, 1, sigma3-sigma23_bar)
            add(M, 4, 4, sigma33)

    def _init_force_vector(self):
        self._force_vector = np.zeros([self._MN * 5])
        self._init_force_vector_concentrated_force()
        self._init_force_vector_pressure_constant()
        self._init_force_vector_pressure_function()

    def _init_force_vector_concentrated_force(self):
        MN = self._MN
        basis = self._basis[0]
        for i, f in enumerate(self._force):
            direction, x1, x2, force, h = f
            k1 = self._geometry.k1(x1,x2)
            k2 = self._geometry.k2(x1,x2)
            coeff = (1-k1*h/2)*(1-k2*h/2)*basis(x1,x2).reshape(-1)
            self._force_vector[direction*MN:direction*MN+MN] += coeff * force

    def _init_force_vector_pressure_constant(self):
        MN = self._MN
        N = self._N
        v2 = np.array([self._quad_x2(weights=bv) for bv in self._basis_x2_values[0]])
        for i, f in enumerate(self._pressure_constant):
            direction, p, h = f
            coeff = np.zeros(len(self._quad_x1.x))
            for i,x in enumerate(self._quad_x1.x):
                k1 = self._geometry.k1(x)
                k2 = self._geometry.k2(x)
                a1 = self._geometry.a1(x)
                a2 = self._geometry.a2(x)
                coeff[i] = (1-k1*h/2)*(1-k2*h/2)*a1*a2
            v1 = np.array([self._quad_x1(weights=coeff*bv)
                           for bv in self._basis_x1_values[0]])
            i = np.arange(MN)
            i1 = i // N
            i2 = i % N
            self._force_vector[direction*MN:direction*MN+MN] += v1[i1]*v2[i2]*p

    def _init_force_vector_pressure_function(self):
        MN = self._MN
        N = self._N
        b1v = self._basis_x1_values[0]
        b2v = self._basis_x2_values[0]

        for i, f in enumerate(self._pressure_function):
            direction, fp, h = f
            coeff = np.zeros(len(self._quad_x1.x))
            for i, x1 in enumerate(self._quad_x1.x):
                k1 = self._geometry.k1(x1)
                k2 = self._geometry.k2(x1)
                a1 = self._geometry.a1(x1)
                a2 = self._geometry.a2(x1)
                coeff[i] = (1-k1*h/2)*(1-k2*h/2)*a1*a2
            for i in range(MN):
                i1 = i // N
                i2 = i % N
                # res = self._quad_x1(lambda x1: self._quad_x2(lambda x2:fp(x1,x2)*b2v[i2])*b1v[i1], weights=coeff)
                res = self._quad_x1(lambda x1: self._quad_x2
                                    (lambda x2: fp(*np.meshgrid(x1, x2)).T *
                                     b2v[i2]), weights=coeff * b1v[i1])
                self._force_vector[direction*MN+i] = res


    @property
    def M(self):
        self.initialize()
        return self._mass_matrix

    @property
    def K(self):
        self.initialize()
        return self._stiffness_matrix

    @property
    def F(self):
        self.initialize()
        return self._force_vector

    @property
    def M_dense(self):
        if self._mass_matrix_dense is not None:
            return self._mass_matrix_dense
        self._mass_matrix_dense = self.M.toarray()
        return self._mass_matrix_dense

    @property
    def K_dense(self):
        if self._stiffness_matrix_dense is not None:
            return self._stiffness_matrix_dense
        self._stiffness_matrix_dense = self.K.toarray()
        return self._stiffness_matrix_dense

    def frequency_dense(self, n=20):
        freq = eigvals(self.K_dense, self.M_dense)
        freq = np.sqrt(freq).real / 2 / np.pi
        freq = sorted(freq)
        return np.asarray(freq[:n])

    def frequency(self, n=20, frequency_shift=0.1, part=None, auto_filter=0.0,
                  ncv=None, maxiter=None):
        # auto_filter可以过滤虚部过大的特征值，当虚部比实部大于auto_filter时，即滤去它
        freq = slinalg.eigs(self.K, n, self.M, frequency_shift,
                            v0=np.ones(self._MN * 5), ncv=ncv, maxiter=maxiter,
                            return_eigenvectors=False)
        freq = np.sqrt(freq) / 2 / np.pi
        freq = sorted(freq, key=lambda x: np.real(x))
        if auto_filter is not None:
            freq_filtered = []
            for f in freq:
                if abs(np.real(f) * auto_filter) >= abs(np.imag(f)):
                    freq_filtered.append(f)
        else:
            freq_filtered = freq
        return np.asarray(get_part(freq_filtered, part))

    def base_state(self):
        freq = None
        shape = np.zeros([self._MN * 5])
        if not self._is_initialized:
            self._init_basis()
        return Frame(self._geometry, self._basis[0].functions, freq, shape)

    def modal(self, n=20, frequency_shift=0.1, part=None, auto_filter=0.0,
              ncv=None, maxiter=None):
        freq, mode_shape = slinalg.eigs(self.K, n, self.M, frequency_shift,
                                        v0=np.ones(self._MN * 5), ncv=ncv,
                                        maxiter=maxiter)
        freq = np.sqrt(freq) / 2 / np.pi
        index_array = np.argsort(freq.real)
        step = Step()
        for idx in index_array:
            f = freq[idx]
            x = mode_shape.T[idx]
            if auto_filter is not None:
                if abs(np.real(f) * auto_filter) < abs(np.imag(f)):
                    continue
            step.append(Frame(self._geometry, self._basis[0].functions,
                              get_part(f, part), get_part(x, part)))
        return step

    def spsolve(self, freq, part=None):
        w = freq * 2 * np.pi
        A = -w ** 2 * self.M + self.K
        F = self.F
        x = slinalg.spsolve(A, F)
        return Frame(self._geometry, self._basis[0].functions,
                     get_part(freq, part), get_part(x, part))

    def solve(self, freq, part=None):
        # I don't know why it runs faster
        w = freq * 2 * np.pi
        A = -w ** 2 * self.M + self.K
        F = self.F
        x = linalg.solve(A.toarray(), F)
        return Frame(self._geometry, self._basis[0].functions,
                     get_part(freq, part), get_part(x, part))

    def frequency_response(self, freq_range, part=None):
        step = Step()
        for f in freq_range:
            frame = self.solve(f, part)
            step.append(frame)
        return step

    def frequency_response_parallel(self, freq_range, part=None, cpus=4):
        from multiprocessing import Pool
        w = freq_range * (2 * np.pi)
        M = self.M
        K = self.K
        f = self.F
        results = []
        with Pool(processes=cpus) as pool:
            for i in range(len(w)):
                results.append(pool.apply_async(dynamic_solve,(M,K,w[i],f)))
            for i in range(len(results)):
                results[i] = results[i].get()
        step = Step()
        functions = self._basis[0].functions
        for i,x in enumerate(results):
            step.append(Frame(self._geometry, functions,
                              get_part(freq_range[i], part), get_part(x, part)))
        return step

def dynamic_solve(M,K,w,f):
    A = -w ** 2 * M + K
    x = linalg.solve(A.toarray(), f)
    return x

class Step:
    def __init__(self, frames=[]):
        self._frames = list(frames)

    def append(self, frame):
        self._frames.append(frame)

    def __repr__(self):
        return "Step object with {n} frames".format(n=len(self._frames))

    def __getitem__(self, *args, **kwargs):
        return self._frames.__getitem__(*args, **kwargs)

    @property
    def frequency(self):
        return np.array([i.frequency for i in self._frames])

    def U(self, x, y):
        return np.array([i.U(x, y) for i in self._frames])

    def V(self, x, y):
        return np.array([i.V(x, y) for i in self._frames])

    def W(self, x, y):
        return np.array([i.W(x, y) for i in self._frames])

    def Φx(self, x, y):
        return np.array([i.Φx(x, y) for i in self._frames])
    Px = Φx

    def Φy(self, x, y):
        return np.array([i.Φy(x, y) for i in self._frames])
    Py = Φy


class Frame:
    def __init__(self, geometry, basis, freq, x):
        self._geometry = geometry
        self._basis = basis.reshape(-1)
        self._freq = freq
        self._x = np.asarray(x)

    def __repr__(self):
        return "Frame object at {freq}Hz".format(freq=self._freq)

    @property
    def geometry(self):
        return self._geometry

    @property
    def frequency(self):
        return self._freq

    def _generate_func(self, coeff):
        def func(x1, x2):
            res = 0
            for i in range(len(self._basis)):
                res += coeff[i] * self._basis[i](x1, x2)
            return res
        return func

    @property
    def U(self):
        MN = len(self._basis)
        return self._generate_func(self._x[0 * MN: 1 * MN])

    @property
    def V(self):
        MN = len(self._basis)
        return self._generate_func(self._x[1 * MN: 2 * MN])

    @property
    def W(self):
        MN = len(self._basis)
        return self._generate_func(self._x[2 * MN: 3 * MN])

    @property
    def Φx(self):
        MN = len(self._basis)
        return self._generate_func(self._x[3 * MN: 4 * MN])
    Px = Φx

    @property
    def Φy(self):
        MN = len(self._basis)
        return self._generate_func(self._x[4 * MN: 5 * MN])
    Py = Φy

    @property
    def real(self):
        return Frame(self._geometry, self._basis,
                     self._freq.real, self._x.real)


if __name__ == '__main__':
    material = Material(210e9, 210e9, 210e9 / 2.6, 210e9 / 2.6, 210e9 / 2.6, 0.3,
                        rho=7800, name='Steel')
    profile = Profile([material], [0], [-0.005, 0.005])
    material_beam = Material(210e9, 210e9, 210e9 / 2.6, 210e9 / 2.6, 210e9 / 2.6, 0.3,
                        rho=7800, name='Steel')
    material_none = Material(210, 210, 210 / 2.6, 210 / 2.6, 210 / 2.6, 0.3,
                        rho=7800/1e9, name='None')
    profile_beam  = Profile_beam([material_beam,material_none,material_beam], [0,0,0], [-0.02,-0.005,0.005, 0.02], 0.01)
    profile_beam  = Profile_beam([material_beam], [0], [-0.02, 0.02], 0.01)

    # material = Material(144.84e9, 9.65e9, 3.45e9, 4.14e9, 4.14e9, 0.3,
    #                    rho=1389.79, name='AS/3501')
    # profile = Profile([material] * 8,
    #            [np.pi / 6, 0, np.pi / 2, -np.pi / 4,
    #                np.pi / 4, 0, np.pi / 2, -np.pi / 6],
    #            np.linspace(-0.5, 0.5, 9, endpoint=True))
    import sympy as sym

    t = sym.symbols('t', real=True)
    x = t
    y = 0.5-0.25*t
    curve = Curve(x, y)

    shell = ShellOfRevolution(curve, [0, 2], 20, 21,
                                profile, nquads=[160] * 2)
    shell.add_boundary_condition(1e18,x1=0,direction=[0,1,2])
#    shell.add_stringer(profile_beam, x2=np.pi/4*np.arange(8))
    shell.add_stringer(profile_beam, x2=np.pi/2*np.arange(4),range_x=[0,1])
#    shell.add_stringer(profile_beam, x2=0)
#    shell.add_stringer(profile_beam, x1=[0.2,0.4,0.6,0.8])
#    lb = legendre_basis(30,[0,1])
#    shell.set_basis(bx1=lb)
    modal = shell.modal()
