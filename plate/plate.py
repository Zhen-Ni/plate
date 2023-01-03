#!/usr/bin/env python3

import copy
import numpy as np
import sympy as sym

from .trail_function import Basis, trigonometric_basis
from .quadrature import quad_gauss
from .misc import round_array


__all__ = ('Plate', 'SupersonicPlate', 'PressurizedPlate',
           'PressurizedSupersonicPlate',
           'BC_S', 'BC_C', 'BC_F',
           'AIR_DENSITY', 'SOUND_SPEED')


BC_F = ()
BC_S = 0, 1, 2
BC_C = 0, 1, 2, 3, 4

AIR_DENSITY = 1.2
SOUND_SPEED = 340


class PlateMeta(type):
    def __init__(self, clsname, bases, dct):
        if '_assemble' not in dct:
            setattr(self, '_assemble', self.empty_func)
        if '_reset' not in dct:
            setattr(self, '_reset', self.empty_func)
        setattr(self, 'assemble', self.get_assemble())
        setattr(self, 'reset', self.get_reset())

    @staticmethod
    def empty_func(self):
        pass

    def get_assemble(DerivedPlate):
        def assemble(self):
            super(DerivedPlate, self).assemble()
            DerivedPlate._assemble(self)
        return assemble

    def get_reset(DerivedPlate):
        def reset(self):
            DerivedPlate._reset(self)
            super(DerivedPlate, self).reset()
        return reset


class PlateBase():
    """Dynamic model for plate.

    Parameters
    ----------
    a, b: float
        Length and width of the plate. The parameter a corresponds to
        the x direction of the plate while b corresponds to the y
        direction

    M, N: int
        The number of basis functions used for x and y direction respectively.

    profile: laminate.Profile
        The basic parameters of the plate.

    dtype: data-type, optional
        The desired data-type for modelling. If None is given, the same
        data-type with `profile` will be used.

    basis_x, basis_y: None or trail_function.Basis
        Basis function for the x and direction respectively. If None,
        trigonometric basis will be used. (default to None)

    """

    def __init__(self, a, b, M, N, profile, dtype=None, *,
                 basis_x=None, basis_y=None, quad_x=None, quad_y=None):
        self._a = a
        self._b = b
        self._M = M
        self._N = N
        self._profile = profile
        
        self._dtype = None
        self.set_dtype(dtype)

        # The total number of basis funcitons
        self._MN = self._M * self._N

        # Setup basis functions
        # self._basis_x = [\varphi(x),d\varphi(x)/{dx}]
        # each number of basis functions = M
        self._basis_x = [None] * 2
        # self._basis_y = [\varphi(y),d\varphi(y)/{dy}]
        # each number of basis functions = N
        self._basis_y = [None] * 2
        self.set_basis(basis_x, basis_y)

        # setup self._quad_x and self._quad_y
        # number of quadrature points is the same as number of
        # basis functions by default
        self._quad_x = None
        self._quad_y = None
        self.set_quadrature(quad_x, quad_y)

        # The corresponding values of the basis functions on the
        # quadrature points. These variables are used for acceleration
        # of the program.
        self._basis_x_values = [None] * 2
        self._basis_y_values = [None] * 2

        # The values of the corresponding methods. Setup cache
        # manually. Used for acceleration of the program.
        self._integrate_basis_x_values = {}
        self._integrate_basis_y_values = {}

        # The matrixes
        self._mass_matrix = {}
        self._stiffness_matrix = {}
        self._damping_matrix = {}
        self._force_vector = {}
        self._M_assembled = None
        self._K_assembled = None
        self._C_assembled = None
        self._F_assembled = None

        # Initialization includes setting orders and basis functions.
        # Thus, the basis values and integrate basis values should be
        # reset when uninitialing. The derived values, such as the
        # mass and stiffness matrix, needs to be reset as well.
        self._is_initialized = False

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        self.set_dtype(dtype)

    def set_dtype(self, dtype=None):
        self.uninitialize()
        self._dtype = self._profile.dtype if dtype is None else dtype

    @property
    def profile(self):
        return self._profile

    @profile.setter
    def profile(self, profile):
        self.set_profile(profile)

    def set_profile(self, profile):
        self.uninitialize()
        self._profile = profile

    def set_basis(self, basis_x=None, basis_y=None):
        """Setup basis functions for the x and y direction.

        The input basis can be trail_function.Basis object or None. If None
        is used here, trigonometric basis will be generated and used. Note
        that self._M and self._N will be used in this method to truncate these
        basis funcitons.

        Parameters
        ----------
        basis_x, basis_y: None or trail_function.Basis
            Basis function for the x and direction respectively. If None,
            trigonometric basis will be used.
        """
        if basis_x is None:
            basis_x = trigonometric_basis(self._M, (0, self._a))
        if basis_y is None:
            basis_y = trigonometric_basis(self._N, (0, self._b))
        fx = Basis(basis_x.symbolic_functions[:self._M], name='x')
        fy = Basis(basis_y.symbolic_functions[:self._N], name='y')
        x, y = sym.symbols('x, y')
        fx.symbol = x
        fy.symbol = y
        dfx = fx.diff()
        dfy = fy.diff()
        if not (len(dfx) == self._M and len(dfy) == self._N):
            raise ValueError('number of basis less than defined')
        self.uninitialize()
        self._basis_x = [fx, dfx]
        self._basis_y = [fy, dfy]
        return None

    def set_quadrature(self, quad_x=None, quad_y=None):
        """Setup the quadrature methods for the two axis.

        Parameters
        ----------
        quad_x, quad_y: quadrature.Quadrature or None
            Quadrature object used to performance quadrature along the x
            or the y axis. If None is given, Gauss Quadrature will be used
            and the number of quadrature points will be set to the twice
            of the number of corresponding basis functions.
        """
        if quad_x is None:
            quad_x = quad_gauss(self._M * 2)
        if quad_y is None:
            quad_y = quad_gauss(self._N * 2)
        self.uninitialize()
        self._quad_x = copy.copy(quad_x).set_limit(0, self._a)
        self._quad_y = copy.copy(quad_y).set_limit(0, self._b)
        return None

    def _init_basis_values(self):
        """Initialize values of the basis functions on quadrature points."""
        # Note that basis functions are ALWAYS REAL!
        self._basis_x_values[0] = np.zeros([self._M, len(self._quad_x.x)])
        self._basis_x_values[1] = np.zeros([self._M, len(self._quad_x.x)])
        for i in range(self._M):
            self._basis_x_values[0][i] = self._basis_x[0][i](self._quad_x.x)
            self._basis_x_values[1][i] = self._basis_x[1][i](self._quad_x.x)
        self._basis_y_values[0] = np.zeros([self._N, len(self._quad_y.x)])
        self._basis_y_values[1] = np.zeros([self._N, len(self._quad_y.x)])
        for i in range(self._N):
            self._basis_y_values[0][i] = self._basis_y[0][i](self._quad_y.x)
            self._basis_y_values[1][i] = self._basis_y[1][i](self._quad_y.x)

    def _integrate_basis_x(self, order_i, order_j):
        """Get the integration along the x axis.

        The function to be integrated is:
        self._basis_x[order_i] * self._basis_x[order_j]
        where order_i and order_j are the order of derivation, should
        be 0 or 1.
        """
        # Use cached result if possible
        if (res := self._integrate_basis_x_values.get(
                (order_i, order_j))) is not None:
            return res
        # Note that basis functions are ALWAYS REAL!
        res = np.zeros([self._M, self._M])
        if order_i == order_j:
            for i, vi in enumerate(self._basis_x_values[order_i]):
                for j, vj in enumerate(self._basis_x_values[order_j][:i+1]):
                    resij = np.sum(vi*vj*self._quad_x.w)
                    res[i, j] = resij
                    if i != j:
                        res[j, i] = resij
        else:
            for i, vi in enumerate(self._basis_x_values[order_i]):
                for j, vj in enumerate(self._basis_x_values[order_j]):
                    resij = np.sum(vi*vj*self._quad_x.w)
                    res[i, j] = resij
        round_array(res)
        self._integrate_basis_x_values[(order_i, order_j)] = res
        return res

    def _integrate_basis_y(self, order_i, order_j):
        """Get the integration along the y axis.

        The function to be integrated is:
        self._basis_y[order_i] * self._basis_y[order_j]
        where order_i and order_j are the order of derivation, should
        be 0 or 1.
        """
        # Use cached result if possible
        if (res := self._integrate_basis_y_values.get(
                (order_i, order_j))) is not None:
            return res
        # Note that basis functions are ALWAYS REAL!
        res = np.zeros([self._N, self._N])
        if order_i == order_j:
            for i, vi in enumerate(self._basis_y_values[order_i]):
                for j, vj in enumerate(self._basis_y_values[order_j][:i+1]):
                    resij = np.sum(vi*vj*self._quad_y.w)
                    res[i, j] = resij
                    if i != j:
                        res[j, i] = resij
        else:
            for i, vi in enumerate(self._basis_y_values[order_i]):
                for j, vj in enumerate(self._basis_y_values[order_j]):
                    resij = np.sum(vi*vj*self._quad_y.w)
                    res[i, j] = resij
        round_array(res)
        self._integrate_basis_y_values[(order_i, order_j)] = res
        return res

    def initialize(self):
        """Initialize the basis settings of the plate. This method is called
        automatically when assemble matrixes."""
        if not self._is_initialized:
            self._initialize()

    def _initialize(self):
        self._init_basis_values()
        self._is_initialized = True

    def uninitialize(self):
        self._is_initialized = False
        self._is_assembled = False
        self._mass_matrix = {}
        self._stiffness_matrix = {}
        self._damping_matrix = {}
        self._force_vector = {}
        self._M_assembled = None
        self._K_assembled = None
        self._C_assembled = None
        self._F_assembled = None

        self._integrate_basis_x_values = {}
        self._integrate_basis_y_values = {}
        self._basis_x_values = [None] * 2
        self._basis_y_values = [None] * 2

    def reset(self):
        """The method `reset` should be implemented in the derived class."""
        pass

    def _pop_mass_matrix(self, name):
        self._mass_matrix.pop(name, None)
        self._M_assembled = None

    def _pop_stiffness_matrix(self, name):
        self._stiffness_matrix.pop(name, None)
        self._K_assembled = None

    def _pop_damping_matrix(self, name):
        self._damping_matrix.pop(name, None)
        self._C_assembled = None

    def _pop_force_vector(self, name):
        self._force_vector.pop(name, None)
        self._F_assembled = None

    def assemble(self):
        self.initialize()

    def get_M(self):
        if self._M_assembled is None:
            self.assemble()
            self._M_assembled = np.zeros([self._MN * 5, self._MN * 5],
                                         dtype=self.dtype)
            for name, value in self._mass_matrix.items():
                self._M_assembled += value
        return self._M_assembled

    def get_K(self):
        if self._K_assembled is None:
            self.assemble()
            self._K_assembled = np.zeros([self._MN * 5, self._MN * 5],
                                         dtype=self.dtype)
            for name, value in self._stiffness_matrix.items():
                self._K_assembled += value
        return self._K_assembled

    def get_C(self):
        if self._C_assembled is None:
            self.assemble()
            self._C_assembled = np.zeros([self._MN * 5, self._MN * 5],
                                         dtype=self.dtype)
            for name, value in self._damping_matrix.items():
                self._C_assembled += value
        return self._C_assembled

    def get_F(self):
        if self._F_assembled is None:
            self.assemble()
            self._F_assembled = np.zeros([self._MN * 5], dtype=self.dtype)
            for name, value in self._force_vector.items():
                self._F_assembled += value
        return self._F_assembled

    def _block_access(self, matrix, i, j):
        MN = self._MN
        i *= MN
        j *= MN
        return matrix[i: i + MN, j: j + MN]

    def _block_add(self, matrix, i, j, val):
        MN = self._MN
        i *= MN
        j *= MN
        matrix[i: i + MN, j: j + MN] += np.asarray(val, dtype=self._dtype)

    def _expand_basis(self, vvx, vvy):
        # Optimization can be done for this method as it will be called lots
        # of times with the same vvx and vvy. (eg. use something like
        # lru_cache. lru_cache is not used here as vvx and vvy are np.ndarray
        # objects which cannot be pickled.)
        i = np.arange(self._MN)
        ix = i // self._N
        iy = i % self._N
        vv = vvx[ix][:, ix] * vvy[iy][:, iy]
        return vv

    def __getstate__(self):
        if self._is_initialized:
            plate = type(self).__new__(type(self))
            plate.__dict__.update(self.__dict__)
            plate.uninitialize()
        else:
            plate = self
        state = plate.__dict__
        return state


class Plate(PlateBase, metaclass=PlateMeta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Additional terms
        # spring
        # format: pairs of (value, dof, x, y)
        # `x` and `y` can be set to None. If both are None, the boundary
        # condition corresponds to elastic foundation. If only one None,
        # it means boundary condition along the given line. If both `x`
        # and `y` are numbers, it means an isolated spring.
        self._spring = []
        # additional mass
        # format: pairs of (value, dof, x, y)
        # similar to format of spring
        self._inertia = []
        # dashpot
        # format: pairs of (value, dof, x, y)
        # similar to format of spring
        self._dashpot = []
        # External force
        # format: pairs of (value, dof, x, y)
        self._force = []
        # user-defined damping matrix
        # format: list of damping matrixes
        self._user_C = []

    def assemble(self):
        """Assemble mass, stiffness, damping matrix and force vector.

        The derived class should write its own _assemble method to assemble
        its own energy functionals. Do not overwrite this method in the derived
        class."""
        # This method is overwrite by the metaclass to avoid polymorphism.
        super().assemble()
        self._assemble()

    def _assemble(self):
        # Assemble basic mass and stiffness matrixes for plate.
        if self._mass_matrix.get('fsdt') is None:
            self._assemble_mass_matrix_fsdt()
        if self._stiffness_matrix.get('fsdt') is None:
            self._assemble_stiffness_matrix_fsdt()
        # Assemble stiffness matrix for springs. Area spring, line spring and
        # point spring are considered here. The boundary conditions are
        # implemented by using the penalty method.
        if self._stiffness_matrix.get('spring') is None:
            self._assemble_stiffness_matrix_spring()
        # Assemble mass matrix for additional mass. Area mass, line mass and
        # point mass are considered here.
        if self._mass_matrix.get('inertia') is None:
            self._assemble_mass_matrix_inertia()
        # Assemble damping matrix for additional dashpot. Area dashpot,
        # linedashpot and point dashpot are all considered here.
        if self._damping_matrix.get('dashpot') is None:
            self._assemble_damping_matrix_dashpot()
        # Assemble damping matrix for user-defined damping.
        if self._damping_matrix.get('user_C') is None:
            self._assemble_damping_matrix_user_C()
        # Assemble force vector
        if self._force_vector.get('force') is None:
            self._assemble_force_vector_force()

    def reset(self):
        """Remove all additional mass, stiffness, damping and force.

        The derived class should write its own _reset method to remove its
        additional settings. Do not overwrite this method in the derived
        class."""
        # This method is overwrite by the metaclas to avoid spolymorphism.
        self._reset()
        super().reset()

    def _reset(self):
        self.reset_spring()
        self.reset_inertia()
        self.reset_dashpot()
        self.reset_force()

    def add_spring(self, value, dof, *, x=None, y=None):
        """Add a spring to the plate.

        The spring can either be an aera spring, line spring, or a point
        spring, depending on whether the x and y coordinate of the spring
        is defined.

        Parameters
        ----------
        value: float
            Stiffness of the spring.
        dof: int or list of int
            The degree(s) of freedom the spring applies to. Should be in
            [0, 1, 2, 3, 4, 5]
        x: float or None, optional
            The x coordinate of the spring. If None is given, an area
            spring (y is None) or line spring (y is float) is defined.
        y: float or None, optional
            The y coordinate of the spring. If None is given, an area
            spring (x is None) or line spring (x is float) is defined.
        """
        self._pop_stiffness_matrix('spring')
        if np.iterable(dof):
            for dofi in dof:
                self.add_spring(value, dofi, x=x, y=y)
        else:
            self._spring.append((value, dof, x, y))

    def reset_spring(self):
        """Remove all springs."""
        self._pop_stiffness_matrix('spring')
        self._spring = []

    def add_inertia(self, value, dof=(0, 1, 2), *, x=None, y=None):
        """Add inertia to the plate.
        The inertia can be a point mass, line mass or area mass."""
        self._pop_mass_matrix('inertia')
        if np.iterable(dof):
            for dofi in dof:
                self.add_inertia(value, dofi, x=x, y=y)
        else:
            self._inertia.append((value, dof, x, y))

    def reset_inertia(self):
        """"Remove all inertia."""
        self._pop_mass_matrix('inertia')
        self._inertia = []

    def add_dashpot(self, value, dof, *, x=None, y=None):
        """Add dashpot to the plate."""
        self._pop_damping_matrix('dashpot')
        if np.iterable(dof):
            for dofi in dof:
                self.add_dashpot(value, dofi, x=x, y=y)
        else:
            self._dashpot.append((value, dof, x, y))

    def reset_dashpot(self):
        """Remove all dashpots."""
        self._pop_damping_matrix('dashpot')
        self._dashpot = []

    def add_force(self, value, dof=2, *, x=None, y=None):
        """Add force to the plate."""
        self._pop_force_vector('force')
        if np.iterable(dof):
            for dofi in dof:
                self.add_force(value, dof, x=x, y=y)
        else:
            self._force.append((value, dof, x, y))

    def reset_force(self):
        """Remove all force."""
        self._pop_force_vector('force')
        self._force = []

    def add_user_C(self, C):
        """Add an additional term to the final damping matrix."""
        self._pop_damping_matrix('user_C')
        self._user_C.append(C)

    def reset_user_C(self):
        """Remove additional damping matrixes."""
        self._pop_damping_matrix('user')
        self._user_C = []

    def _assemble_mass_matrix_fsdt(self):
        M = np.zeros([self._MN * 5] * 2, dtype=self.dtype)
        I_ = self._profile.I()
        vvx = self._integrate_basis_x(0, 0)
        vvy = self._integrate_basis_y(0, 0)
        vv = self._expand_basis(vvx, vvy)

        sigma = np.einsum('i,jk', I_, vv)
        add = self._block_add
        add(M, 0, 0, sigma[0])
        add(M, 1, 1, sigma[0])
        add(M, 2, 2, sigma[0])
        add(M, 0, 3, sigma[1])
        add(M, 1, 4, sigma[1])
        add(M, 3, 0, sigma[1])
        add(M, 4, 1, sigma[1])
        add(M, 3, 3, sigma[2])
        add(M, 4, 4, sigma[2])
        self._mass_matrix['fsdt'] = M

    def _assemble_stiffness_matrix_fsdt(self):
        K = np.zeros([self._MN * 5] * 2, dtype=self.dtype)
        # get matrix R
        ABDAs = np.zeros([8, 8], dtype=self.dtype)
        ABDAs[:3, :3] = np.asarray(self._profile.A(), dtype=self._dtype)
        ABDAs[3:6, :3] = ABDAs[:3, 3:6] = np.asarray(self._profile.B(),
                                                     dtype=self._dtype)
        ABDAs[3:6, 3:6] = np.asarray(self._profile.D(), dtype=self._dtype)
        ABDAs[6:, 6:] = np.asarray(self._profile.kappa() * self._profile.As(),
                                   dtype=self._dtype)
        # Optimization can be done here as T.T.dot(ABDAs).dot(T) is actually
        # reordering the elements in ABDAs. The result can be easily obtained
        # by using sympy.
        # >>> A = [[Symbol('a{}{}'.format(i,j)) for j in range(8)]
        #           for i in range(8)]
        # >>> R = T.T * A * T
        # >>> R
        # Matrix([
        # [0,0,0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        # [0,0,0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        # [0,0,0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        # [0,0,0, a77, a76, a70, a72, a77, a73, a75, a72, a71, a76, a75, a74],
        # [0,0,0, a67, a66, a60, a62, a67, a63, a65, a62, a61, a66, a65, a64],
        # [0,0,0, a07, a06, a00, a02, a07, a03, a05, a02, a01, a06, a05, a04],
        # [0,0,0, a27, a26, a20, a22, a27, a23, a25, a22, a21, a26, a25, a24],
        # [0,0,0, a77, a76, a70, a72, a77, a73, a75, a72, a71, a76, a75, a74],
        # [0,0,0, a37, a36, a30, a32, a37, a33, a35, a32, a31, a36, a35, a34],
        # [0,0,0, a57, a56, a50, a52, a57, a53, a55, a52, a51, a56, a55, a54],
        # [0,0,0, a27, a26, a20, a22, a27, a23, a25, a22, a21, a26, a25, a24],
        # [0,0,0, a17, a16, a10, a12, a17, a13, a15, a12, a11, a16, a15, a14],
        # [0,0,0, a67, a66, a60, a62, a67, a63, a65, a62, a61, a66, a65, a64],
        # [0,0,0, a57, a56, a50, a52, a57, a53, a55, a52, a51, a56, a55, a54],
        # [0,0,0, a47, a46, a40, a42, a47, a43, a45, a42, a41, a46, a45, a44]])
        T = _get_T()
        R = T.T.dot(ABDAs).dot(T)
        for i in range(5):
            for j in range(i + 1):
                block = self._assemble_stiffness_matrix_fsdt_helper(i, j, R)
                self._block_add(K, i, j, block)
                if i != j:
                    self._block_add(K, j, i, block.T)
        self._stiffness_matrix['fsdt'] = K

    def _assemble_stiffness_matrix_fsdt_helper(self, i, j, R):
        block = np.zeros([self._MN, self._MN], dtype=self.dtype)
        for ii in range(3):
            for jj in range(3):
                Riijj = R[i+5*ii, j+5*jj]
                vv1 = self._integrate_basis_x(ii == 1, jj == 1)
                vv2 = self._integrate_basis_y(ii == 2, jj == 2)
                vv = self._expand_basis(vv1, vv2)
                block += vv * Riijj
        return block

    def _assemble_stiffness_matrix_spring(self):
        block = np.zeros([self._MN * 5] * 2, dtype=self.dtype)
        for value, dof, x, y in self._spring:
            if x is None:
                vvx = self._integrate_basis_x(0, 0)
            else:
                vx = self._basis_x[0](x)
                vvx = np.einsum('i,j', vx, vx)
            if y is None:
                vvy = self._integrate_basis_y(0, 0)
            else:
                vy = self._basis_y[0](y)
                vvy = np.einsum('i,j', vy, vy)
            vv = self._expand_basis(vvx, vvy)
            k = vv * value
            self._block_add(block, dof, dof, k)
        self._stiffness_matrix['spring'] = block

    def _assemble_mass_matrix_inertia(self):
        block = np.zeros([self._MN * 5] * 2, dtype=self.dtype)
        for value, dof, x, y in self._inertia:
            if x is None:
                vvx = self._integrate_basis_x(0, 0)
            else:
                vx = self._basis_x[0](x)
                vvx = np.einsum('i,j', vx, vx)
            if y is None:
                vvy = self._integrate_basis_y(0, 0)
            else:
                vy = self._basis_y[0](y)
                vvy = np.einsum('i,j', vy, vy)
            vv = self._expand_basis(vvx, vvy)
            k = vv * value
            self._block_add(block, dof, dof, k)
        self._mass_matrix['inertia'] = block

    def _assemble_damping_matrix_dashpot(self):
        block = np.zeros([self._MN * 5] * 2, dtype=self.dtype)
        for value, dof, x, y in self._dashpot:
            if x is None:
                vvx = self._integrate_basis_x(0, 0)
            else:
                vx = self._basis_x[0](x)
                vvx = np.einsum('i,j', vx, vx)
            if y is None:
                vvy = self._integrate_basis_y(0, 0)
            else:
                vy = self._basis_y[0](y)
                vvy = np.einsum('i,j', vy, vy)
            vv = self._expand_basis(vvx, vvy)
            k = vv * value
            self._block_add(block, dof, dof, k)
        self._damping_matrix['dashpot'] = block

    def _assemble_force_vector_force(self):
        block = np.zeros([self._MN * 5], dtype=self.dtype)
        for value, dof, x, y in self._force:
            if x is None:
                # The following four statements are equal:
                # vx = np.sum(self._quad_x.w * self._basis_x_values[0],
                #             axis=-1)
                # vx = np.einsum('j,ij',
                #                self._quad_x.w, self._basis_x_values[0])
                # vx = np.dot(self._basis_x_values[0], self._quad_x.w)
                # vx = self._basis_x_values[0].dot(self._quad_x.w)
                # But the computational time are 5.48us, 2.7us, 985ns and 574ns
                # respectively. However, the function dot uses multiple cpu
                # cores while the other two methods uses only one core.
                # Anyway, use the Einstein summation convention is efficient
                # and the most clear.
                vx = np.einsum('j,ij', self._quad_x.w, self._basis_x_values[0])
            else:
                vx = self._basis_x[0](x)
            if y is None:
                # As explained before. Use Einstein summation convention.
                vy = np.einsum('j,ij', self._quad_y.w, self._basis_y_values[0])
            else:
                vy = self._basis_y[0](y)
            v = np.einsum('i,j', vx, vy).reshape(-1)
            p = v * value
            block[dof * self._MN: dof * self._MN + self._MN] += p
        self._force_vector['force'] = block

    def _assemble_damping_matrix_user_C(self):
        block = np.zeros([self._MN * 5] * 2, dtype=self.dtype)
        for ci in self._user_C:
            block += ci
        self._damping_matrix['user_C'] = block


class SupersonicPlate(Plate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # supersonic airflow
        # format: (scaled_dynamic_pressure, flow angle, mach_number, velocity)
        # where scaled_dynamic_pressure = (air_density * velocity ** 2) /
        # (mach_number ** 2 - 1) ** .5
        self._airflow = None

    def get_lambda(self, velocity, density=AIR_DENSITY,
                   sound_speed=SOUND_SPEED):
        """Get the non-dimensional aerodynamic pressure parameter.

        The definitions of the non-dimensional aerodynamic pressure
        parameter is given by Ref[1].

        Reference
        ---------
        [1] ZHOU K, HUANG X, ZHANG Z, et al. Aero-thermo-elastic flutter
        analysis of coupled plate structures in supersonic flow with
        general boundary conditions[J/OL]. Journal of Sound and Vibration,
        2018, 430: 36-58.
        """
        D = self._profile._material[0].Q()[0, 0]
        h = self._profile._z[-1] - self._profile._z[0]
        D *= h ** 3 / 12
        L = self._a
        M = velocity / sound_speed
        lambda_ = density * velocity ** 2 * L ** 3 / (D * (M ** 2 - 1) ** .5)
        return lambda_

    def get_airflow_velocity(self, lambda_,
                             density=AIR_DENSITY, sound_speed=SOUND_SPEED):
        """Calculate the velocity of the airflow from given non-dimensional
        aerodynamic pressure."""
        D = self._profile._material[0].Q()[0, 0]
        h = self._profile._z[-1] - self._profile._z[0]
        D *= h ** 3 / 12
        L = self._a
        # The equation of the non-dimensional aerodynamic pressure can
        # be written as a quadratic polynomial of (velocity**2).
        a2 = (density * L ** 3) ** 2
        a1 = -(lambda_ * D / sound_speed) ** 2
        a0 = (lambda_ * D) ** 2
        # Only the maximum real root is calculated
        delta = float(a1 ** 2 - 4 * a0 * a2)
        velocity2 = (-a1 + delta ** .5) / (2 * a2)
        return velocity2 ** .5

    def add_airflow(self, velocity, flow_angle=0,
                    density=AIR_DENSITY, sound_speed=SOUND_SPEED):
        if self._airflow is not None:
            raise AttributeError('airflow has already been set')
        self._pop_stiffness_matrix('airflow')
        self._pop_damping_matrix('airflow')
        mach_number = velocity / sound_speed
        scaled_dynamic_pressure = density * velocity ** 2
        scaled_dynamic_pressure /= (mach_number ** 2 - 1) ** .5
        self._airflow = (scaled_dynamic_pressure, flow_angle,
                         mach_number, velocity)

    def add_airflow_lambda(self, lambda_, flow_angle=0,
                           density=AIR_DENSITY, sound_speed=SOUND_SPEED):
        if self._airflow is not None:
            raise AttributeError('airflow has already been set')
        self._pop_stiffness_matrix('airflow')
        self._pop_damping_matrix('airflow')
        velocity = self.get_airflow_velocity(lambda_)
        mach_number = velocity / sound_speed
        D = self._profile._material[0].Q()[0, 0]
        h = self._profile._z[-1] - self._profile._z[0]
        D *= h ** 3 / 12
        L = self._a
        scaled_dynamic_pressure = lambda_ * D / L ** 3
        self._airflow = (scaled_dynamic_pressure, flow_angle,
                         mach_number, velocity)

    def reset_airflow(self):
        self._pop_stiffness_matrix('airflow')
        self._pop_damping_matrix('airflow')
        self._airflow = None

    def _reset(self):
        self.reset_airflow()

    def _assemble(self):
        if self._stiffness_matrix.get('airflow') is None:
            self._assemble_stiffness_matrix_airflow()
        if self._damping_matrix.get('airflow') is None:
            self._assemble_damping_matrix_airflow()

    def _assemble_stiffness_matrix_airflow(self):
        if self._airflow is None:
            return
        K = np.zeros([self._MN * 5] * 2, dtype=self.dtype)
        # A is the scaled_dynamic_pressure
        A, flow_angle, mach_number, velocity = self._airflow
        vvx = self._integrate_basis_x(0, 1)
        vvy = self._integrate_basis_y(0, 0)
        vv_cos = self._expand_basis(vvx, vvy)
        # vv_cos = vv_cos + vv_cos.T
        vv_cos *= np.cos(flow_angle)
        vvx = self._integrate_basis_x(0, 0)
        vvy = self._integrate_basis_y(0, 1)
        vv_sin = self._expand_basis(vvx, vvy)
        # vv_sin = vv_sin + vv_sin.T
        vv_sin *= np.sin(flow_angle)
        vv = vv_cos + vv_sin
        vv = vv * A
        self._block_add(K, 2, 2, vv)
        self._stiffness_matrix['airflow'] = K

    def _assemble_damping_matrix_airflow(self):
        if self._airflow is None:
            return
        C = np.zeros([self._MN * 5] * 2, dtype=self.dtype)
        A, flow_angle, mach_number, velocity = self._airflow
        vvx = self._integrate_basis_x(0, 0)
        vvy = self._integrate_basis_y(0, 0)
        vv = self._expand_basis(vvx, vvy)
        A = A * (mach_number ** 2 - 2) / (mach_number ** 2 - 1) / velocity
        self._block_add(C, 2, 2, A * vv)
        self._damping_matrix['airflow'] = C


class PressurizedPlate(Plate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # arbitrary pressure
        # format: list of pairs: (function, flag_vectorize)
        # The function accepts two arguments as the x and y coordinate and
        # returns the pressure on the point. The flag_vectorize identicates
        # whether the function can be vectorized.
        self._pressure = []

    def add_pressure(self, function, vectorize=True):
        """Add arbitrary pressure to the plate.
        
        The pressure is in the z direction.
        """
        self._pop_force_vector('pressure')
        self._pressure.append((function, vectorize))

    def reset_pressure(self):
        """Remove all pressure."""
        self._pop_force_vector('pressure')
        self._pressure = []

    def _reset(self):
        self.reset_pressure()

    def _assemble(self):
        if self._force_vector.get('pressure') is None:
            self._assemble_force_vector_pressure()

    def _assemble_force_vector_pressure(self):
        block = np.zeros([self._MN * 5], dtype=self.dtype)
        w = np.einsum('i,j', self._quad_x.w, self._quad_y.w)
        vx = self._basis_x_values[0]
        vy = self._basis_y_values[0]
        v = np.einsum('ik,jl', vx, vy)
        v.resize(self._MN, vx.shape[1], vy.shape[1])
        wv = w * v
        for p_func, vectorize in self._pressure:
            if vectorize:
                x, y = np.meshgrid(self._quad_x.x, self._quad_y.x)
                pv = p_func(x, y).T
            else:
                pv = np.zeros(w.shape, dtype=self.dtype)
                for i, xi in enumerate(self._quad_x.x):
                    for j, yi in enumerate(self._quad_y.x):
                        pv[i, j] = p_func(xi, yi)
            # The following statement is equal to
            # `pi = np.sum(pv * wv, axis=(1, 2))`
            # but much faster (about twice the speed)
            pi = np.einsum('jk, ijk', pv, wv)
            block[self._MN * 2: self._MN * 3] += pi
        self._force_vector['force'] = block


class PressurizedSupersonicPlate(SupersonicPlate, PressurizedPlate):
    pass


# Strain-displacement relation.
# \varepsilon = T \cdot \mathbf{u}
def _get_T():
    T = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
    return T
