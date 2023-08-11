import numpy as np
cimport numpy as np
cimport cython
from .base cimport OdeSolver
from .base import DenseOutput
from .common import (validate_max_step, validate_tol, select_initial_step,
                     warn_extraneous, validate_first_step)
from . import dop853_coefficients


cdef class RungeKutta(OdeSolver):
    """
    Parameters
    ----------
    fun : callable
        Right-hand side of the system. The calling signature is ``fun(t, y)``.
        Here ``t`` is a scalar, and there are two options for the ndarray ``y``:
        It can either have shape (n,); then ``fun`` must return array_like with
        shape (n,). Alternatively it can have shape (n, k); then ``fun``
        must return an array_like with shape (n, k), i.e., each column
        corresponds to a single column in ``y``. The choice between the two
        options is determined by `vectorized` argument (see below).
    t0 : float
        Initial time.
    y0 : array_like, shape (n,)
        Initial state.
    t_bound : float
        Boundary time - the integration won't continue beyond it. It also
        determines the direction of the integration.
    first_step : float or None, optional
        Initial step size. Default is ``None`` which means that the algorithm
        should choose.
    max_step : float, optional
        Maximum allowed step size. Default is np.inf, i.e., the step size is not
        bounded and determined solely by the solver.
    rtol, atol : float and array_like, optional
        Relative and absolute tolerances. The solver keeps the local error
        estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a
        relative accuracy (number of correct digits), while `atol` controls
        absolute accuracy (number of correct decimal places). To achieve the
        desired `rtol`, set `atol` to be smaller than the smallest value that
        can be expected from ``rtol * abs(y)`` so that `rtol` dominates the
        allowable error. If `atol` is larger than ``rtol * abs(y)`` the
        number of correct digits is not guaranteed. Conversely, to achieve the
        desired `atol` set `rtol` such that ``rtol * abs(y)`` is always smaller
        than `atol`. If components of y have different scales, it might be
        beneficial to set different `atol` values for different components by
        passing array_like with shape (n,) for `atol`. Default values are
        1e-3 for `rtol` and 1e-6 for `atol`.
    vectorized : bool, optional
        Whether `fun` is implemented in a vectorized fashion. Default is False.

    Attributes
    ----------
    n : int
        Number of equations.
    status : string
        Current status of the solver: 'running', 'finished' or 'failed'.
    t_bound : float
        Boundary time.
    direction : float
        Integration direction: +1 or -1.
    t : float
        Current time.
    y : ndarray
        Current state.
    t_old : float
        Previous time. None if no steps were made yet.
    step_size : float
        Size of the last successful step. None if no steps were made yet.
    nfev : int
        Number evaluations of the system's right-hand side.
    njev : int
        Number of evaluations of the Jacobian. Is always 0 for this solver
        as it does not use the Jacobian.
    nlu : int
        Number of LU decompositions. Is always 0 for this solver.
    """

    cdef public:
        double rtol, atol
        double[::1] rtol_ary, atol_ary
        bint tols_are_scalar
        double max_step, h_abs, h_previous, error_exponent
        double[::1] y_old, y_new, f, f_new, _error, _scale
        double SAFETY, MAX_FACTOR, MIN_FACTOR
        double min_rel_step
        # defined by subclasses
        Py_ssize_t order, error_estimator_order, n_stages

    cdef:
        # defined by subclasses
        double[::1] C, B, E
        double[:, ::1] K, A, P

    def __init__(self, fun, t0, y0, t_bound, max_step=np.inf,
                 rtol=1e-3, atol=1e-6, vectorized=False,
                 first_step=None, **extraneous):
        warn_extraneous(extraneous)
        super().__init__(fun, t0, y0, t_bound, vectorized,
                         support_complex=True)

        self.y_old = np.empty_like(self.y)
        self.max_step = validate_max_step(max_step)

        rtol, atol = validate_tol(rtol, atol, self.n)
        if atol.ndim > 0:
            self.tols_are_scalar = False
            self.atol_ary = atol
            self.rtol_ary = rtol * np.ones_like(atol)
            self.atol = 0.
            self.rtol = 0.
        else:
            self.tols_are_scalar = True
            self.atol = atol
            self.rtol = rtol

        self.f = self.fun(self.t, self.y.base)
        if first_step is None:
            self.h_abs = select_initial_step(
                lambda t, y: np.asarray(self.fun(t, np.asarray(y))),
                self.t, self.y, self.f, self.direction,
                self.error_estimator_order, self.rtol, self.atol)
        else:
            self.h_abs = validate_first_step(first_step, t0, t_bound)
        self.K = np.empty((self.n_stages + 1, self.n), dtype=self.dtype)
        self.error_exponent = -1 / (self.error_estimator_order + 1)
        self.h_previous = 0.
        self.y_new = np.empty_like(self.y)
        self.f_new = np.empty_like(self.f)
        self._error = np.empty_like(self.f)
        self._scale = np.empty_like(self.f)

        self.min_rel_step = self.direction * 10. * np.finfo(self.dtype).eps
        self.n_accept = 0
        self.n_reject = 0

        # Multiply steps computed from asymptotic behaviour of errors by this.
        self.SAFETY = 0.9
        self.MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
        self.MAX_FACTOR = 10  # Maximum allowed increase in a step size.

    # grant Python-side access to Butcher tableau arrays, converted to numpy arrays

    @property
    def A(self):
        return np.array(self.A)

    @property
    def B(self):
        return np.array(self.B)

    @property
    def C(self):
        return np.array(self.C)

    @property
    def E(self):
        return np.array(self.E)

    @property
    def P(self):
        return np.array(self.P)

    # for testing
    def estimate_error(self, h):
        error = np.empty(self.n, self.dtype)
        self._estimate_error(h, error)
        return error

    def estimate_error_norm(self, h, scale):
        error = np.empty(self.n, self.dtype)
        return self._estimate_error_norm(h, scale, error)

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef _estimate_error(self, double h, double[::1] error):
        cdef:
            double K_dot_E
            Py_ssize_t i, j

        for i in range(self.n):
            K_dot_E = 0.
            for j in range(self.n_stages+1):
                K_dot_E += self.K[j, i] * self.E[j]
            error[i] = h * K_dot_E

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _estimate_error_norm(self, double h, double[::1] scale, double[::1] error):
        cdef:
            double _norm = 0.
            Py_ssize_t i

        self._estimate_error(h, error)  # FIXME: fuse with next looop

        for i in range(self.n):
            _norm += (error[i] / scale[i])**2

        return (_norm / self.n)**0.5

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef _compute_scale(self, double[::1] y, double[::1] y_new, double[::1] scale):
        cdef:
            Py_ssize_t i
            double size

        for i in range(self.n):
            size = max(abs(y[i]), abs(y_new[i]))
            if self.tols_are_scalar:
                scale[i] = self.atol + self.rtol * size
            else:
                scale[i] = self.atol_ary[i] + self.rtol_ary[i] * size

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef _step_impl(self):
        cdef:
            double t = self.t
            double max_step = self.max_step
            double min_step = abs(t * self.min_rel_step)
            double h_abs, h, t_new, error_norm, factor

        if self.h_abs > max_step:
            h_abs = max_step
        elif self.h_abs < min_step:
            h_abs = min_step
        else:
            h_abs = self.h_abs

        cdef bint step_accepted = False
        cdef bint step_rejected = False

        while not step_accepted:
            if h_abs < min_step:
                return False, self.TOO_SMALL_STEP

            h = h_abs * self.direction
            t_new = t + h

            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound

            h = t_new - t
            h_abs = abs(h)

            self.rk_step(t, self.y, self.f, h, self.y_new, self.f_new)
            self._compute_scale(self.y, self.y_new, self._scale)
            error_norm = self._estimate_error_norm(h, self._scale, self._error)

            if error_norm < 1:
                if error_norm == 0:
                    factor = self.MAX_FACTOR
                else:
                    factor = min(self.MAX_FACTOR,
                                 self.SAFETY * error_norm ** self.error_exponent)

                if step_rejected:
                    factor = min(1, factor)

                h_abs *= factor

                self.n_accept += 1
                step_accepted = True
            else:
                h_abs *= max(self.MIN_FACTOR,
                             self.SAFETY * error_norm ** self.error_exponent)
                self.n_reject += 1
                step_rejected = True

        self.h_previous = h
        self.t = t_new
        self.h_abs = h_abs

        cdef Py_ssize_t i
        for i in range(self.n):
            self.y_old[i] = self.y[i]
            self.y[i] = self.y_new[i]
            self.f[i] = self.f_new[i]

        return True, None

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef rk_step(self, t: double, y: double[::1], f: double[::1], h: double,
                 y_new: double[::1], f_new: double[::1]):
        """Perform a single Runge-Kutta step.

        This function computes a prediction of an explicit Runge-Kutta method and
        also estimates the error of a less accurate method.

        Notation for Butcher tableau is as in [1]_.

        Some of the below are class attributes, not passed parameters.

        Parameters
        ----------
        fun : callable
            Right-hand side of the system.
        t : float
            Current time.
        y : ndarray, shape (n,)
            Current state.
        f : ndarray, shape (n,)
            Current value of the derivative, i.e., ``fun(x, y)``.
        h : float
            Step to use.
        A : ndarray, shape (n_stages, n_stages)
            Coefficients for combining previous RK stages to compute the next
            stage. For explicit methods the coefficients at and above the main
            diagonal are zeros.
        B : ndarray, shape (n_stages,)
            Coefficients for combining RK stages for computing the final
            prediction.
        C : ndarray, shape (n_stages,)
            Coefficients for incrementing time for consecutive RK stages.
            The value for the first stage is always zero.
        K : ndarray, shape (n_stages + 1, n)
            Storage array for putting RK stages here. Stages are stored in rows.
            The last row is a linear combination of the previous rows with
            coefficients

        Returns
        -------
        y_new : ndarray, shape (n,)
            Solution at t + h computed with a higher accuracy.
        f_new : ndarray, shape (n,)
            Derivative ``fun(t + h, y_new)``.

        References
        ----------
        .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
                Equations I: Nonstiff Problems", Sec. II.4.
        """
        cdef:
            Py_ssize_t i, j, s
            double _dydh, t_new
            double[::1] _f

        for i in range(self.n):
            self.K[0, i] = f[i]

        for s in range(1, self.n_stages):
            for i in range(self.n):
                _dydh = 0.
                for j in range(s):
                    _dydh += self.K[j, i] * self.A[s, j]
                y_new[i] = y[i] + h * _dydh

            t_new = t + self.C[s] * h
            _f = self.fun(t_new, y_new.base)
            for i in range(self.n):
                self.K[s, i] = _f[i]

        for i in range(self.n):
            _dydh = 0.
            for s in range(self.n_stages):
                _dydh += self.K[s, i] * self.B[s]
            y_new[i] = y[i] + h * _dydh

        _f = self.fun(t + h, y_new.base)
        for i in range(self.n):
            f_new[i] = _f[i]
            self.K[self.n_stages, i] = f_new[i]

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef _dense_output_impl(self):
        cdef:
            double K_dot_P
            Py_ssize_t i, j, k
            double[:, ::1] Q = np.empty((self.n, self.P.shape[1]), self.dtype)

        for i in range(self.n):
            for k in range(self.P.shape[1]):
                K_dot_P = 0.
                for j in range(self.n_stages+1):
                    K_dot_P += self.K[j, i] * self.P[j, k]
                Q[i, k] = K_dot_P

        return RkDenseOutput(self.t_old, self.t, self.y_old, Q)


cdef class RK23(RungeKutta):
    __doc__ = f"""Explicit Runge-Kutta method of order 3(2).

    This uses the Bogacki-Shampine pair of formulas [1]_. The error is controlled
    assuming accuracy of the second-order method, but steps are taken using the
    third-order accurate formula (local extrapolation is done). A cubic Hermite
    polynomial is used for the dense output.

    Can be applied in the complex domain.

    {RungeKutta.__doc__}

    References
    ----------
    .. [1] P. Bogacki, L.F. Shampine, "A 3(2) Pair of Runge-Kutta Formulas",
           Appl. Math. Lett. Vol. 2, No. 4. pp. 321-325, 1989.
    """

    def __init__(self, fun, t0, y0, t_bound, max_step=np.inf,
                 rtol=1e-3, atol=1e-6, vectorized=False,
                 first_step=None, **extraneous):
        self.order = 3
        self.error_estimator_order = 2
        self.n_stages = 3
        self.C = np.array([0, 1/2, 3/4])
        self.A = np.array([
            [0, 0, 0],
            [1/2, 0, 0],
            [0, 3/4, 0]
        ])
        self.B = np.array([2/9, 1/3, 4/9])
        self.E = np.array([5/72, -1/12, -1/9, 1/8])
        self.P = np.array([
            [1, -4 / 3, 5 / 9],
            [0, 1, -2/3],
            [0, 4/3, -8/9],
            [0, -1, 1]
        ])

        super().__init__(fun, t0, y0, t_bound, max_step, rtol, atol,
                         vectorized, first_step, **extraneous)


cdef class RK45(RungeKutta):
    __doc__ = f"""Explicit Runge-Kutta method of order 5(4).

    This uses the Dormand-Prince pair of formulas [1]_. The error is controlled
    assuming accuracy of the fourth-order method accuracy, but steps are taken
    using the fifth-order accurate formula (local extrapolation is done).
    A quartic interpolation polynomial is used for the dense output [2]_.

    Can be applied in the complex domain.

    {RungeKutta.__doc__}

    References
    ----------
    .. [1] J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta
           formulae", Journal of Computational and Applied Mathematics, Vol. 6,
           No. 1, pp. 19-26, 1980.
    .. [2] L. W. Shampine, "Some Practical Runge-Kutta Formulas", Mathematics
           of Computation,, Vol. 46, No. 173, pp. 135-150, 1986.
    """

    def __init__(self, fun, t0, y0, t_bound, max_step=np.inf,
                 rtol=1e-3, atol=1e-6, vectorized=False,
                 first_step=None, **extraneous):
        self.order = 5
        self.error_estimator_order = 4
        self.n_stages = 6

        self.C = np.array([0, 1/5, 3/10, 4/5, 8/9, 1])
        self.A = np.array([
            [0, 0, 0, 0, 0],
            [1/5, 0, 0, 0, 0],
            [3/40, 9/40, 0, 0, 0],
            [44/45, -56/15, 32/9, 0, 0],
            [19372/6561, -25360/2187, 64448/6561, -212/729, 0],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]
        ])
        self.B = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])
        self.E = np.array([-71/57600, 0, 71/16695, -71/1920, 17253/339200, -22/525,
                      1/40])
        # Corresponds to the optimum value of c_6 from [2]_.
        self.P = np.array([
            [1, -8048581381/2820520608, 8663915743/2820520608,
             -12715105075/11282082432],
            [0, 0, 0, 0],
            [0, 131558114200/32700410799, -68118460800/10900136933,
             87487479700/32700410799],
            [0, -1754552775/470086768, 14199869525/1410260304,
             -10690763975/1880347072],
            [0, 127303824393/49829197408, -318862633887/49829197408,
             701980252875 / 199316789632],
            [0, -282668133/205662961, 2019193451/616988883, -1453857185/822651844],
            [0, 40617522/29380423, -110615467/29380423, 69997945/29380423]])

        super().__init__(fun, t0, y0, t_bound, max_step, rtol, atol,
                         vectorized, first_step, **extraneous)


cdef class DOP853(RungeKutta):
    __doc__ = f"""Explicit Runge-Kutta method of order 8.

    This is a Python implementation of "DOP853" algorithm originally written
    in Fortran [1]_, [2]_. Note that this is not a literate translation, but
    the algorithmic core and coefficients are the same.

    Can be applied in the complex domain.

    {RungeKutta.__doc__}

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.
    .. [2] `Page with original Fortran code of DOP853
            <http://www.unige.ch/~hairer/software.html>`_.
    """

    cdef:
        double[::1] E3, E5, C_EXTRA
        double[:, ::1] D, A_EXTRA, K_extended
        Py_ssize_t n_stages_extended

    def __init__(self, fun, t0, y0, t_bound, max_step=np.inf,
                 rtol=1e-3, atol=1e-6, vectorized=False,
                 first_step=None, **extraneous):
        self.n_stages = dop853_coefficients.N_STAGES
        self.n_stages_extended = dop853_coefficients.N_STAGES_EXTENDED
        self.order = 8
        self.error_estimator_order = 7
        # copy to ensure contiguous
        self.A = dop853_coefficients.A  # [:self.n_stages, :self.n_stages].copy()
        self.B = dop853_coefficients.B.copy()
        self.C = dop853_coefficients.C  # [:self.n_stages].copy()
        self.E3 = dop853_coefficients.E3.copy()
        self.E5 = dop853_coefficients.E5.copy()
        self.D = dop853_coefficients.D.copy()

        self.A_EXTRA = dop853_coefficients.A[self.n_stages + 1:].copy()
        self.C_EXTRA = dop853_coefficients.C[self.n_stages + 1:].copy()

        super().__init__(fun, t0, y0, t_bound, max_step, rtol, atol,
                         vectorized, first_step, **extraneous)
        self.K_extended = np.empty(
            (self.n_stages_extended, self.n), dtype=self.dtype)
        self.K = self.K_extended[:self.n_stages + 1]

    @property
    def D(self):
        return np.array(self.D)

    @property
    def E3(self):
        return np.array(self.E3)

    @property
    def E5(self):
        return np.array(self.E5)

    # for testing only, unused in solver
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef _estimate_error(self, double h, double[::1] error):
        # err5 = np.dot(K.T, self.E5)
        # err3 = np.dot(K.T, self.E3)
        # denom = np.hypot(np.abs(err5), 0.1 * np.abs(err3))
        # correction_factor = np.ones_like(err5)
        # mask = denom > 0
        # correction_factor[mask] = np.abs(err5[mask]) / denom[mask]
        # return h * err5 * correction_factor
        cdef:
            double err_3, err_5, denom
            Py_ssize_t i

        for i in range(self.n):
            err_3 = 0.
            err_5 = 0.
            for j in range(self.n_stages+1):
                err_3 += self.K[j, i] * self.E3[j]
                err_5 += self.K[j, i] * self.E5[j]

            error[i] = h * err_5

            denom = (abs(err_3)**2 + 0.01 * abs(err_3)**2)**0.5
            if denom > 0:
                error[i] *= abs(err_5) / denom

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double _estimate_error_norm(self, double h, double[::1] scale, double[::1] error):
        cdef:
            double err_norm_3 = 0.
            double err_norm_5 = 0.
            double tmp3, tmp5, denom
            Py_ssize_t i, j

        for i in range(self.n):
            tmp3 = 0.
            tmp5 = 0.
            for j in range(self.n_stages+1):
                tmp3 += self.K[j, i] * self.E3[j]
                tmp5 += self.K[j, i] * self.E5[j]
            err_norm_3 += (tmp3 / scale[i])**2
            err_norm_5 += (tmp5 / scale[i])**2

        if err_norm_3 == 0. and err_norm_5 == 0.:
            return 0.0
        else:
            denom = err_norm_5 + 0.01 * err_norm_3
            return abs(h) * err_norm_5 / (denom * self.n)**0.5

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef _dense_output_impl(self):
        cdef:
            Py_ssize_t i, j, s
            double _dydh, f_old, delta_y, dot
            double h = self.h_previous
            double[::1] _f
            double[:, ::1] F = np.empty(
                (dop853_coefficients.INTERPOLATOR_POWER, self.n), self.dtype)

        for s in range(self.n_stages + 1, self.K_extended.shape[0]):
            for i in range(self.n):
                _dydh = 0.
                for j in range(s):
                    _dydh += self.K_extended[j, i] * self.A[s, j]
                self.y_new[i] = self.y[i] + h * _dydh

            _f = self.fun(self.t_old + self.C[s] * h, self.y_new.base)
            for i in range(self.n):
                self.K_extended[s, i] = _f[i]

        for i in range(self.n):
            f_old = self.K_extended[0, i]
            delta_y = self.y[i] - self.y_old[i]

            F[0, i] = delta_y
            F[1, i] = h * f_old - delta_y
            F[2, i] = 2 * delta_y - h * (self.f[i] + f_old)

            for j in range(3, F.shape[0]):
                dot = 0
                for k in range(self.n_stages_extended):
                    dot += self.D[j-3, k] * self.K_extended[k, i]
                F[j, i] = h * dot

        return Dop853DenseOutput(self.t_old, self.t, self.y_old, F)

    # def _dense_output_impl(self):
    #     K = self.K_extended
    #     h = self.h_previous
    #     for s, (a, c) in enumerate(zip(self.A_EXTRA, self.C_EXTRA),
    #                                start=self.n_stages + 1):
    #         dy = np.dot(K[:s].T, a[:s]) * h
    #         K[s] = self.fun(self.t_old + c * h, self.y_old + dy)

    #     F = np.empty((dop853_coefficients.INTERPOLATOR_POWER, self.n),
    #                  dtype=self.y_old.dtype)

    #     f_old = K[0]
    #     delta_y = np.array(self.y) - np.array(self.y_old)

    #     F[0] = delta_y
    #     F[1] = h * f_old - delta_y
    #     F[2] = 2 * delta_y - h * (self.f + f_old)
    #     F[3:] = h * np.dot(self.D, K)

    #     return Dop853DenseOutput(self.t_old, self.t, self.y_old, F)


class RkDenseOutput(DenseOutput):
    def __init__(self, t_old, t, y_old, Q):
        super().__init__(t_old, t)
        self.h = t - t_old
        self.Q = Q
        self.order = Q.shape[1] - 1
        self.y_old = y_old

    def _call_impl(self, t):
        x = (t - self.t_old) / self.h
        if t.ndim == 0:
            p = np.tile(x, self.order + 1)
            p = np.cumprod(p)
        else:
            p = np.tile(x, (self.order + 1, 1))
            p = np.cumprod(p, axis=0)
        y = self.h * np.dot(self.Q, p)
        if y.ndim == 2:
            y += self.y_old[:, None]
        else:
            y += self.y_old

        return y


class Dop853DenseOutput(DenseOutput):
    def __init__(self, t_old, t, y_old, F):
        super().__init__(t_old, t)
        self.h = t - t_old
        self.F = F
        self.y_old = y_old

    def _call_impl(self, t):
        x = (t - self.t_old) / self.h

        if t.ndim == 0:
            y = np.zeros_like(self.y_old)
        else:
            x = x[:, None]
            y = np.zeros((len(x), len(self.y_old)), dtype=self.y_old.dtype)

        for i, f in enumerate(reversed(self.F)):
            y += f
            if i % 2 == 0:
                y *= x
            else:
                y *= 1 - x
        y += self.y_old

        return y.T
