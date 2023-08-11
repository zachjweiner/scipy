cdef class OdeSolver:
    cdef public:
        double t_old, t, t_bound
        bint vectorized
        double direction
        Py_ssize_t n
        str status
        int nfev, njev, nlu, n_accept, n_reject
        dtype
        _fun, fun, fun_single, fun_vectorized

    cdef:
        double[::1] y
