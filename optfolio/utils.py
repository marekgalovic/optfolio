import numba as nb


@nb.jit(nopython=True)
def is_close_to_zero(x):
    return x <= 1e-8
