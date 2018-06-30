from autograd.extend import defvjp
import autograd.numpy as np
from .sparse_wrapper import dot

# I am very sure that I have not done this correctly.
#
# def dot_left_vjp(ans, sparse, dense):
#     return lambda g: g * dot(sparse, dense)
#
# def dot_right_vjp(ans, dense, sparse):
#     return lambda g: g * dot(sparse, dense)
# defvjp(dot, dot_left_vjp)


def _dot_vjp_0(g, ans, sparse, dense):
    if max(anp.ndim(sparse), anp.ndim(dense)) > 2:
        raise NotImplementedError("Current dot vjps only support ndim <= 2.")

    if anp.ndim(sparse) == 0:
        return anp.sum(dense * g)
    if anp.ndim(sparse) == 1 and anp.ndim(dense) == 1:
        return g * dense
    if anp.ndim(sparse) == 2 and anp.ndim(dense) == 1:
        return g[:, None] * dense
    if anp.ndim(sparse) == 1 and anp.ndim(dense) == 2:
        return anp.dot(dense, g)
    return dot(g, dense.T)


def _dot_vjp_1(g, ans, sparse, dense):
    if max(anp.ndim(sparse), anp.ndim(dense)) > 2:
        raise NotImplementedError("Current dot vjps only support ndim <= 2.")

    if anp.ndim(dense) == 0:
        return anp.sum(sparse * g)
    if anp.ndim(sparse) == 1 and anp.ndim(dense) == 1:
        return g * sparse
    if anp.ndim(sparse) == 2 and anp.ndim(dense) == 1:
        return anp.dot(g, sparse)
    if anp.ndim(sparse) == 1 and anp.ndim(dense) == 2:
        return sparse[:, None] * g
    return anp.dot(sparse.T, g)

defvjp(dot, _dot_vjp_0, None)

