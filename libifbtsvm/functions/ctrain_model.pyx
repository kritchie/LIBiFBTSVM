
import numpy as np

cimport numpy as np


cdef ctrain_model(int max_iter, float epsilon, np.ndarray H, np.ndarray G, int C, np.ndarray CCx):

    cdef int i, j
    cdef int lenH = np.size(H, axis=1)
    cdef int length_g = len(G)
    cdef int iterations = 0

    cdef float _proj_grad_max_old = float('inf')
    cdef float _proj_grad_min_old = float('-inf')

    cdef list projected_grads = []
    cdef list _keep = []
    cdef list Q = []

    cdef np.ndarray identity_matrix = np.eye(lenH, dtype=int)
    cdef np.ndarray weights = np.zeros((lenH, 1))
    cdef np.ndarray x_new = np.arange(length_g)
    cdef np.ndarray x_old = np.arange(length_g)

    cdef np.ndarray alpha_new = np.zeros(length_g)
    cdef np.ndarray alpha_old = np.zeros(length_g)
    cdef np.ndarray _substr_buf = np.zeros_like(weights)

    # TODO : choose linear solver, for now use Numpy for simplicity
    identity_matrix[-1][-1] = 0
    cdef np.ndarray _Q = np.linalg.solve((np.dot(np.transpose(H), H) + C * identity_matrix), np.transpose(G))

    for i in range(length_g):
        Q.append(np.dot(G[i, :], _Q[:, i]))

    for i in range(max_iter):
        _proj_grad_max_new = float('-inf')
        _proj_grad_min_new = float('inf')

        np.random.shuffle(x_old)

        for j in range(len(x_old)):

            pos = x_old[j]
            _grad = -np.matmul(G[pos, :], weights) - 1
            gradient = 0

            if alpha_new[pos] == 0:

                if _grad > _proj_grad_max_old:
                    continue

                elif _grad < 0:
                    _keep.append(pos)
                    gradient = _grad

            elif alpha_new[pos] == CCx[pos]:

                if _grad < _proj_grad_min_old:
                    continue

                elif _grad > 0:
                    _keep.append(pos)
                    gradient = _grad

            else:
                gradient = _grad

            _proj_grad_max_new = np.maximum(_proj_grad_max_new, gradient)
            _proj_grad_min_new = np.minimum(_proj_grad_min_new, gradient)

            if np.absolute(gradient) > 1e-12:
                alpha_old[pos] = alpha_new[pos]
                alpha_new[pos] = np.minimum(np.maximum(alpha_new[pos] - _grad / Q[pos], 0), CCx[pos])

                weights_aux = _Q[:, pos] * (alpha_new[pos] - alpha_old[pos])
                weights_aux = np.expand_dims(weights_aux, axis=1)
                weights = np.subtract(weights, weights_aux, out=_substr_buf)

                if gradient != 0:
                    projected_grads.append(gradient)

        x_old = x_new[_keep]

        iterations += 1

        if _proj_grad_max_new - _proj_grad_min_new <= epsilon:
            if len(x_old) == length_g:
                break

            else:
                x_old = np.arange(length_g)
                x_new = np.arange(length_g)
                _proj_grad_max_old = float('inf')
                _proj_grad_min_old = float('-inf')

        _proj_grad_max_old = float('inf') if _proj_grad_max_new <= 0 else _proj_grad_max_new
        _proj_grad_min_old = float('-inf') if _proj_grad_min_new >= 0 else _proj_grad_min_new

    return alpha_new, weights, iterations, projected_grads


def train_model(max_iter, epsilon, H, G, C, CCx):
    return ctrain_model(max_iter, epsilon, H, G, C, CCx)
