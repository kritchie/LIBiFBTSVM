
import numpy as np

from libifbtsvm.models.ifbtsvm import (
    Hyperparameters,
    Hyperplane,
)


def train_model(parameters: Hyperparameters, H: np.ndarray, G: np.ndarray, C: float, CCx: np.ndarray) -> Hyperplane:

    identity_matrix = np.eye(H.shape[1], dtype=int)
    identity_matrix[-1][-1] = 0

    # TODO : choose linear solver, for now use Numpy for simplicity
    _Q = np.linalg.solve((np.dot(np.transpose(H), H) + C * identity_matrix), np.transpose(G))
    Q = []

    length_g = len(G)

    for i in range(length_g):
        Q.append(np.dot(G[i, :], _Q[:, i]))

    x_new = np.arange(length_g)
    x_old = np.arange(length_g)

    alphas_new = np.zeros(length_g)
    alpha_old = np.zeros(length_g)

    weights = np.zeros((H.shape[1], 1))
    _proj_grad_max_old = float('inf')
    _proj_grad_min_old = float('-inf')

    _projected_grads = []

    iterations = 0

    for i in range(parameters.max_evaluations):
        _proj_grad_max_new = float('-inf')
        _proj_grad_min_new = float('inf')

        np.random.shuffle(x_old)

        for j in range(len(x_old)):

            pos = x_old[j]
            _grad = -np.matmul(G[pos, :], weights) - 1
            gradient = 0

            if alphas_new[pos] == 0:

                if _grad > _proj_grad_max_old:
                    np.delete(x_new, pos)
                    continue

                elif _grad < 0:
                    gradient = _grad

            elif alphas_new[pos] == CCx[pos]:

                if _grad < _proj_grad_min_old:
                    np.delete(x_new, pos)
                    continue

                elif _grad > 0:
                    gradient = _grad

            else:
                gradient = _grad

            _proj_grad_max_new = np.maximum(_proj_grad_max_new, gradient)
            _proj_grad_min_new = np.minimum(_proj_grad_min_new, gradient)

            if np.absolute(gradient) > 1e-12:
                alpha_old[pos] = alphas_new[pos]
                alphas_new[pos] = np.minimum(np.maximum(alphas_new[pos] - _grad / Q[pos], 0), CCx[pos])

                weights_aux = _Q[:, pos] * (alphas_new[pos] - alpha_old[pos])
                weights_aux = np.expand_dims(weights_aux, axis=1)
                weights = np.subtract(weights, weights_aux)

                if gradient != 0:
                    _projected_grads.append(gradient)

        x_old = x_new

        iterations += 1

        if _proj_grad_max_new - _proj_grad_min_new <= parameters.epsilon:
            if len(x_old) == length_g:
                break

            else:
                x_old = np.arange(length_g)
                x_new = np.arange(length_g)
                _proj_grad_max_old = float('inf')
                _proj_grad_min_old = float('-inf')

        _proj_grad_max_old = float('inf') if _proj_grad_max_new <= 0 else _proj_grad_max_new
        _proj_grad_min_old = float('-inf') if _proj_grad_min_new >= 0 else _proj_grad_min_new

    hyperplane = Hyperplane(alpha=alphas_new, weights=weights,
                            iterations=iterations, proj_gradients=_projected_grads)

    return hyperplane
