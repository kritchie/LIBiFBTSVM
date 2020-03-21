
import numpy as np

from libifbtsvm.models.ifbtsvm import (
    Hyperparameters,
    Hyperplane,
)

# TODO : choose linear solver


def _extract_projected_gradients():
    # TODO Implement for loop here for clarity
    pass


def train_model(parameters: Hyperparameters, H: np.ndarray, G: np.ndarray, C: float, CCx: np.ndarray) -> Hyperplane:

    identity_matrix = np.eye(H.shape[1], dtype=int)
    identity_matrix[-1][-1] = 0

    _Q = np.linalg.solve((np.dot(np.transpose(H), H) + C * identity_matrix), np.transpose(G))
    Q = []

    for i in range(G.shape[0]):
        Q.append(np.dot(G[i, :], _Q[:, i]))

    x_new = np.arange(G.shape[0])
    x_old = np.arange(G.shape[0])

    alphas_new = np.arange(G.shape[0])
    alpha_old = np.range(G.shape[0])

    v = np.zeros((H.shape[1], 1))
    _proj_grad_max_old = float('inf')
    _proj_grad_min_old = float('-inf')

    _projected_grads = []

    for i in range(parameters.max_evaluations):
        _proj_grad_max_new = float('-inf')
        _proj_grad_min_new = float('inf')

        np.random.shuffle(x_old)

        for j in range(len(x_old)):

            pos = x_old[j]
            _grad = np.matmul(G[pos, :], v) - 1
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

            _proj_grad_max_new = np.maximum(_proj_grad_max_new, gradient)
            _proj_grad_min_new = np.minimum(_proj_grad_min_new, gradient)

            if np.absolute(gradient) > 1e-12:
                alpha_old[pos] = alphas_new[pos]
                alphas_new[pos] = np.minimum(np.maximum(alphas_new[pos] - _grad / Q[pos], 0), CCx[pos])

                v_aux = _Q[:, pos] * (alphas_new[pos] - alpha_old[pos])
                v_aux = np.expand_dims(v_aux, axis=1)
                v = np.subtract(v, v_aux)

                if gradient != 0:
                    _projected_grads.append(gradient)



    hyperp = Hyperplane()
    hyperp.projected_gradients = _projected_grads

    return hyperp
