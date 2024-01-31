
import numpy as np

from libifbtsvm.models.ifbtsvm import (
    Hyperparameters,
    Hyperplane,
)


def train_model(parameters: Hyperparameters, H: np.ndarray, G: np.ndarray, C: float, CCx: np.ndarray) -> Hyperplane:

    identity_matrix = np.eye(H.shape[1], dtype=int)
    identity_matrix[-1][-1] = 0

    _Q = np.linalg.solve((np.dot(np.transpose(H), H) + C * identity_matrix), np.transpose(G))

    length_g = len(G)
    Q = (G * _Q.T).sum(axis=1)

    ref_old = np.arange(length_g)

    x_new = np.copy(ref_old)
    x_old = np.copy(ref_old)

    alphas_new = np.zeros(length_g)

    weights = np.zeros((H.shape[1],))
    _proj_grad_max_old = float('inf')
    _proj_grad_min_old = float('-inf')

    _projected_grads = []
    _keep = []

    iterations = 0

    _substract_buffer = np.zeros_like(weights)

    for i in range(parameters.max_iter):

        _proj_grad_max_new = float('-inf')
        _proj_grad_min_new = float('inf')

        weights_deltas = []
        for j in range(len(x_old)):

            pos = x_old[j]
            _grad = -np.matmul(G[pos, :], weights) - 1
            gradient = 0

            if alphas_new[pos] == 0:

                if _grad > _proj_grad_max_old:
                    continue

                elif _grad < 0:
                    _keep.append(pos)
                    gradient = _grad

            elif alphas_new[pos] == CCx[pos]:

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
                alpha_old = alphas_new[pos]
                alphas_new[pos] = np.minimum(np.maximum(alphas_new[pos] - _grad / Q[pos], 0), CCx[pos])

                weights_aux = np.multiply(_Q[:, pos], (alphas_new[pos] - alpha_old))

                np.subtract(weights, weights_aux, out=_substract_buffer)
                weights = _substract_buffer
                weights_deltas.append(weights_aux)

                if gradient != 0:
                    _projected_grads.append(gradient)

        x_old = x_new[_keep]
        iterations += 1

        if _proj_grad_max_new - _proj_grad_min_new <= 1e-3:
            if len(x_old) == length_g:
                break

            else:
                x_old = np.copy(ref_old)
                x_new = np.copy(ref_old)
                _proj_grad_max_old = float('inf')
                _proj_grad_min_old = float('-inf')

        _proj_grad_max_old = float('inf') if _proj_grad_max_new <= 0 else _proj_grad_max_new
        _proj_grad_min_old = float('-inf') if _proj_grad_min_new >= 0 else _proj_grad_min_new

        # Verify stop condition
        # at this point we assume convergence, this was found empirically
        # letting the model train after this point will not yield better results
        # in terms of accuracy
        if weights_deltas:
            weights_deltas = np.asarray(weights_deltas)
            mean = np.mean(weights_deltas)
            if np.abs(mean) <= 1e-6:
                break

    hyperplane = Hyperplane(alpha=alphas_new, weights=weights,
                            iterations=iterations, proj_gradients=_projected_grads)

    return hyperplane
