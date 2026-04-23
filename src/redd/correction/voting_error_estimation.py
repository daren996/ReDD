import math

import numpy as np


def estimate_mv_error(y_gt, preds, k=3, condition=None):
    if condition is not None:
        mask = y_gt == condition
        y_gt = y_gt[mask]
        preds = preds[mask, :]

    n_samples = y_gt.shape[0]
    err = (preds != y_gt[:, np.newaxis]).astype(float)
    l_bar = np.mean(err)

    d01 = (preds[:, 0] != preds[:, 1]).astype(float)
    d02 = (preds[:, 0] != preds[:, 2]).astype(float)
    d12 = (preds[:, 1] != preds[:, 2]).astype(float)
    d_bar = np.mean(np.concatenate([d01, d02, d12]))

    eps_numerators = []
    for i in range(n_samples):
        eps_num = 0.0
        for j in range(3):
            for jp in range(3):
                indicator_diff = 1.0 if preds[i, j] != preds[i, jp] else 0.0
                eps_num += 0.5 * (err[i, j] + err[i, jp]) * indicator_diff
        eps_numerators.append(eps_num / 9.0)
    eps_avg = np.mean(eps_numerators)
    epsilon_3 = eps_avg / (2 * l_bar) if l_bar > 0 else 0.0

    w = np.mean(err, axis=1)
    numerator = np.mean(w > 0.5)
    denominator = np.mean(w**2)
    eta3 = numerator / denominator if denominator > 0 else 0.0

    bracket = l_bar + (3 * (k - 1)) / (2 * k) * (epsilon_3 * l_bar - 0.5 * d_bar)
    return eta3 * bracket


def estimate_mv_error_fn(y_gt, preds, k=3):
    fn_conditional = estimate_mv_error(y_gt, preds, k=k, condition=1)
    p_y1 = np.mean(y_gt == 1)
    post_correction_error = fn_conditional * p_y1
    return round(post_correction_error, 4)


def chernoff_bound(n_classifiers, p_e):
    if p_e <= 0.0 or p_e >= 1.0:
        return 0.0
    delta = (1 / (2 * p_e)) - 1
    exponent = -((delta**2) * n_classifiers * p_e) / 2
    return math.exp(exponent)


class VotingErrorEstimation:
    estimate_mv_error = staticmethod(estimate_mv_error)
    estimate_mv_error_fn = staticmethod(estimate_mv_error_fn)
    chernoff_bound = staticmethod(chernoff_bound)
