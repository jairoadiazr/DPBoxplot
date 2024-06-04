import numpy as np
from copy import deepcopy

def get_epsilon(n_quantiles, epsilon, swap, cdp):
    layers = np.log2(n_quantiles) + 1

    if swap:
        composition = 2 * layers
    else:
        composition = layers

    if cdp:
        epsilon = np.sqrt((2 * epsilon) / composition)
    else:
        epsilon = epsilon / composition

    return epsilon


def split_by_number(array, m):
    return array[array <= m], array[array >= m]


def gaussian_noise(array, bounds, scale=0.00001):
    data = deepcopy(array)
    return np.sort(data + np.random.normal(0, scale, len(array))), (bounds[0] - 4 * scale, bounds[1] + 4 * scale)


def approximate_quantiles_algo(array, quantiles, bounds, epsilon, swap=False, cdp=False):
    epsilon = get_epsilon(len(quantiles), epsilon, swap, cdp)
    array, bounds = gaussian_noise(array, bounds)

    def algo_helper(array, quantiles, bounds):
        m = len(quantiles)
        a, b = bounds
        if m == 0:
            return []

        if m == 1:
            return [
                single_quantile(array, bounds, quantiles[0], epsilon=epsilon, swap=False)]

        q_mid = quantiles[m // 2]
        v = single_quantile(array, bounds, q_mid, epsilon=epsilon, swap=False)
        d_l, d_u = split_by_number(array, v)
        q_l, q_u = np.array_split(quantiles[quantiles != q_mid], 2)
        q_l, q_u = q_l / q_mid, (q_u - q_mid) / (1 - q_mid)

        return algo_helper(d_l, q_l, (a, v)) + [v] + algo_helper(d_u, q_u, (v, b))

    return algo_helper(np.sort(array), quantiles, bounds)
    
def racing_sample(log_terms):
    """Numerically stable method for sampling from an exponential distribution.

    Args:
      log_terms: Array of terms of form log(coefficient) - (exponent term).

    Returns:
      A sample from the exponential distribution determined by terms. See
      Algorithm 1 from the paper "Duff: A Dataset-Distance-Based
      Utility Function Family for the Exponential Mechanism"
      (https://arxiv.org/pdf/2010.04235.pdf) for details; each element of terms is
      analogous to a single log(lambda(A_k)) - (eps * k/2) in their algorithms.
    """
    return np.argmin(np.log(np.log(1.0 / np.random.uniform(size=log_terms.shape))) - log_terms)


def single_quantile(sorted_array, bounds, quantile, epsilon, swap):
    a, d = bounds
    n = len(sorted_array)
    sorted_array = np.clip(sorted_array, a, d)
    sorted_array = np.concatenate(([a], sorted_array, [d]))
    intervals = np.diff(sorted_array)

    sensitivity = max(quantile, 1 - quantile)
    if swap:
        sensitivity = 1.0

    utility = -np.abs(np.arange(0, n + 1) - (quantile * n))
    idx_left = racing_sample(np.log(intervals) + (epsilon / (2.0 * sensitivity) * utility))
    return np.random.uniform(sorted_array[idx_left], sorted_array[idx_left + 1])