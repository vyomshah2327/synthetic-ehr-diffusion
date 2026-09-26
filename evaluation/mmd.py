import numpy as np


def _sq_dists(X, Y):
    """Squared euclidean distances between rows of X and Y, without
    materialising an (n, m, d) intermediate."""
    XX = (X ** 2).sum(axis=1, keepdims=True)
    YY = (Y ** 2).sum(axis=1, keepdims=True)
    d = XX + YY.T - 2 * X @ Y.T
    np.maximum(d, 0, out=d)  # clip small negatives from float error
    return d


def _rbf_kernel(X, Y, sigma):
    """RBF kernel between rows of X and Y."""
    return np.exp(-_sq_dists(X, Y) / (2 * sigma ** 2))


def mmd_score(real: np.ndarray, synthetic: np.ndarray, subsample=2000,
              seed=0) -> float:
    """
    Maximum Mean Discrepancy with RBF kernel (median heuristic for sigma).
    Lower = distributions are closer = better.

    real, synthetic: (N, seq_len, n_features)
    """
    rng = np.random.default_rng(seed)

    n = min(subsample, len(real), len(synthetic))
    idx_r = rng.choice(len(real), n, replace=False)
    idx_s = rng.choice(len(synthetic), n, replace=False)

    X = real[idx_r].reshape(n, -1).astype(np.float64)
    Y = synthetic[idx_s].reshape(n, -1).astype(np.float64)

    # median heuristic for bandwidth.
    # NOTE: computed via _sq_dists rather than broadcasting. The naive
    # all_data[:, None] - all_data[None, :] builds a (2n, 2n, d) array,
    # which for n=2000, d=840 is ~107 GB and OOMs the session.
    all_data = np.concatenate([X, Y], axis=0)
    pairwise = _sq_dists(all_data, all_data)
    sigma = np.sqrt(np.median(pairwise[pairwise > 0]) / 2)
    del pairwise, all_data

    K_xx = _rbf_kernel(X, X, sigma)
    K_yy = _rbf_kernel(Y, Y, sigma)
    K_xy = _rbf_kernel(X, Y, sigma)

    mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    return float(mmd)