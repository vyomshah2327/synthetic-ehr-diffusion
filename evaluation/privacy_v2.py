import numpy as np
from sklearn.neighbors import NearestNeighbors


def _nn_dist_cross(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """For each point in A, distance to its nearest neighbor in B (A != B)."""
    nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', n_jobs=-1).fit(B)
    dist, _ = nn.kneighbors(A)
    return dist[:, 0]


def _nn_dist_within(A: np.ndarray) -> np.ndarray:
    """For each point in A, distance to its nearest OTHER neighbor in A
    (excludes the point itself, which would otherwise be distance 0)."""
    nn = NearestNeighbors(n_neighbors=2, algorithm='ball_tree', n_jobs=-1).fit(A)
    dist, _ = nn.kneighbors(A)
    return dist[:, 1]


def nnaa_score_paper(real_train: np.ndarray, real_test: np.ndarray,
                      synthetic: np.ndarray, subsample=2000, seed=0):
    """Nearest-Neighbor Adversarial Accuracy (NNAA), matching the TimeDiff
    paper's exact definition (Appendix A.5.4, Eq. 15-20):

        AA_test  = 1/2 * ( mean_i[ d_ES(i) > d_EE(i) ] + mean_i[ d_SE(i) > d_SS(i) ] )
        AA_train = 1/2 * ( mean_i[ d_TS(i) > d_TT(i) ] + mean_i[ d_ST(i) > d_SS(i) ] )
        NNAA     = |AA_test - AA_train|

    where T=real_train, E=real_test, S=synthetic, d_XY(i) is the distance
    from point i in X to its nearest neighbor in Y (or nearest OTHER point
    within X itself for d_TT/d_SS/d_EE).

    On this scale, NNAA close to 0 is the target (paper's TimeDiff numbers:
    ~0.002-0.006). This is NOT the same scale as this repo's earlier
    nnaa_score_repo() below, which returns values centered on 0.5 -- the
    two are different metrics that happen to share a name; do not compare
    across them.

    real_train, real_test, synthetic: (N, seq_len, n_features)
    Returns: (aa_test, aa_train, nnaa) -- all floats.
    """
    rng = np.random.default_rng(seed)
    n = min(subsample, len(real_train), len(real_test), len(synthetic))

    Tr = real_train[rng.choice(len(real_train), n, replace=False)].reshape(n, -1).astype(np.float32)
    Te = real_test[rng.choice(len(real_test), n, replace=False)].reshape(n, -1).astype(np.float32)
    S = synthetic[rng.choice(len(synthetic), n, replace=False)].reshape(n, -1).astype(np.float32)

    d_TS = _nn_dist_cross(Tr, S)
    d_ST = _nn_dist_cross(S, Tr)
    d_ES = _nn_dist_cross(Te, S)
    d_SE = _nn_dist_cross(S, Te)

    d_TT = _nn_dist_within(Tr)
    d_SS = _nn_dist_within(S)
    d_EE = _nn_dist_within(Te)

    aa_test = 0.5 * ((d_ES > d_EE).mean() + (d_SE > d_SS).mean())
    aa_train = 0.5 * ((d_TS > d_TT).mean() + (d_ST > d_SS).mean())
    nnaa = abs(aa_test - aa_train)

    return float(aa_test), float(aa_train), float(nnaa)


def nnaa_score_repo(real_train: np.ndarray, real_test: np.ndarray,
                     synthetic: np.ndarray, subsample=2000) -> float:
    """ORIGINAL repo metric, kept for continuity/reference only -- this is
    NOT the paper's NNAA despite the shared name. Fraction of synthetic
    samples whose nearest neighbor is in real_test rather than real_train.
    On THIS scale, 0.5 = private (no bias toward either set), 1.0 = likely
    memorizing training data. Do not compare this number to the paper.

    real_train, real_test, synthetic: (N, seq_len, n_features)
    """
    n = min(subsample, len(synthetic), len(real_train), len(real_test))

    S = synthetic[:n].reshape(n, -1).astype(np.float32)
    Tr = real_train[:n].reshape(n, -1).astype(np.float32)
    Te = real_test[:n].reshape(n, -1).astype(np.float32)

    nn_train = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', n_jobs=-1).fit(Tr)
    nn_test = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', n_jobs=-1).fit(Te)

    dist_train, _ = nn_train.kneighbors(S)
    dist_test, _ = nn_test.kneighbors(S)

    closer_to_test = (dist_test < dist_train).mean()
    return float(closer_to_test)
