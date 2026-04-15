import numpy as np
from scipy import stats
from sklearn.metrics.pairwise import rbf_kernel

def calculate_wasserstein(p: np.ndarray, q: np.ndarray) -> float:
    """
    Calculate the 1rd Wasserstein distance between two distributions.
    This is more stable than KL divergence for GANs.
    """
    # p and q are 1D arrays (features)
    return float(stats.wasserstein_distance(p, q))

def calculate_mmd(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
    """
    Calculate Maximum Mean Discrepancy (MMD) with RBF kernel.
    X, Y: (n_samples, n_features)
    """
    # Limit number of samples for MMD calculation as it's O(n^2)
    n_samples = min(len(X), len(Y), 1000)
    X = X[:n_samples]
    Y = Y[:n_samples]

    K_XX = rbf_kernel(X, X, gamma=gamma)
    K_YY = rbf_kernel(Y, Y, gamma=gamma)
    K_XY = rbf_kernel(X, Y, gamma=gamma)

    # MMD^2 = 1/n^2 * sum(K_XX) + 1/m^2 * sum(K_YY) - 2/nm * sum(K_XY)
    mmd_sq = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
    return float(np.sqrt(max(mmd_sq, 0)))

def calculate_distribution_metrics(real: np.ndarray, fake: np.ndarray) -> dict:
    """
    Helper to calculate mean Wasserstein across all features.
    """
    n_features = real.shape[1]
    wd_scores = []
    
    for i in range(n_features):
        wd = calculate_wasserstein(real[:, i], fake[:, i])
        wd_scores.append(wd)
        
    return {
        "mean_wd": float(np.mean(wd_scores)),
        "mmd": calculate_mmd(real, fake)
    }
