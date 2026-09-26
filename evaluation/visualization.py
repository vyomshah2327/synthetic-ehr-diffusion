import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os


def _subsample(real, synthetic, n=1000):
    n = min(n, len(real), len(synthetic))
    r = real[np.random.choice(len(real), n, replace=False)]
    s = synthetic[np.random.choice(len(synthetic), n, replace=False)]
    return r, s


def plot_tsne(real, synthetic, title, save_path, n=1000):
    r, s = _subsample(real, synthetic, n)
    X = np.concatenate([r, s], axis=0).reshape(len(r) + len(s), -1)
    labels = ['Real'] * len(r) + ['Synthetic'] * len(s)

    proj = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'Real': '#2196F3', 'Synthetic': '#FF5722'}
    for label in ['Real', 'Synthetic']:
        mask = np.array(labels) == label
        ax.scatter(proj[mask, 0], proj[mask, 1], label=label,
                   c=colors[label], alpha=0.5, s=10)
    ax.set_title(f't-SNE — {title}')
    ax.legend()
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_pca(real, synthetic, title, save_path, n=1000):
    r, s = _subsample(real, synthetic, n)
    X = np.concatenate([r, s], axis=0).reshape(len(r) + len(s), -1)
    labels = ['Real'] * len(r) + ['Synthetic'] * len(s)

    proj = PCA(n_components=2).fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'Real': '#2196F3', 'Synthetic': '#FF5722'}
    for label in ['Real', 'Synthetic']:
        mask = np.array(labels) == label
        ax.scatter(proj[mask, 0], proj[mask, 1], label=label,
                   c=colors[label], alpha=0.5, s=10)
    ax.set_title(f'PCA — {title}')
    ax.legend()
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_sample_traces(real, synthetic, title, save_path, n=5, feature_names=None):
    """Side-by-side line plots of n random samples, first 6 features."""
    n_show_features = min(6, real.shape[-1])
    fig, axes = plt.subplots(n, 2, figsize=(14, n * 3))

    r_idx = np.random.choice(len(real),      n, replace=False)
    s_idx = np.random.choice(len(synthetic), n, replace=False)

    for i in range(n):
        for j, (sample, label) in enumerate([(real[r_idx[i]], 'Real'),
                                              (synthetic[s_idx[i]], 'Synthetic')]):
            ax = axes[i, j]
            for f in range(n_show_features):
                name = feature_names[f] if feature_names else f'Feature {f}'
                ax.plot(sample[:, f], label=name, alpha=0.8)
            ax.set_title(f'{label} Sample {i+1}')
            ax.set_xlabel('Hour')
            if i == 0:
                ax.legend(fontsize=7, ncol=2)

    fig.suptitle(f'Sample Traces — {title}', fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
