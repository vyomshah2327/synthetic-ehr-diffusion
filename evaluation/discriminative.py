import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class GRUClassifier(nn.Module):
    def __init__(self, n_features, hidden_dim=128, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden_dim, num_layers,
                           batch_first=True, dropout=0.1 if num_layers > 1 else 0.0)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1]).squeeze(-1)


def discriminative_score(real: np.ndarray, synthetic: np.ndarray,
                          n_steps=2000, batch_size=256,
                          hidden_dim=128, num_layers=2, device='cuda') -> float:
    """
    Train a GRU to classify real (1) vs synthetic (0).
    Returns test accuracy. Closer to 0.5 = better (harder to distinguish).

    real, synthetic: (N, seq_len, n_features)

    hidden_dim/num_layers let you swap in a weaker classifier (e.g. the
    paper's spec of n_features // 2, 1 layer) to check how much of a given
    score is "our classifier is stronger than theirs" vs. a real generation
    gap, without needing to retrain the generator.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    n = min(len(real), len(synthetic))

    X = np.concatenate([real[:n], synthetic[:n]], axis=0)
    y = np.array([1.0] * n + [0.0] * n)

    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    split = int(0.8 * len(X))
    X_train = torch.tensor(X[:split], dtype=torch.float32)
    y_train = torch.tensor(y[:split], dtype=torch.float32)
    X_test = torch.tensor(X[split:], dtype=torch.float32)
    y_test = torch.tensor(y[split:], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                               batch_size=batch_size, shuffle=True)

    n_features = real.shape[-1]
    model = GRUClassifier(n_features, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    model.train()
    step = 0
    while step < n_steps:
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if step >= n_steps:
                break

    model.eval()
    with torch.no_grad():
        pred_test = model(X_test.to(device)).cpu().numpy()
    acc = ((pred_test > 0.5) == y_test.numpy()).mean()
    return float(acc)
