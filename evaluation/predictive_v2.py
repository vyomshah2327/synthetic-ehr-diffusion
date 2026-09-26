import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class GRUPredictor(nn.Module):
    """Predicts the next-step vector at EVERY timestep (not just the last).
    This matches the paper's methodology (citing GT-GAN Appendix D): 'the
    mean absolute error based on the next step vector prediction'."""

    def __init__(self, n_features, hidden_dim=128, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden_dim, num_layers,
                           batch_first=True, dropout=0.1 if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        # x: (B, L-1, n_features) -> returns (B, L-1, n_features), one
        # prediction per input timestep (predicting the NEXT step at each
        # position), not just a single prediction for the final step.
        out, _ = self.gru(x)
        return self.fc(out)


def _train_predictor(train_source: np.ndarray, n_steps=2000, batch_size=256,
                      device='cuda', hidden_dim=128, num_layers=2):
    """Trains a GRUPredictor using teacher forcing: input = seq[:, :-1, :],
    target = seq[:, 1:, :] (next-step vector at every position)."""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    n_features = train_source.shape[-1]

    X = torch.tensor(train_source[:, :-1, :], dtype=torch.float32)
    y = torch.tensor(train_source[:, 1:, :], dtype=torch.float32)

    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

    model = GRUPredictor(n_features, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss()

    model.train()
    step = 0
    while step < n_steps:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if step >= n_steps:
                break
    return model, device


def _eval_mae(model, device, eval_data: np.ndarray) -> float:
    X_eval = torch.tensor(eval_data[:, :-1, :], dtype=torch.float32)
    y_eval = torch.tensor(eval_data[:, 1:, :], dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        pred = model(X_eval.to(device)).cpu().numpy()

    return float(np.abs(pred - y_eval.numpy()).mean())


def predictive_score(synthetic: np.ndarray, real_test: np.ndarray,
                      n_steps=2000, batch_size=256, device='cuda') -> float:
    """TSTR (Train on Synthetic, Test on Real). Train GRU on synthetic
    (every-timestep next-step prediction), evaluate MAE on real test set.
    Lower MAE = better, but the target is the TRTR baseline (see
    trtr_score below), NOT zero -- real physiological time series have
    inherent unpredictability, so 0 MAE would not be realistic.

    synthetic, real_test: (N, seq_len, n_features)
    """
    model, device = _train_predictor(synthetic, n_steps=n_steps,
                                      batch_size=batch_size, device=device)
    return _eval_mae(model, device, real_test)


def trtr_score(real_train: np.ndarray, real_test: np.ndarray,
               n_steps=2000, batch_size=256, device='cuda') -> float:
    """TRTR (Train on Real, Test on Real) baseline. Same architecture and
    procedure as predictive_score, but trained on real data instead of
    synthetic. This is the reference point TSTR should be close to --
    a TSTR score far below TRTR usually signals an easier/degenerate task
    rather than genuinely better-than-real fidelity, and a TSTR score far
    above TRTR signals the synthetic data failed to preserve predictable
    structure.

    real_train, real_test: (N, seq_len, n_features)
    """
    model, device = _train_predictor(real_train, n_steps=n_steps,
                                      batch_size=batch_size, device=device)
    return _eval_mae(model, device, real_test)
