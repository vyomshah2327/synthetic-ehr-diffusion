import torch
import torch.nn as nn
import math


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.out_dim = (dim // 2) * 2  # always even

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / max(half - 1, 1)
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, half*2)


class TimestepMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.sin_emb = SinusoidalTimestepEmbedding(dim)
        emb_dim = self.sin_emb.out_dim  # (dim//2)*2, handles odd dim
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        return self.mlp(self.sin_emb(t))


class BRNNDenoiser(nn.Module):
    """
    Bidirectional GRU denoising backbone for TimeDiff.
    Input:  noisy sequence x (B, seq_len, n_features) +
            noisy mask m (B, seq_len, n_features) +
            timestep t (B,)
    Output: predicted noise (B, seq_len, n_features),
            predicted mask logits (B, seq_len, n_features)
    """

    def __init__(self, n_features, seq_len, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        self.time_mlp = TimestepMLP(n_features)

        # input: noisy_x + noisy_mask + time_emb = n_features * 3
        self.brnn = nn.GRU(
            input_size=n_features * 3,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.noise_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, n_features),
        )

        self.mask_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, n_features),
        )

    def forward(self, x_noisy, m_noisy, t):
        """
        x_noisy: (B, seq_len, n_features)
        m_noisy: (B, seq_len, n_features) binary mask corrupted toward 0.5
        t:       (B,) integer timesteps
        returns: noise_pred (B, seq_len, n_features),
                 mask_logits (B, seq_len, n_features)
        """
        B, S, F = x_noisy.shape

        t_emb = self.time_mlp(t)
        t_emb = t_emb.unsqueeze(1).expand(B, S, F)

        x = torch.cat([x_noisy, m_noisy, t_emb], dim=-1)  # (B, S, F*3)

        out, _ = self.brnn(x)                   # (B, S, hidden_dim*2)
        noise_pred = self.noise_proj(out)        # (B, S, n_features)
        mask_logits = self.mask_proj(out)        # (B, S, n_features)
        return noise_pred, mask_logits
