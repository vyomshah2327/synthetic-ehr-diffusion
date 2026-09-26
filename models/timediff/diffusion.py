import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm


class TimeDiff(nn.Module):
    """
    DDPM diffusion process wrapping the BRNNDenoiser backbone.
    Jointly handles:
    - Gaussian diffusion for continuous EHR values
    - Multinomial diffusion for binary missingness mask
    """

    def __init__(self, model, T=1000, beta_schedule='cosine', device='cuda',
                 real_cov=None, lambda_corr=0.0):
        super().__init__()
        self.model = model
        self.T = T
        self.device = device
        self.lambda_corr = lambda_corr

        if lambda_corr > 0:
            assert real_cov is not None, "real_cov is required when lambda_corr > 0"

        betas = self._get_betas(beta_schedule)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = F.pad(alpha_bars[:-1], (1, 0), value=1.0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sqrt_alpha_bars', alpha_bars.sqrt())
        self.register_buffer('sqrt_one_minus_alpha_bars', (1.0 - alpha_bars).sqrt())
        self.register_buffer('sqrt_recip_alphas', (1.0 / alphas).sqrt())
        self.register_buffer(
            'posterior_variance',
            betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        )

        if real_cov is not None:
            self.register_buffer('real_cov', torch.as_tensor(real_cov, dtype=torch.float32))
        else:
            self.real_cov = None

    def _get_betas(self, schedule):
        if schedule == 'linear':
            return torch.linspace(1e-4, 0.02, self.T)
        elif schedule == 'cosine':
            steps = self.T + 1
            t = torch.linspace(0, self.T, steps)
            f = torch.cos((t / self.T + 0.008) / 1.008 * math.pi / 2) ** 2
            f = f / f[0]
            betas = 1 - f[1:] / f[:-1]
            return betas.clamp(0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

    def q_sample(self, x0, t, noise=None):
        """Forward Gaussian process: add noise to x0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alpha_bars[t][:, None, None]
        sqrt_one_minus_ab = self.sqrt_one_minus_alpha_bars[t][:, None, None]
        return sqrt_ab * x0 + sqrt_one_minus_ab * noise, noise

    def q_sample_mask(self, m0, t):
        """Forward multinomial process: corrupt binary mask toward Bernoulli(0.5)."""
        alpha_bar = self.alpha_bars[t][:, None, None]
        probs = alpha_bar * m0 + (1.0 - alpha_bar) * 0.5
        return torch.bernoulli(probs)

    def _corr_loss(self, x0_hat):
        """
        MSE between the batch's predicted feature-covariance matrix and the
        real training set's covariance matrix. loss_x is a per-position,
        per-feature MSE, so it is fully satisfied by matching each feature's
        marginal independently -- it carries no signal about cross-feature
        relationships (e.g. Hgb/Hct correlation) at all. This term is what
        supplies that signal.

        x0_hat is reconstructed from the model's noise prediction, so at high
        t the estimate is divided by a near-zero sqrt(alpha_bar) and can blow
        up; clamp before this is called (see p_losses) rather than here, so
        the clamp is visible next to the reconstruction that needs it.
        """
        B, S, Fdim = x0_hat.shape
        pred_flat = x0_hat.reshape(B * S, Fdim)
        pred_c = pred_flat - pred_flat.mean(dim=0, keepdim=True)
        pred_cov = (pred_c.T @ pred_c) / (pred_flat.shape[0] - 1)
        return F.mse_loss(pred_cov, self.real_cov)

    def p_losses(self, x0, m0, t):
        """Combined Gaussian + multinomial training loss."""
        x_noisy, noise = self.q_sample(x0, t)
        m_noisy = self.q_sample_mask(m0, t)

        noise_pred, mask_logits = self.model(x_noisy, m_noisy, t)

        loss_x = F.mse_loss(noise_pred, noise)
        loss_m = F.binary_cross_entropy_with_logits(mask_logits, m0)
        loss = loss_x + loss_m

        if self.lambda_corr > 0:
            sqrt_ab = self.sqrt_alpha_bars[t][:, None, None]
            sqrt_one_minus_ab = self.sqrt_one_minus_alpha_bars[t][:, None, None]
            # reconstruct x0 estimate from the predicted noise (standard DDPM
            # algebra), clamped since high-t samples divide by a near-zero
            # sqrt_ab and would otherwise dominate the covariance estimate
            x0_hat = ((x_noisy - sqrt_one_minus_ab * noise_pred) / sqrt_ab).clamp(-2.0, 2.0)
            loss = loss + self.lambda_corr * self._corr_loss(x0_hat)

        return loss

    @torch.no_grad()
    def p_sample(self, xt, mt, t_scalar):
        """Single reverse step: denoise both x and mask from t to t-1."""
        B = xt.shape[0]
        t = torch.full((B,), t_scalar, device=self.device, dtype=torch.long)

        noise_pred, mask_logits = self.model(xt, mt, t)

        # Gaussian reverse step
        beta_t = self.betas[t_scalar]
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t_scalar]
        sqrt_one_minus_ab_t = self.sqrt_one_minus_alpha_bars[t_scalar]
        mean = sqrt_recip_alpha_t * (xt - beta_t / sqrt_one_minus_ab_t * noise_pred)

        if t_scalar == 0:
            x_next = mean.clamp(0.0, 1.0)
        elif t_scalar == 1:
            x_next = mean.clamp(-1.5, 1.5)  # no noise on last stochastic step
        else:
            posterior_var = self.posterior_variance[t_scalar]
            x_next = (mean + posterior_var.sqrt() * torch.randn_like(xt)).clamp(-1.5, 1.5)

        # Multinomial reverse step: predict m0, sample m_{t-1}
        m0_pred = torch.sigmoid(mask_logits)
        if t_scalar == 0:
            m_next = (m0_pred > 0.5).float()
        else:
            alpha_bar_prev = self.alpha_bars[t_scalar - 1]
            m_probs = alpha_bar_prev * m0_pred + (1.0 - alpha_bar_prev) * 0.5
            m_next = torch.bernoulli(m_probs)

        return x_next, m_next

    @torch.no_grad()
    def sample(self, n_samples, seq_len, n_features, batch_size=256,
               return_mask=False):
        """
        Generate synthetic samples via full reverse diffusion.

        Returns numpy array of shape (n_samples, seq_len, n_features).
        If return_mask=True, returns (values, mask) instead.

        NOTE: values are returned UNMASKED. p_losses computes loss_x over all
        positions (m0 only feeds the separate BCE mask head), so the model is
        trained to reconstruct the full imputed x0 -- including carried-forward
        values at unobserved positions. Zeroing by the mask here would produce
        ~76% zeros against ~25% in the real data, which makes real vs synthetic
        trivially separable.
        """
        self.model.eval()
        all_samples = []
        all_masks = []

        for start in range(0, n_samples, batch_size):
            B = min(batch_size, n_samples - start)
            xt = torch.randn(B, seq_len, n_features, device=self.device)
            mt = torch.bernoulli(torch.full((B, seq_len, n_features), 0.5, device=self.device))

            for t in tqdm(reversed(range(self.T)), desc='Sampling', total=self.T, leave=False):
                xt, mt = self.p_sample(xt, mt, t)

            all_samples.append(xt.cpu())
            if return_mask:
                all_masks.append(mt.cpu())

        values = torch.cat(all_samples, dim=0).numpy()
        if return_mask:
            return values, torch.cat(all_masks, dim=0).numpy()
        return values
