import argparse
import os
import shutil
import yaml
import torch
import numpy as np
import pandas as pd
import pickle

from models.timediff.brnn import BRNNDenoiser
from models.timediff.diffusion import TimeDiff
from models.timediff.ema import EMA
from models.ehrm_gan.dual_vae import DualVAE
from models.ehrm_gan.coupled_generator import CoupledGenerator


def load_scaler(dataset_name, data_dir='data/processed'):
    path = os.path.join(data_dir, f'{dataset_name}_scaler.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)


def inverse_normalize(data, scaler):
    N, S, F = data.shape
    flat = data.reshape(-1, F)
    restored = scaler['min'] + flat * (scaler['max'] - scaler['min'])
    return restored.reshape(N, S, F)


def clear_pycache(root='.'):
    for dirpath, dirs, _ in os.walk(root):
        for d in dirs:
            if d == '__pycache__':
                shutil.rmtree(os.path.join(dirpath, d))


def apply_generated_mask(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Forward/backward-fill synthetic values using the model's OWN jointly-
    generated mask, exactly mirroring how real hourly matrices were built in
    build_hourly_matrix() (mat_df.ffill().bfill()) from raw irregular
    observations.

    Previously sample() intentionally returned raw, fully-continuous values
    at every position -- the model generates a plausible mask via its own
    multinomial diffusion head, but that mask was discarded (generate.py
    called diffusion.sample() without return_mask=True). This meant
    synthetic data never reproduced the flat/carried-forward stretches real
    data has at ~75-85% of positions (see flatness diagnostic), which gave
    the discriminative classifier an easy, artificial tell.

    values, mask: (N, seq_len, n_features), mask in {0, 1} (1 = "observed"
    per the model's own generated mask, 0 = "unobserved" -> should be held
    flat at the last observed value, matching real preprocessing).
    """
    N, S, Fdim = values.shape
    filled = np.empty_like(values)
    mask_bool = mask.astype(bool)

    for i in range(N):
        df = pd.DataFrame(values[i], columns=range(Fdim))
        obs = mask_bool[i]
        # positions the model marked unobserved get NaN'd out, then
        # ffill/bfill -- identical operation to build_hourly_matrix()
        df = df.mask(~obs)
        df = df.ffill().bfill()
        # any feature never marked observed anywhere in this sequence has
        # no value to fill from -- fall back to the model's raw (unmasked)
        # diffusion output for that column rather than leaving NaN
        still_nan = df.isna()
        if still_nan.any().any():
            raw = pd.DataFrame(values[i], columns=range(Fdim))
            df = df.where(~still_nan, raw)
        filled[i] = df.values

    return filled


def generate_timediff(cfg, checkpoint, n_samples, device, apply_mask=True):
    model = BRNNDenoiser(
        n_features=cfg['n_features'],
        seq_len=cfg['seq_len'],
        hidden_dim=cfg['brnn_hidden_dim'],
        num_layers=cfg['brnn_layers'],
    ).to(device)

    diffusion = TimeDiff(model, T=cfg['T'], beta_schedule=cfg['beta_schedule'], device=device).to(device)
    ema = EMA(model, decay=cfg['ema_decay'])

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    ema.load_state_dict(ckpt['ema'])
    ema.apply_shadow(model)

    if apply_mask:
        samples, mask = diffusion.sample(n_samples, cfg['seq_len'], cfg['n_features'], return_mask=True)
        print(f'  Generated mask observation rate: {mask.mean():.4f} '
              f'(compare to real training data\'s observation rate)')
        samples = apply_generated_mask(samples, mask)
        samples = np.clip(samples, 0.0, 1.0)  # ffill/bfill can't introduce new out-of-range values,
                                                # but the raw fallback for never-observed columns can
    else:
        samples = diffusion.sample(n_samples, cfg['seq_len'], cfg['n_features'])
        mask = None
    return samples, mask  # (N, seq_len, n_features), (N, seq_len, n_features) or None


def generate_ehrm_gan(cfg, checkpoint, n_samples, device):
    n_features = cfg['n_features']
    seq_len = cfg['seq_len']
    latent_dim = cfg['vae_latent_dim']

    vae = DualVAE(n_features, seq_len, cfg['vae_hidden_dim'], latent_dim).to(device)
    generator = CoupledGenerator(latent_dim, cfg['crn_hidden_dim'], seq_len, cfg['crn_layers']).to(device)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    generator.load_state_dict(ckpt['generator'])

    vae_path = os.path.join(os.path.dirname(checkpoint), 'vae_pretrained.pt')
    vae.load_state_dict(torch.load(vae_path, map_location=device, weights_only=False))
    vae.eval()
    generator.eval()

    all_samples = []
    batch_size = 256
    split = vae.split

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            B = min(batch_size, n_samples - start)
            noise_a = torch.randn(B, seq_len, latent_dim, device=device)
            noise_b = torch.randn(B, seq_len, latent_dim, device=device)
            za, zb = generator(noise_a, noise_b)
            xa = vae.dec_a(za)
            xb = vae.dec_b(zb)
            x = torch.cat([xa, xb], dim=-1)
            all_samples.append(x.cpu().numpy())

    return np.concatenate(all_samples, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['timediff', 'ehrm_gan'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--n_samples', type=int, default=10000)
    parser.add_argument('--output', required=True)
    parser.add_argument('--data_dir', default='data/processed')
    parser.add_argument('--no_mask_fill', action='store_true',
                         help='Disable mask-based ffill/bfill post-processing (old behavior)')
    args = parser.parse_args()

    # clear pycache so latest source files are always used
    clear_pycache()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg['lr'] = float(cfg['lr'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # delete existing output file if present so we always regenerate fresh
    if os.path.exists(args.output):
        os.remove(args.output)
        print(f'Deleted existing: {args.output}')

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)

    if args.model == 'timediff':
        samples, mask = generate_timediff(cfg, args.checkpoint, args.n_samples, device,
                                           apply_mask=not args.no_mask_fill)
    else:
        samples = generate_ehrm_gan(cfg, args.checkpoint, args.n_samples, device)
        mask = None

    print(f'Synthetic stats — min: {samples.min():.4f} max: {samples.max():.4f} mean: {samples.mean():.4f}')
    np.save(args.output, samples)
    print(f'Saved {len(samples)} samples to {args.output}')

    if mask is not None:
        mask_output = args.output.replace('.npy', '_mask.npy')
        np.save(mask_output, mask)
        print(f'Saved generated mask to {mask_output} '
              f'(this is the model\'s own generated observation mask, '
              f'not a proxy -- true synthetic miss rate = 1 - mask.mean())')


if __name__ == '__main__':
    main()
