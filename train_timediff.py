import argparse
import os
import yaml
import torch
import numpy as np
import wandb
from tqdm import tqdm

from data.dataset import get_dataloader
from models.timediff.brnn import BRNNDenoiser
from models.timediff.diffusion import TimeDiff
from models.timediff.ema import EMA


def train(config_path, run_name):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    run_name = run_name or cfg.get('run_name', 'timediff_run')
    save_dir = os.path.join('results', run_name)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    wandb.init(project='synthetic-ehr', name=run_name, config=cfg)

    data_dir = cfg.get('data_dir', os.path.join('data', 'processed'))
    train_loader = get_dataloader(data_dir, cfg['dataset'], 'train', cfg['batch_size'])

    model = BRNNDenoiser(
        n_features=cfg['n_features'],
        seq_len=cfg['seq_len'],
        hidden_dim=cfg['brnn_hidden_dim'],
        num_layers=cfg['brnn_layers'],
    ).to(device)

    # Cross-feature covariance target for the optional correlation loss.
    # loss_x (per-position MSE) only ever supervises each feature's own
    # marginal, so without this term the model has no training signal that
    # cross-feature relationships (e.g. Hgb/Hct) are wrong.
    lambda_corr = cfg.get('lambda_corr', 0.0)
    real_cov = None
    if lambda_corr > 0:
        real_data = np.load(os.path.join(data_dir, f"{cfg['dataset']}_train.npy"))  # (N, seq_len, n_features)
        real_flat = real_data.reshape(-1, real_data.shape[-1])
        real_flat = real_flat - real_flat.mean(axis=0, keepdims=True)
        real_cov = torch.tensor(
            (real_flat.T @ real_flat) / (real_flat.shape[0] - 1), dtype=torch.float32
        )
        print(f"Precomputed real feature covariance for corr loss (lambda_corr={lambda_corr})")

    diffusion = TimeDiff(
        model=model,
        T=cfg['T'],
        beta_schedule=cfg['beta_schedule'],
        device=device,
        real_cov=real_cov,
        lambda_corr=lambda_corr,
    ).to(device)

    ema = EMA(model, decay=cfg['ema_decay'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)

    warmup_steps = 500
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, cfg['epochs'] * len(train_loader) - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # resume from checkpoint if exists
    start_epoch = 0
    ckpt_path = os.path.join(save_dir, 'last.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

    global_step = 0
    best_loss = float('inf')

    for epoch in range(start_epoch, cfg['epochs']):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            x, m = batch
            x = x.to(device).permute(0, 2, 1)  # (B, seq_len, n_features)
            m = m.to(device).permute(0, 2, 1)  # (B, seq_len, n_features)
            t = torch.randint(0, cfg['T'], (x.shape[0],), device=device)

            loss = diffusion.p_losses(x, m, t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            ema.update(model)

            epoch_loss += loss.item()
            global_step += 1

            if global_step % 50 == 0:
                wandb.log({'loss': loss.item(), 'lr': scheduler.get_last_lr()[0]}, step=global_step)

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{cfg['epochs']} | Loss: {avg_loss:.6f}")

        # save checkpoint every save_every epochs
        if (epoch + 1) % cfg['save_every'] == 0 or (epoch + 1) == cfg['epochs']:
            ckpt = {
                'epoch': epoch,
                'model': model.state_dict(),
                'ema': ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'cfg': cfg,
            }
            torch.save(ckpt, ckpt_path)
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(ckpt, os.path.join(save_dir, 'best.pt'))
                print(f"  Saved best model (loss={best_loss:.6f})")

    wandb.finish()
    print("Training complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--run_name', default=None)
    args = parser.parse_args()
    train(args.config, args.run_name)
