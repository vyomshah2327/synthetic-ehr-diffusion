import argparse
import os
import numpy as np
import csv

from evaluation.discriminative import discriminative_score
from evaluation.predictive_v2 import predictive_score, trtr_score
from evaluation.mmd import mmd_score
from evaluation.privacy_v2 import nnaa_score_paper, nnaa_score_repo
from evaluation.visualization import plot_tsne, plot_pca, plot_sample_traces

FEATURE_NAMES = [
    'HeartRate', 'SysBP', 'DiasBP', 'MeanBP', 'RespRate', 'SpO2', 'Temp',
    'Glucose', 'WBC', 'Hgb', 'Hct', 'Plt', 'Na', 'K', 'Cl', 'CO2',
    'BUN', 'Creatinine', 'Mg', 'Phos', 'Ca', 'ALT', 'AST', 'AlkPhos',
    'Bili', 'Albumin', 'Lactate', 'PaO2', 'PaCO2', 'pH',
    'FiO2', 'PEEP', 'TidalVol', 'GCSMotor', 'GCSVerbal'
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='mimic3 | mimic4 | eicu')
    parser.add_argument('--model', required=True, help='timediff | ehrm_gan')
    parser.add_argument('--synthetic', required=True, help='path to synthetic .npy file')
    parser.add_argument('--data_dir', default='data/processed')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    print(f"\nEvaluating {args.model} on {args.dataset}")
    print("=" * 50)

    real_train = np.load(os.path.join(args.data_dir, f'{args.dataset}_train.npy'))
    real_test = np.load(os.path.join(args.data_dir, f'{args.dataset}_test.npy'))
    synthetic = np.load(args.synthetic)

    # The scaler was fit on train, so train is exactly [0,1] but test can
    # exceed it. MIMIC-III/IV MeanBP (feature 3) has raw-data outliers that
    # reach ~15x the training range. The model can only ever emit ~[0,1], so
    # comparing against unclipped test data is structurally unfair.
    n_bad = int((real_test > 1.0).sum() + (real_test < 0.0).sum())
    if n_bad:
        print(f'  Clipped {n_bad} out-of-range test values '
              f'({100 * n_bad / real_test.size:.4f}%)')
        real_train = np.clip(real_train, 0.0, 1.0)
        real_test = np.clip(real_test, 0.0, 1.0)
        synthetic = np.clip(synthetic, 0.0, 1.0)

    # Drop features that are constant in the real data. These are extraction
    # failures in the loaders (mimic4: K, CO2, Creatinine, AlkPhos, Albumin;
    # eicu: TidalVol, GCSMotor, GCSVerbal) -- they carry no information, but
    # because the model generates variation in them they make real vs synthetic
    # trivially separable for a sequence classifier.
    keep = real_train.std(axis=(0, 1)) > 1e-9
    if not keep.all():
        dropped = np.where(~keep)[0]
        print(f'  Dropping {len(dropped)} constant feature(s): {dropped.tolist()}')
        real_train = real_train[..., keep]
        real_test = real_test[..., keep]
        synthetic = synthetic[..., keep]
    print(f'  Using {keep.sum()}/{len(keep)} features')

    print(f"Real train: {real_train.shape} | Real test: {real_test.shape} | Synthetic: {synthetic.shape}")

    # 1. Discriminative score -- unchanged, already matches the paper's
    # methodology (GRU classifier, real=1/synthetic=0, |acc-0.5| scale).
    print("\nComputing discriminative score...")
    disc = discriminative_score(real_test, synthetic, device=args.device)
    disc_paper_scale = abs(disc - 0.5)
    print(f"  Discriminative Score: {disc:.4f} (0.5 = perfect)")
    print(f"  Discriminative Score, |acc-0.5| scale: {disc_paper_scale:.4f} "
          f"(0 = perfect -- this is the scale the paper's numbers are reported on)")

    n_feat = synthetic.shape[-1]
    print("\nComputing discriminative score (paper-spec classifier: "
          f"hidden_dim={n_feat // 2}, 1 layer)...")
    disc_paper_spec = discriminative_score(
        real_test, synthetic, device=args.device,
        hidden_dim=max(n_feat // 2, 1), num_layers=1, n_steps=5000,
    )
    print(f"  Discriminative Score (paper-spec classifier): {disc_paper_spec:.4f} (0.5 = perfect)")

    # 2. Predictive score -- NOW matches the paper's every-timestep next-step
    # vector prediction (was previously last-timestep-only). Also computes
    # the TRTR baseline: TSTR should be close to TRTR, not to zero.
    print("\nComputing predictive score (TSTR, every-timestep, paper-matching)...")
    pred_tstr = predictive_score(synthetic, real_test, device=args.device)
    print(f"  Predictive Score TSTR (MAE): {pred_tstr:.6f} (lower = better, "
          f"but compare against TRTR below, not against 0)")

    print("Computing TRTR baseline (train on real, test on real)...")
    pred_trtr = trtr_score(real_train, real_test, device=args.device)
    print(f"  Predictive Score TRTR (MAE): {pred_trtr:.6f} (reference target for TSTR)")
    print(f"  TSTR / TRTR ratio: {pred_tstr / pred_trtr:.4f} (near 1.0 = TimeDiff preserves "
          f"real predictability; far below 1.0 usually signals an easier/degenerate task "
          f"rather than better-than-real fidelity)")

    # 3. MMD -- unchanged. Not reported in the paper's tables, so there is no
    # paper baseline to compare against; useful only as a relative/internal
    # metric across your own datasets and runs.
    print("\nComputing MMD...")
    mmd = mmd_score(real_test, synthetic)
    print(f"  MMD: {mmd:.6f} (lower = better; no paper baseline exists for this metric)")

    # 4. Privacy -- NNAA, now computed BOTH ways for a clean before/after:
    #    - nnaa_score_paper: matches the paper's exact AA_test/AA_train/NNAA
    #      formula (Eq. 15-20). Target ~0 (paper's TimeDiff: 0.002-0.006).
    #    - nnaa_score_repo: the original repo metric (kept for continuity).
    #      NOT the same scale -- target ~0.5 on this one.
    # NNAA is sensitive to set size: unequal train/test/synthetic set sizes
    # bias the score, so all three are equalized to the same n.
    print("Computing NNAA privacy metric (paper-matching + original repo metric)...")
    aa_test, aa_train, nnaa_paper = nnaa_score_paper(real_train, real_test, synthetic)
    print(f"  AA_test:  {aa_test:.4f} (paper's TimeDiff: ~0.5-0.6)")
    print(f"  AA_train: {aa_train:.4f} (paper's TimeDiff: ~0.5-0.6)")
    print(f"  NNAA (paper-matching, |AA_test - AA_train|): {nnaa_paper:.4f} "
          f"(0 = perfect -- paper's TimeDiff: 0.002-0.006)")

    nnaa_repo = nnaa_score_repo(real_train, real_test, synthetic)
    print(f"  NNAA (original repo metric, different scale): {nnaa_repo:.4f} "
          f"(0.5 = private on THIS scale -- do not compare to the line above)")

    # 5. Visualizations
    plot_dir = os.path.join('results', 'plots', args.dataset, args.model)
    title = args.dataset.upper()

    print("Generating t-SNE plot...")
    plot_tsne(real_test, synthetic, title, os.path.join(plot_dir, 'tsne.png'))

    print("Generating PCA plot...")
    plot_pca(real_test, synthetic, title, os.path.join(plot_dir, 'pca.png'))

    print("Generating sample traces...")
    plot_sample_traces(real_test, synthetic, title,
                        os.path.join(plot_dir, 'traces.png'),
                        feature_names=FEATURE_NAMES)

    # 6. Append to metrics_v2.csv (separate file from the original
    # metrics.csv so old and new runs never get mixed in the same table).
    metrics_path = os.path.join('results', 'metrics_v2.csv')
    file_exists = os.path.exists(metrics_path)
    with open(metrics_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                'Dataset', 'Model',
                'Disc_Score', 'Disc_Score_PaperScale', 'Disc_Score_PaperSpecClassifier',
                'Pred_TSTR_MAE', 'Pred_TRTR_MAE', 'Pred_TSTR_TRTR_Ratio',
                'MMD',
                'AA_test', 'AA_train', 'NNAA_Paper', 'NNAA_Repo',
            ])
        writer.writerow([
            args.dataset, args.model,
            f'{disc:.4f}', f'{disc_paper_scale:.4f}', f'{disc_paper_spec:.4f}',
            f'{pred_tstr:.6f}', f'{pred_trtr:.6f}', f'{pred_tstr / pred_trtr:.4f}',
            f'{mmd:.6f}',
            f'{aa_test:.4f}', f'{aa_train:.4f}', f'{nnaa_paper:.4f}', f'{nnaa_repo:.4f}',
        ])

    print(f"\nResults appended to {metrics_path}")
    print("Plots saved to", plot_dir)
    print("Done.")


if __name__ == '__main__':
    main()
