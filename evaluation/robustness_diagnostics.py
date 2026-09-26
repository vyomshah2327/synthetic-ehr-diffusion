"""
Two free (no generator retraining) diagnostics for a discriminative score
that's much worse than the paper's reported value:

1. boundary_density: p_sample() hard-clamps the last two reverse steps to
   [0,1] / [-1.5,1.5]. If synthetic values pile up at exactly 0/1 more than
   real data does, that alone is a trivial tell for the classifier and has
   nothing to do with cross-feature correlation.

2. feature_ablation_sweep: for each feature, shuffle that feature's column
   across samples in the SYNTHETIC set only (breaks its relationship to the
   rest of that row, keeps its own marginal), then rerun the discriminative
   test. If accuracy DROPS after shuffling a feature, that feature's
   real-vs-synthetic correlation with the rest of the row was one of the
   things the classifier was using to tell them apart -- shuffling it away
   accidentally made the (already wrong) synthetic row harder to catch.
   Features with the biggest drop are the best targets for a covariance loss.

Run after generating synthetic samples, before deciding whether to spend
compute on retraining with a correlation loss.
"""
import argparse
import os
import numpy as np

from evaluation.discriminative import discriminative_score

FEATURE_NAMES = [
    'HeartRate', 'SysBP', 'DiasBP', 'MeanBP', 'RespRate', 'SpO2', 'Temp',
    'Glucose', 'WBC', 'Hgb', 'Hct', 'Plt', 'Na', 'K', 'Cl', 'CO2',
    'BUN', 'Creatinine', 'Mg', 'Phos', 'Ca', 'ALT', 'AST', 'AlkPhos',
    'Bili', 'Albumin', 'Lactate', 'PaO2', 'PaCO2', 'pH',
    'FiO2', 'PEEP', 'TidalVol', 'GCSMotor', 'GCSVerbal'
]


def boundary_density(real, synthetic, feature_names, eps=1e-3):
    """Fraction of values within eps of 0 or 1, per feature, real vs synthetic."""
    real_frac = ((real <= eps) | (real >= 1 - eps)).mean(axis=(0, 1))
    synth_frac = ((synthetic <= eps) | (synthetic >= 1 - eps)).mean(axis=(0, 1))
    rows = sorted(
        zip(feature_names, real_frac, synth_frac),
        key=lambda r: abs(r[2] - r[1]),
        reverse=True,
    )
    print(f"\n{'Feature':<12} {'Real boundary %':>16} {'Synth boundary %':>18} {'Delta':>8}")
    for name, r, s in rows:
        print(f"{name:<12} {100 * r:>15.2f}% {100 * s:>17.2f}% {100 * (s - r):>7.2f}%")
    return rows


def feature_ablation_sweep(real, synthetic, feature_names, device='cuda',
                            n_steps=500, seed=0):
    """
    Baseline discriminative accuracy, then per-feature: shuffle that column
    across samples in `synthetic` only, rerun, report the delta from
    baseline. Uses fewer steps than the primary metric since this is a
    directional sweep, not the number you'd report -- rerun the top
    suspects at full n_steps to confirm before committing to a fix.
    """
    rng = np.random.default_rng(seed)

    print(f"\nBaseline discriminative score (n_steps={n_steps})...")
    baseline = discriminative_score(real, synthetic, device=device, n_steps=n_steps)
    print(f"  Baseline accuracy: {baseline:.4f}")

    results = []
    for i, name in enumerate(feature_names):
        shuffled = synthetic.copy()
        perm = rng.permutation(len(shuffled))
        shuffled[:, :, i] = shuffled[perm, :, i]
        acc = discriminative_score(real, shuffled, device=device, n_steps=n_steps)
        delta = acc - baseline
        results.append((name, acc, delta))
        print(f"  [{i:2d}] {name:<12} shuffled -> acc={acc:.4f}  delta={delta:+.4f}")

    print("\nFeatures ranked by biggest accuracy DROP when shuffled "
          "(most likely to be carrying a bad cross-feature correlation):")
    for name, acc, delta in sorted(results, key=lambda r: r[2]):
        if delta < 0:
            print(f"  {name:<12} delta={delta:+.4f}")

    return baseline, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='mimic3 | mimic4 | eicu')
    parser.add_argument('--synthetic', required=True, help='path to synthetic .npy file')
    parser.add_argument('--data_dir', default='data/processed')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--n_steps', type=int, default=500,
                         help='steps per classifier in the ablation sweep (kept small -- '
                              'this is a directional check, not the reported metric)')
    args = parser.parse_args()

    real_test = np.load(os.path.join(args.data_dir, f'{args.dataset}_test.npy'))
    real_train = np.load(os.path.join(args.data_dir, f'{args.dataset}_train.npy'))
    synthetic = np.load(args.synthetic)

    real_test = np.clip(real_test, 0.0, 1.0)
    synthetic = np.clip(synthetic, 0.0, 1.0)

    keep = real_train.std(axis=(0, 1)) > 1e-9
    real_test = real_test[..., keep]
    synthetic = synthetic[..., keep]
    feature_names = [f for f, k in zip(FEATURE_NAMES, keep) if k]

    print("=" * 60)
    print(f"Boundary-clamp check: {args.dataset}")
    print("=" * 60)
    boundary_density(real_test, synthetic, feature_names)

    print("\n" + "=" * 60)
    print(f"Feature ablation sweep: {args.dataset}")
    print("=" * 60)
    feature_ablation_sweep(real_test, synthetic, feature_names,
                            device=args.device, n_steps=args.n_steps)


if __name__ == '__main__':
    main()
