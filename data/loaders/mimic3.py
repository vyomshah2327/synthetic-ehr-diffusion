import os
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

MIMIC3_ITEMIDS = {
    'HeartRate': [211, 220045],
    'SysBP': [51, 442, 455, 6701, 220179, 220050],
    'DiasBP': [8368, 8440, 8441, 8555, 220180, 220051],
    'MeanBP': [456, 52, 6702, 443, 220052, 220181, 225312],
    'RespRate': [615, 618, 220210, 224690],
    'SpO2': [646, 220277],
    'Temp': [223761, 678, 223762, 676],
    'Glucose': [807, 811, 1529, 3745, 3744, 225664, 220621, 226537],
    'WBC': [861, 1127, 1542, 220546],
    'Hgb': [814, 220228],
    'Hct': [813, 220545],
    'Plt': [828, 220457, 227457],
    'Na': [837, 220645],
    'K': [829, 220640],
    'Cl': [1532, 220602],
    'CO2': [788, 227443],
    'BUN': [1162, 225624],
    'Creatinine': [791, 220615],
    'Mg': [821, 220635],
    'Phos': [853, 225677],
    'Ca': [786, 225625],
    'ALT': [769, 220644],
    'AST': [770, 220587],
    'AlkPhos': [773, 220586],
    'Bili': [848, 225690],
    'Albumin': [763, 226981],
    'Lactate': [818, 225668],
    'PaO2': [490, 220224],
    'PaCO2': [489, 220235],
    'pH': [780, 220274],
    'FiO2': [2981, 3420, 3422, 223835],
    'PEEP': [505, 506, 686, 220339, 224700],
    'TidalVol': [639, 654, 681, 682, 683, 684, 224685, 224684],
    'GCSMotor': [454, 223901],
    'GCSVerbal': [723, 223900],
}

# Physiologically plausible ranges, used to drop obvious data-entry errors
# (e.g. HeartRate charted as 9999999, DiasBP charted as -12) before they
# reach normalize()'s raw train.min()/train.max(), which has no outlier
# resistance at all -- a single corrupted row is enough to compress the
# entire real clinical range for that feature into a sliver near 0 or 1.
# The original TimeDiff paper avoids this because their preprocessing goes
# through the official MIMIC "concepts" Postgres views (per their README),
# which already validate ranges as part of building those views; this
# reproduction extracts straight from raw chartevents by itemid, so that
# validation has to happen here instead. Bounds are intentionally generous
# (order-of-magnitude guards against entry errors, not tight clinical
# cutoffs) so genuine extreme-but-real ICU values are never dropped.
PLAUSIBLE_RANGES = {
    'HeartRate': (0, 300), 'SysBP': (0, 300), 'DiasBP': (0, 225), 'MeanBP': (0, 250),
    'RespRate': (0, 70), 'SpO2': (0, 100), 'Temp': (25, 115), 'Glucose': (0, 2000),
    'WBC': (0, 500), 'Hgb': (0, 25), 'Hct': (0, 75), 'Plt': (0, 2000),
    'Na': (80, 200), 'K': (0, 15), 'Cl': (50, 150), 'CO2': (0, 60),
    'BUN': (0, 300), 'Creatinine': (0, 40), 'Mg': (0, 15), 'Phos': (0, 20),
    'Ca': (0, 20), 'ALT': (0, 10000), 'AST': (0, 10000), 'AlkPhos': (0, 5000),
    'Bili': (0, 50), 'Albumin': (0, 6), 'Lactate': (0, 30), 'PaO2': (0, 700),
    'PaCO2': (0, 150), 'pH': (6.5, 8.0), 'FiO2': (0.2, 100), 'PEEP': (0, 30),
    'TidalVol': (0, 2000), 'GCSMotor': (1, 6), 'GCSVerbal': (1, 5),
}

FEATURES = list(MIMIC3_ITEMIDS.keys())
ALL_ITEMIDS = [iid for ids in MIMIC3_ITEMIDS.values() for iid in ids]
ITEMID_TO_FEATURE = {iid: feat for feat, ids in MIMIC3_ITEMIDS.items() for iid in ids}

_d = sorted({i for i in ALL_ITEMIDS if ALL_ITEMIDS.count(i) > 1})
assert not _d, f'duplicate itemids: {_d}'
assert set(PLAUSIBLE_RANGES.keys()) == set(FEATURES), 'PLAUSIBLE_RANGES must cover every feature'

SEQ_LEN = 24


def load_icustays(raw_dir):
    print("Loading ICU stays...")
    stays = pd.read_csv(os.path.join(raw_dir, 'ICUSTAYS.csv.gz'), compression='gzip',
                         usecols=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'INTIME', 'LOS'])
    stays['INTIME'] = pd.to_datetime(stays['INTIME'])
    stays = stays[stays['LOS'] >= 1.0]  # at least 24 hours
    print(f"  ICU stays >= 24h: {len(stays)}")
    return stays


def load_mortality(raw_dir):
    print("Loading mortality labels...")
    admissions = pd.read_csv(os.path.join(raw_dir, 'ADMISSIONS.csv.gz'), compression='gzip',
                              usecols=['HADM_ID', 'HOSPITAL_EXPIRE_FLAG'])
    return admissions.set_index('HADM_ID')['HOSPITAL_EXPIRE_FLAG'].to_dict()


def extract_features(raw_dir, stays):
    print("Loading CHARTEVENTS (this will take a while)...")
    stay_ids = set(stays['ICUSTAY_ID'].tolist())
    intime_map = stays.set_index('ICUSTAY_ID')['INTIME'].to_dict()

    chunks = pd.read_csv(
        os.path.join(raw_dir, 'CHARTEVENTS.csv.gz'),
        compression='gzip',
        usecols=['ICUSTAY_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM'],
        chunksize=1_000_000,
        low_memory=False
    )

    n_dropped_oob = 0
    n_total = 0
    records = []
    for chunk in tqdm(chunks, desc='Reading CHARTEVENTS'):
        chunk = chunk.dropna(subset=['ICUSTAY_ID', 'VALUENUM'])
        chunk = chunk[chunk['ICUSTAY_ID'].isin(stay_ids)]
        chunk = chunk[chunk['ITEMID'].isin(ALL_ITEMIDS)]
        if len(chunk) == 0:
            continue
        chunk['CHARTTIME'] = pd.to_datetime(chunk['CHARTTIME'])
        chunk['FEATURE'] = chunk['ITEMID'].map(ITEMID_TO_FEATURE)

        lo = chunk['FEATURE'].map(lambda f: PLAUSIBLE_RANGES[f][0]).values
        hi = chunk['FEATURE'].map(lambda f: PLAUSIBLE_RANGES[f][1]).values
        n_total += len(chunk)
        in_range = (chunk['VALUENUM'].values >= lo) & (chunk['VALUENUM'].values <= hi)
        n_dropped_oob += (~in_range).sum()
        chunk = chunk[in_range]

        records.append(chunk[['ICUSTAY_ID', 'CHARTTIME', 'FEATURE', 'VALUENUM']])

    if not records:
        raise ValueError("No matching chart events found.")

    df = pd.concat(records, ignore_index=True)
    print(f"  Total matched events: {len(df)}")
    print(f"  Dropped {n_dropped_oob}/{n_total} ({100*n_dropped_oob/max(n_total,1):.3f}%) "
          f"as physiologically implausible")
    return df, intime_map


def build_hourly_matrix(df, intime_map, stay_ids):
    print("Building hourly matrices...")
    samples = []
    masks = []
    valid_stay_ids = []

    feat_pos = {f: i for i, f in enumerate(FEATURES)}
    grouped = dict(tuple(df.groupby('ICUSTAY_ID')))

    for stay_id in tqdm(stay_ids, desc='Processing stays'):
        if stay_id not in intime_map:
            continue
        sub = grouped.get(stay_id)
        if sub is None or len(sub) == 0:
            continue
        sub = sub.copy()
        intime = intime_map[stay_id]

        sub['HOUR'] = ((sub['CHARTTIME'] - intime).dt.total_seconds() / 3600).astype(int)
        sub = sub[(sub['HOUR'] >= 0) & (sub['HOUR'] < SEQ_LEN)]

        matrix = np.full((SEQ_LEN, len(FEATURES)), np.nan)
        for h, feat, val in zip(sub['HOUR'].values, sub['FEATURE'].values,
                                 sub['VALUENUM'].values):
            f = feat_pos[feat]
            if np.isnan(matrix[h, f]):
                matrix[h, f] = val

        # capture mask before imputation (1=originally observed, 0=missing)
        mask = (~np.isnan(matrix)).astype(np.float32)

        # forward fill then backward fill, then column mean for remainder
        mat_df = pd.DataFrame(matrix, columns=FEATURES)
        mat_df = mat_df.ffill().bfill()
        matrix = mat_df.values

        # drop if >50% missing after ffill
        if np.isnan(matrix).mean() > 0.5:
            continue

        with np.errstate(all='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            col_means = np.nanmean(matrix, axis=0)
        for f in range(matrix.shape[1]):
            nan_mask = np.isnan(matrix[:, f])
            matrix[nan_mask, f] = col_means[f] if not np.isnan(col_means[f]) else 0.0

        samples.append(matrix)
        masks.append(mask)
        valid_stay_ids.append(stay_id)

    return np.array(samples), np.array(masks), valid_stay_ids


def normalize(train, val, test):
    mins = train.min(axis=(0, 1), keepdims=True)
    maxs = train.max(axis=(0, 1), keepdims=True)
    rng = np.where(maxs - mins == 0, 1.0, maxs - mins)

    train_n = (train - mins) / rng
    val_n = (val - mins) / rng
    test_n = (test - mins) / rng

    scaler = {'min': mins.squeeze(), 'max': maxs.squeeze()}
    return train_n, val_n, test_n, scaler


def split_data(samples, labels, train_r=0.7, val_r=0.15, seed=42):
    """Returns INDICES so features and masks share one identical split."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(samples))
    pos = idx[labels[idx] == 1]
    neg = idx[labels[idx] == 0]

    def split_idx(arr):
        n1 = int(len(arr) * train_r)
        n2 = int(len(arr) * val_r)
        return arr[:n1], arr[n1:n1+n2], arr[n1+n2:]

    tr_p, va_p, te_p = split_idx(pos)
    tr_n, va_n, te_n = split_idx(neg)
    return (np.concatenate([tr_p, tr_n]),
            np.concatenate([va_p, va_n]),
            np.concatenate([te_p, te_n]))


def main(raw_dir, out_dir, debug=False, debug_stays=500):
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(42)

    stays = load_icustays(raw_dir)
    if debug:
        stays = stays.sample(min(debug_stays, len(stays)), random_state=42)
        print(f"DEBUG MODE: using {len(stays)} stays")
    mortality = load_mortality(raw_dir)
    df, intime_map = extract_features(raw_dir, stays)

    stay_ids = stays['ICUSTAY_ID'].tolist()
    samples, masks, valid_stay_ids = build_hourly_matrix(df, intime_map, stay_ids)
    print(f"Valid samples: {len(samples)}")

    sid_to_hadm = stays.set_index('ICUSTAY_ID')['HADM_ID'].to_dict()
    labels = np.array([mortality.get(sid_to_hadm.get(sid), 0) for sid in valid_stay_ids])

    train_idx, val_idx, test_idx = split_data(samples, labels)
    train, val, test = samples[train_idx], samples[val_idx], samples[test_idx]
    print(f"Train: {train.shape} | Val: {val.shape} | Test: {test.shape}")

    train_n, val_n, test_n, scaler = normalize(train, val, test)

    np.save(os.path.join(out_dir, 'mimic3_train.npy'), train_n)
    np.save(os.path.join(out_dir, 'mimic3_val.npy'), val_n)
    np.save(os.path.join(out_dir, 'mimic3_test.npy'), test_n)
    np.save(os.path.join(out_dir, 'mimic3_train_mask.npy'), masks[train_idx])
    np.save(os.path.join(out_dir, 'mimic3_val_mask.npy'), masks[val_idx])
    np.save(os.path.join(out_dir, 'mimic3_test_mask.npy'), masks[test_idx])

    with open(os.path.join(out_dir, 'mimic3_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    print(f"Saved to {out_dir}")
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_stays', type=int, default=500)
    args = parser.parse_args()
    main(args.raw_dir, args.out_dir, args.debug, args.debug_stays)
