import os
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

FEATURES = [
    'HeartRate', 'SysBP', 'DiasBP', 'MeanBP', 'RespRate', 'SpO2', 'Temp',
    'Glucose', 'WBC', 'Hgb', 'Hct', 'Plt', 'Na', 'K', 'Cl', 'CO2',
    'BUN', 'Creatinine', 'Mg', 'Phos', 'Ca', 'ALT', 'AST', 'AlkPhos',
    'Bili', 'Albumin', 'Lactate', 'PaO2', 'PaCO2', 'pH',
    'FiO2', 'PEEP', 'TidalVol', 'GCSMotor', 'GCSVerbal'
]

VITAL_COLS = {
    'HeartRate': 'heartrate',
    'SysBP': 'systemicsystolic',
    'DiasBP': 'systemicdiastolic',
    'MeanBP': 'systemicmean',
    'RespRate': 'respiration',
    'SpO2': 'sao2',
    'Temp': 'temperature',
}

LAB_COLS = {
    'Glucose': 'glucose',
    'WBC': 'WBC x 1000',
    'Hgb': 'Hgb',
    'Hct': 'Hct',
    'Plt': 'platelets x 1000',
    'Na': 'sodium',
    'K': 'potassium',
    'Cl': 'chloride',
    'CO2': 'bicarbonate',
    'BUN': 'BUN',
    'Creatinine': 'creatinine',
    'Mg': 'magnesium',
    'Phos': 'phosphate',
    'Ca': 'calcium',
    'ALT': 'ALT (SGPT)',
    'AST': 'AST (SGOT)',
    'AlkPhos': 'alkaline phos.',
    'Bili': 'total bilirubin',
    'Albumin': 'albumin',
    'Lactate': 'lactate',
    'PaO2': 'paO2',
    'PaCO2': 'paCO2',
    'pH': 'pH',
}

RESP_COLS = {
    'FiO2': 'FiO2',
    'PEEP': 'PEEP',
    'TidalVol': 'tidal volume (set)',
    'GCSMotor': 'GCS - Motor',
    'GCSVerbal': 'GCS - Verbal',
}

# Physiologically plausible ranges, used to drop obvious data-entry errors
# before they reach normalize()'s raw train.min()/train.max(), which has no
# outlier resistance at all -- a single corrupted row is enough to compress
# the entire real clinical range for that feature into a sliver near 0 or 1.
# The original TimeDiff paper avoids this because their preprocessing goes
# through the official MIMIC "concepts" Postgres views (per their README),
# which already validate ranges as part of building those views; eICU has
# no equivalent pre-validated view, so it's applied directly here. Bounds
# are intentionally generous (order-of-magnitude guards against entry
# errors, not tight clinical cutoffs) so genuine extreme-but-real ICU
# values are never dropped.
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
assert set(PLAUSIBLE_RANGES.keys()) == set(FEATURES), 'PLAUSIBLE_RANGES must cover every feature'

SEQ_LEN = 24


def _apply_range_filter(df, feat_cols):
    """In-place: set out-of-range values to NaN for each feature column present in df."""
    n_dropped = 0
    n_total = 0
    for feat in feat_cols:
        if feat not in df.columns:
            continue
        lo, hi = PLAUSIBLE_RANGES[feat]
        col = df[feat]
        n_total += col.notna().sum()
        oob = col.notna() & ((col < lo) | (col > hi))
        n_dropped += oob.sum()
        df.loc[oob, feat] = np.nan
    if n_total:
        print(f"    dropped {n_dropped}/{n_total} ({100*n_dropped/n_total:.3f}%) "
              f"as physiologically implausible")
    return df


def load_patients(raw_dir):
    print("Loading patients...")
    patients = pd.read_csv(os.path.join(raw_dir, 'patient.csv.gz'), compression='gzip',
                            usecols=['patientunitstayid', 'unitdischargestatus', 'unitdischargeoffset'])
    # keep stays >= 24 hours (offset is in minutes)
    patients = patients[patients['unitdischargeoffset'] >= 1440]
    patients['mortality'] = (patients['unitdischargestatus'] == 'Expired').astype(int)
    print(f"  Stays >= 24h: {len(patients)}")
    return patients


def load_vitals(raw_dir, stay_ids):
    print("Loading vital signs...")
    chunks = pd.read_csv(
        os.path.join(raw_dir, 'vitalPeriodic.csv.gz'),
        compression='gzip',
        usecols=['patientunitstayid', 'observationoffset'] + list(VITAL_COLS.values()),
        chunksize=200_000,
    )
    records = []
    for chunk in tqdm(chunks, desc='Reading vitals'):
        chunk = chunk[chunk['patientunitstayid'].isin(stay_ids)]
        if len(chunk) == 0:
            continue
        chunk = chunk.rename(columns={v: k for k, v in VITAL_COLS.items()})
        chunk['HOUR'] = (chunk['observationoffset'] / 60).astype(int)
        chunk = chunk[(chunk['HOUR'] >= 0) & (chunk['HOUR'] < SEQ_LEN)]
        records.append(chunk[['patientunitstayid', 'HOUR'] + list(VITAL_COLS.keys())])
    if not records:
        return pd.DataFrame()
    df = pd.concat(records, ignore_index=True)
    del records
    df = _apply_range_filter(df, VITAL_COLS.keys())
    return df


def load_labs(raw_dir, stay_ids):
    print("Loading lab values...")
    lab_path = os.path.join(raw_dir, 'lab.csv')
    if not os.path.exists(lab_path):
        lab_path = os.path.join(raw_dir, 'lab.csv.gz')
    compression = None if lab_path.endswith('.csv') else 'gzip'
    chunks = pd.read_csv(
        lab_path,
        compression=compression,
        usecols=['patientunitstayid', 'labresultoffset', 'labname', 'labresult'],
        chunksize=200_000,
    )
    records = []
    lab_names = set(LAB_COLS.values())
    for chunk in tqdm(chunks, desc='Reading labs'):
        chunk = chunk[chunk['patientunitstayid'].isin(stay_ids)]
        chunk = chunk[chunk['labname'].isin(lab_names)]
        if len(chunk) > 0:
            records.append(chunk)
    if not records:
        return pd.DataFrame()
    df = pd.concat(records, ignore_index=True)
    df['FEATURE'] = df['labname'].map({v: k for k, v in LAB_COLS.items()})
    df['HOUR'] = (df['labresultoffset'] / 60).astype(int)
    df = df[(df['HOUR'] >= 0) & (df['HOUR'] < SEQ_LEN)]
    # pivot to wide format
    df = df.groupby(['patientunitstayid', 'HOUR', 'FEATURE'])['labresult'].mean().reset_index()
    df = df.pivot_table(index=['patientunitstayid', 'HOUR'], columns='FEATURE',
                         values='labresult', aggfunc='mean').reset_index()
    df = _apply_range_filter(df, LAB_COLS.keys())
    return df


def load_resp(raw_dir, stay_ids):
    print("Loading respiratory data...")
    try:
        chunks = pd.read_csv(
            os.path.join(raw_dir, 'respiratoryCharting.csv.gz'),
            compression='gzip',
            usecols=['patientunitstayid', 'respchartoffset', 'respchartvaluelabel', 'respchartvalue'],
            chunksize=500_000,
        )
        records = []
        resp_names = set(RESP_COLS.values())
        for chunk in tqdm(chunks, desc='Reading resp'):
            chunk = chunk[chunk['patientunitstayid'].isin(stay_ids)]
            chunk = chunk[chunk['respchartvaluelabel'].isin(resp_names)]
            chunk['respchartvalue'] = pd.to_numeric(chunk['respchartvalue'], errors='coerce')
            chunk = chunk.dropna(subset=['respchartvalue'])
            if len(chunk) > 0:
                records.append(chunk)
        if not records:
            return pd.DataFrame()
        df = pd.concat(records, ignore_index=True)
        df['FEATURE'] = df['respchartvaluelabel'].map({v: k for k, v in RESP_COLS.items()})
        df['HOUR'] = (df['respchartoffset'] / 60).astype(int)
        df = df[(df['HOUR'] >= 0) & (df['HOUR'] < SEQ_LEN)]
        df = df.groupby(['patientunitstayid', 'HOUR', 'FEATURE'])['respchartvalue'].mean().reset_index()
        df = df.pivot_table(index=['patientunitstayid', 'HOUR'], columns='FEATURE',
                             values='respchartvalue', aggfunc='mean').reset_index()
        df = _apply_range_filter(df, RESP_COLS.keys())
        return df
    except Exception as e:
        print(f"  Warning: could not load respiratory data: {e}")
        return pd.DataFrame()


def build_samples(stay_ids, vitals_df, labs_df, resp_df):
    print("Building hourly matrices...")
    samples = []
    masks = []
    valid_ids = []

    g_vit = dict(tuple(vitals_df.groupby('patientunitstayid'))) if len(vitals_df) else {}
    g_lab = dict(tuple(labs_df.groupby('patientunitstayid'))) if len(labs_df) else {}
    g_res = dict(tuple(resp_df.groupby('patientunitstayid'))) if len(resp_df) else {}

    for stay_id in tqdm(stay_ids, desc='Processing stays'):
        matrix = np.full((SEQ_LEN, len(FEATURES)), np.nan)

        # vitals
        if len(vitals_df) > 0:
            sub = g_vit.get(stay_id)
            if sub is None:
                sub = pd.DataFrame()
            for _, row in sub.iterrows():
                h = int(row['HOUR'])
                for feat in VITAL_COLS.keys():
                    f_idx = FEATURES.index(feat)
                    if feat in row and not pd.isna(row[feat]):
                        if np.isnan(matrix[h, f_idx]):
                            matrix[h, f_idx] = row[feat]

        # labs
        if len(labs_df) > 0:
            sub = g_lab.get(stay_id)
            if sub is None:
                sub = pd.DataFrame()
            for _, row in sub.iterrows():
                h = int(row['HOUR'])
                for feat in LAB_COLS.keys():
                    if feat in row.index and not pd.isna(row[feat]):
                        f_idx = FEATURES.index(feat)
                        if np.isnan(matrix[h, f_idx]):
                            matrix[h, f_idx] = row[feat]

        # resp
        if len(resp_df) > 0:
            sub = g_res.get(stay_id)
            if sub is None:
                sub = pd.DataFrame()
            for _, row in sub.iterrows():
                h = int(row['HOUR'])
                for feat in RESP_COLS.keys():
                    if feat in row.index and not pd.isna(row[feat]):
                        f_idx = FEATURES.index(feat)
                        if np.isnan(matrix[h, f_idx]):
                            matrix[h, f_idx] = row[feat]

        # capture mask before imputation (1=originally observed, 0=missing)
        mask = (~np.isnan(matrix)).astype(np.float32)

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
        valid_ids.append(stay_id)

    return np.array(samples), np.array(masks), valid_ids


def normalize(train, val, test):
    mins = train.min(axis=(0, 1), keepdims=True)
    maxs = train.max(axis=(0, 1), keepdims=True)
    rng = np.where(maxs - mins == 0, 1.0, maxs - mins)
    scaler = {'min': mins.squeeze(), 'max': maxs.squeeze()}
    return (train - mins) / rng, (val - mins) / rng, (test - mins) / rng, scaler


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

    patients = load_patients(raw_dir)
    if debug:
        patients = patients.sample(min(debug_stays, len(patients)), random_state=42)
        print(f"DEBUG MODE: using {len(patients)} stays")
    stay_ids = set(patients['patientunitstayid'].tolist())

    vitals_df = load_vitals(raw_dir, stay_ids)
    labs_df = load_labs(raw_dir, stay_ids)
    resp_df = load_resp(raw_dir, stay_ids)

    samples, masks, valid_ids = build_samples(sorted(stay_ids), vitals_df, labs_df, resp_df)
    print(f"Valid samples: {len(samples)}")

    mortality_map = patients.set_index('patientunitstayid')['mortality'].to_dict()
    labels = np.array([mortality_map.get(sid, 0) for sid in valid_ids])

    train_idx, val_idx, test_idx = split_data(samples, labels)
    train, val, test = samples[train_idx], samples[val_idx], samples[test_idx]
    print(f"Train: {train.shape} | Val: {val.shape} | Test: {test.shape}")

    train_n, val_n, test_n, scaler = normalize(train, val, test)

    np.save(os.path.join(out_dir, 'eicu_train.npy'), train_n)
    np.save(os.path.join(out_dir, 'eicu_val.npy'), val_n)
    np.save(os.path.join(out_dir, 'eicu_test.npy'), test_n)
    np.save(os.path.join(out_dir, 'eicu_train_mask.npy'), masks[train_idx])
    np.save(os.path.join(out_dir, 'eicu_val_mask.npy'), masks[val_idx])
    np.save(os.path.join(out_dir, 'eicu_test_mask.npy'), masks[test_idx])

    with open(os.path.join(out_dir, 'eicu_scaler.pkl'), 'wb') as f:
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
