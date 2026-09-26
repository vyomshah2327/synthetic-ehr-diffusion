import os
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

# MIMIC-IV uses same itemids as MIMIC-III for chartevents
MIMIC4_ITEMIDS = {
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

# NEW: hosp.labevents itemids for features that are actually lab-drawn
# rather than ICU-charted. These are the standard, widely-published MIMIC-IV
# lab itemids (same ones used across common open research tooling). They are
# cross-checked against your own hosp/d_labitems.csv.gz at runtime -- see
# verify_lab_itemids() -- which will WARN (not crash) if any of these don't
# match your actual dictionary file, so double check that warning output.
LAB_ITEMIDS = {
    'Na': [50983],
    'K': [50971],
    'Cl': [50902],
    'CO2': [50882],       # Bicarbonate
    'BUN': [51006],
    'Creatinine': [50912],
    'Mg': [50960],
    'Phos': [50970],
    'Ca': [50893],
    'WBC': [51301],
    'Hgb': [51222],
    'Hct': [51221],
    'Plt': [51265],
    'ALT': [50861],
    'AST': [50878],
    'AlkPhos': [50863],
    'Bili': [50885],
    'Albumin': [50862],
    'Lactate': [50813],
    'Glucose': [50931],
    'PaO2': [50821],
    'PaCO2': [50818],
    'pH': [50820],
}
LAB_ALL_ITEMIDS = [iid for ids in LAB_ITEMIDS.values() for iid in ids]
LAB_ITEMID_TO_FEATURE = {iid: feat for feat, ids in LAB_ITEMIDS.items() for iid in ids}

# Physiologically plausible ranges, used to drop obvious data-entry errors
# (e.g. HeartRate charted as 9999999, DiasBP charted as -12) before they
# reach normalize()'s raw train.min()/train.max(), which has no outlier
# resistance at all -- a single corrupted row is enough to compress the
# entire real clinical range for that feature into a sliver near 0 or 1.
# The original TimeDiff paper avoids this because their preprocessing goes
# through the official MIMIC "concepts" Postgres views (per their README),
# which already validate ranges as part of building those views; this
# reproduction extracts straight from raw chartevents/labevents by itemid,
# so that validation has to happen here instead. Bounds are intentionally
# generous (order-of-magnitude guards against entry errors, not tight
# clinical cutoffs) so genuine extreme-but-real ICU values are never dropped.
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

FEATURES = list(MIMIC4_ITEMIDS.keys())
ALL_ITEMIDS = [iid for ids in MIMIC4_ITEMIDS.values() for iid in ids]
ITEMID_TO_FEATURE = {iid: feat for feat, ids in MIMIC4_ITEMIDS.items() for iid in ids}

_d = sorted({i for i in ALL_ITEMIDS if ALL_ITEMIDS.count(i) > 1})
assert not _d, f'duplicate itemids: {_d}'
assert set(PLAUSIBLE_RANGES.keys()) == set(FEATURES), 'PLAUSIBLE_RANGES must cover every feature'
assert set(LAB_ITEMIDS.keys()) <= set(FEATURES), 'LAB_ITEMIDS keys must be a subset of FEATURES'

SEQ_LEN = 24


def verify_lab_itemids(raw_dir):
    """Cross-check LAB_ITEMIDS against the user's own hosp/d_labitems.csv.gz.
    Warns (does not raise) on mismatches so a run never fails silently on
    wrong itemids, but also never blocks progress if the dictionary file is
    missing or laid out slightly differently than expected."""
    d_path = os.path.join(raw_dir, 'hosp', 'd_labitems.csv.gz')
    if not os.path.exists(d_path):
        print(f"  [verify_lab_itemids] WARNING: {d_path} not found, skipping verification")
        return
    try:
        d_labitems = pd.read_csv(d_path, compression='gzip')
        known_itemids = set(d_labitems['itemid'].tolist())
    except Exception as e:
        print(f"  [verify_lab_itemids] WARNING: could not read d_labitems.csv.gz ({e}), skipping verification")
        return

    missing = [(feat, iid) for feat, ids in LAB_ITEMIDS.items() for iid in ids if iid not in known_itemids]
    if missing:
        print("  [verify_lab_itemids] WARNING: the following assumed lab itemids were NOT found")
        print("  in your d_labitems.csv.gz -- double check these before trusting results:")
        for feat, iid in missing:
            print(f"    feature={feat!r} itemid={iid}")
    else:
        print(f"  [verify_lab_itemids] OK: all {len(LAB_ALL_ITEMIDS)} lab itemids found in d_labitems.csv.gz")


def load_icustays(raw_dir):
    print("Loading ICU stays...")
    stays = pd.read_csv(os.path.join(raw_dir, 'icu', 'icustays.csv.gz'), compression='gzip',
                         usecols=['subject_id', 'hadm_id', 'stay_id', 'intime', 'los'])
    stays['intime'] = pd.to_datetime(stays['intime'])
    stays = stays[stays['los'] >= 1.0]
    print(f"  ICU stays >= 24h: {len(stays)}")
    return stays


def load_mortality(raw_dir):
    print("Loading mortality labels...")
    admissions = pd.read_csv(os.path.join(raw_dir, 'hosp', 'admissions.csv.gz'), compression='gzip',
                              usecols=['hadm_id', 'hospital_expire_flag'])
    return admissions.set_index('hadm_id')['hospital_expire_flag'].to_dict()


def extract_chartevents(raw_dir, stays):
    print("Loading CHARTEVENTS (this will take a while)...")
    stay_ids = set(stays['stay_id'].tolist())
    intime_map = stays.set_index('stay_id')['intime'].to_dict()

    chunks = pd.read_csv(
        os.path.join(raw_dir, 'icu', 'chartevents.csv.gz'),
        compression='gzip',
        usecols=['stay_id', 'itemid', 'charttime', 'valuenum'],
        chunksize=1_000_000,
        low_memory=False
    )

    n_dropped_oob = 0
    n_total = 0
    records = []
    for chunk in tqdm(chunks, desc='Reading CHARTEVENTS'):
        chunk = chunk.dropna(subset=['stay_id', 'valuenum'])
        chunk = chunk[chunk['stay_id'].isin(stay_ids)]
        chunk = chunk[chunk['itemid'].isin(ALL_ITEMIDS)]
        if len(chunk) == 0:
            continue
        chunk['charttime'] = pd.to_datetime(chunk['charttime'])
        chunk['FEATURE'] = chunk['itemid'].map(ITEMID_TO_FEATURE)

        lo = chunk['FEATURE'].map(lambda f: PLAUSIBLE_RANGES[f][0]).values
        hi = chunk['FEATURE'].map(lambda f: PLAUSIBLE_RANGES[f][1]).values
        n_total += len(chunk)
        in_range = (chunk['valuenum'].values >= lo) & (chunk['valuenum'].values <= hi)
        n_dropped_oob += (~in_range).sum()
        chunk = chunk[in_range]

        records.append(chunk[['stay_id', 'charttime', 'FEATURE', 'valuenum']])

    if not records:
        raise ValueError("No matching chart events found.")

    df = pd.concat(records, ignore_index=True)
    print(f"  CHARTEVENTS matched: {len(df)}")
    print(f"  Dropped {n_dropped_oob}/{n_total} ({100*n_dropped_oob/max(n_total,1):.3f}%) "
          f"as physiologically implausible")
    return df, intime_map


def extract_labevents(raw_dir, stays):
    """NEW: extract lab-origin features from hosp/labevents.csv.gz.
    labevents is keyed by hadm_id (not stay_id), so we join through the
    stays table. If an hadm_id maps to multiple ICU stays we attach the lab
    value to all of them (same as how a lab draw applies to the whole
    admission); intime for hour-offset is still taken per-stay."""
    labevents_path = os.path.join(raw_dir, 'hosp', 'labevents.csv.gz')
    if not os.path.exists(labevents_path):
        print(f"  [extract_labevents] {labevents_path} not found -- skipping lab features "
              f"(Na/K/Cl/CO2/BUN/Creatinine/Mg/Phos/Ca/WBC/Hgb/Hct/Plt/ALT/AST/AlkPhos/"
              f"Bili/Albumin/Lactate/Glucose/PaO2/PaCO2/pH will fall back to chartevents-only "
              f"coverage, which is sparse/degenerate for several of them)")
        return None

    print("Loading LABEVENTS (this will take a while)...")
    hadm_to_stays = stays.groupby('hadm_id')['stay_id'].apply(list).to_dict()
    intime_map = stays.set_index('stay_id')['intime'].to_dict()
    valid_hadm_ids = set(stays['hadm_id'].tolist())

    chunks = pd.read_csv(
        labevents_path,
        compression='gzip',
        usecols=['hadm_id', 'itemid', 'charttime', 'valuenum'],
        chunksize=1_000_000,
        low_memory=False
    )

    n_dropped_oob = 0
    n_total = 0
    records = []
    for chunk in tqdm(chunks, desc='Reading LABEVENTS'):
        chunk = chunk.dropna(subset=['hadm_id', 'valuenum'])
        chunk['hadm_id'] = chunk['hadm_id'].astype('Int64')
        chunk = chunk[chunk['hadm_id'].isin(valid_hadm_ids)]
        chunk = chunk[chunk['itemid'].isin(LAB_ALL_ITEMIDS)]
        if len(chunk) == 0:
            continue
        chunk['charttime'] = pd.to_datetime(chunk['charttime'])
        chunk['FEATURE'] = chunk['itemid'].map(LAB_ITEMID_TO_FEATURE)

        lo = chunk['FEATURE'].map(lambda f: PLAUSIBLE_RANGES[f][0]).values
        hi = chunk['FEATURE'].map(lambda f: PLAUSIBLE_RANGES[f][1]).values
        n_total += len(chunk)
        in_range = (chunk['valuenum'].values >= lo) & (chunk['valuenum'].values <= hi)
        n_dropped_oob += (~in_range).sum()
        chunk = chunk[in_range]

        # explode hadm_id -> one or more stay_ids
        rows = []
        for hadm_id, grp in chunk.groupby('hadm_id'):
            for stay_id in hadm_to_stays.get(hadm_id, []):
                g = grp.copy()
                g['stay_id'] = stay_id
                rows.append(g[['stay_id', 'charttime', 'FEATURE', 'valuenum']])
        if rows:
            records.append(pd.concat(rows, ignore_index=True))

    if not records:
        print("  [extract_labevents] WARNING: no matching lab events found -- check LAB_ITEMIDS")
        return None

    df = pd.concat(records, ignore_index=True)
    print(f"  LABEVENTS matched: {len(df)}")
    print(f"  Dropped {n_dropped_oob}/{n_total} ({100*n_dropped_oob/max(n_total,1):.3f}%) "
          f"as physiologically implausible")
    return df


def extract_features(raw_dir, stays):
    chart_df, intime_map = extract_chartevents(raw_dir, stays)
    lab_df = extract_labevents(raw_dir, stays)
    if lab_df is not None:
        df = pd.concat([chart_df, lab_df], ignore_index=True)
        print(f"  Total matched events (chartevents + labevents): {len(df)}")
    else:
        df = chart_df
        print(f"  Total matched events (chartevents only): {len(df)}")
    return df, intime_map


def build_hourly_matrix(df, intime_map, stay_ids):
    print("Building hourly matrices...")
    samples = []
    masks = []
    valid_stay_ids = []

    for stay_id in tqdm(stay_ids, desc='Processing stays'):
        if stay_id not in intime_map:
            continue
        intime = intime_map[stay_id]
        sub = df[df['stay_id'] == stay_id].copy()
        if len(sub) == 0:
            continue

        sub['HOUR'] = ((sub['charttime'] - intime).dt.total_seconds() / 3600).astype(int)
        sub = sub[(sub['HOUR'] >= 0) & (sub['HOUR'] < SEQ_LEN)]

        matrix = np.full((SEQ_LEN, len(FEATURES)), np.nan)
        for _, row in sub.iterrows():
            h = int(row['HOUR'])
            f = FEATURES.index(row['FEATURE'])
            if np.isnan(matrix[h, f]):
                matrix[h, f] = row['valuenum']

        # capture mask before imputation (1=originally observed, 0=missing)
        mask = (~np.isnan(matrix)).astype(np.float32)

        mat_df = pd.DataFrame(matrix, columns=FEATURES)
        mat_df = mat_df.ffill().bfill()
        matrix = mat_df.values

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

    verify_lab_itemids(raw_dir)

    stays = load_icustays(raw_dir)
    if debug:
        stays = stays.sample(min(debug_stays, len(stays)), random_state=42)
        print(f"DEBUG MODE: using {len(stays)} stays")
    mortality = load_mortality(raw_dir)
    df, intime_map = extract_features(raw_dir, stays)

    stay_ids = stays['stay_id'].tolist()
    samples, masks, valid_stay_ids = build_hourly_matrix(df, intime_map, stay_ids)
    print(f"Valid samples: {len(samples)}")

    labels = np.array([mortality.get(
        stays[stays['stay_id'] == sid]['hadm_id'].values[0], 0
    ) for sid in valid_stay_ids])

    train_idx, val_idx, test_idx = split_data(samples, labels)
    train, val, test = samples[train_idx], samples[val_idx], samples[test_idx]
    print(f"Train: {train.shape} | Val: {val.shape} | Test: {test.shape}")

    train_n, val_n, test_n, scaler = normalize(train, val, test)

    np.save(os.path.join(out_dir, 'mimic4_train.npy'), train_n)
    np.save(os.path.join(out_dir, 'mimic4_val.npy'), val_n)
    np.save(os.path.join(out_dir, 'mimic4_test.npy'), test_n)
    np.save(os.path.join(out_dir, 'mimic4_train_mask.npy'), masks[train_idx])
    np.save(os.path.join(out_dir, 'mimic4_val_mask.npy'), masks[val_idx])
    np.save(os.path.join(out_dir, 'mimic4_test_mask.npy'), masks[test_idx])

    with open(os.path.join(out_dir, 'mimic4_scaler.pkl'), 'wb') as f:
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
