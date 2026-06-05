"""
NHS avoidable-deaths forecasting — core library (v2 architecture).

Design (see plan): the target is a slowly-drifting level + near-noise. The only
structure beyond the level is day-of-week. So we:
  1. Anchor on a day-of-week-adjusted rolling-mean BASELINE (computed from the RAW
     target, causal w.r.t. the 3-day reporting lag).
  2. Have TabPFN predict the RESIDUAL (deviation from that baseline) from fresh
     operational crowding drivers + calendar + recent level/deviation features.
  3. Blend with a shrinkage weight alpha:  pred = baseline + alpha * tabpfn_residual.
     alpha is tuned per horizon block (days 1-5 and 6-10) and may go to 0, collapsing
     to the safe baseline.

All target history used by the baseline/features respects the 3-day lag (only data
up to D-3 is used for the target). Operational predictors are available up to day D.
"""

import numpy as np
import pandas as pd

TARGET = "estimated_avoidable_deaths"
DATE_COL = "midday_day"

# Feature groups (exact column names verified present in data/processed_tabpfn.csv).
CAL_FEATURES = [
    "sin_dow", "cos_dow", "is_weekend", "is_bank_holiday",
    "sin_month", "cos_month", "winter_indicator",
]
# Operational crowding/delay drivers (the proximal causes of avoidable deaths, fresh to day D).
OPS_FEATURES = ["ptntiae", "nofdtas", "ahmlh", "ahmlh.1", "escltnbo", "chrtnrq"]
# Short rolling/momentum derivatives of the drivers (only kept if present).
OPS_DERIV = [
    "ptntiae_roll_mean_3", "nofdtas_roll_mean_3", "escltnbo_roll_mean_3",
    "chrtnrq_roll_mean_3", "nofdtas_roll_max_3", "chrtnrq_roll_max_14",
    "ptntiae_change_3", "nofdtas_change_3", "escltnbo_change_3",
]
# Recent level / deviation features built on the lagged target (raw, causal).
LEVEL_FEATURES = [
    "ead_roll_mean_7", "ead_roll_mean_14", "ead_roll_mean_30",
    "ead_deviation_from_mean", "ead_z_score_30", "ead_recent_vs_baseline",
    "ead_roll_sd_7", "ead_roll_max_7", "ead_roll_q90_30",
]

# Compact, domain-ranked ~10-feature set (TabPFN does best with few high-signal cols):
# day-of-week + the freshest ED-crowding drivers + where the level sits vs. baseline.
REDUCED_FEATURES = [
    "cos_dow", "sin_dow", "is_bank_holiday",
    "ptntiae", "nofdtas", "escltnbo", "ahmlh.1", "chrtnrq",
    "ead_deviation_from_mean", "ead_z_score_30",
]

# Operational drivers selected by mutual information with the *baseline residual*
# (the part of avoidable deaths the level-baseline misses). These system-pressure
# metrics predict the residual far better than the raw ED-crowding counts, and are
# clinically robust leading indicators. The residual model is fit per horizon, so a
# single union of ACUTE drivers (best for days 1-5) and BACKLOG / waiting-list drivers
# (best for days 6-10) lets each horizon's fit specialise automatically.
OPS_PLUS = [
    # --- acute real-time pressure (strongest for days 1-5) ---
    "pabnu_roll_mean_3",          # P2 (Available) Beds Not Utilised
    "opel", "opel_roll_mean_3",   # Operational Pressures Escalation Level
    "srsblsr_roll_mean_3",        # Sirona South Bristol % staffing reduction
    "cbmrm.1_roll_mean_3",        # Cat-2 BNSSG mean ambulance response
    "cbrh.1_roll_mean_3",         # Cat-2 BNSSG 90th ambulance response
    "aissmtr.1_roll_mean_3",      # 999 answered in 60s since midnight
    "aisilm.1_roll_mean_3",       # 999 answered in 60s, last 15 min (%)
    "astasmtr_roll_mean_3",       # avg speed to answer since midnight
    "nowcotcs_roll_mean_3",       # waiting calls on the 999 stack
    # --- structural backlog / waiting lists (strongest for days 6-10) ---
    "pawlu_roll_mean_3",          # P1 Acute Waiting List (Booked+Unbooked)  (highest MI)
    "dptauwfcamfaiasan_roll_mean_3",  # DtA P1 TOTAL waiting for capacity
    "dapbwfcmfartla_roll_mean_3", # DtA P1 booked, waiting for capacity, medically fit
    "dpbauwfcmfaniaa_roll_mean_3",# DtA P1 booked & un-booked waiting for capacity
    "htcsm_roll_mean_3",          # Handover to Clear 15 mins (since midnight)
    # --- direct ED-crowding drivers ---
    "ptntiae", "nofdtas",         # Patients in A&E, Decisions-To-Admit
]


def load_data(path):
    df = pd.read_csv(path, parse_dates=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    return df


def build_feature_list(df, extra=None, feature_set="full"):
    """Curated, domain-motivated feature set; only columns that actually exist.
    feature_set: 'full' (~31 cols) or 'reduced' (~10 cols)."""
    if feature_set == "reduced":
        wanted = REDUCED_FEATURES + (extra or [])
    elif feature_set == "opsplus":
        wanted = CAL_FEATURES + OPS_PLUS + LEVEL_FEATURES + (extra or [])
    else:
        wanted = CAL_FEATURES + OPS_FEATURES + OPS_DERIV + LEVEL_FEATURES + (extra or [])
    feats = [c for c in wanted if c in df.columns]
    # de-dup, preserve order
    seen, out = set(), []
    for c in feats:
        if c not in seen:
            seen.add(c); out.append(c)
    return out


SENTINEL_THRESHOLD = -999.0   # source data uses -9999 dummies for missing values

def get_arrays(df, feature_cols, verbose=False):
    y = df[TARGET].values.astype(float)
    dow = df[DATE_COL].dt.weekday.values
    month = df[DATE_COL].dt.month.values
    X = df[feature_cols].values.astype(float)
    # Guard against -9999 sentinel/dummy values (used for missing data in the source
    # data, including the assessment period and partial-recording variables). The
    # Kalman imputation in the R prep treats -9999 as a real number, so catch any
    # implausible value here and impute it instead of feeding it to the model. All
    # legitimate feature values are > -100, and rolling-mean dilution of one -9999 is
    # still < -999, so this threshold catches raw and smoothed sentinels alike.
    n_sentinel = int((X <= SENTINEL_THRESHOLD).sum())
    if n_sentinel:
        X[X <= SENTINEL_THRESHOLD] = np.nan
        if verbose:
            print(f"  [sentinel guard] replaced {n_sentinel} dummy (<= {SENTINEL_THRESHOLD}) "
                  f"feature values with column means")
    # mean-impute any residual NaNs in X
    cm = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(cm, inds[1])
    return X, y, dow, month


# ----------------------------------------------------------------------------
# Baseline: day-of-week-adjusted rolling mean of the lagged (>=3 days) target.
# ----------------------------------------------------------------------------

def _level(y, i, w):
    """Mean of w target values ending at index i-3 (the most recent legal value)."""
    lo = max(0, i - 2 - w)
    seg = y[lo:i - 2]            # indices lo .. i-3 inclusive
    return seg.mean() if len(seg) else y[max(0, i - 3)]


def _dow_dev(y, dow, i, dow_window):
    """Additive day-of-week deviations from the mean over a trailing causal window
    of known target values (indices <= i-3)."""
    lo = max(0, (i - 2) - dow_window)
    idx = np.arange(lo, i - 2)   # <= i-3
    if len(idx) < 14:
        return np.zeros(7)
    vals = y[idx]; ds = dow[idx]; g = vals.mean()
    dev = np.zeros(7)
    for d in range(7):
        m = ds == d
        if m.any():
            dev[d] = vals[m].mean() - g
    return dev


def baseline_vector(y, dow, i, horizons, level_window, dow_window):
    """Baseline forecast at origin index i for each horizon in `horizons`."""
    lvl = _level(y, i, level_window)
    dev = _dow_dev(y, dow, i, dow_window)
    return np.array([lvl + dev[dow[i + h]] for h in horizons])


def baseline_scalar(y, dow, i, h, level_window, dow_window):
    lvl = _level(y, i, level_window)
    dev = _dow_dev(y, dow, i, dow_window)
    return lvl + dev[dow[i + h]]


# ----------------------------------------------------------------------------
# Residual TabPFN model, fit per horizon.
# ----------------------------------------------------------------------------

def make_regressor(TabPFNRegressor, n_estimators, random_state=42):
    return TabPFNRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
    )


def fit_predict_horizon(TabPFNRegressor, X, y, dow, i, h,
                        level_window, dow_window, train_window,
                        n_estimators, random_state=42):
    """Train one TabPFN on residual targets for horizon h and return the raw
    residual prediction at origin i (caller adds baseline + alpha scaling)."""
    # Legal training pairs: t -> t+h where the target t+h was observable by day D=i
    # (t+h <= i-3) and within the train window.
    last_t = i - 3 - h
    if last_t < 1:
        return 0.0
    first_t = max(1, last_t - train_window + 1)
    ts = np.arange(first_t, last_t + 1)
    if len(ts) < 30:
        return 0.0
    X_tr = X[ts]
    base_tr = np.array([baseline_scalar(y, dow, t, h, level_window, dow_window) for t in ts])
    y_resid = y[ts + h] - base_tr
    model = make_regressor(TabPFNRegressor, n_estimators, random_state)
    model.fit(X_tr, y_resid)
    return float(model.predict(X[i:i + 1])[0])


# ----------------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------------

def block_mse(pred_mat, act_mat):
    """Return (mse_1_5, mse_6_10) averaged over all periods (competition definition)."""
    p, a = np.asarray(pred_mat), np.asarray(act_mat)
    mse15 = np.nanmean((a[:, :5] - p[:, :5]) ** 2)
    mse610 = np.nanmean((a[:, 5:] - p[:, 5:]) ** 2)
    return float(mse15), float(mse610)


def stratified_origins(df, n_per_month=None, min_i=40, horizon=10, winter_weight=2):
    """Pick validation origins stratified across calendar months, over-weighting
    Oct-Mar (the assessment season). Returns sorted list of origin indices."""
    n = len(df)
    month = df[DATE_COL].dt.month.values
    rng = np.random.default_rng(42)
    origins = []
    valid = np.arange(min_i, n - horizon)
    for m in range(1, 13):
        pool = valid[month[valid] == m]
        if len(pool) == 0:
            continue
        k = (n_per_month or 6)
        if m in (10, 11, 12, 1, 2, 3):
            k *= winter_weight
        k = min(k, len(pool))
        origins.extend(rng.choice(pool, size=k, replace=False).tolist())
    return sorted(set(origins))
