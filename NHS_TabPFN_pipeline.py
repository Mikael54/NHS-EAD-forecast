"""
NHS TabPFN Forecasting Pipeline (v2)
====================================
Architecture for the SPHERE-PPL NHS avoidable-deaths contest.

The target is a slowly-drifting level plus near-unpredictable noise; the only
structure beyond the level is day-of-week. So we forecast with three layers:

  1. BASELINE  — a day-of-week-adjusted rolling mean of the lagged (>=3 day) target,
     computed from the RAW target. This alone beats a plain absolute-value TabPFN.
  2. RESIDUAL  — TabPFN predicts the deviation from the baseline using fresh
     operational crowding/delay drivers (Patients in A&E, DTAs, ambulance handovers,
     escalation beds, cohorting), calendar encodings and recent level/deviation features.
  3. BLEND     — pred = baseline + alpha * tabpfn_residual, with a per-horizon-block
     shrinkage weight alpha (the residual is mostly noise; over-confident corrections
     raise MSE). alpha may be 0, collapsing to the safe baseline.

The two prize blocks (days 1-5 and days 6-10) are configured independently.
All target history respects the 3-day reporting lag; operational predictors are
available up to day D.

Engine: local `tabpfn` by default (free, reproducible, no API key). `tabpfn_client`
(cloud) is available via --api.

Usage:
  python NHS_TabPFN_pipeline.py --mode validate            # rolling validation, prints MSE 1-5 / 6-10
  python NHS_TabPFN_pipeline.py --mode validate --stratified --max-origins 40
  python NHS_TabPFN_pipeline.py --mode assess              # full rolling forecast -> submission files
  python NHS_TabPFN_pipeline.py --mode assess --api        # use cloud tabpfn_client instead of local
"""

import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd

import nhs_forecast_core as C

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION (per prize block; populated/overridden by model_config.json)
# ============================================================================

DATA_PATH = "data/processed_tabpfn.csv"
CONFIG_PATH = "data/model_config.json"
SUBMISSION_DIR = "submission"
RESULTS_DIR = "results"

# Defaults: baseline locked by free grid search (level_window=45, full-history dow);
# alpha set from validation (Step B). Edit via model_config.json.
DEFAULT_CONFIG = {
    "block_15": {
        "horizons": [1, 2, 3, 4, 5], "feature_set": "opsplus",
        "level_window": 45, "dow_window": 99999,
        "train_window": 365, "n_estimators": 8, "alpha": 0.8,
    },
    "block_610": {
        "horizons": [6, 7, 8, 9, 10], "feature_set": "full",
        "level_window": 45, "dow_window": 99999,
        "train_window": 365, "n_estimators": 8, "alpha": 0.0,
    },
    "clip_min_frac": 0.5,   # floor predictions at clip_min_frac * historical min target
}


def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
        print(f"Loaded model config from {CONFIG_PATH}")
        return cfg
    return DEFAULT_CONFIG


def load_tabpfn(use_api: bool):
    if use_api:
        try:
            from tabpfn_client import TabPFNRegressor, set_access_token
            key_file = os.path.join(os.path.dirname(__file__), ".tabpfn_key")
            if os.path.exists(key_file):
                with open(key_file) as f:
                    key = f.read().strip()
                if key:
                    set_access_token(key)
            print("Engine: tabpfn_client (cloud API)")
            return TabPFNRegressor
        except ImportError:
            print("tabpfn_client unavailable; falling back to local tabpfn.")
    from tabpfn import TabPFNRegressor
    print("Engine: local tabpfn (CPU, no API)")
    return TabPFNRegressor


# ============================================================================
# CORE: forecast one origin (rolling — residual models retrained per origin)
# ============================================================================

def forecast_origin(TabPFNRegressor, Xmap, y, dow, i, cfg, clip_min):
    """Return a 10-vector forecast for origin index i (predicting i+1..i+10).
    Xmap maps a feature_set name -> its feature matrix, so each prize block can use
    its own feature set."""
    preds = np.full(10, np.nan)
    for block in ("block_15", "block_610"):
        b = cfg[block]
        X = Xmap[b.get("feature_set", "full")]
        for h in b["horizons"]:
            base = C.baseline_scalar(y, dow, i, h, b["level_window"], b["dow_window"])
            if b["alpha"] != 0.0:
                rp = C.fit_predict_horizon(
                    TabPFNRegressor, X, y, dow, i, h,
                    b["level_window"], b["dow_window"], b["train_window"], b["n_estimators"])
                preds[h - 1] = base + b["alpha"] * rp
            else:
                preds[h - 1] = base
    return np.clip(preds, clip_min, None)


# ============================================================================
# ROLLING FORECAST over a set of origins
# ============================================================================

def build_feature_maps(df, cfg, verbose=True):
    """Build one feature matrix per distinct feature_set used across the blocks."""
    sets = {cfg[b].get("feature_set", "full") for b in ("block_15", "block_610")}
    Xmap, y, dow = {}, None, None
    for s in sets:
        feats = C.build_feature_list(df, feature_set=s)
        X, y, dow, _ = C.get_arrays(df, feats)
        Xmap[s] = X
        if verbose:
            print(f"  feature set '{s}': {len(feats)} features")
    return Xmap, y, dow


def rolling_forecast(TabPFNRegressor, df, cfg, origins, verbose=True):
    Xmap, y, dow = build_feature_maps(df, cfg, verbose)
    clip_min = cfg.get("clip_min_frac", 0.5) * np.nanmin(y)

    pred_mat = np.full((len(origins), 10), np.nan)
    act_mat = np.full((len(origins), 10), np.nan)
    dates = []
    t0 = time.time()
    for oi, i in enumerate(origins):
        pred_mat[oi] = forecast_origin(TabPFNRegressor, Xmap, y, dow, i, cfg, clip_min)
        act_mat[oi] = y[i + 1:i + 11]
        dates.append(df[C.DATE_COL].iloc[i])
        if verbose and (oi % 10 == 0 or oi == len(origins) - 1):
            el = time.time() - t0
            m15, m610 = C.block_mse(pred_mat[:oi + 1], act_mat[:oi + 1])
            eta = el / (oi + 1) * (len(origins) - oi - 1)
            print(f"  {oi+1}/{len(origins)} origin={dates[-1].date()} "
                  f"MSE1-5={m15:.4f} MSE6-10={m610:.4f} "
                  f"elapsed={el/60:.1f}m ETA={eta/60:.1f}m", flush=True)
    return pred_mat, act_mat, dates


# ============================================================================
# OUTPUT
# ============================================================================

def write_submission(pred_mat, act_mat=None):
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    cols = ["forecast_id"] + [f"day_{d}" for d in range(1, 11)]
    rows = [{"forecast_id": i + 1, **{f"day_{d+1}": float(pred_mat[i, d]) for d in range(10)}}
            for i in range(len(pred_mat))]
    pd.DataFrame(rows, columns=cols).to_csv(os.path.join(SUBMISSION_DIR, "pred_matrix.csv"), index=False)
    print(f"Predictions -> {SUBMISSION_DIR}/pred_matrix.csv ({len(rows)} rows)")

    if act_mat is not None and not np.all(np.isnan(act_mat)):
        mse_rows = []
        for i in range(len(pred_mat)):
            a, p = act_mat[i], pred_mat[i]
            mse_rows.append({"forecast_id": i + 1,
                             "mse_1_5": float(np.nanmean((a[:5] - p[:5]) ** 2)),
                             "mse_6_10": float(np.nanmean((a[5:] - p[5:]) ** 2))})
        mse_df = pd.DataFrame(mse_rows)
        mse_df.to_csv(os.path.join(SUBMISSION_DIR, "mse_summary.csv"), index=False)
        m15, m610 = C.block_mse(pred_mat, act_mat)
        print(f"MSE summary -> {SUBMISSION_DIR}/mse_summary.csv")
        print(f"  Mean MSE days 1-5 : {m15:.4f}  (RMSE {m15**0.5:.4f})")
        print(f"  Mean MSE days 6-10: {m610:.4f}  (RMSE {m610**0.5:.4f})")


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="NHS TabPFN Forecasting Pipeline v2")
    ap.add_argument("--mode", choices=["validate", "assess"], default="validate")
    ap.add_argument("--api", action="store_true", help="use tabpfn_client cloud API")
    ap.add_argument("--max-origins", type=int, default=None,
                    help="cap number of origins (validate/assess)")
    ap.add_argument("--stratified", action="store_true",
                    help="validate on a stratified, winter-weighted subset of origins")
    ap.add_argument("--assess-start", default="2025-10-01",
                    help="first day of the assessment forecast window (D+1)")
    ap.add_argument("--assess-end", default="2026-03-31",
                    help="last day of the assessment forecast window (D+10)")
    args = ap.parse_args()

    cfg = load_config()
    df = C.load_data(DATA_PATH)
    n = len(df)
    print(f"Loaded {n} rows, {df[C.DATE_COL].min().date()} .. {df[C.DATE_COL].max().date()}")
    TabPFNRegressor = load_tabpfn(args.api)

    if args.mode == "validate":
        if args.stratified:
            origins = C.stratified_origins(df, n_per_month=6)
        else:
            lo = int(n * 0.40)
            origins = list(range(max(lo, 60), n - 10))
        if args.max_origins:
            origins = origins[:args.max_origins]
        print(f"Validation over {len(origins)} origins")
        pred_mat, act_mat, _ = rolling_forecast(TabPFNRegressor, df, cfg, origins)
        write_submission(pred_mat, act_mat)

    elif args.mode == "assess":
        # Emit exactly the competition's forecast periods: each origin i forecasts
        # df dates [i+1 .. i+10]; keep those whose whole 10-day window lies within
        # [assess_start, assess_end] (Oct 2025-Mar 2026 -> 173 periods).
        dates = df[C.DATE_COL]
        start, end = pd.Timestamp(args.assess_start), pd.Timestamp(args.assess_end)
        origins = [i for i in range(60, n - 10)
                   if dates.iloc[i + 1] >= start and dates.iloc[i + 10] <= end]
        if not origins:
            print(f"WARNING: no origins fall in [{start.date()}, {end.date()}] "
                  f"(data ends {dates.max().date()}). Did you regenerate the data with the "
                  f"assessment period included? Falling back to all available origins.")
            origins = list(range(60, n - 10))
        if args.max_origins:
            origins = origins[-args.max_origins:]
        print(f"Assessment forecast: {len(origins)} periods "
              f"({dates.iloc[origins[0]+1].date()}..{dates.iloc[origins[-1]+10].date()}), rolling retrain")
        pred_mat, act_mat, _ = rolling_forecast(TabPFNRegressor, df, cfg, origins)
        write_submission(pred_mat, act_mat)


if __name__ == "__main__":
    main()
