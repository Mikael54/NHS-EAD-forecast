"""Generate the demonstration submission with the FINAL locked config:
  days 1-5  : baseline + 0.8 * TabPFN-residual on opsplus features (retrained every 21 days)
  days 6-10 : baseline only (alpha=0)
over a contiguous validation region, and write submission/pred_matrix.csv + mse_summary.csv.
Prints final block MSEs (overall + winter). Only the 1-5 block needs TabPFN fits.
"""
import argparse, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import nhs_forecast_core as C

LW, DW = 45, 99999
A15, A610 = 0.8, 0.0


def get_cls(api):
    if api:
        import os
        from tabpfn_client import TabPFNRegressor, set_access_token
        if os.path.exists(".tabpfn_key"):
            set_access_token(open(".tabpfn_key").read().strip())
        return TabPFNRegressor
    from tabpfn import TabPFNRegressor
    return TabPFNRegressor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", action="store_true")
    ap.add_argument("--n-est", type=int, default=8)
    ap.add_argument("--retrain-every", type=int, default=21)
    ap.add_argument("--train-window", type=int, default=365)
    ap.add_argument("--test-start-frac", type=float, default=0.45)
    args = ap.parse_args()
    Reg = get_cls(args.api)

    df = C.load_data("data/processed_tabpfn.csv")
    y = df[C.TARGET].values.astype(float)
    dow = df[C.DATE_COL].dt.weekday.values
    month = df[C.DATE_COL].dt.month.values
    n = len(y)
    clip_min = 0.5 * np.nanmin(y)

    feats15 = C.build_feature_list(df, feature_set="opsplus")
    X15, _, _, _ = C.get_arrays(df, feats15)
    print(f"days 1-5 residual on {len(feats15)} opsplus features; alpha={A15}")

    origins = np.arange(int(n * args.test_start_frac), n - 10)
    winter = np.isin(month[origins], [10, 11, 12, 1, 2, 3])
    buckets = origins // args.retrain_every

    pred = np.full((len(origins), 10), np.nan)
    act = np.array([y[i + 1:i + 11] for i in origins])

    # days 6-10: baseline only
    for hi, h in enumerate(range(6, 11)):
        pred[:, 5 + hi] = [C.baseline_scalar(y, dow, i, h, LW, DW) for i in origins]

    # days 1-5: baseline + A15 * residual (retrain every N, batched predict)
    import time
    t0 = time.time()
    for hi, h in enumerate(range(1, 6)):
        base_h = np.array([C.baseline_scalar(y, dow, i, h, LW, DW) for i in origins])
        resid_h = np.zeros(len(origins))
        for bk in np.unique(buckets):
            sel = np.where(buckets == bk)[0]
            i0 = origins[sel[0]]
            last_t = i0 - 3 - h
            first_t = max(1, last_t - args.train_window + 1)
            ts = np.arange(first_t, last_t + 1)
            if len(ts) < 30:
                continue
            btr = np.array([C.baseline_scalar(y, dow, t, h, LW, DW) for t in ts])
            m = Reg(n_estimators=args.n_est, random_state=42)
            m.fit(X15[ts], y[ts + h] - btr)
            resid_h[sel] = m.predict(X15[origins[sel]])
        pred[:, hi] = base_h + A15 * resid_h
        print(f"  horizon {h} done, elapsed={ (time.time()-t0)/60:.1f}m", flush=True)

    pred = np.clip(pred, clip_min, None)

    # write submission
    cols = ["forecast_id"] + [f"day_{i}" for i in range(1, 11)]
    pd.DataFrame(
        [{"forecast_id": k + 1, **{f"day_{j+1}": float(pred[k, j]) for j in range(10)}}
         for k in range(len(pred))], columns=cols
    ).to_csv("submission/pred_matrix.csv", index=False)
    pd.DataFrame(
        [{"forecast_id": k + 1,
          "mse_1_5": float(np.nanmean((act[k, :5] - pred[k, :5]) ** 2)),
          "mse_6_10": float(np.nanmean((act[k, 5:] - pred[k, 5:]) ** 2))}
         for k in range(len(pred))]
    ).to_csv("submission/mse_summary.csv", index=False)

    def blk(cols, mask):
        return float(np.nanmean((act[mask][:, cols] - pred[mask][:, cols]) ** 2))
    c15, c610 = slice(0, 5), slice(5, 10)
    allm = np.ones(len(origins), bool)
    print(f"\nWrote submission over {len(origins)} contiguous origins "
          f"({df[C.DATE_COL].iloc[origins[0]].date()} .. {df[C.DATE_COL].iloc[origins[-1]].date()}), "
          f"{int(winter.sum())} winter")
    print(f"{'set':>8}{'MSE_1_5':>9}{'MSE_6_10':>9}")
    print(f"{'ALL':>8}{blk(c15,allm):>9.4f}{blk(c610,allm):>9.4f}")
    print(f"{'WINTER':>8}{blk(c15,winter):>9.4f}{blk(c610,winter):>9.4f}")


if __name__ == "__main__":
    main()
