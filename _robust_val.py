"""Robust, cheap validation: retrain residual models every RETRAIN_EVERY origins
(not every origin), evaluate over ALL test origins (hundreds), sweep alpha per block.
This gives a low-variance estimate of whether the TabPFN residual genuinely helps,
unlike the noisy 24/42-origin rolling samples.
"""
import argparse, warnings, numpy as np
warnings.filterwarnings("ignore")
import nhs_forecast_core as C

LW, DW = 45, 99999


def get_cls(api):
    if api:
        import os
        from tabpfn_client import TabPFNRegressor, set_access_token
        if os.path.exists(".tabpfn_key"):
            set_access_token(open(".tabpfn_key").read().strip())
        return TabPFNRegressor
    from tabpfn import TabPFNRegressor
    return TabPFNRegressor


def run(featset, block_cols, df, y, dow, month, Reg, train_window, n_est,
        retrain_every, test_start_frac, n_est_label):
    feats = C.build_feature_list(df, feature_set=featset)
    X, _, _, _ = C.get_arrays(df, feats)
    n = len(y)
    test0 = int(n * test_start_frac)
    origins = np.arange(max(test0, 120), n - 10)
    winter = np.isin(month[origins], [10, 11, 12, 1, 2, 3])
    H = list(range(block_cols[0] + 1, block_cols[1] + 1))  # horizons in this block
    buckets = origins // retrain_every

    base = np.array([[C.baseline_scalar(y, dow, i, h, LW, DW) for h in H] for i in origins])
    act = np.array([[y[i + h] for h in H] for i in origins])
    resid = np.zeros((len(origins), len(H)))

    # one model per (horizon, retrain-bucket); BATCH-predict all origins in the bucket.
    for hi, h in enumerate(H):
        for bk in np.unique(buckets):
            sel = np.where(buckets == bk)[0]
            i0 = origins[sel[0]]                    # train as known at bucket start (causal)
            last_t = i0 - 3 - h
            first_t = max(1, last_t - train_window + 1)
            ts = np.arange(first_t, last_t + 1)
            if len(ts) < 30:
                continue
            btr = np.array([C.baseline_scalar(y, dow, t, h, LW, DW) for t in ts])
            m = Reg(n_estimators=n_est, random_state=42)
            m.fit(X[ts], y[ts + h] - btr)
            resid[sel, hi] = m.predict(X[origins[sel]])   # one batched predict
    return base, resid, act, winter, len(origins), int(winter.sum())


def sweep(base, resid, act, mask, label):
    best = (9, 0)
    line = []
    for a in np.round(np.arange(0, 1.01, 0.1), 1):
        p = base[mask] + a * resid[mask]
        mse = np.nanmean((act[mask] - p) ** 2)
        line.append((a, mse))
        if mse < best[0]:
            best = (mse, a)
    print(f"  {label}: baseline={line[0][1]:.4f}  best={best[0]:.4f}@a={best[1]}  "
          + " ".join(f"{a}:{m:.4f}" for a, m in line if a in (0.0,0.2,0.4,0.6,0.8,1.0)))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", action="store_true")
    ap.add_argument("--train-window", type=int, default=365)
    ap.add_argument("--n-est", type=int, default=4)
    ap.add_argument("--retrain-every", type=int, default=14)
    ap.add_argument("--test-start-frac", type=float, default=0.45)
    args = ap.parse_args()
    Reg = get_cls(args.api)
    df = C.load_data("data/processed_tabpfn.csv")
    y = df[C.TARGET].values.astype(float); dow = df[C.DATE_COL].dt.weekday.values
    month = df[C.DATE_COL].dt.month.values

    for featset, cols, name in [("opsplus", (0, 5), "days 1-5"), ("full", (5, 10), "days 6-10")]:
        b, r, a, w, no, nw = run(featset, cols, df, y, dow, month, Reg,
                                 args.train_window, args.n_est, args.retrain_every,
                                 args.test_start_frac, args.n_est)
        print(f"\n=== {name} ({featset}); {no} test origins, {nw} winter; "
              f"retrain_every={args.retrain_every}, tw={args.train_window}, n_est={args.n_est} ===")
        sweep(b, r, a, np.ones(no, bool), "ALL   ")
        sweep(b, r, a, w, "WINTER")


if __name__ == "__main__":
    main()
