"""Torch-free unit tests for the bug classes locked in PLAN.md section 6.

Run with: python -m pytest tests/ -q   (or python tests/test_core.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import AssetData, compute_targets, targets_to_arrays
from src.evaluate import dm_test, newey_west_lrv, qlike
from src.models_econ import RollingConfig, rolling_direct_forecast


def _toy_asset(n: int = 30) -> AssetData:
    idx = pd.bdate_range("2020-01-01", periods=n)
    var_d = np.arange(1.0, n + 1.0)          # var_d[i] = i + 1, easy to check
    df = pd.DataFrame({"var_d": var_d}, index=idx)
    df["rv_ann"] = np.sqrt(252.0 * df["var_d"])
    df["log_rv"] = np.log(df["rv_ann"])
    return AssetData("TOY", "equity", df, "gk")


def test_target_alignment():
    """target_var[i, h] must equal mean(var_d[i+1 : i+1+h]) exactly."""
    asset = _toy_asset(30)
    targets = compute_targets(asset, horizons=(1, 5, 22))
    var_d = asset.df["var_d"].to_numpy()
    for h in (1, 5, 22):
        tv, _ = targets_to_arrays(asset, targets, h)
        for i in range(30):
            expected = (var_d[i + 1: i + 1 + h].mean()
                        if i + h <= 29 else np.nan)
            if np.isnan(expected):
                assert np.isnan(tv[i]), f"h={h} i={i}: expected NaN"
            else:
                assert abs(tv[i] - expected) < 1e-12, f"h={h} i={i}"
    # spot checks by hand: h=1 at i=0 -> var_d[1] = 2
    tv1, _ = targets_to_arrays(asset, targets, 1)
    assert tv1[0] == 2.0
    # h=5 at i=0 -> mean(2..6) = 4
    tv5, _ = targets_to_arrays(asset, targets, 5)
    assert tv5[0] == 4.0
    # h=22 at i=7 -> mean(var_d[8..29]) = mean(9..30) = 19.5
    tv22, _ = targets_to_arrays(asset, targets, 22)
    assert tv22[7] == 19.5


def test_lognormal_correction_recovers_mean():
    """On simulated lognormal data, exp(m + s2/2) recovers E[Y]."""
    rng = np.random.default_rng(42)
    n = 4000
    mu, sigma = -1.0, 0.5
    y_log = rng.normal(mu, sigma, size=n)
    X = np.ones((n, 1))
    true_mean = np.exp(mu + 0.5 * sigma**2)
    cfg = RollingConfig(train_window=1000, refit_every=22)
    test_idx = np.arange(3000, 3200)
    preds = rolling_direct_forecast(y_log, X, test_idx, cfg, h=1,
                                    log_space=True)
    assert np.isfinite(preds).all()
    rel_err = abs(preds.mean() - true_mean) / true_mean
    assert rel_err < 0.05, f"lognormal correction off by {rel_err:.3f}"
    # Omitting the correction would give exp(mu), materially lower:
    naive = np.exp(mu)
    assert preds.mean() > naive * 1.05


def test_rolling_no_lookahead():
    """A shock at the first test origin must not affect its own forecast
    (training pairs are restricted to j <= i - h)."""
    rng = np.random.default_rng(0)
    n = 2000
    y = rng.normal(0.0, 1.0, size=n)
    X = np.ones((n, 1))
    test_idx = np.arange(1500, 1510)
    base = rolling_direct_forecast(y.copy(), X, test_idx,
                                   RollingConfig(), h=5, log_space=False)
    y2 = y.copy()
    y2[1496:1510] = 100.0     # targets inside (i - h, i] and beyond
    bumped = rolling_direct_forecast(y2, X, test_idx,
                                     RollingConfig(), h=5, log_space=False)
    # first forecast trains on j <= 1495 only, so it must be unchanged
    assert abs(base[0] - bumped[0]) < 1e-12


def test_dm_newey_west_vs_statsmodels():
    """NW long-run variance matches statsmodels HAC on a constant-only OLS."""
    import statsmodels.api as sm

    rng = np.random.default_rng(7)
    e = rng.normal(size=600)
    d = e[4:] + 0.6 * e[3:-1] + 0.3 * e[2:-2]     # MA(2)-ish differential
    lag = 4
    T = len(d)
    lrv = newey_west_lrv(d, lag=lag)
    ols = sm.OLS(d, np.ones((T, 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": lag, "use_correction": False})
    se_ours = np.sqrt(lrv / T)
    se_sm = float(ols.bse[0])
    assert abs(se_ours - se_sm) / se_sm < 1e-6

    stat, pval = dm_test(np.abs(d) + 0.5, np.abs(d), h=5)
    assert np.isfinite(stat) and np.isfinite(pval)
    assert stat > 0 and pval < 0.05      # loss_a is strictly worse


def test_qlike_minimum_at_truth():
    t = np.array([2.0, 2.0, 2.0])
    assert qlike(t, t).max() < 1e-12
    assert (qlike(t, t * 1.5) > 0).all()
    assert (qlike(t, t * 0.5) > 0).all()


if __name__ == "__main__":
    for fn in [test_target_alignment, test_lognormal_correction_recovers_mean,
               test_rolling_no_lookahead, test_dm_newey_west_vs_statsmodels,
               test_qlike_minimum_at_truth]:
        fn()
        print(f"PASS {fn.__name__}")
