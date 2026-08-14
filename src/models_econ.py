"""Econometric benchmarks: HAR, Log-HAR, AR(1)-log, EWMA, GARCH(1,1).

All models forecast target_var (mean daily variance over the horizon) at each
test origin. Rolling re-estimation per PLAN.md 0.3: 1000 training pairs,
refit every 22 test origins, training origins j <= i - h.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import AssetData, targets_to_arrays

MIN_TRAIN_ROWS = 100


@dataclass
class ForecastResult:
    model: str
    asset: str
    h: int
    frame: pd.DataFrame   # index origin_date; columns: fcst_var float64


@dataclass
class RollingConfig:
    train_window: int = 1000
    refit_every: int = 22
    horizons: tuple[int, ...] = (1, 5, 22)


def rolling_direct_forecast(y: np.ndarray, X: np.ndarray, test_idx: np.ndarray,
                            cfg: RollingConfig, h: int,
                            log_space: bool) -> np.ndarray:
    """Direct-h rolling OLS. Implements PLAN.md 0.3 and, when log_space, the
    lognormal correction of 0.4 with s2 = training residual variance.

    y and X are full-length (n,) and (n, k) arrays aligned to positional
    origins; NaN rows are excluded from training automatically.
    """
    n_feat = X.shape[1]
    preds = np.full(len(test_idx), np.nan)
    beta = None
    s2 = 0.0
    i0 = int(test_idx[0])
    for k, i in enumerate(test_idx):
        i = int(i)
        if beta is None or (i - i0) % cfg.refit_every == 0:
            j_hi = i - h                       # inclusive: j <= i - h
            j_lo = max(0, j_hi - cfg.train_window + 1)
            js = np.arange(j_lo, j_hi + 1)
            rows_ok = np.isfinite(y[js]) & np.isfinite(X[js]).all(axis=1)
            js = js[rows_ok]
            if len(js) >= max(MIN_TRAIN_ROWS, n_feat + 2):
                Xt, yt = X[js], y[js]
                new_beta, _, _, _ = np.linalg.lstsq(Xt, yt, rcond=None)
                resid = yt - Xt @ new_beta
                beta = new_beta
                s2 = float(np.sum(resid**2) / max(1, len(js) - n_feat))
            # else keep previous beta (or stay None and emit NaN)
        if beta is None or not np.isfinite(X[i]).all():
            continue
        m = float(X[i] @ beta)
        preds[k] = np.exp(m + 0.5 * s2) if log_space else m
    return preds


def _har_features(x: pd.Series) -> np.ndarray:
    """[1, x[i], mean(x[i-4:i+1]), mean(x[i-21:i+1])]; windows end AT i."""
    f5 = x.rolling(5).mean()
    f22 = x.rolling(22).mean()
    return np.column_stack([np.ones(len(x)), x.to_numpy(),
                            f5.to_numpy(), f22.to_numpy()])


def forecast_har(asset: AssetData, targets: pd.DataFrame, test_idx: np.ndarray,
                 cfg: RollingConfig) -> list[ForecastResult]:
    X = _har_features(asset.df["var_d"])
    floor = max(1e-12, float(np.nanmin(asset.df["var_d"])) / 10.0)
    out = []
    for h in cfg.horizons:
        tv, _ = targets_to_arrays(asset, targets, h)
        preds = rolling_direct_forecast(tv, X, test_idx, cfg, h, log_space=False)
        preds = np.maximum(preds, floor)
        frame = pd.DataFrame({"fcst_var": preds},
                             index=asset.df.index[test_idx])
        out.append(ForecastResult("HAR", asset.ticker, h, frame))
    return out


def forecast_log_har(asset: AssetData, targets: pd.DataFrame,
                     test_idx: np.ndarray,
                     cfg: RollingConfig) -> list[ForecastResult]:
    """Reference benchmark. Log-space HAR on log_rv, lognormal-corrected."""
    X = _har_features(asset.df["log_rv"])
    out = []
    for h in cfg.horizons:
        _, trv = targets_to_arrays(asset, targets, h)
        y = np.log(trv)
        fcst_rv = rolling_direct_forecast(y, X, test_idx, cfg, h, log_space=True)
        frame = pd.DataFrame({"fcst_var": fcst_rv**2 / asset.ann_factor},
                             index=asset.df.index[test_idx])
        out.append(ForecastResult("LogHAR", asset.ticker, h, frame))
    return out


def forecast_ar1_log(asset: AssetData, targets: pd.DataFrame,
                     test_idx: np.ndarray,
                     cfg: RollingConfig) -> list[ForecastResult]:
    x = asset.df["log_rv"].to_numpy()
    X = np.column_stack([np.ones(len(x)), x])
    out = []
    for h in cfg.horizons:
        _, trv = targets_to_arrays(asset, targets, h)
        y = np.log(trv)
        fcst_rv = rolling_direct_forecast(y, X, test_idx, cfg, h, log_space=True)
        frame = pd.DataFrame({"fcst_var": fcst_rv**2 / asset.ann_factor},
                             index=asset.df.index[test_idx])
        out.append(ForecastResult("AR1", asset.ticker, h, frame))
    return out


def forecast_ewma(asset: AssetData, targets: pd.DataFrame,
                  test_idx: np.ndarray, cfg: RollingConfig,
                  lam: float = 0.94) -> list[ForecastResult]:
    """RiskMetrics EWMA on var_d itself; flat forecast for all horizons."""
    v = asset.df["var_d"].to_numpy()
    n = len(v)
    s = np.empty(n)
    s0 = float(v[:50].mean())
    prev = s0
    for i in range(n):
        prev = lam * prev + (1.0 - lam) * v[i]
        s[i] = prev
    out = []
    for h in cfg.horizons:
        frame = pd.DataFrame({"fcst_var": s[test_idx]},
                             index=asset.df.index[test_idx])
        out.append(ForecastResult("EWMA", asset.ticker, h, frame))
    return out


def forecast_garch(asset: AssetData, targets: pd.DataFrame,
                   test_idx: np.ndarray,
                   cfg: RollingConfig) -> list[ForecastResult]:
    """GARCH(1,1) on 100*ret_cc; conditional-variance forecast mapped to the
    var_d scale via the training-window mean ratio c (PLAN.md 2.2 and 6.8).
    """
    from arch import arch_model
    from arch.utility.exceptions import DataScaleWarning, ConvergenceWarning

    warnings.filterwarnings("ignore", category=DataScaleWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    r = 100.0 * asset.df["ret_cc"].to_numpy()
    var_d = asset.df["var_d"].to_numpy()
    finite_pos = np.where(np.isfinite(r))[0]

    max_h = max(cfg.horizons)
    preds = {h: np.full(len(test_idx), np.nan) for h in cfg.horizons}
    params = None
    c = 1.0
    n_fail = 0
    i0 = int(test_idx[0])

    def window_positions(i: int) -> np.ndarray:
        upto = finite_pos[finite_pos <= i]
        return upto[-cfg.train_window:]

    for k, i in enumerate(test_idx):
        i = int(i)
        pos = window_positions(i)
        if len(pos) < MIN_TRAIN_ROWS:
            continue
        r_win = r[pos]
        if params is None or (i - i0) % cfg.refit_every == 0:
            am = arch_model(r_win, vol="GARCH", p=1, q=1,
                            dist="normal", mean="Constant")
            try:
                res = am.fit(disp="off", show_warning=False)
                if res.convergence_flag == 0 or params is None:
                    params = np.asarray(res.params)
                    cond_var = (res.conditional_volatility**2) / 1e4
                    c = float(np.mean(var_d[pos]) / np.mean(cond_var))
                    c = float(np.clip(c, 1e-3, 1e3))
                    if res.convergence_flag != 0:
                        n_fail += 1
                else:
                    n_fail += 1
            except Exception:
                n_fail += 1
        if params is None:
            continue
        fixed = arch_model(r_win, vol="GARCH", p=1, q=1,
                           dist="normal", mean="Constant").fix(params)
        fc = fixed.forecast(horizon=max_h, reindex=False)
        var_path = fc.variance.to_numpy()[-1] / 1e4     # steps 1..max_h
        for h in cfg.horizons:
            preds[h][k] = c * float(np.mean(var_path[:h]))

    if n_fail:
        print(f"    GARCH {asset.ticker}: {n_fail} refit failures "
              f"(previous fit reused), final scale c={c:.3g}")

    out = []
    for h in cfg.horizons:
        frame = pd.DataFrame({"fcst_var": preds[h]},
                             index=asset.df.index[test_idx])
        out.append(ForecastResult("GARCH", asset.ticker, h, frame))
    return out
