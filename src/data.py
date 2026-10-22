"""Download, clean, and compute realized-variance estimators from daily OHLC.

Conventions (PLAN.md section 0):
- var_d[i]: daily variance proxy for day i (unannualized, squared-log-return units).
- rv_ann[i] = sqrt(252 * var_d[i]), log_rv[i] = ln(rv_ann[i]).
- target_var[i, h] = mean(var_d[i+1 : i+1+h]) computed at forecast origin i.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

START_DATE = "2005-01-01"
MIN_ROWS = 2000
HORIZONS = (1, 5, 22)
TRAIN_WINDOW = 1000
CONTEXT_LEN = 512
FEATURE_WARMUP = 21
TEST_DAYS_FULL = 750
TEST_DAYS_QUICK = 200
BURN_IN = 250

UNIVERSE: list[tuple[str, str]] = [
    ("SPY", "equity"), ("QQQ", "equity"), ("DIA", "equity"), ("IWM", "equity"),
    ("XLF", "equity"), ("XLK", "equity"), ("XLE", "equity"), ("XLV", "equity"),
    ("XLI", "equity"), ("XLP", "equity"), ("XLU", "equity"), ("EEM", "equity"),
    ("EFA", "equity"),
    ("EURUSD=X", "fx"), ("GBPUSD=X", "fx"), ("USDJPY=X", "fx"),
    ("AUDUSD=X", "fx"), ("USDCAD=X", "fx"), ("USDCHF=X", "fx"), ("NZDUSD=X", "fx"),
    ("ES=F", "futures"), ("NQ=F", "futures"), ("YM=F", "futures"),
    ("GC=F", "futures"), ("SI=F", "futures"), ("CL=F", "futures"),
    ("ZN=F", "futures"), ("ZB=F", "futures"), ("HG=F", "futures"), ("NG=F", "futures"),
]

QUICK_TICKERS = ["SPY", "QQQ", "EEM", "EURUSD=X", "USDJPY=X", "CL=F"]

REQUIRED_COLS = ["open", "high", "low", "close", "volume"]


@dataclass
class AssetData:
    ticker: str
    asset_class: str            # "equity" | "fx" | "futures" | "crypto"
    df: pd.DataFrame            # schema in PLAN.md 1.2
    rv_estimator: str           # "gk" | "parkinson" | "rv5m" (what var_d uses)
    notes: dict = field(default_factory=dict)
    ann_factor: float = 252.0   # annualization constant: 252 Track A, 365 crypto


def safe_ticker(ticker: str) -> str:
    return ticker.replace("=", "_")


def download_ohlc(ticker: str, start: str, cache_dir: Path, force: bool = False) -> pd.DataFrame:
    """Download one ticker via yfinance, normalize columns, cache to parquet."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"ohlc_{safe_ticker(ticker)}.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)

    import yfinance as yf

    raw = yf.download(ticker, start=start, auto_adjust=False, progress=False)
    if raw is None or len(raw) == 0:
        raise ValueError(f"empty download for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)
    raw.columns = [str(c).strip().lower().replace(" ", "_") for c in raw.columns]
    missing = [c for c in REQUIRED_COLS if c not in raw.columns]
    if missing:
        raise ValueError(f"{ticker}: missing columns {missing} after normalize")
    keep = REQUIRED_COLS + (["adj_close"] if "adj_close" in raw.columns else [])
    raw = raw[keep].astype("float64")
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
    raw.index.name = "date"
    raw = raw[~raw.index.duplicated(keep="first")].sort_index()
    raw.to_parquet(path)
    return raw


def clean_ohlc(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Drop invalid OHLC rows per PLAN.md 1.3. Returns open/high/low/close only."""
    df = raw[["open", "high", "low", "close"]].copy()
    df = df.dropna()
    df = df[(df > 0).all(axis=1)]

    mx = df[["open", "close"]].max(axis=1)
    mn = df[["open", "close"]].min(axis=1)
    tol = 1e-8
    # Snap tiny violations, then drop the remaining hard violations.
    snap_hi = (df["high"] < mx) & ((mx - df["high"]) <= tol * mx)
    df.loc[snap_hi, "high"] = mx[snap_hi]
    snap_lo = (df["low"] > mn) & ((df["low"] - mn) <= tol * mn)
    df.loc[snap_lo, "low"] = mn[snap_lo]
    ok = (df["high"] >= mx) & (df["low"] <= mn)
    df = df[ok]
    df = df[df["high"] > df["low"]]
    return df


def garman_klass_var(df: pd.DataFrame) -> pd.Series:
    hl = np.log(df["high"] / df["low"])
    co = np.log(df["close"] / df["open"])
    return 0.5 * hl**2 - (2.0 * math.log(2.0) - 1.0) * co**2


def parkinson_var(df: pd.DataFrame) -> pd.Series:
    hl = np.log(df["high"] / df["low"])
    return hl**2 / (4.0 * math.log(2.0))


def yang_zhang_var(df: pd.DataFrame, window: int = 22) -> pd.Series:
    o = np.log(df["open"] / df["close"].shift(1))
    c = np.log(df["close"] / df["open"])
    rs = (np.log(df["high"] / df["close"]) * np.log(df["high"] / df["open"])
          + np.log(df["low"] / df["close"]) * np.log(df["low"] / df["open"]))
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    var_o = o.rolling(window).var()
    var_c = c.rolling(window).var()
    rs_mean = rs.rolling(window).mean()
    return var_o + k * var_c + (1.0 - k) * rs_mean


def build_asset(ticker: str, asset_class: str, cache_dir: Path,
                force: bool = False,
                end_date: str | None = None) -> AssetData | None:
    """end_date (inclusive, 'YYYY-MM-DD') truncates the cached download so a
    run can be pinned to a fixed data window."""
    try:
        raw = download_ohlc(ticker, START_DATE, cache_dir, force=force)
    except Exception as exc:
        print(f"  DROP {ticker}: download failed ({exc})")
        return None
    if end_date is not None:
        raw = raw[raw.index <= pd.Timestamp(end_date)]

    df = clean_ohlc(raw, ticker)
    if len(df) < MIN_ROWS:
        print(f"  DROP {ticker}: only {len(df)} rows after cleaning")
        return None

    df["var_gk"] = garman_klass_var(df)
    df["var_park"] = parkinson_var(df)
    df["var_yz"] = yang_zhang_var(df)

    estimator = "gk"
    zero_oc_frac = float((np.abs(np.log(df["close"] / df["open"])) < 1e-6).mean())
    if asset_class == "fx" and zero_oc_frac > 0.10:
        estimator = "parkinson"

    if estimator == "gk":
        var_d = df["var_gk"].copy()
        neg = ~(var_d > 0)
        gk_fallback_frac = float(neg.mean())
        var_d[neg] = df["var_park"][neg]
    else:
        var_d = df["var_park"].copy()
        gk_fallback_frac = 0.0

    keep = np.isfinite(var_d) & (var_d > 0)
    dropped_nonpos = int((~keep).sum())
    df = df[keep].copy()
    var_d = var_d[keep]

    lo, hi = np.percentile(var_d, [0.01, 99.9])
    df["var_d"] = var_d.clip(lo, hi)
    df["ret_cc"] = np.log(df["close"] / df["close"].shift(1))
    df["rv_ann"] = np.sqrt(252.0 * df["var_d"])
    df["log_rv"] = np.log(df["rv_ann"])

    assert np.isfinite(df["log_rv"].to_numpy()).all(), f"{ticker}: non-finite log_rv"

    if len(df) < MIN_ROWS:
        print(f"  DROP {ticker}: only {len(df)} rows after estimator cleaning")
        return None

    notes = {
        "rows": len(df),
        "estimator": estimator,
        "zero_oc_frac": round(zero_oc_frac, 4),
        "gk_fallback_frac": round(gk_fallback_frac, 4),
        "dropped_nonpos_var": dropped_nonpos,
        "first_date": str(df.index[0].date()),
        "last_date": str(df.index[-1].date()),
    }
    return AssetData(ticker=ticker, asset_class=asset_class, df=df,
                     rv_estimator=estimator, notes=notes)


def load_universe(cache_dir: Path, quick: bool = False,
                  force: bool = False,
                  end_date: str | None = None) -> list[AssetData]:
    universe = UNIVERSE
    if quick:
        wanted = set(QUICK_TICKERS)
        universe = [(t, c) for t, c in UNIVERSE if t in wanted]
    assets = []
    for ticker, asset_class in universe:
        asset = build_asset(ticker, asset_class, cache_dir, force=force,
                            end_date=end_date)
        if asset is not None:
            print(f"  OK   {ticker} ({asset_class}, {asset.rv_estimator}, "
                  f"{len(asset.df)} rows)")
            assets.append(asset)
    return assets


def compute_targets(asset: AssetData,
                    horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    """Long frame: origin_date, h, target_var, target_rv.

    target_var at origin i = mean(var_d[i+1 : i+1+h]); implemented as
    rolling(h).mean().shift(-h), which at row i equals the rolling value at
    row i+h, i.e. the mean over rows i+1 .. i+h.
    """
    var_d = asset.df["var_d"]
    frames = []
    for h in horizons:
        tv = var_d.rolling(h).mean().shift(-h)
        sub = pd.DataFrame({
            "origin_date": asset.df.index,
            "h": np.int8(h),
            "target_var": tv.to_numpy(),
        })
        sub = sub.dropna(subset=["target_var"])
        sub["target_rv"] = np.sqrt(asset.ann_factor * sub["target_var"])
        frames.append(sub)
    out = pd.concat(frames, ignore_index=True)
    out["h"] = out["h"].astype("int8")
    return out


def targets_to_arrays(asset: AssetData, targets: pd.DataFrame,
                      h: int) -> tuple[np.ndarray, np.ndarray]:
    """Full-length arrays (n,) of target_var and target_rv, NaN where undefined."""
    n = len(asset.df)
    sub = targets[targets["h"] == h]
    pos = asset.df.index.get_indexer(pd.DatetimeIndex(sub["origin_date"]))
    tv = np.full(n, np.nan)
    trv = np.full(n, np.nan)
    tv[pos] = sub["target_var"].to_numpy()
    trv[pos] = sub["target_rv"].to_numpy()
    return tv, trv


def select_test_origins(n_rows: int, h: int, n_test: int) -> np.ndarray:
    """Last n_test valid origins (positional). Validity per PLAN.md 0.2."""
    min_i = max(FEATURE_WARMUP, CONTEXT_LEN - 1, TRAIN_WINDOW + FEATURE_WARMUP + h)
    valid = np.arange(min_i, n_rows - h)   # i <= n - 1 - h
    return valid[-n_test:]


def write_universe_md(assets: list[AssetData], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Universe and estimator choices",
        "",
        "| ticker | class | rows | estimator | zero open-close frac | GK fallback frac | span |",
        "|---|---|---|---|---|---|---|",
    ]
    for a in assets:
        n = a.notes
        lines.append(
            f"| {a.ticker} | {a.asset_class} | {n['rows']} | {a.rv_estimator} "
            f"| {n['zero_oc_frac']} | {n['gk_fallback_frac']} "
            f"| {n['first_date']} to {n['last_date']} |")
    lines += [
        "",
        "Rules: GK default; per-day Parkinson fallback where GK <= 0; FX assets",
        "switch wholesale to Parkinson when > 10 percent of days have near-zero",
        "open-close moves; H == L rows dropped; var_d winsorized at the 0.01 and",
        "99.9 percentiles per asset.",
        "",
    ]
    (out_dir / "universe.md").write_text("\n".join(lines), encoding="utf-8")
