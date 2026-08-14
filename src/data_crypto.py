"""Track B: true 5-min realized volatility from Binance monthly 1m klines.

Data source (SPEC.md Track B): free zip archives at
https://data.binance.vision/data/spot/monthly/klines/{SYMBOL}/1m/{SYMBOL}-1m-{YYYY-MM}.zip

Conventions (PLAN_TRACK_B.md):
- 1m closes are indexed at the bar CLOSE instant (open_time + 1 minute),
  tz-naive UTC.
- 5-min closes: resample("5min", label="right", closed="right").last(),
  never forward-filled; a 5-min return is valid only when both endpoint
  closes exist, so gap-bridging returns are excluded from RV and n_valid.
- Day assignment: the return stamped t belongs to the UTC day of
  (t - 1 microsecond), so stamps 00:05 .. 24:00 give exactly 288 slots/day.
- Keep a day iff n_valid >= 260 (SPEC-literal >= 90 percent rule, no
  288/n rescaling of RV on kept days; stated choice).
- Annualization: 365 (crypto trades 7 days a week). HAR lags and horizons
  stay in observation units (5/22 and h = 1, 5, 22), identical to Track A.
- Cached zips under data/binance/{symbol}/ are immutable; parsed per-month
  1m closes are cached as parquet under data/binance/parsed/ and only those
  are regenerated under force.
"""
from __future__ import annotations

import io
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .data import MIN_ROWS, AssetData

BINANCE_BASE = "https://data.binance.vision/data/spot/monthly/klines"
CRYPTO_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT"]
QUICK_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
LISTING_MONTH = {
    "BTCUSDT": "2017-08", "ETHUSDT": "2017-08", "BNBUSDT": "2017-11",
    "XRPUSDT": "2018-05", "SOLUSDT": "2020-08",
}
ANN_CRYPTO = 365.0
INTERVALS_PER_DAY = 288
MIN_VALID_INTERVALS = 260        # keep day iff n_valid >= 260 (>= 90 percent)

# --quick needs roughly TRAIN_WINDOW + warmup + max_h + 200 test origins of
# daily rows (about 1265); 46 complete months is about 1400 days.
QUICK_MONTHS = 46
QUICK_MIN_ROWS = 1265

WINSOR_PCT = (0.01, 99.9)
MAX_START_ADVANCES = 3           # tolerated 404s at the listed start month
DOWNLOAD_RETRIES = 3
US_THRESHOLD = 1e14              # timestamps >= 1e14 are microseconds


def last_complete_month() -> str:
    """Last complete UTC month as 'YYYY-MM' (never the current month)."""
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    return str(pd.Period(now, freq="M") - 1)


def month_range(start_ym: str, end_ym: str) -> list[str]:
    """Inclusive list of 'YYYY-MM' from start_ym to end_ym."""
    p = pd.period_range(start_ym, end_ym, freq="M")
    return [str(m) for m in p]


def zip_path_for(symbol: str, ym: str, cache_dir: Path) -> Path:
    return Path(cache_dir) / symbol / f"{symbol}-1m-{ym}.zip"


def download_month(symbol: str, ym: str, cache_dir: Path,
                   force: bool = False) -> Path | None:
    """Download one monthly zip into the immutable cache.

    Returns the cached path, or None on HTTP 404 (month not published,
    e.g. before listing). Raises on other HTTP errors after retries.
    Writes to *.part then os.replace so interrupted downloads never leave
    a truncated zip under the final name.
    """
    import requests

    path = zip_path_for(symbol, ym, cache_dir)
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BINANCE_BASE}/{symbol}/1m/{symbol}-1m-{ym}.zip"

    last_exc: Exception | None = None
    for attempt in range(DOWNLOAD_RETRIES):
        try:
            resp = requests.get(url, timeout=(10, 120))
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            part = path.with_suffix(".zip.part")
            part.write_bytes(resp.content)
            os.replace(part, path)
            return path
        except Exception as exc:
            last_exc = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"{symbol} {ym}: download failed after "
                       f"{DOWNLOAD_RETRIES} attempts: {last_exc}")


def load_month_1m_closes(zip_path: Path) -> pd.Series:
    """Parse one monthly kline zip to a Series of 1m closes.

    Index = bar close instant (open_time + 1 minute), tz-naive UTC, sorted,
    duplicates dropped (keep first). Handles both millisecond (pre-2025) and
    microsecond (2025-01 onward) timestamps and optional header rows.
    Raises zipfile.BadZipFile on a corrupt archive.
    """
    zip_path = Path(zip_path)
    ym = zip_path.stem[-7:]                      # {SYMBOL}-1m-YYYY-MM
    with zipfile.ZipFile(zip_path) as zf:
        if zf.testzip() is not None:
            raise zipfile.BadZipFile(f"corrupt member in {zip_path.name}")
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"{zip_path.name}: expected 1 csv, got {names}")
        with zf.open(names[0]) as f:
            df = pd.read_csv(io.TextIOWrapper(f, encoding="utf-8"),
                             header=None, usecols=[0, 4], dtype=str)
    df.columns = ["open_time", "close"]
    first = str(df.iloc[0, 0]).strip()
    if not first.isdigit():                      # header row in newer files
        df = df.iloc[1:]
    ts = pd.to_numeric(df["open_time"], errors="raise").astype("int64")
    close = pd.to_numeric(df["close"], errors="raise").astype("float64")
    unit = "us" if float(ts.iloc[0]) >= US_THRESHOLD else "ms"
    open_dt = pd.to_datetime(ts.to_numpy(), unit=unit)

    period = pd.Period(ym, freq="M")
    if open_dt.min() < period.start_time or open_dt.max() > period.end_time:
        raise ValueError(
            f"{zip_path.name}: timestamps outside {ym} "
            f"({open_dt.min()} .. {open_dt.max()}); unit detection ({unit}) "
            f"or file content is wrong")
    if not (np.isfinite(close.to_numpy()).all() and (close > 0).all().item()):
        raise ValueError(f"{zip_path.name}: nonpositive or non-finite closes")

    s = pd.Series(close.to_numpy(), index=open_dt + pd.Timedelta(minutes=1),
                  name="close")
    s = s[~s.index.duplicated(keep="first")].sort_index()
    return s


def five_min_closes(closes_1m: pd.Series) -> pd.Series:
    """5-min closes on a right-labeled, right-closed grid; NO forward fill."""
    return closes_1m.resample("5min", label="right", closed="right").last()


def daily_rv(closes_5m: pd.Series) -> pd.DataFrame:
    """Per UTC day: rv = sum of squared valid 5-min log returns, n_valid.

    A return is valid iff finite, i.e. both endpoint 5-min closes exist
    (gap-bridging returns are excluded). The return stamped t belongs to
    the UTC day of (t - 1 microsecond): stamps 00:05 .. 24:00, 288 slots.
    """
    r = np.log(closes_5m).diff()
    valid = np.isfinite(r.to_numpy())
    day = (r.index - pd.Timedelta(microseconds=1)).normalize()
    tmp = pd.DataFrame({
        "r2": np.where(valid, np.square(r.to_numpy()), 0.0),
        "valid": valid.astype("int64"),
    }, index=day)
    g = tmp.groupby(level=0).sum()
    out = pd.DataFrame({"rv": g["r2"], "n_valid": g["valid"]})
    out.index.name = "date"
    return out


def _month_closes(symbol: str, ym: str, cache_dir: Path,
                  force: bool = False) -> pd.Series | None:
    """Cached per-month 1m closes: parquet if present, else zip -> parquet.

    force regenerates the parsed parquet from the zip; zips themselves are
    immutable and only fetched when missing. Returns None on 404.
    """
    cache_dir = Path(cache_dir)
    parsed = cache_dir / "parsed" / f"{symbol}-{ym}.parquet"
    if parsed.exists() and not force:
        frame = pd.read_parquet(parsed)
        return frame["close"]

    zp = download_month(symbol, ym, cache_dir)
    if zp is None:
        return None
    try:
        s = load_month_1m_closes(zp)
    except zipfile.BadZipFile:
        print(f"  {symbol} {ym}: corrupt zip, re-downloading once")
        zp.unlink()
        zp = download_month(symbol, ym, cache_dir)
        if zp is None:
            return None
        s = load_month_1m_closes(zp)
    parsed.parent.mkdir(parents=True, exist_ok=True)
    frame = s.to_frame()
    frame.index.name = "ts"
    frame.to_parquet(parsed)
    return s


def _daily_frame_from_1m(closes_1m: pd.Series) -> tuple[pd.DataFrame, dict]:
    """1m closes -> final daily frame (var_d, ret_cc, rv_ann, log_rv, OHLC).

    Pure transformation shared by build_crypto_asset and the unit tests.
    Returns (df, stats) where stats holds dropped-day counters and the
    winsorization clip fraction.
    """
    closes_1m = closes_1m[~closes_1m.index.duplicated(keep="first")].sort_index()
    assert closes_1m.index.tz is None, "1m index must be tz-naive UTC"

    rv = daily_rv(five_min_closes(closes_1m))

    day = (closes_1m.index - pd.Timedelta(microseconds=1)).normalize()
    g = closes_1m.groupby(day)
    daily = pd.DataFrame({
        "open": g.first(), "high": g.max(),
        "low": g.min(), "close": g.last(),
    })
    daily.index.name = "date"
    df = daily.join(rv, how="inner")

    n_all = len(df)
    keep = df["n_valid"] >= MIN_VALID_INTERVALS
    dropped_missing = int((~keep).sum())
    df = df[keep]
    keep = np.isfinite(df["rv"]) & (df["rv"] > 0)
    dropped_nonpos = int((~keep).sum())
    df = df[keep].copy()

    lo, hi = np.percentile(df["rv"], list(WINSOR_PCT))
    df["var_d"] = df["rv"].clip(lo, hi)
    clip_frac = float(((df["rv"] < lo) | (df["rv"] > hi)).mean())
    df["ret_cc"] = np.log(df["close"] / df["close"].shift(1))
    df["rv_ann"] = np.sqrt(ANN_CRYPTO * df["var_d"])
    df["log_rv"] = np.log(df["rv_ann"])
    assert np.isfinite(df["log_rv"].to_numpy()).all(), "non-finite log_rv"

    stats = {
        "days_raw": n_all,
        "dropped_missing": dropped_missing,
        "dropped_nonpos": dropped_nonpos,
        "clip_frac": round(clip_frac, 5),
    }
    return df, stats


def build_crypto_asset(symbol: str, cache_dir: Path, force: bool = False,
                       start_ym: str | None = None,
                       min_rows: int = MIN_ROWS) -> AssetData | None:
    """Download + parse all months for one symbol and build its AssetData.

    A 404 at the start of the range advances the start month (listing-date
    slack, up to MAX_START_ADVANCES); a 404 strictly inside the range raises
    because Binance monthly archives have no legitimate interior holes.
    """
    start = start_ym or LISTING_MONTH[symbol]
    months = month_range(start, last_complete_month())
    parts: list[pd.Series] = []
    months_used: list[str] = []
    advances = 0
    for ym in months:
        s = _month_closes(symbol, ym, cache_dir, force=force)
        if s is None:
            if months_used:
                raise RuntimeError(
                    f"{symbol}: 404 for {ym} inside the expected range")
            advances += 1
            if advances > MAX_START_ADVANCES:
                raise RuntimeError(
                    f"{symbol}: no data in first {advances} months from "
                    f"{start}; LISTING_MONTH looks wrong")
            print(f"  {symbol} {ym}: 404 at start of range, advancing")
            continue
        months_used.append(ym)
        parts.append(s)
    if not parts:
        print(f"  DROP {symbol}: no months downloaded")
        return None

    closes_1m = pd.concat(parts)
    df, stats = _daily_frame_from_1m(closes_1m)
    if len(df) < min_rows:
        print(f"  DROP {symbol}: only {len(df)} usable days "
              f"(< {min_rows})")
        return None

    notes = {
        "rows": len(df),
        "months": len(months_used),
        "first_month": months_used[0],
        "last_month": months_used[-1],
        "dropped_missing_days": stats["dropped_missing"],
        "dropped_nonpos_rv": stats["dropped_nonpos"],
        "clip_frac": stats["clip_frac"],
        "first_date": str(df.index[0].date()),
        "last_date": str(df.index[-1].date()),
    }
    return AssetData(ticker=symbol, asset_class="crypto", df=df,
                     rv_estimator="rv5m", notes=notes,
                     ann_factor=ANN_CRYPTO)


def load_crypto_universe(cache_dir: Path, quick: bool = False,
                         force: bool = False) -> list[AssetData]:
    """All five symbols; under quick, 2 symbols and only the trailing
    QUICK_MONTHS complete months (enough history for 200 test origins)."""
    symbols = QUICK_SYMBOLS if quick else CRYPTO_SYMBOLS
    end = pd.Period(last_complete_month(), freq="M")
    assets = []
    for symbol in symbols:
        start_ym = LISTING_MONTH[symbol]
        min_rows = MIN_ROWS
        if quick:
            quick_start = str(end - (QUICK_MONTHS - 1))
            start_ym = max(start_ym, quick_start)
            min_rows = QUICK_MIN_ROWS
        asset = build_crypto_asset(symbol, cache_dir, force=force,
                                   start_ym=start_ym, min_rows=min_rows)
        if asset is not None:
            print(f"  OK   {symbol} (crypto, rv5m, {len(asset.df)} rows, "
                  f"{asset.notes['months']} months)")
            assets.append(asset)
    return assets


def write_crypto_universe_md(assets: list[AssetData], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Track B universe: Binance 5-min realized volatility",
        "",
        "| symbol | rows | months | dropped days (missing rule) | dropped rv<=0 "
        "| winsor clip frac | span |",
        "|---|---|---|---|---|---|---|",
    ]
    for a in assets:
        n = a.notes
        lines.append(
            f"| {a.ticker} | {n['rows']} | {n['months']} "
            f"| {n['dropped_missing_days']} | {n['dropped_nonpos_rv']} "
            f"| {n['clip_frac']} | {n['first_date']} to {n['last_date']} |")
    lines += [
        "",
        "Rules and stated choices:",
        "- RV_t = sum of squared valid 5-min log returns over the UTC day",
        "  (288 slots); a return is valid only if both endpoint 5-min closes",
        "  exist, so gap-bridging returns are excluded.",
        "- Days with n_valid < 260 (> 10 percent missing) are dropped; kept",
        "  days are NOT rescaled by 288/n (stated choice, bias <= 10 percent",
        "  and typically far smaller).",
        "- Annualization uses 365. var_d winsorized at the 0.01 and 99.9",
        "  percentiles per asset, same as Track A.",
        "- HAR lags and horizons are in observation units (days of a 7-day",
        "  week for crypto): lags 5/22, h = 1, 5, 22, identical to Track A.",
        "",
    ]
    (out_dir / "universe_crypto.md").write_text("\n".join(lines),
                                                encoding="utf-8")
