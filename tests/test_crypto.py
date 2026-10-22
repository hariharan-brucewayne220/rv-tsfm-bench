"""Torch-free, network-free unit tests for Track B (PLAN_TRACK_B.md section 6).

Run with: python -m pytest tests/ -q   (or python tests/test_crypto.py)
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import AssetData, compute_targets
from src.data_crypto import (ANN_CRYPTO, INTERVALS_PER_DAY,
                             MIN_VALID_INTERVALS, _daily_frame_from_1m,
                             daily_rv, five_min_closes, load_month_1m_closes,
                             month_range)
from src.evaluate import add_ensemble, add_mz_rows
from src.models_econ import (ForecastResult, RollingConfig, forecast_ewma,
                             forecast_har)
from src.run import has_cached, results_to_frame


def _synthetic_1m_closes(n_days: int, start: str = "2023-01-01",
                         seed: int = 7) -> pd.Series:
    """Gap-free 1m closes indexed at bar-close instants 00:01 .. 24:00."""
    rng = np.random.default_rng(seed)
    n = n_days * 1440
    idx = pd.date_range(start=pd.Timestamp(start) + pd.Timedelta(minutes=1),
                        periods=n, freq="1min")
    logp = np.cumsum(rng.normal(0.0, 1e-4, size=n)) + np.log(100.0)
    return pd.Series(np.exp(logp), index=idx, name="close")


def test_daily_rv_exact_and_midnight_boundary():
    """Gap-free days give exactly 288 valid returns and the hand-computed RV;
    the day boundary uses the (t - 1 microsecond) rule."""
    c1m = _synthetic_1m_closes(3)
    c5 = five_min_closes(c1m)
    rv = daily_rv(c5)

    # day 1 and day 2 are complete (day 1 loses its first return to diff
    # only when there is no prior close; here day 2 and 3 are fully interior)
    d2 = pd.Timestamp("2023-01-02")
    assert rv.loc[d2, "n_valid"] == INTERVALS_PER_DAY

    # hand check: RV = sum of squared 5-min log returns with stamps
    # 00:05 .. 24:00 of that day (right-labeled)
    r = np.log(c5).diff()
    lo = pd.Timestamp("2023-01-02 00:00:00.000001")
    hi = pd.Timestamp("2023-01-03 00:00:00")
    hand = float((r[(r.index >= lo) & (r.index <= hi)] ** 2).sum())
    assert abs(rv.loc[d2, "rv"] - hand) < 1e-18

    # the return stamped exactly at midnight belongs to the PREVIOUS day
    midnight = pd.Timestamp("2023-01-03 00:00:00")
    assert np.isfinite(r[midnight])
    d3 = pd.Timestamp("2023-01-03")
    r_d3 = r[(r.index > midnight) & (r.index <= d3 + pd.Timedelta(days=1))]
    assert abs(rv.loc[d3, "rv"] - float((r_d3 ** 2).sum())) < 1e-18


def test_missing_interval_rule_and_gap_bridging():
    """A gap of k empty 5-min windows invalidates k + 1 returns (the window
    returns plus the one bridging the gap), so 27 missing windows keep the
    day (n_valid = 260) and 28 drop it (n_valid = 259). The bridging return
    is excluded from both RV and n_valid."""
    for n_gap, expect_kept in ((27, True), (28, False)):
        c1m = _synthetic_1m_closes(3)
        # knock out n_gap consecutive 5-min windows in day 2 (10:00 onward)
        gap_lo = pd.Timestamp("2023-01-02 10:00:00.000001")
        gap_hi = gap_lo + pd.Timedelta(minutes=5 * n_gap)
        c1m = c1m[(c1m.index < gap_lo) | (c1m.index > gap_hi)]
        rv = daily_rv(five_min_closes(c1m))
        d2 = pd.Timestamp("2023-01-02")
        # n_gap empty windows kill n_gap returns, plus the first return after
        # the gap bridges it and is excluded too
        assert rv.loc[d2, "n_valid"] == INTERVALS_PER_DAY - n_gap - 1
        kept = rv.loc[d2, "n_valid"] >= MIN_VALID_INTERVALS
        assert kept == expect_kept, f"n_gap={n_gap}"

        # gap-bridging exclusion: the large jump across the hole must not
        # enter rv; rebuild rv by hand from valid 5-min returns only
        r = np.log(five_min_closes(c1m)).diff()
        lo = pd.Timestamp("2023-01-02 00:00:00.000001")
        hi = pd.Timestamp("2023-01-03 00:00:00")
        rd = r[(r.index >= lo) & (r.index <= hi)]
        hand = float((rd.dropna() ** 2).sum())
        assert abs(rv.loc[d2, "rv"] - hand) < 1e-18


def _write_kline_zip(path: Path, ts_values: np.ndarray, closes: np.ndarray,
                     header: bool) -> None:
    lines = []
    if header:
        lines.append("open_time,open,high,low,close,volume,close_time,"
                     "quote_volume,count,taker_buy_volume,"
                     "taker_buy_quote_volume,ignore")
    for t, c in zip(ts_values, closes):
        lines.append(f"{t},1,1,1,{c:.8f},0,0,0,0,0,0,0")
    csv_name = path.stem + ".csv"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(csv_name, "\n".join(lines) + "\n")


def test_timestamp_unit_and_header_detection(tmp_path):
    """The same synthetic month encoded as ms/us, with and without header,
    parses to identical series."""
    open_times = pd.date_range("2024-03-01 00:00", periods=3 * 1440,
                               freq="1min")
    rng = np.random.default_rng(1)
    closes = 100.0 + np.cumsum(rng.normal(0, 0.01, size=len(open_times)))
    epoch_ns = open_times.as_unit("ns").asi8

    variants = {}
    for unit, div in (("ms", 10**6), ("us", 10**3)):
        for header in (False, True):
            p = tmp_path / f"SYN-1m-2024-03-{unit}-{int(header)}.zip"
            # load_month_1m_closes reads ym from stem[-7:], so name the file
            # to end with the month
            p = tmp_path / f"{unit}{int(header)}-1m-2024-03.zip"
            _write_kline_zip(p, epoch_ns // div, closes, header)
            variants[(unit, header)] = load_month_1m_closes(p)

    base = variants[("ms", False)]
    assert (base.index == open_times + pd.Timedelta(minutes=1)).all()
    for key, s in variants.items():
        assert (s.index == base.index).all(), key
        assert np.allclose(s.to_numpy(), base.to_numpy(), atol=1e-8), key

    # month-range assert catches a wrong unit: file named for the wrong month
    bad = tmp_path / f"bad-1m-2024-04.zip"
    _write_kline_zip(bad, epoch_ns // 10**6, closes, False)
    try:
        load_month_1m_closes(bad)
        raised = False
    except ValueError:
        raised = True
    assert raised, "timestamps outside the named month must raise"


def test_ann_round_trip_365():
    """fcst_var -> fcst_rv -> fcst_var through results_to_frame, add_ensemble
    and add_mz_rows is exact with ann=365 and would NOT be with mixed 252/365."""
    ann = ANN_CRYPTO
    idx = pd.date_range("2024-01-01", periods=400, freq="D")
    rng = np.random.default_rng(3)
    var_a = np.exp(rng.normal(-8.0, 0.3, size=len(idx)))
    var_b = np.exp(rng.normal(-8.1, 0.3, size=len(idx)))
    results = [
        ForecastResult("TTM", "BTCUSDT", 1,
                       pd.DataFrame({"fcst_var": var_a}, index=idx)),
        ForecastResult("LogHAR", "BTCUSDT", 1,
                       pd.DataFrame({"fcst_var": var_b}, index=idx)),
    ]
    store = results_to_frame(results, ann=ann)
    assert np.allclose(store["fcst_rv"] ** 2 / ann, store["fcst_var"],
                       rtol=1e-12)
    # mixing constants would break the identity
    assert not np.allclose(store["fcst_rv"] ** 2 / 252.0, store["fcst_var"],
                           rtol=1e-3)

    store = add_ensemble(store, ann=ann)
    ens = store[store["model"] == "EW-ENS"]
    assert len(ens) == len(idx)
    assert np.allclose(ens["fcst_rv"] ** 2 / ann, ens["fcst_var"], rtol=1e-12)
    a = store[store["model"] == "TTM"].set_index("origin_date")["fcst_rv"]
    b = store[store["model"] == "LogHAR"].set_index("origin_date")["fcst_rv"]
    e = ens.set_index("origin_date")["fcst_rv"]
    assert np.allclose(e, 0.5 * a + 0.5 * b, rtol=1e-12)

    # MZ rows: targets correlated with forecasts so the regression is sane
    tgt_var = var_b * np.exp(rng.normal(0.0, 0.1, size=len(idx)))
    targets = pd.DataFrame({
        "asset": "BTCUSDT", "h": np.int8(1), "origin_date": idx,
        "target_var": tgt_var, "target_rv": np.sqrt(ann * tgt_var),
    })
    store = add_mz_rows(store, targets, ann=ann)
    mz = store[store["recalibrated"]]
    assert len(mz) > 0
    assert np.allclose(mz["fcst_rv"] ** 2 / ann, mz["fcst_var"], rtol=1e-12)


def test_crypto_assetdata_contract():
    """A crypto-style AssetData flows through compute_targets and the econ
    models with zero crypto-specific logic, and target_rv uses 365."""
    n = 700
    idx = pd.date_range("2022-01-01", periods=n, freq="D")   # 7-day weeks
    rng = np.random.default_rng(11)
    var_d = np.exp(rng.normal(-8.0, 0.5, size=n))
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, size=n)))
    df = pd.DataFrame({"var_d": var_d, "close": close}, index=idx)
    df["ret_cc"] = np.log(df["close"] / df["close"].shift(1))
    df["rv_ann"] = np.sqrt(ANN_CRYPTO * df["var_d"])
    df["log_rv"] = np.log(df["rv_ann"])
    asset = AssetData(ticker="TOYUSDT", asset_class="crypto", df=df,
                      rv_estimator="rv5m", ann_factor=ANN_CRYPTO)

    targets = compute_targets(asset, horizons=(1, 5, 22))
    sub = targets[targets["h"] == 1]
    assert np.allclose(sub["target_rv"],
                       np.sqrt(ANN_CRYPTO * sub["target_var"]), rtol=1e-12)

    cfg = RollingConfig(train_window=300, refit_every=22)
    test_idx = np.arange(500, 560)
    for fn in (forecast_har, forecast_ewma):
        out = fn(asset, targets, test_idx, cfg)
        assert {r.h for r in out} == {1, 5, 22}
        for r in out:
            v = r.frame["fcst_var"].to_numpy()
            assert np.isfinite(v).all() and (v > 0).all()
            assert (r.frame.index == idx[test_idx]).all()


def test_daily_frame_from_1m_end_to_end():
    """The shared 1m -> daily transformation produces the full schema on
    synthetic data, keeps complete days, and matches daily_rv."""
    c1m = _synthetic_1m_closes(40)
    df, stats = _daily_frame_from_1m(c1m)
    for col in ("open", "high", "low", "close", "var_d", "ret_cc",
                "rv_ann", "log_rv", "n_valid"):
        assert col in df.columns, col
    # interior days all complete; only day 1 can lose its first return
    assert len(df) >= 39
    assert (df["n_valid"] >= MIN_VALID_INTERVALS).all()
    assert np.isfinite(df["log_rv"].to_numpy()).all()
    assert np.allclose(df["rv_ann"],
                       np.sqrt(ANN_CRYPTO * df["var_d"]), rtol=1e-12)
    assert stats["dropped_missing"] in (0, 1)

    # month_range sanity
    assert month_range("2024-11", "2025-02") == \
        ["2024-11", "2024-12", "2025-01", "2025-02"]


def test_has_cached_requires_full_origin_coverage():
    """Regression: a quick run's 200 cached origins must not satisfy a full
    run's 750-origin grid (bug found during Track B testing: the full Track A
    run skipped the 6 quick assets because has_cached only checked that any
    rows existed per (asset, model, h))."""
    full = pd.date_range("2020-01-01", periods=750, freq="D")
    quick = full[-200:]
    rows = []
    for h in (1, 5, 22):
        for d in quick:
            rows.append({"asset": "BTCUSDT", "model": "TTM", "h": np.int8(h),
                         "origin_date": d, "fcst_var": 1e-4, "fcst_rv": 0.2,
                         "recalibrated": False})
    store = pd.DataFrame(rows)

    assert has_cached(store, "BTCUSDT", "TTM", quick)          # quick rerun: skip
    assert not has_cached(store, "BTCUSDT", "TTM", full)       # full run: recompute
    assert not has_cached(store, "BTCUSDT", "LogHAR", quick)   # other model absent

    # missing a single horizon invalidates the cache
    partial = store[store["h"] != 5]
    assert not has_cached(partial, "BTCUSDT", "TTM", quick)


if __name__ == "__main__":
    import tempfile
    for fn in [test_daily_rv_exact_and_midnight_boundary,
               test_missing_interval_rule_and_gap_bridging,
               test_ann_round_trip_365,
               test_crypto_assetdata_contract,
               test_daily_frame_from_1m_end_to_end,
               test_has_cached_requires_full_origin_coverage]:
        fn()
        print(f"PASS {fn.__name__}")
    with tempfile.TemporaryDirectory() as td:
        test_timestamp_unit_and_header_detection(Path(td))
        print("PASS test_timestamp_unit_and_header_detection")
