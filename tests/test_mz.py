"""MZ recalibration schemes and the like-for-like comparison vs recalibrated
Log-HAR. Torch-free, synthetic data only."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import (MZ_SCHEMES, add_mz_rows, build_tables,
                          mz_recalibrate, split_mz_name)

ANN = 252.0


def _series(n: int = 700, seed: int = 0):
    """AR(1) log-vol target and a noisy, unbiased log-space forecast."""
    rng = np.random.default_rng(seed)
    lv = np.empty(n)
    lv[0] = np.log(0.2)
    for t in range(1, n):
        lv[t] = np.log(0.2) + 0.95 * (lv[t - 1] - np.log(0.2)) \
            + rng.normal(0, 0.15)
    idx = pd.bdate_range("2020-01-01", periods=n)
    target_rv = np.exp(lv)
    fcst_rv = np.exp(lv + rng.normal(0, 0.2, size=n))
    return idx, target_rv, fcst_rv


def _frames(idx, target_rv, fcst_rv):
    fcst = pd.DataFrame({"fcst_rv": fcst_rv}, index=idx)
    targets = pd.DataFrame({"origin_date": idx, "target_rv": target_rv})
    return fcst, targets


def _recal(scheme, fcst, targets, h):
    spec = MZ_SCHEMES[scheme]
    return mz_recalibrate(fcst, targets, h, window=spec["window"],
                          min_obs=spec["min_obs"], ann=ANN, scheme=scheme)


def test_split_mz_name():
    assert split_mz_name("TTM+MZL") == ("TTM", "+MZL")
    assert split_mz_name("EW-ENS+MZ") == ("EW-ENS", "+MZ")
    assert split_mz_name("LogHAR") == ("LogHAR", "")


def test_mz_no_lookahead_both_schemes():
    """A target at position p may only affect forecasts at k >= p + h."""
    idx, trv, frv = _series()
    for scheme in MZ_SCHEMES:
        for h in (1, 5, 22):
            fcst, targets = _frames(idx, trv, frv)
            base = _recal(scheme, fcst, targets, h)["fcst_rv"].to_numpy()
            p = 400
            trv2 = trv.copy()
            trv2[p] *= 5.0
            fcst2, targets2 = _frames(idx, trv2, frv)
            pert = _recal(scheme, fcst2, targets2, h)["fcst_rv"].to_numpy()
            same = np.isclose(base, pert, equal_nan=True)
            assert same[:p + h].all(), (scheme, h)
            assert not same[p + h:].all(), (scheme, h)


def test_expanding_level_removes_affine_bias():
    idx, trv, _ = _series(seed=1)
    biased = 0.05 + 1.8 * trv                    # level and scale bias
    fcst, targets = _frames(idx, trv, biased)
    rec = _recal("expanding_level", fcst, targets, 1)["fcst_rv"].to_numpy()
    ok = np.isfinite(rec)
    assert ok.sum() > 300
    assert np.allclose(rec[ok], trv[ok], rtol=1e-6)


def _store_with_scaled_copy(scale: float, n: int = 700, h: int = 1):
    idx, trv, frv = _series(n=n, seed=2)
    rows = []
    for model, f in (("LogHAR", frv), ("Scaled", scale * frv)):
        rows.append(pd.DataFrame({
            "asset": "SYN", "model": model, "h": np.int8(h),
            "origin_date": idx, "fcst_rv": f, "fcst_var": f**2 / ANN,
            "recalibrated": False}))
    store = pd.concat(rows, ignore_index=True)
    targets = pd.DataFrame({"asset": "SYN", "h": np.int8(h),
                            "origin_date": idx, "target_rv": trv,
                            "target_var": trv**2 / ANN})
    return store, targets


def test_like_for_like_removes_pure_scale_bias(tmp_path):
    """A forecast that is Log-HAR times a constant loses on raw QLIKE, but
    both MZ schemes map it to exactly the recalibrated Log-HAR forecast, so
    its like-for-like ratio is 1 and the DM test has nothing to find."""
    store, targets = _store_with_scaled_copy(1.4)
    store = add_mz_rows(store, targets, ann=ANN)
    assert set(store["model"]) == {"LogHAR", "Scaled", "LogHAR+MZ",
                                   "Scaled+MZ", "LogHAR+MZL", "Scaled+MZL"}
    tables = build_tables(store, targets, tmp_path)
    per = tables["mz_like_for_like_per_asset"]
    s = per[per["model"] == "Scaled"].set_index("scheme")
    assert set(s.index) == set(MZ_SCHEMES)
    assert (s["ratio_raw"] > 1.02).all()
    assert np.allclose(s["ratio_mz"], 1.0, atol=1e-9)
    ref = per[per["model"] == "LogHAR"]
    assert np.allclose(ref["ratio_mz"], 1.0)
    assert (tmp_path / "tables" / "mz_like_for_like.md").exists()
    assert "mz_effect" in tables                 # original arm still built
