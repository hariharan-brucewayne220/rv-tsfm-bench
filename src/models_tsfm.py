"""Zero-shot TSFM wrappers: Chronos-Bolt (tiny, small) and TTM (granite r2).

Both forecast the log_rv series directly. Per PLAN.md 3.2 the predicted log
target for horizon h is the mean over steps 1..h of the predicted log_rv
path, then the lognormal correction of 0.4 uses a trailing window of
realized log errors (origins j <= i - h only), seeded by a burn-in pass on
the 250 origins before the test window.
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .data import AssetData, targets_to_arrays
from .models_econ import ForecastResult

SEED = 42
MIN_CALIB_OBS = 30
_MODEL_CACHE: dict[str, object] = {}


@dataclass
class TSFMConfig:
    context_length: int = 512
    batch_size: int = 64          # windows per forward pass on CPU
    horizons: tuple[int, ...] = (1, 5, 22)
    max_pred_len: int = 22        # one forward pass covers all horizons
    calib_window: int = 250       # trailing log-error window for PLAN.md 0.4
    num_threads: int = max(1, (os.cpu_count() or 2) - 1)


def _setup_torch(cfg: TSFMConfig) -> None:
    torch.manual_seed(SEED)
    torch.set_num_threads(cfg.num_threads)


def build_context_matrix(log_rv: np.ndarray, origins: np.ndarray,
                         L: int) -> np.ndarray:
    """(num_origins, L); row k = log_rv[origins[k]-L+1 : origins[k]+1]."""
    if origins.min() < L - 1:
        raise ValueError("origin with insufficient history for context")
    return np.stack([log_rv[o - L + 1: o + 1] for o in origins]).astype(
        np.float32)


def lognormal_calibrate(log_point_path: np.ndarray, log_target: np.ndarray,
                        origins: np.ndarray, h: int,
                        calib_window: int) -> np.ndarray:
    """s2 estimate per origin from trailing realized log errors.

    log_point_path: m[k] = predicted log target at origins[k].
    log_target: ln(target_rv) realized at origins[k] (NaN if unavailable).
    Only errors at origins j with j <= origins[k] - h enter s2 at k.
    """
    err = log_point_path - log_target
    s2 = np.full(len(origins), np.nan)
    for k in range(len(origins)):
        usable = np.where((origins <= origins[k] - h) & np.isfinite(err))[0]
        if len(usable) < MIN_CALIB_OBS:
            continue
        e = err[usable[-calib_window:]]
        s2[k] = float(np.var(e, ddof=1))
    return s2


def _emit_results(model_name: str, asset: AssetData, targets: pd.DataFrame,
                  paths: np.ndarray, origins: np.ndarray,
                  test_idx: np.ndarray, cfg: TSFMConfig) -> list[ForecastResult]:
    """Shared post-processing: horizon means, calibration, back-transform."""
    test_set = set(int(i) for i in test_idx)
    is_test = np.array([int(o) in test_set for o in origins])
    out = []
    for h in cfg.horizons:
        m = paths[:, :h].mean(axis=1)
        _, trv = targets_to_arrays(asset, targets, h)
        log_target = np.log(trv[origins])
        s2 = lognormal_calibrate(m, log_target, origins, h, cfg.calib_window)
        fcst_rv = np.exp(m + 0.5 * s2)
        frame = pd.DataFrame(
            {"fcst_var": fcst_rv[is_test]**2 / asset.ann_factor},
            index=asset.df.index[origins[is_test]])
        out.append(ForecastResult(model_name, asset.ticker, h, frame))
    return out


def _load_chronos(model_id: str):
    if model_id not in _MODEL_CACHE:
        from chronos import BaseChronosPipeline
        _MODEL_CACHE[model_id] = BaseChronosPipeline.from_pretrained(
            model_id, device_map="cpu", torch_dtype=torch.float32,
            local_files_only=True)
    return _MODEL_CACHE[model_id]


TTM_REVISION = "512-96-ft-r2.1"


def _daily_freq_token() -> int:
    """Daily frequency token per tsfm_public's canonical mapping."""
    from tsfm_public.toolkit.time_series_preprocessor import (
        DEFAULT_FREQUENCY_MAPPING)
    token = DEFAULT_FREQUENCY_MAPPING["d"]
    assert token == 8, f"unexpected daily freq token {token}, expected 8"
    return token


def _load_ttm(model_id: str):
    cache_key = f"{model_id}@{TTM_REVISION}"
    if cache_key not in _MODEL_CACHE:
        try:
            from tsfm_public import TinyTimeMixerForPrediction
        except ImportError:
            from tsfm_public.models.tinytimemixer import (
                TinyTimeMixerForPrediction)
        model = TinyTimeMixerForPrediction.from_pretrained(
            model_id, revision=TTM_REVISION, local_files_only=True)
        if model.config.context_length != 512 or \
                model.config.prediction_length != 96:
            raise RuntimeError(
                f"TTM revision mismatch: context "
                f"{model.config.context_length}, prediction "
                f"{model.config.prediction_length}; expected 512/96. "
                f"Check revision={TTM_REVISION!r} in the local HF cache.")
        model.eval()
        _MODEL_CACHE[cache_key] = model
    return _MODEL_CACHE[cache_key]


def forecast_chronos(asset: AssetData, targets: pd.DataFrame,
                     origins: np.ndarray, test_idx: np.ndarray,
                     cfg: TSFMConfig, model_id: str) -> list[ForecastResult]:
    """Chronos-Bolt mean point-forecast path on raw log_rv."""
    _setup_torch(cfg)
    pipe = _load_chronos(model_id)
    log_rv = asset.df["log_rv"].to_numpy()
    ctx = build_context_matrix(log_rv, origins, cfg.context_length)

    paths = np.empty((len(origins), cfg.max_pred_len), dtype=np.float64)
    with torch.inference_mode():
        for lo in range(0, len(origins), cfg.batch_size):
            batch = torch.tensor(ctx[lo:lo + cfg.batch_size],
                                 dtype=torch.float32)
            _quantiles, mean = pipe.predict_quantiles(
                batch, prediction_length=cfg.max_pred_len,
                quantile_levels=[0.5])
            assert mean.shape == (batch.shape[0], cfg.max_pred_len), \
                f"unexpected Chronos mean shape {tuple(mean.shape)}"
            paths[lo:lo + batch.shape[0]] = mean.cpu().numpy()

    name = "ChronosBoltTiny" if "tiny" in model_id else "ChronosBoltSmall"
    return _emit_results(name, asset, targets, paths, origins, test_idx, cfg)


def forecast_ttm(asset: AssetData, targets: pd.DataFrame,
                 origins: np.ndarray, test_idx: np.ndarray, cfg: TSFMConfig,
                 model_id: str = "ibm-granite/granite-timeseries-ttm-r2"
                 ) -> list[ForecastResult]:
    """TTM zero-shot point forecast with per-window standardization.

    Uses the 512-96-ft-r2.1 revision, which requires a freq_token input;
    daily data uses token 8 per tsfm_public's DEFAULT_FREQUENCY_MAPPING.
    """
    _setup_torch(cfg)
    model = _load_ttm(model_id)
    freq_token = _daily_freq_token()
    log_rv = asset.df["log_rv"].to_numpy()
    ctx = build_context_matrix(log_rv, origins, cfg.context_length)

    mu = ctx.mean(axis=1, keepdims=True)
    sd = np.maximum(ctx.std(axis=1, keepdims=True), 1e-8)
    ctx_std = (ctx - mu) / sd

    paths = np.empty((len(origins), cfg.max_pred_len), dtype=np.float64)
    with torch.inference_mode():
        for lo in range(0, len(origins), cfg.batch_size):
            batch = torch.tensor(ctx_std[lo:lo + cfg.batch_size],
                                 dtype=torch.float32).unsqueeze(-1)
            out = model(past_values=batch,
                        freq_token=torch.tensor(
                            [freq_token] * batch.shape[0]))
            pred = out.prediction_outputs
            assert pred.shape == (batch.shape[0], 96, 1), \
                f"unexpected TTM output shape {tuple(pred.shape)}"
            raw = pred[:, :cfg.max_pred_len, 0].cpu().numpy()
            b = slice(lo, lo + batch.shape[0])
            paths[b] = raw * sd[b] + mu[b]

    return _emit_results("TTM", asset, targets, paths, origins, test_idx, cfg)
