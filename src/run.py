"""CLI orchestrator: download -> forecast -> evaluate.

Usage:
    python -m src.run --quick
    python -m src.run --stage evaluate
    python -m src.run --models LogHAR,TTM --force-forecast
"""
from __future__ import annotations

import os

# huggingface.co is blocked from Python on this machine (SPEC.md environment
# constraints); models are pre-downloaded to the HF cache. Force offline mode
# before any model library import.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import SEED
from .data import (AssetData, BURN_IN, HORIZONS, TEST_DAYS_FULL,
                   TEST_DAYS_QUICK, compute_targets, load_universe,
                   select_test_origins, write_universe_md)
from .models_econ import (ForecastResult, RollingConfig, forecast_ar1_log,
                          forecast_ewma, forecast_garch, forecast_har,
                          forecast_log_har)

ECON_MODELS = ["HAR", "LogHAR", "AR1", "EWMA", "GARCH"]
TSFM_MODELS = ["ChronosBoltTiny", "ChronosBoltSmall", "TTM"]
ALL_MODELS = ECON_MODELS + TSFM_MODELS
KEYS = ["asset", "model", "h", "origin_date"]

CHRONOS_IDS = {
    "ChronosBoltTiny": "amazon/chronos-bolt-tiny",
    "ChronosBoltSmall": "amazon/chronos-bolt-small",
}


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def results_to_frame(results: list[ForecastResult],
                     ann: float = 252.0) -> pd.DataFrame:
    rows = []
    for r in results:
        frame = r.frame.dropna(subset=["fcst_var"])
        sub = pd.DataFrame({
            "asset": r.asset,
            "model": r.model,
            "h": np.int8(r.h),
            "origin_date": frame.index,
            "fcst_var": frame["fcst_var"].to_numpy(),
        })
        sub["fcst_rv"] = np.sqrt(ann * sub["fcst_var"])
        sub["recalibrated"] = False
        rows.append(sub)
    if not rows:
        return pd.DataFrame(
            columns=KEYS + ["fcst_var", "fcst_rv", "recalibrated"])
    return pd.concat(rows, ignore_index=True)


def load_store(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=KEYS + ["fcst_var", "fcst_rv", "recalibrated"])


def upsert_store(path: Path, new_rows: pd.DataFrame) -> pd.DataFrame:
    store = load_store(path)
    if len(store) == 0:
        combined = new_rows.copy()
    else:
        combined = pd.concat([store, new_rows], ignore_index=True)
    combined = combined.drop_duplicates(subset=KEYS, keep="last")
    combined.to_parquet(path, index=False)
    return combined


def has_cached(store: pd.DataFrame, asset: str, model: str,
               expected_dates: pd.DatetimeIndex) -> bool:
    """True only if the cached rows cover every expected test origin for
    every horizon. A plain existence check is not enough: a quick run caches
    200 origins per (asset, model, h) and a later full run (750 origins)
    must NOT treat that subset as complete."""
    sub = store[(store["asset"] == asset) & (store["model"] == model)]
    want = set(pd.DatetimeIndex(expected_dates))
    for h in HORIZONS:
        have = set(pd.DatetimeIndex(sub.loc[sub["h"] == h, "origin_date"]))
        if not want <= have:
            return False
    return True


def run_forecasts(assets: list[AssetData], models: list[str], n_test: int,
                  store_path: Path, force: bool) -> None:
    cfg = RollingConfig(horizons=HORIZONS)
    econ_fns = {
        "HAR": forecast_har,
        "LogHAR": forecast_log_har,
        "AR1": forecast_ar1_log,
        "EWMA": forecast_ewma,
        "GARCH": forecast_garch,
    }
    store = load_store(store_path)
    wanted_tsfm = [m for m in models if m in TSFM_MODELS]
    wanted_econ = [m for m in models if m in ECON_MODELS]

    tsfm_cfg = None
    if wanted_tsfm:
        from .models_tsfm import TSFMConfig
        tsfm_cfg = TSFMConfig(horizons=HORIZONS)

    for a_num, asset in enumerate(assets, 1):
        targets = compute_targets(asset, HORIZONS)
        n = len(asset.df)
        # One shared origin grid for all horizons (matches the per-model
        # single test_idx signatures in PLAN.md 2.2); validity uses the
        # strictest horizon so every origin is valid for every h.
        test_idx = select_test_origins(n, max(HORIZONS), n_test)
        expected_dates = asset.df.index[test_idx]

        for model in wanted_econ:
            if not force and has_cached(store, asset.ticker, model,
                                        expected_dates):
                print(f"[{a_num}/{len(assets)}] {asset.ticker} {model}: cached, skip")
                continue
            t0 = time.time()
            results = econ_fns[model](asset, targets, test_idx, cfg)
            store = upsert_store(store_path,
                                 results_to_frame(results,
                                                  ann=asset.ann_factor))
            print(f"[{a_num}/{len(assets)}] {asset.ticker} {model}: "
                  f"{time.time() - t0:.1f}s")

        if wanted_tsfm:
            burn_start = int(test_idx[0]) - BURN_IN
            origins = np.arange(max(511, burn_start), int(test_idx[-1]) + 1)
            from .models_tsfm import forecast_chronos, forecast_ttm
            for model in wanted_tsfm:
                if not force and has_cached(store, asset.ticker, model,
                                            expected_dates):
                    print(f"[{a_num}/{len(assets)}] {asset.ticker} {model}: "
                          f"cached, skip")
                    continue
                t0 = time.time()
                if model == "TTM":
                    results = forecast_ttm(asset, targets, origins, test_idx,
                                           tsfm_cfg)
                else:
                    results = forecast_chronos(asset, targets, origins,
                                               test_idx, tsfm_cfg,
                                               CHRONOS_IDS[model])
                store = upsert_store(store_path,
                                     results_to_frame(results,
                                                      ann=asset.ann_factor))
                print(f"[{a_num}/{len(assets)}] {asset.ticker} {model}: "
                      f"{time.time() - t0:.1f}s")


def run_evaluation(assets: list[AssetData], store_path: Path,
                   out_dir: Path, suffix: str = "", ann: float = 252.0,
                   example_asset: str | None = None) -> None:
    from .evaluate import add_ensemble, add_mz_rows, build_tables, make_plots

    store = load_store(store_path)
    if len(store) == 0:
        print("no forecasts in store; run the forecast stage first")
        return

    targets_frames = []
    for asset in assets:
        t = compute_targets(asset, HORIZONS)
        t.insert(0, "asset", asset.ticker)
        targets_frames.append(t)
    targets = pd.concat(targets_frames, ignore_index=True)

    print("evaluate: ensemble")
    store = add_ensemble(store, ann=ann)
    print("evaluate: MZ recalibration pass")
    store = add_mz_rows(store, targets, ann=ann)
    store.to_parquet(store_path, index=False)

    print("evaluate: tables")
    tables = build_tables(store, targets, out_dir, suffix=suffix)
    print("evaluate: plots")
    example = example_asset
    if example is None or not any(a.ticker == example for a in assets):
        example = "SPY" if any(a.ticker == "SPY" for a in assets) else \
            assets[0].ticker
    make_plots(store, targets, out_dir, example_asset=example, suffix=suffix)
    for name in tables:
        print(f"  wrote table {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="rv-tsfm-bench pipeline")
    parser.add_argument("--quick", action="store_true",
                        help="reduced scope: 6 assets (track a) or 2 crypto "
                             "symbols with ~3 years of data (track b), "
                             "200 test days")
    parser.add_argument("--track", default="a", choices=["a", "b", "both"],
                        help="a = daily OHLC range RV, b = Binance 5-min RV")
    parser.add_argument("--stage", default="all",
                        choices=["download", "forecast", "evaluate", "all"])
    parser.add_argument("--models", default="all",
                        help="comma list from: " + ",".join(ALL_MODELS))
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-forecast", action="store_true")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    set_seeds(SEED)
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.models == "all":
        models = list(ALL_MODELS)
    else:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = [m for m in models if m not in ALL_MODELS]
        if unknown:
            parser.error(f"unknown models {unknown}")

    n_test = TEST_DAYS_QUICK if args.quick else TEST_DAYS_FULL
    tracks = ["a", "b"] if args.track == "both" else [args.track]

    print(f"track={args.track} stage={args.stage} quick={args.quick} "
          f"models={models}")
    for track in tracks:
        if track == "a":
            print("track a: loading universe")
            assets = load_universe(data_dir, quick=args.quick,
                                   force=args.force_download)
            write_universe_md(assets, out_dir)
            store_path = out_dir / "forecasts.parquet"
            suffix, ann, example = "", 252.0, None
        else:
            from .data_crypto import (ANN_CRYPTO, load_crypto_universe,
                                      write_crypto_universe_md)
            print("track b: loading crypto universe")
            assets = load_crypto_universe(data_dir / "binance",
                                          quick=args.quick,
                                          force=args.force_download)
            write_crypto_universe_md(assets, out_dir)
            store_path = out_dir / "forecasts_crypto.parquet"
            suffix, ann, example = "_crypto", ANN_CRYPTO, "BTCUSDT"

        if not assets:
            print(f"track {track}: no assets, skipping")
            continue
        if args.stage == "download":
            continue

        if args.stage in ("forecast", "all"):
            run_forecasts(assets, models, n_test, store_path,
                          force=args.force_forecast)

        if args.stage in ("evaluate", "all"):
            run_evaluation(assets, store_path, out_dir, suffix=suffix,
                           ann=ann, example_asset=example)

    print("done")


if __name__ == "__main__":
    main()
