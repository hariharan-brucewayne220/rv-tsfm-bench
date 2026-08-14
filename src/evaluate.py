"""Losses, Diebold-Mariano test, Mincer-Zarnowitz recalibration, tables, plots.

Torch-free by design: evaluation re-runs from cached forecast parquet without
any model inference.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

VAR_FLOOR = 1e-12
REFERENCE = "LogHAR"
KEYS = ["asset", "model", "h", "origin_date"]


def qlike(target_var: np.ndarray, fcst_var: np.ndarray) -> np.ndarray:
    f = np.maximum(np.asarray(fcst_var, dtype=float), VAR_FLOOR)
    r = np.asarray(target_var, dtype=float) / f
    return r - np.log(r) - 1.0


def mse_rv(target_rv: np.ndarray, fcst_rv: np.ndarray) -> np.ndarray:
    return (np.asarray(target_rv, dtype=float)
            - np.asarray(fcst_rv, dtype=float))**2


def newey_west_lrv(d: np.ndarray, lag: int) -> float:
    """Bartlett-kernel long-run variance of the mean of d."""
    d = np.asarray(d, dtype=float)
    dc = d - d.mean()
    T = len(d)
    lrv = float(np.sum(dc**2)) / T
    for l in range(1, min(lag, T - 1) + 1):
        w = 1.0 - l / (lag + 1.0)
        gamma = float(np.sum(dc[l:] * dc[:-l])) / T
        lrv += 2.0 * w * gamma
    return lrv


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, h: int) -> tuple[float, float]:
    """DM statistic and p-value for H0: equal expected loss.

    Negative stat means model a (loss_a) is better. Newey-West with Bartlett
    kernel at lag h-1 plus the Harvey-Leybourne-Newbold small-sample
    correction; p-value from Student t with T-1 df.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    ok = np.isfinite(d)
    d = d[ok]
    T = len(d)
    if T < 30:
        return np.nan, np.nan
    lrv = newey_west_lrv(d, lag=h - 1)
    if lrv <= 0:
        return np.nan, np.nan
    dm = d.mean() / np.sqrt(lrv / T)
    hln = np.sqrt((T + 1.0 - 2.0 * h + h * (h - 1.0) / T) / T)
    stat = dm * hln
    pval = 2.0 * stats.t.sf(abs(stat), df=T - 1)
    return float(stat), float(pval)


def mz_recalibrate(fcst: pd.DataFrame, targets: pd.DataFrame, h: int,
                   window: int = 250, min_obs: int = 60,
                   ann: float = 252.0) -> pd.DataFrame:
    """Rolling log-log MZ recalibration of one (asset, model, h) forecast frame.

    fcst: index origin_date, column fcst_rv. targets: origin_date, target_rv
    for this asset and h. For origin at position k (consecutive test origins),
    fits ln(target) = a + b ln(fcst) on the trailing `window` pairs at
    positions p <= k - h (fully realized, no lookahead), expanding until
    `window` pairs exist; NaN before `min_obs` pairs.
    """
    tgt = targets.set_index("origin_date")["target_rv"]
    merged = fcst[["fcst_rv"]].join(tgt, how="inner").sort_index()
    x = np.log(np.maximum(merged["fcst_rv"].to_numpy(), 1e-12))
    y = np.log(merged["target_rv"].to_numpy())
    T = len(merged)
    recal = np.full(T, np.nan)
    for k in range(T):
        e = k - h
        if e < 0:
            continue
        lo = max(0, e - window + 1)
        xs, ys = x[lo:e + 1], y[lo:e + 1]
        ok = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[ok], ys[ok]
        if len(xs) < min_obs or np.std(xs) < 1e-10 or not np.isfinite(x[k]):
            continue
        b, a = np.polyfit(xs, ys, 1)
        resid = ys - (a + b * xs)
        s2 = float(np.sum(resid**2) / max(1, len(xs) - 2))
        recal[k] = np.exp(a + b * x[k] + 0.5 * s2)
    out = pd.DataFrame({"fcst_rv": recal}, index=merged.index)
    out["fcst_var"] = out["fcst_rv"]**2 / ann
    return out


def add_ensemble(forecasts: pd.DataFrame,
                 ttm_model: str = "TTM",
                 ref_model: str = REFERENCE,
                 ann: float = 252.0) -> pd.DataFrame:
    """EW-ENS = 0.5*TTM + 0.5*LogHAR in forecast RV space (SPEC), inner-joined
    on (asset, h, origin_date). Components are already lognormal-corrected;
    no further correction (PLAN.md 0.4).
    """
    raw = forecasts[~forecasts["recalibrated"]]
    a = raw[raw["model"] == ttm_model]
    b = raw[raw["model"] == ref_model]
    if len(a) == 0 or len(b) == 0:
        return forecasts
    m = a.merge(b, on=["asset", "h", "origin_date"], suffixes=("_a", "_b"))
    ens = pd.DataFrame({
        "asset": m["asset"],
        "model": "EW-ENS",
        "h": m["h"],
        "origin_date": m["origin_date"],
        "fcst_rv": 0.5 * m["fcst_rv_a"] + 0.5 * m["fcst_rv_b"],
    })
    ens["fcst_var"] = ens["fcst_rv"]**2 / ann
    ens["recalibrated"] = False
    keep = forecasts[forecasts["model"] != "EW-ENS"]
    return pd.concat([keep, ens[keep.columns]], ignore_index=True)


def add_mz_rows(forecasts: pd.DataFrame, targets: pd.DataFrame,
                window: int = 250, min_obs: int = 60,
                ann: float = 252.0) -> pd.DataFrame:
    """Append '<model>+MZ' recalibrated rows for every raw (asset, model, h)."""
    raw = forecasts[~forecasts["recalibrated"]]
    new_rows = []
    for (asset, model, h), grp in raw.groupby(["asset", "model", "h"]):
        tgt = targets[(targets["asset"] == asset) & (targets["h"] == h)]
        fr = grp.set_index("origin_date")[["fcst_rv"]]
        rec = mz_recalibrate(fr, tgt, int(h), window=window, min_obs=min_obs,
                             ann=ann)
        rec = rec.dropna()
        if len(rec) == 0:
            continue
        sub = pd.DataFrame({
            "asset": asset,
            "model": f"{model}+MZ",
            "h": np.int8(h),
            "origin_date": rec.index,
            "fcst_var": rec["fcst_var"].to_numpy(),
            "fcst_rv": rec["fcst_rv"].to_numpy(),
            "recalibrated": True,
        })
        new_rows.append(sub)
    keep = forecasts[~forecasts["recalibrated"]]
    return pd.concat([keep] + new_rows, ignore_index=True)


def compute_losses(forecasts: pd.DataFrame,
                   targets: pd.DataFrame) -> pd.DataFrame:
    merged = forecasts.merge(targets, on=["asset", "h", "origin_date"],
                             how="inner")
    merged = merged[np.isfinite(merged["fcst_var"])]
    merged["qlike"] = qlike(merged["target_var"].to_numpy(),
                            merged["fcst_var"].to_numpy())
    merged["mse"] = mse_rv(merged["target_rv"].to_numpy(),
                           merged["fcst_rv"].to_numpy())
    return merged


def df_to_markdown(df: pd.DataFrame, float_fmt: str = "{:.4g}") -> str:
    """Minimal markdown table writer (no tabulate dependency)."""
    d = df.reset_index()

    def fmt(v):
        if isinstance(v, float):
            return "" if not np.isfinite(v) else float_fmt.format(v)
        return str(v)

    header = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |"
            for row in d.itertuples(index=False)]
    return "\n".join([header, sep] + rows) + "\n"


def _write_table(df: pd.DataFrame, name: str, tables_dir: Path,
                 title: str) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(tables_dir / f"{name}.parquet")
    md = f"# {title}\n\n" + df_to_markdown(df)
    (tables_dir / f"{name}.md").write_text(md, encoding="utf-8")


def _per_asset_ratios(losses: pd.DataFrame, loss_col: str,
                      ref: str = REFERENCE) -> pd.DataFrame:
    """Per (asset, model, h): mean loss ratio vs the reference model, computed
    on the origins common to both model and reference."""
    ref_losses = losses[losses["model"] == ref][
        ["asset", "h", "origin_date", loss_col]].rename(
            columns={loss_col: "ref_loss"})
    rows = []
    for (asset, model, h), grp in losses.groupby(["asset", "model", "h"]):
        m = grp.merge(ref_losses, on=["asset", "h", "origin_date"])
        if len(m) < 30:
            continue
        rows.append({"asset": asset, "model": model, "h": int(h),
                     "ratio": float(m[loss_col].mean() / m["ref_loss"].mean()),
                     "n": len(m)})
    return pd.DataFrame(rows)


def build_tables(forecasts: pd.DataFrame, targets: pd.DataFrame,
                 out_dir: Path, suffix: str = "") -> dict[str, pd.DataFrame]:
    out_dir = Path(out_dir)
    tables_dir = out_dir / f"tables{suffix}"
    losses = compute_losses(forecasts, targets)
    losses.to_parquet(out_dir / f"losses{suffix}.parquet")

    tables: dict[str, pd.DataFrame] = {}
    raw_losses = losses[~losses["recalibrated"]]

    # a. pooled average loss
    pooled = raw_losses.groupby(["model", "h"])[["qlike", "mse"]].mean()
    pooled_q = pooled["qlike"].unstack("h")
    pooled_m = pooled["mse"].unstack("h")
    pooled_tbl = pd.concat({"QLIKE": pooled_q, "MSE_rv": pooled_m}, axis=1)
    tables["pooled_loss"] = pooled_tbl
    _write_table(pooled_tbl, "pooled_loss", tables_dir,
                 "Pooled average loss per model per horizon")

    # b. per-asset loss ratio vs Log-HAR, averaged across assets
    ratios = _per_asset_ratios(raw_losses, "qlike")
    tables["asset_ratios"] = ratios
    agg = ratios.groupby(["model", "h"])["ratio"].agg(["mean", "median"])
    ratio_tbl = agg.unstack("h")
    tables["ratio_vs_loghar"] = ratio_tbl
    _write_table(ratio_tbl, "ratio_vs_loghar", tables_dir,
                 "Per-asset QLIKE ratio vs Log-HAR (mean and median across assets)")

    # c. DM tests vs Log-HAR
    ref_l = raw_losses[raw_losses["model"] == REFERENCE][
        ["asset", "h", "origin_date", "qlike"]].rename(
            columns={"qlike": "ref_qlike"})
    dm_rows = []
    for (asset, model, h), grp in raw_losses.groupby(["asset", "model", "h"]):
        if model == REFERENCE:
            continue
        m = grp.merge(ref_l, on=["asset", "h", "origin_date"]).sort_values(
            "origin_date")
        stat, pval = dm_test(m["qlike"].to_numpy(),
                             m["ref_qlike"].to_numpy(), int(h))
        dm_rows.append({"asset": asset, "model": model, "h": int(h),
                        "dm_stat": stat, "pval": pval})
    dm_df = pd.DataFrame(dm_rows)
    tables["dm_per_asset"] = dm_df
    if len(dm_df):
        dm_df["sig_better"] = (dm_df["pval"] < 0.05) & (dm_df["dm_stat"] < 0)
        dm_df["sig_worse"] = (dm_df["pval"] < 0.05) & (dm_df["dm_stat"] > 0)
        dm_sum = dm_df.groupby(["model", "h"]).agg(
            pct_better=("sig_better", lambda s: 100.0 * s.mean()),
            pct_worse=("sig_worse", lambda s: 100.0 * s.mean()),
            n_assets=("asset", "count"))
        dm_tbl = dm_sum[["pct_better", "pct_worse"]].unstack("h")
        tables["dm_summary"] = dm_tbl
        _write_table(dm_tbl, "dm_summary", tables_dir,
                     "DM vs Log-HAR: percent of assets significantly "
                     "better / worse at 5 percent")
        dm_df.to_parquet(tables_dir / "dm_per_asset.parquet")

    # d. MZ effect: ratio vs raw Log-HAR before vs after recalibration,
    #    computed on the origins where the +MZ forecast exists.
    mz_losses = losses[losses["recalibrated"]]
    if len(mz_losses):
        mz_rows = []
        for (asset, model_mz, h), grp in mz_losses.groupby(
                ["asset", "model", "h"]):
            base = model_mz[:-3]                      # strip "+MZ"
            raw_grp = raw_losses[(raw_losses["asset"] == asset)
                                 & (raw_losses["model"] == base)
                                 & (raw_losses["h"] == h)]
            ref_sub = ref_l[(ref_l["asset"] == asset) & (ref_l["h"] == h)][
                ["origin_date", "ref_qlike"]]
            common = grp[["origin_date", "qlike"]].merge(
                raw_grp[["origin_date", "qlike"]], on="origin_date",
                suffixes=("_mz", "_raw")).merge(ref_sub, on="origin_date")
            if len(common) < 30:
                continue
            ref_mean = common["ref_qlike"].mean()
            mz_rows.append({
                "asset": asset, "model": base, "h": int(h),
                "ratio_before": float(common["qlike_raw"].mean() / ref_mean),
                "ratio_after": float(common["qlike_mz"].mean() / ref_mean)})
        mz_df = pd.DataFrame(mz_rows)
        if len(mz_df):
            mz_tbl = mz_df.groupby(["model", "h"])[
                ["ratio_before", "ratio_after"]].mean().unstack("h")
            tables["mz_effect"] = mz_tbl
            _write_table(mz_tbl, "mz_effect", tables_dir,
                         "Mean QLIKE ratio vs Log-HAR before vs after "
                         "MZ recalibration (common origins)")

    return tables


def make_plots(forecasts: pd.DataFrame, targets: pd.DataFrame,
               out_dir: Path, example_asset: str = "SPY",
               suffix: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    losses = compute_losses(forecasts[~forecasts["recalibrated"]], targets)
    ratios = _per_asset_ratios(losses, "qlike")
    if len(ratios) == 0:
        return

    # 1. bar chart of avg QLIKE ratio vs Log-HAR by horizon
    agg = ratios[ratios["model"] != REFERENCE].groupby(
        ["model", "h"])["ratio"].mean().unstack("h")
    fig, ax = plt.subplots(figsize=(9, 5))
    agg.plot.bar(ax=ax)
    ax.axhline(1.0, color="black", lw=0.8, ls="--")
    ax.set_ylabel("mean per-asset QLIKE ratio vs Log-HAR")
    ax.set_title("Average QLIKE loss ratio vs Log-HAR by horizon")
    fig.tight_layout()
    fig.savefig(out_dir / f"fig_ratio_bars{suffix}.png", dpi=130)
    plt.close(fig)

    # 2. per-asset scatter: TTM ratio vs Chronos ratio, h=1
    r1 = ratios[ratios["h"] == 1]
    ttm = r1[r1["model"] == "TTM"].set_index("asset")["ratio"]
    chronos_name = ("ChronosBoltSmall"
                    if (r1["model"] == "ChronosBoltSmall").any()
                    else "ChronosBoltTiny")
    chr_r = r1[r1["model"] == chronos_name].set_index("asset")["ratio"]
    both = pd.concat({"ttm": ttm, "chronos": chr_r}, axis=1).dropna()
    if len(both):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(both["ttm"], both["chronos"], s=25)
        for a, row in both.iterrows():
            ax.annotate(a, (row["ttm"], row["chronos"]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        ax.axhline(1.0, color="grey", lw=0.7, ls="--")
        ax.axvline(1.0, color="grey", lw=0.7, ls="--")
        ax.set_xlabel("TTM QLIKE ratio vs Log-HAR (h=1)")
        ax.set_ylabel(f"{chronos_name} QLIKE ratio vs Log-HAR (h=1)")
        ax.set_title("Per-asset loss ratios: TTM vs Chronos")
        fig.tight_layout()
        fig.savefig(out_dir / f"fig_ttm_vs_chronos{suffix}.png", dpi=130)
        plt.close(fig)

    # 3. example asset: realized RV vs forecasts, h=1, last 250 test days
    raw = forecasts[~forecasts["recalibrated"]]
    ex = raw[(raw["asset"] == example_asset) & (raw["h"] == 1)]
    tgt = targets[(targets["asset"] == example_asset) & (targets["h"] == 1)]
    if len(ex) and len(tgt):
        fig, ax = plt.subplots(figsize=(10, 5))
        tgt = tgt.sort_values("origin_date").tail(250)
        ax.plot(tgt["origin_date"], tgt["target_rv"], color="black",
                lw=0.9, label="realized RV")
        for model in [REFERENCE, "TTM", "EW-ENS"]:
            sub = ex[ex["model"] == model].sort_values("origin_date")
            sub = sub[sub["origin_date"].isin(tgt["origin_date"])]
            if len(sub):
                ax.plot(sub["origin_date"], sub["fcst_rv"], lw=0.9,
                        label=model)
        ax.set_ylabel("annualized vol")
        ax.set_title(f"{example_asset}: realized RV vs forecasts (h=1)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"fig_example_{example_asset}.png", dpi=130)
        plt.close(fig)
