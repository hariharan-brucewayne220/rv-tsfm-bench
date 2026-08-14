# rv-tsfm-bench: Do Time Series Foundation Models Beat HAR for Realized Volatility?

A weekend-scale, free-data replication of the core questions in arXiv:2607.05291
(Brini, "Forecasting Realized Volatility with Time Series Foundation Models:
A Comparison with Econometric Benchmarks", July 2026).

## Research questions (from the paper, kept verbatim in spirit)

1. Do zero-shot time series foundation models (TSFMs) beat well-specified HAR-family
   econometric models for realized volatility (RV) forecasting?
2. Is any TSFM advantage genuine volatility-dynamics information, or mostly better
   forecast scaling? (Test via Mincer-Zarnowitz recalibration of all forecasts.)
3. Does a simple equal-weight ensemble of the best TSFM and Log-HAR beat both components?

Paper's headline findings to check against: pooled losses favor TSFMs but the edge is
concentrated in outlier assets; per-asset, only TTM narrowly beats Log-HAR at all
horizons; much of the short-horizon edge disappears after MZ recalibration; the
TTM + Log-HAR equal-weight average is the most robust choice.

## Deviation from the paper (data)

The paper uses the VOLARE dataset (5-min RV from tick data; access requires manual
account approval, not feasible this weekend). We instead use **daily range-based RV
proxies computed from free daily OHLC data**:

- RV proxy = Garman-Klass daily variance estimator, annualized vol = sqrt(252 * GK).
  Also compute Parkinson and Yang-Zhang as robustness columns.
- Forecast target: log(RV) internally where a model needs it; losses evaluated on
  variance level with QLIKE and on RV with MSE (both reported).

## Universe (~30 assets, 3 classes, daily OHLC via yfinance, 2005-01-01 to latest)

- Equities (ETFs): SPY, QQQ, DIA, IWM, XLF, XLK, XLE, XLV, XLI, XLP, XLU, EEM, EFA
- FX (Yahoo `=X` pairs): EURUSD=X, GBPUSD=X, USDJPY=X, AUDUSD=X, USDCAD=X, USDCHF=X, NZDUSD=X
- Futures (Yahoo continuous front month `=F`): ES=F, NQ=F, YM=F, GC=F, SI=F, CL=F, ZN=F, ZB=F, HG=F, NG=F

Drop any asset with < 2000 usable observations after cleaning. FX has no true open/close
gaps; use Parkinson for FX if GK misbehaves on zero open-close moves (decide in code,
document choice).

## Models

Econometric benchmarks (rolling re-estimation, window = 1000 obs, re-fit every 22 obs):
1. HAR (Corsi 2009): RV_{t+h} ~ RV_t + mean(RV_{t-4..t}) + mean(RV_{t-21..t})
2. Log-HAR: same in logs, forecasts bias-corrected back with half the residual variance
   (lognormal correction). **This is the reference benchmark.**
3. AR(1) on log RV
4. EWMA / RiskMetrics (lambda = 0.94)
5. GARCH(1,1) on daily close-close returns (arch package), forecast variance mapped to RV scale
   via in-sample mean ratio (document this)

Zero-shot TSFMs (no fine-tuning, context window 512, quantile median as point forecast):
6. Chronos-Bolt tiny (amazon/chronos-bolt-tiny)
7. Chronos-Bolt small (amazon/chronos-bolt-small)
8. TTM - Tiny Time Mixers (ibm-granite/granite-timeseries-ttm-r2, zero-shot)

TSFMs forecast the log-RV series directly (paper feeds RV levels; log stabilizes scale
for range-based proxies; note the deviation). Exponentiate with lognormal correction
using recent residual variance, same treatment as Log-HAR for fairness.

Ensemble:
9. EW-ENS: 0.5 * TTM + 0.5 * Log-HAR (in forecast RV space)

## Horizons and evaluation

- Horizons: h = 1, 5, 22 days. Multi-day target = mean RV over the h days. NOTE (learned
  from the full paper text 2026-07-11): the paper's HEADLINE target is point-in-time RV
  h days ahead; the h-day-average target is its robustness arm (Tab. 15), which is the
  arm our results are comparable to. The paper reports the same headline ranking under
  both targets (a few mid-table models reorder).
- Out-of-sample window: last ~750 trading days per asset (~3 years), rolling origin, step 1 day.
- Losses: QLIKE (on variance) and MSE (on annualized vol). QLIKE is primary.
- Reporting, mirroring the paper's key tables:
  a. Pooled average loss per model per horizon.
  b. Per-asset loss ratio vs Log-HAR, then averaged across assets (equal weight per asset).
  c. Diebold-Mariano tests: each model vs Log-HAR, per asset; report % of assets where
     model significantly better / worse at 5%.
  d. Mincer-Zarnowitz recalibration: regress realized on forecast in logs over a rolling
     window, re-evaluate recalibrated forecasts; compare loss ratios before/after.
  e. Ensemble row for EW-ENS in all tables.
- Plots: bar chart of avg loss ratio vs Log-HAR by horizon; per-asset scatter of
  TTM ratio vs Chronos ratio; one example asset RV vs forecasts time series.

## Track B: true high-frequency realized volatility (real 5-min RV, no proxies)

Motivation: the headline result should also hold on RV computed from real intraday data,
not only on range-based proxies. Binance publishes full-history 1-minute klines as free
zip archives, no account needed (verified reachable from this machine, including from
Python): https://data.binance.vision/data/spot/monthly/klines/{SYMBOL}/1m/{SYMBOL}-1m-{YYYY-MM}.zip

- Symbols: BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, SOLUSDT (start: earliest available,
  BTC/ETH from 2017-08, SOL from 2020-08).
- Aggregate 1-min closes to 5-min log returns; daily RV_t = sum of squared 5-min returns
  over the UTC day (24/7 market, 288 intervals/day). Annualize with 365.
- Days with > 10% missing 5-min intervals are dropped.
- Same models, horizons, and evaluation as Track A. Report Track A and Track B in
  separate tables, never pooled.

## Environment constraints (IMPORTANT, verified on this machine)

- huggingface.co is NOT reachable from Python (TLS connection reset by ISP-level
  filtering; PowerShell/schannel works fine). All three model repos are pre-downloaded
  into the standard HF cache (~/.cache/huggingface/hub) with correct refs/snapshots
  layout. Code MUST load models with `from_pretrained(..., local_files_only=True)`
  (or trust the cache fallback) and MUST NOT attempt network access to huggingface.co,
  request non-main revisions, or call snapshot_download.
- TTM: use ibm-granite/granite-timeseries-ttm-r2, main revision only (512 context,
  96-step prediction head; slice the first h steps as needed).
- yfinance and data.binance.vision are reachable from Python (verified).
- Verified offline model loading works (2026-07-11): chronos BaseChronosPipeline
  predict_quantiles takes the context tensor as first POSITIONAL arg named `inputs`
  (shape [batch, context_len], returns quantiles [batch, h, n_q]); TTM r2 forward is
  `model(past_values=[batch, 512, 1])` returning prediction_outputs [batch, 96, 1].
  Set env HF_HUB_OFFLINE=1 in run.py before importing model libs.

## Engineering constraints

- CPU-only machine (torch 2.8 CPU, Python 3.10). Batch TSFM inference aggressively:
  build all context windows for an asset as a batch tensor, one forward pass per
  asset-horizon (Chronos-Bolt supports batch; TTM likewise).
- `--quick` flag: 6 assets, 200 test days, for CI/smoke.
- Cache downloaded OHLC to data/ as parquet; cache model forecasts to results/ as parquet
  so evaluation can re-run without re-inference.
- Repo layout:
  rv-tsfm-bench/
    SPEC.md, README.md, requirements.txt
    src/data.py        (download, clean, RV estimators)
    src/models_econ.py (HAR family, AR1, EWMA, GARCH)
    src/models_tsfm.py (Chronos-Bolt, TTM wrappers)
    src/evaluate.py    (losses, DM test, MZ recalibration, tables)
    src/run.py         (CLI orchestrator: download -> forecast -> evaluate)
    results/           (parquet + generated tables/figures)
- Style: plain functions + dataclasses, no framework. Deterministic seeds. No em dashes
  in any docs or comments. Commits without Co-Authored-By trailer.

## Definition of done

- `python -m src.run --quick` completes end-to-end on CPU in < 15 min.
- Full run completes overnight; results tables in results/ as markdown + parquet.
- README with findings vs the paper's findings (agree/disagree table).
