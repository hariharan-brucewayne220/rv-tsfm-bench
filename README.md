# rv-tsfm-bench

Do zero-shot time series foundation models beat HAR-family econometric models
for realized volatility forecasting? A weekend-scale, free-data replication of
the core questions in arXiv:2607.05291 (Brini, 2026), on two tracks of real
market data:

- **Track A**: 30 assets (13 equity ETFs, 7 FX pairs, 10 futures), daily
  range-based RV (Garman-Klass / Parkinson) from free OHLC, 2005 to 2026,
  750 out-of-sample days per asset.
- **Track B**: true 5-minute realized volatility computed from Binance's full
  1-minute trade history for BTC, ETH, BNB, XRP, SOL (the same construction
  as the paper's VOLARE dataset, on a 24/7 market), 750 out-of-sample days
  per symbol.

See SPEC.md for the full specification and PLAN.md for implementation
conventions (alignment, rolling re-estimation, lognormal corrections).

## Setup

```
pip install -r requirements.txt
```

CPU-only torch is sufficient. Python 3.10.

## Usage

```
python -m src.run --quick            # 6 assets, 200 test days, smoke run
python -m src.run                    # full Track A universe, 750 test days
python -m src.run --track b          # Track B: crypto 5-min RV from Binance
python -m src.run --track b --quick  # BTC+ETH, 200 test days
python -m src.run --stage evaluate   # re-evaluate from cached forecasts
python -m src.run --models LogHAR,TTM --force-forecast
```

Stages: `download`, `forecast`, `evaluate`, `all` (default). Raw OHLC is
cached in `data/` as parquet; forecasts are cached in
`results/forecasts.parquet` keyed by (asset, model, h, origin_date), so
re-runs skip completed model-asset pairs unless `--force-forecast`.

## Models

- Econometric (rolling window 1000, refit every 22): HAR, Log-HAR (reference
  benchmark), AR(1) on log RV, EWMA (lambda 0.94), GARCH(1,1).
- Zero-shot TSFMs (context 512): Chronos-Bolt tiny, Chronos-Bolt small,
  TTM (granite-timeseries-ttm-r2).
- EW-ENS: equal-weight average of TTM and Log-HAR in forecast RV space.

## Evaluation

QLIKE on variance (primary) and MSE on annualized vol; horizons h = 1, 5, 22
(multi-day target = mean daily variance over the h days). Diebold-Mariano
tests vs Log-HAR with Newey-West (Bartlett, lag h-1) and the
Harvey-Leybourne-Newbold correction. Mincer-Zarnowitz recalibration pass
(rolling log-log OLS, trailing 250 realized pairs) to separate genuine
dynamics information from forecast scaling.

Outputs land in `results/`: `universe.md`, `forecasts.parquet`,
`losses.parquet`, `tables/*.md` + `tables/*.parquet`, and three figures.

## Findings

Full tables in `results/tables/` (Track A) and `results/tables_crypto/`
(Track B). Headline numbers below are per-asset mean QLIKE ratios vs Log-HAR
(lower is better, 1.00 = tie) and Diebold-Mariano significance at 5 percent.

![QLIKE ratio vs Log-HAR by horizon, crypto 5-min RV](results/fig_ratio_bars_crypto.png)

All numbers below use the paper's exact TTM checkpoint (branch 512-96-ft-r2.1
of granite-timeseries-ttm-r2). An earlier run with the repo's main branch
produced materially worse TTM results (QLIKE 9 to 22 percent higher); see
finding 4.

**1. At the 1-day horizon, zero-shot foundation models are genuinely
competitive, and on true high-frequency RV they win.** Track A: TTM and
Chronos-Bolt-Small both sit at parity with Log-HAR (ratios 0.998 and 1.000),
with Chronos significantly better on 10 percent of assets and neither model
significantly worse anywhere. Track B (real 5-min crypto RV): TTM 0.876
(DM significantly better on 3 of 5 symbols, worse on none) and Chronos small
0.922 (significantly better on BTC and ETH). Both crypto edges survive
Mincer-Zarnowitz recalibration (TTM 0.876 to 0.914, Chronos 0.922 to 0.929),
so on high-frequency RV this is genuine dynamics information, not scaling.
On Track A, TTM's parity does not survive recalibration (0.998 to 1.027):
there, its raw edge is calibration, exactly the paper's own decomposition of
TTM at short horizons.

**2. At 5-day and 22-day horizons, econometrics wins decisively on both
tracks.** Every TSFM sits at QLIKE ratios of 1.14 to 1.83 beyond h = 1.
Recalibration closes much of the gap (Chronos small on crypto at h = 22:
1.66 to 1.22) but never all of it. This is where our results genuinely
diverge from the paper, which found TTM narrowly ahead at all horizons on
its data.

**3. The TTM + Log-HAR ensemble now partially replicates.** EW-ENS beats
Log-HAR at h = 1 on both tracks (0.984 on Track A, 0.903 on Track B, DM
significantly better on 10 and 80 percent of assets respectively), is at
parity at h = 5 (1.039 / 1.011), and loses modestly at h = 22 (1.062 /
1.106). It is also the most robust non-benchmark model: significantly worse
than Log-HAR almost nowhere. The paper's "average TTM with Log-HAR" advice
holds here at short horizons but not at the monthly horizon.

**4. Checkpoint choice matters as much as architecture choice.** Switching
TTM from the repo default main branch to the paper's frequency-tuned
512-96-ft-r2.1 branch, the same architecture and parameter count, improved
TTM's QLIKE by 9 to 22 percent across horizons and flipped the ensemble
finding from "never beats Log-HAR" to "beats it at h = 1 on both tracks".
This amplifies the paper's most durable finding: within-class model
selection dwarfs the foundation-vs-econometric choice. Relatedly, the
median-vs-mean point forecast distinction is vacuous for Chronos-Bolt in the
chronos-forecasting library, whose Bolt implementation returns the 0.5
quantile as the "mean" (verified in the installed library source and
numerically; the paper describes Chronos-Bolt as exposing a mean head).

| Paper finding (on 5-min VOLARE RV) | Our result | Agree? |
|---|---|---|
| Pooled losses favor TSFMs, edge concentrated in a few assets | TSFM wins concentrate at h=1 and in BTC/ETH on Track B | Yes at h=1, no beyond |
| Per-asset, only TTM narrowly beats Log-HAR at all horizons | TTM at parity (A) and clearly ahead (B) at h=1; loses at h=5 and h=22 | Yes at h=1, no beyond |
| Much of the short-horizon TSFM edge is scaling, removed by MZ recalibration | Track A TTM edge vanishes under MZ (calibration); crypto h=1 edges survive (information) | Yes on Track A, stronger than the paper on Track B |
| TTM + Log-HAR equal-weight ensemble is the most robust choice | EW-ENS beats Log-HAR at h=1 both tracks, parity h=5, modest loss h=22; almost never significantly worse | Partly (short horizons yes) |
| Architecture variance dwarfs the foundation-vs-econometric gap | Yes, and even checkpoint variance within one architecture does (finding 4) | Yes, amplified |

Interpretation of the remaining divergence (long horizons): the paper
evaluates 5-minute RV for equities/FX/futures with volatility levels fed to
the models; we evaluate daily range-based proxies (Track A) and 24/7 crypto
5-minute RV (Track B) with log RV inputs. Noisier proxies and a 24/7 market
plausibly both penalize zero-shot models most where scale errors compound,
at the monthly horizon.

### Deviations from the paper (full list, from a close read of the paper)

- Data: daily range-based RV proxies (Track A) and crypto 5-min RV (Track B)
  instead of the paper's VOLARE 5-min RV for 50 equities/FX/futures.
- Forecast target: we use the h-day-average RV target. The paper's headline
  results use the point-in-time target h days ahead and run the h-day average
  as a robustness arm (its Tab. 15, which is the table our ratios should be
  compared against; the paper reports the same headline ranking under both
  targets, though a few mid-table models reorder).
- Model scale and point forecast: the paper feeds raw volatility levels to the
  TSFMs and uses the conditional mean as the point forecast. We feed log RV
  and use the median quantile, then apply a trailing lognormal bias
  calibration identically to all log-space models.
- Context length 512 for all TSFMs (paper: 1000 where the architecture allows;
  its own sensitivity table shows 512 is at or near optimal for Chronos-Bolt).
- TTM checkpoint: final results use the paper's exact branch (512-96-ft-r2.1
  of granite-timeseries-ttm-r2, 855,100 parameters). An initial run used the
  repo's main branch instead and produced 9 to 22 percent worse TTM QLIKE
  (those tables are preserved in results/baseline_pre_paperfix*/); see
  Findings point 4.
- Econometric estimation: direct-h regressions refit every 22 days (paper:
  iterated multistep, re-estimated daily). Benchmark set differs: we include
  EWMA and GARCH(1,1) but not ARFIMA, MEM, or the HAR-J/RS/Q variants, whose
  intraday inputs (bipower variation, semivariance, quarticity) do not exist
  for daily OHLC data.
- We winsorize the input RV series; the paper instead bounds forecasts to each
  asset's in-sample RV support (a guardrail it reports binds in 0.05% of cases).
- We run DM tests and MZ recalibration but not the Model Confidence Set or
  Giacomini-Rossi fluctuation tests.

### Why our results and the paper's reconcile

Our TSFM wrapper (log-space modeling plus a trailing bias calibration applied
to every model) is functionally close to the paper's Mincer-Zarnowitz
recalibrated arm, not its raw arm. And our results match the paper's
MZ-corrected table well: after the paper applies its MZ correction at h = 1,
TimesFM 2.5 has the lowest corrected QLIKE (0.190) with Chronos-Bolt-Base
close behind (0.192), both ahead of recalibrated Log-HAR (0.195), and TTM's
short-horizon edge shrinks to nothing (0.192 to 0.198). We see the same
short-horizon picture natively: Chronos and TTM at or
above parity at h = 1, with Track A TTM's edge attributable to calibration
under our own MZ pass. Where we find more than the paper is Track B: on
crypto 5-minute RV the h = 1 TSFM edges survive recalibration, indicating
genuine short-horizon information in that market.
