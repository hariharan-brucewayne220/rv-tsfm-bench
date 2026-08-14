# PLAN.md: Implementation plan for rv-tsfm-bench

Authoritative spec: SPEC.md. This plan fixes signatures, schemas, alignment
conventions, and known-bug handling before any code is written. Style: plain
functions and dataclasses, deterministic seeds (SEED = 42 for random, numpy,
torch), no em dashes anywhere, CPU-only torch.

## 0. Global conventions

### 0.1 Series definitions (per asset)

All per-asset series share one tz-naive DatetimeIndex of that asset's trading
days after cleaning. Positional index i refers to row i of that index; day t
means the day at position i.

- `var_d[i]`: daily variance proxy for day i (Garman-Klass, or Parkinson for FX,
  see 1.3). Unannualized, in squared-log-return units.
- `rv_ann[i] = sqrt(252 * var_d[i])`: annualized volatility.
- `log_rv[i] = ln(rv_ann[i])`: the series fed to log-space models and TSFMs.
- `ret_cc[i] = ln(close[i] / close[i-1])`: close-close log return (GARCH input).

### 0.2 Target and alignment convention (no lookahead)

A forecast is made at origin i using information available through the close of
day i (inclusive). The target is the mean daily variance over the next h days:

```
target_var[i, h] = mean( var_d[i+1 : i+1+h] )        # Python slice, h elements
```

Exact index arithmetic:

- h = 1:  target = var_d[i+1]
- h = 5:  target = mean(var_d[i+1], ..., var_d[i+5])   = var_d[i+1 : i+6].mean()
- h = 22: target = mean(var_d[i+1], ..., var_d[i+22])  = var_d[i+1 : i+23].mean()

Annualized-vol target for MSE: `target_rv[i, h] = sqrt(252 * target_var[i, h])`.

Feature convention (HAR family), all windows END at i and include i:

```
f_d[i] = x[i]
f_w[i] = mean(x[i-4  : i+1])   # 5 obs, pandas x.rolling(5).mean() at row i
f_m[i] = mean(x[i-21 : i+1])   # 22 obs, pandas x.rolling(22).mean() at row i
```

where x is var_d (HAR) or log_rv (Log-HAR). pandas `rolling(k).mean()` already
ends the window at the current row, so no shifting of features is needed; the
only shift in the system is the target slice starting at i+1. This is the
single place lookahead can enter; unit-test it explicitly (see 6).

Valid origins: `i >= 21` (feature warm-up), `i >= 511` (TSFM context),
`i >= 1000 + 21 + h` (first refit window), and `i <= n - 1 - h` (target fully
observed). Test set: the LAST 750 valid origins per asset,
`i in [n - h - 750, n - h - 1]` intersected with the validity constraints
(200 origins under --quick).

### 0.3 Rolling re-estimation scheme (econometric models)

- Estimation window: 1000 training pairs. Re-fit every 22 test origins.
- At test origin i, a training pair with origin j is admissible only if its
  target is fully observed by day i, i.e. `j + h <= i`, so `j <= i - h`.
- Refit rule: at the first test origin and then whenever
  `(i - i_test_start) % 22 == 0`, re-estimate on training origins
  `j in (i - h - 1000, i - h]` (the most recent 1000 admissible origins).
  Between refits, reuse the last fitted parameters; features are still
  recomputed at every origin i.
- Each horizon is fitted as a DIRECT regression (own coefficients per h);
  no iterated multi-step for HAR/Log-HAR/AR1. GARCH and EWMA produce h-step
  paths analytically (see 3).

### 0.4 Lognormal bias correction (shared rule)

Any model that predicts in log space (Log-HAR, AR1-log, TSFMs) produces
`m = predicted log target` and converts back with

```
forecast = exp(m + 0.5 * s2)
```

where s2 is an estimate of the variance of the log-scale forecast error at
that horizon, using only information available at origin i:

- Log-HAR / AR1: s2 = residual variance of the current fitted regression on
  its 1000-pair training window (per horizon).
- TSFMs: s2 = variance of realized log errors `m_j - ln(target)` over the most
  recent 250 origins j with `j <= i - h` (trailing, lookahead-free). Seed this
  with a burn-in: run TSFM inference on the 250 origins immediately BEFORE the
  test window so s2 exists from the first test origin. Burn-in forecasts are
  used only for calibration, never scored.

Correction is applied exactly once, at the model level. EW-ENS averages
already-corrected RV-space forecasts and gets no further correction.

## 1. src/data.py

### 1.1 Functions

```python
@dataclass
class AssetData:
    ticker: str
    asset_class: str            # "equity" | "fx" | "futures"
    df: pd.DataFrame            # schema in 1.2
    rv_estimator: str           # "gk" | "parkinson" (what var_d uses)

def download_ohlc(ticker: str, start: str, cache_dir: Path, force: bool = False) -> pd.DataFrame
def clean_ohlc(raw: pd.DataFrame, ticker: str) -> pd.DataFrame
def garman_klass_var(df: pd.DataFrame) -> pd.Series
def parkinson_var(df: pd.DataFrame) -> pd.Series
def yang_zhang_var(df: pd.DataFrame, window: int = 22) -> pd.Series
def build_asset(ticker: str, asset_class: str, cache_dir: Path, force: bool = False) -> AssetData | None
def load_universe(cache_dir: Path, quick: bool = False, force: bool = False) -> list[AssetData]
def compute_targets(asset: AssetData, horizons: tuple[int, ...] = (1, 5, 22)) -> pd.DataFrame
```

`load_universe` drops assets with < 2000 usable rows and logs each drop with a
reason. QUICK_TICKERS = SPY, QQQ, EEM, EURUSD=X, USDJPY=X, CL=F (2 per class).

### 1.2 Schemas

`data/ohlc_{safe_ticker}.parquet` (safe_ticker replaces `=` with `_`):
- index: DatetimeIndex `date`, tz-naive, ascending, unique
- columns: `open, high, low, close, volume` float64 (auto_adjust=False download;
  keep raw OHLC for range estimators, `adj_close` column kept but unused)

`AssetData.df` (after clean + estimators), same index:
- `open, high, low, close` float64
- `ret_cc` float64 (NaN on row 0)
- `var_gk, var_park, var_yz` float64 (daily variance; var_yz NaN for first
  window rows)
- `var_d` float64 (the chosen estimator, strictly > 0 on all remaining rows)
- `rv_ann, log_rv` float64

`compute_targets` output, long form:
- columns: `origin_date` (datetime64), `h` (int8), `target_var` (float64),
  `target_rv` (float64); one row per valid origin and horizon.

Panel cache: `data/rv_panel.parquet`, long over assets: `asset` (str) +
AssetData.df columns with `date` as a column.

### 1.3 Cleaning and estimator selection rules

Cleaning, in order: drop rows with any NaN in OHLC; drop rows violating
`high >= max(open, close)` or `low <= min(open, close)` (after a 1e-8 relative
tolerance snap); drop rows with `high == low` (no range, var undefined);
forward-looking gap filter: none (calendar gaps are fine, models are
positional). Then compute estimators.

Estimator choice per asset: default GK. For the FX class, compute the fraction
of days with `abs(ln(close/open)) < 1e-6`; if > 10 percent, use Parkinson for
that asset and record `rv_estimator = "parkinson"`. Document the per-asset
choice in results/universe.md. Any remaining nonpositive var_d rows are
replaced by the Parkinson value for that day; if still nonpositive, drop the
row. Winsorize var_d at the 0.01 and 99.9 percentiles per asset (log and
target stability; document).

## 2. src/models_econ.py

All models consume an AssetData plus precomputed targets and produce forecasts
of `target_var` (mean daily variance over the horizon) at each test origin.

### 2.1 Common types and driver

```python
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
                            log_space: bool) -> np.ndarray
```

`rolling_direct_forecast` implements 0.3 exactly once; HAR, Log-HAR and AR1 are
thin feature-builders over it. OLS via `np.linalg.lstsq` on a column of ones
plus features. When `log_space=True` it returns exp(m + 0.5 * s2_resid).

### 2.2 Model functions

```python
def forecast_har(asset, targets, test_idx, cfg) -> list[ForecastResult]
    # X = [1, var_d[i], mean5(var_d)[i], mean22(var_d)[i]], y = target_var; level space
    # negative predictions floored at 1e-12 (log floor of the asset min var / 10)

def forecast_log_har(asset, targets, test_idx, cfg) -> list[ForecastResult]
    # X on log_rv features, y = ln(target_rv); back-transform per 0.4,
    # then fcst_var = fcst_rv**2 / 252. REFERENCE BENCHMARK.

def forecast_ar1_log(asset, targets, test_idx, cfg) -> list[ForecastResult]
    # X = [1, log_rv[i]], direct per horizon, back-transform per 0.4

def forecast_ewma(asset, targets, test_idx, lam: float = 0.94) -> list[ForecastResult]
    # Recursive on var_d itself: s[i] = lam*s[i-1] + (1-lam)*var_d[i],
    # initialized at mean(var_d[:50]). Forecast for every h is s[i] (flat).
    # No scale mapping needed because it smooths the RV proxy directly. No refit.

def forecast_garch(asset, targets, test_idx, cfg) -> list[ForecastResult]
    # arch_model(100*ret_cc, vol="GARCH", p=1, q=1, dist="normal",
    # mean="Constant"); fit on the 1000 returns ending at origin i (refit every
    # 22 origins, disp="off"); analytic forecast of conditional variance for
    # steps 1..h, mean over steps, divide by 100**2.
    # Scale map: c = mean(var_d over train window) / mean(garch in-sample
    # conditional var over same window, rescaled); fcst_var = c * garch_mean.
    # c recomputed at every refit; guard c with clip to [1e-3, 1e3].
```

GARCH admissibility uses returns through day i only (returns need no h gap;
the h gap in 0.3 applies to regression pairs, not to the GARCH return sample).

## 3. src/models_tsfm.py

### 3.1 Functions

```python
@dataclass
class TSFMConfig:
    context_length: int = 512
    batch_size: int = 64          # windows per forward pass on CPU
    horizons: tuple[int, ...] = (1, 5, 22)
    max_pred_len: int = 22        # one forward pass covers all horizons
    calib_window: int = 250       # trailing log-error window for 0.4
    num_threads: int = max(1, os.cpu_count() - 1)

def build_context_matrix(log_rv: np.ndarray, origins: np.ndarray, L: int) -> np.ndarray
    # shape (num_origins, L); row k = log_rv[origins[k]-L+1 : origins[k]+1]

def forecast_chronos(asset, targets, origins, cfg, model_id: str) -> list[ForecastResult]
def forecast_ttm(asset, targets, origins, cfg) -> list[ForecastResult]
def lognormal_calibrate(log_median_path: np.ndarray, log_rv: np.ndarray,
                        origins: np.ndarray, h: int, calib_window: int) -> np.ndarray
```

`origins` includes the 250 burn-in origins before the test window (0.4); only
test origins are emitted in ForecastResult.

### 3.2 Inference details

Chronos-Bolt (`amazon/chronos-bolt-tiny`, `-small`):
- `BaseChronosPipeline.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float32)`
- Input: `torch.tensor` of shape (B, 512) float32, per-batch chunks of
  cfg.batch_size, under `torch.no_grad()` and `torch.inference_mode()`.
- Call `predict_quantiles(context, prediction_length=22, quantile_levels=[0.5])`.
  Returns `(quantiles, mean)`; quantiles shape (B, 22, 1) with our single
  level. Median path = `quantiles[:, :, 0]`. Assert shapes at runtime; the
  bolt API differs from original Chronos `predict` which returns sample paths
  (B, num_samples, len). Do not use `.predict()` sample semantics.
- Bolt models are deterministic quantile forecasters; still set torch seed.

TTM (`ibm-granite/granite-timeseries-ttm-r2` via tsfm_public):
- `TinyTimeMixerForPrediction.from_pretrained(model_id)`; the r2 default
  revision is context 512, prediction 96. Assert
  `model.config.context_length == 512` at load; if the resolved revision
  differs, pass `revision="main"` and fail loudly rather than pad.
- Input shape (B, 512, 1) float32 (channels last); output
  `.prediction_outputs` shape (B, 96, 1); take steps 0..21 as the log-RV path.
- TTM emits point forecasts only; record that "quantile median" applies to
  Chronos and the TTM point forecast is used as the median analogue.

Shared post-processing, per horizon h: predicted log target
`m[i, h] = mean over steps 1..h of the predicted log_rv path`. This
approximates ln of the mean; the lognormal correction in 0.4 absorbs the
Jensen gap empirically because s2 is estimated from realized errors of this
exact m definition. Then `fcst_rv = exp(m + 0.5*s2)`, `fcst_var = fcst_rv**2/252`.

Normalization: feed raw log_rv (Chronos-Bolt scales internally; TTM r2
zero-shot expects standardized-ish input, so standardize each context row by
its own window mean/std and invert on the output path; document as the TTM
per-window normalization).

CPU throughput: `torch.set_num_threads(cfg.num_threads)`; one model load per
process reused across assets; approx 30 assets x 1000 windows / 64 per batch
is about 470 forwards per model, minutes-scale for tiny/small on CPU.

### 3.3 Ensemble

Computed in evaluate step, not a model class:
`EW-ENS fcst_var = 0.5 * fcst_var_TTM + 0.5 * fcst_var_LogHAR` per
(asset, origin, h), inner-joined on origin_date.

## 4. src/evaluate.py

### 4.1 Functions

```python
def qlike(target_var: np.ndarray, fcst_var: np.ndarray) -> np.ndarray
    # r = target/fcst; r - ln(r) - 1; fcst floored at 1e-12
def mse_rv(target_rv: np.ndarray, fcst_rv: np.ndarray) -> np.ndarray
def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, h: int) -> tuple[float, float]
    # returns (stat, pvalue); Newey-West with Bartlett kernel, lag = h - 1,
    # Harvey-Leybourne-Newbold small-sample correction
def mz_recalibrate(fcst: pd.DataFrame, targets: pd.DataFrame, h: int,
                   window: int = 250, min_obs: int = 60) -> pd.DataFrame
    # rolling OLS ln(target) = a + b ln(fcst) using only pairs with
    # origin j <= i - h; recalibrated = exp(a + b ln f + 0.5 s2_resid);
    # expanding window until 250 pairs available, NaN before min_obs
def build_tables(forecasts: pd.DataFrame, targets: pd.DataFrame,
                 out_dir: Path) -> dict[str, pd.DataFrame]
def make_plots(forecasts, targets, out_dir) -> None
```

### 4.2 Schemas

`results/forecasts.parquet` (long, the single forecast store):
- columns: `asset` str, `model` str, `h` int8, `origin_date` datetime64,
  `fcst_var` float64, `fcst_rv` float64, `recalibrated` bool (False for raw,
  True rows appended by MZ step with model name suffix `+MZ`)

`results/losses.parquet`: same keys plus `target_var, target_rv, qlike, mse`.

Tables written as markdown and parquet under results/tables/:
- `pooled_loss.md`: rows model, cols horizon, mean QLIKE (and MSE variant)
- `ratio_vs_loghar.md`: per-asset QLIKE ratio model/LogHAR, arithmetic mean
  and median across assets, per horizon
- `dm_summary.md`: per model and horizon, percent of assets significantly
  better / worse than Log-HAR at 5 percent
- `mz_effect.md`: mean loss ratio before vs after recalibration
- EW-ENS appears as a model row in all tables.

Plots (matplotlib, no seaborn): `fig_ratio_bars.png`, `fig_ttm_vs_chronos.png`
(per-asset scatter of ratios, h=1), `fig_example_SPY.png` (RV vs Log-HAR, TTM,
EW-ENS forecasts, last 250 test days).

## 5. src/run.py

```python
def main() -> None   # argparse
# flags: --quick, --stage {download,forecast,evaluate,all} default all,
#        --models comma-list default all, --force-download, --force-forecast,
#        --out results/
def set_seeds(seed: int = 42) -> None
```

Pipeline: load_universe -> compute_targets -> for each econ model and TSFM,
skip if its (model, asset, h) rows already exist in results/forecasts.parquet
unless --force-forecast (append-only parquet rewrite via full read + concat +
drop_duplicates on keys) -> ensemble -> evaluate -> tables + plots + README
findings stub. Under --quick: 6 assets, last 200 test origins, same code path.
Progress via plain `print` with asset/model counters, no logging framework.

## 6. Top 8 likely bugs and prescribed handling

1. Garman-Klass negative values. GK = 0.5*ln(H/L)^2 - (2 ln2 - 1)*ln(C/O)^2 can
   go negative on large open-close moves with small range. Handling: per-day
   fallback to Parkinson when GK <= 0; drop the row if Parkinson also <= 0;
   log the per-asset fallback fraction to results/universe.md. Never clip to
   an arbitrary epsilon (distorts QLIKE).

2. FX zero open-close moves. Yahoo FX opens often equal prior closes and some
   days have H == L, making GK degenerate and log_rv = -inf. Handling: the
   10 percent rule in 1.3 switches FX assets to Parkinson wholesale; H == L
   rows are dropped in cleaning; assert np.isfinite(log_rv).all() after build.

3. yfinance MultiIndex columns. Batch downloads return columns like
   ('Close','SPY'); shapes differ between single and multi ticker and across
   yfinance versions. Handling: download ONE ticker per call with
   `auto_adjust=False, progress=False`; if `isinstance(df.columns, pd.MultiIndex)`
   then `df.columns = df.columns.droplevel(1)`; normalize names to lowercase;
   assert the five required columns exist before caching.

4. Chronos-Bolt API shape. Bolt pipelines return quantiles, not sample paths;
   `.predict()` semantics differ from original Chronos and silently produce
   shape (B, 9, L) quantile tensors that look like samples. Handling: use only
   `predict_quantiles(..., quantile_levels=[0.5])`, assert the returned tensor
   is (B, 22, 1), and unit-test one forward on a synthetic sine batch
   asserting monotone quantile ordering when levels [0.1, 0.5, 0.9] are used.

5. TTM context length. r2 checkpoints are fixed (512 context, 96 horizon);
   feeding shorter contexts or expecting variable horizons breaks or silently
   pads. Handling: assert model.config.context_length == 512 and
   prediction_length == 96 at load; require origins to have >= 512 history
   (folded into valid-origin rule 0.2); take output[:, :22, 0] explicitly.

6. Lognormal bias correction errors. Risks: omitting 0.5*s2 (systematic
   underforecast, QLIKE penalizes hard), using h=1 residual variance for all
   horizons, using future errors in the TSFM trailing window, or
   double-correcting the ensemble. Handling: correction lives in exactly two
   functions (rolling_direct_forecast for econ, lognormal_calibrate for TSFM),
   both parameterized by h and both using only origins j <= i - h; ensemble
   averages corrected RV forecasts with no further adjustment; unit test on
   simulated lognormal data recovers the true mean within tolerance.

7. DM test autocorrelation for h > 1. Overlapping h-step targets make loss
   differentials MA(h-1); naive variance overstates significance. Handling:
   Newey-West long-run variance with Bartlett kernel and lag h-1 (0, 4, 21),
   plus the Harvey-Leybourne-Newbold small-sample adjustment; compare against
   statsmodels' HAC on a constant-only regression in a unit test.

8. GARCH-to-RV scale mismatch. GARCH conditional variance of close-close
   returns sits on a different scale than range-based var_d (overnight gaps,
   estimator constants), and arch requires returns scaled by 100 for optimizer
   stability. Handling: fit on 100*ret_cc, divide forecasts by 1e4, then apply
   the training-window mean ratio c = mean(var_d)/mean(cond_var) recomputed at
   every refit, clipped to [1e-3, 1e3]; document c per asset in the run log.
   Convergence failures: catch, reuse previous fit, count occurrences.

Secondary watch items (not top 8): pandas rolling window off-by-one (covered
by the alignment unit test in 0.2), parquet datetime index round-trips, arch
DataScaleWarning spam (suppress via warnings filter, keep others).

## 7. Build order

1. data.py + alignment unit test (targets vs hand-computed slices for
   h = 1, 5, 22 on a 30-row toy frame).
2. models_econ.py (HAR, Log-HAR first; EWMA; AR1; GARCH last).
3. evaluate.py losses + tables on econ-only forecasts (pipeline runs end to
   end without torch).
4. models_tsfm.py (Chronos tiny, then small, then TTM), burn-in calibration.
5. run.py orchestration, --quick smoke, then full run.
6. DM, MZ, plots, README findings table.
