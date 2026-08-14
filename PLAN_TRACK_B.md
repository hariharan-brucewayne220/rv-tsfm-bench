# PLAN_TRACK_B.md: Addendum plan for Track B (Binance true 5-min RV)

Authoritative spec: SPEC.md section "Track B". This addendum extends PLAN.md.
Track A code (data.py, models_econ.py, models_tsfm.py, evaluate.py, run.py) is
complete and reviewed; Track B reuses it unchanged wherever possible. The only
edits to existing files are additive, default-preserving parameters listed in
section 4; with defaults, Track A behavior is byte-identical. Do NOT touch
results/forecasts.parquet or data/ohlc_*.parquet (a Track A full run may be
executing concurrently). Style: plain functions and dataclasses, deterministic
seeds, no em dashes.

## 0. Decisions (with justification)

### 0.1 HAR lags and horizons: keep 5/22 in observation units

For crypto, one observation is one UTC calendar day and the market trades
7 days a week, so "weekly/monthly" could be redefined as 7/30 observations.
We keep the HAR lags at 5 and 22 observations and the horizons at
h = 1, 5, 22 observations, identical to Track A. Justification:

- Observation units is the standard choice when running HAR across mixed
  calendars. The HAR components are heuristic multi-scale smoothers of the
  volatility cascade (Corsi 2009), not literal calendar aggregates; results
  are known to be insensitive to the exact lag lengths.
- The paper being replicated compares models at fixed observation horizons;
  keeping h = 1, 5, 22 makes Track A and Track B tables line up row for row.
- Every downstream constant is defined in observation units and shared:
  refit_every = 22, FEATURE_WARMUP = 21, DM lag = h - 1, MZ gap j <= i - h,
  TRAIN_WINDOW = 1000, calib_window = 250. Changing lags to 7/30 would force
  a parallel set of constants and code paths through data.py, models_econ.py,
  and evaluate.py, violating the reuse constraint for negligible benefit.

State this explicitly in results/tables_crypto/README note and in
universe_crypto.md: "lags and horizons are in observation units (days of a
7-day week for crypto)".

### 0.2 Annualization: 365 for crypto, threaded as one constant

Crypto RV is annualized with 365 (SPEC). Key fact that keeps this cheap: the
annualization constant A is a pure presentation-layer scale linking daily
variance var_d to rv_ann = sqrt(A * var_d). QLIKE (primary loss) is computed
on unannualized variance and is exactly invariant to A. Log-space models see
only a constant shift ln(sqrt(A)) in log_rv, absorbed exactly by the OLS
intercept and by TTM per-window standardization, and near-exactly by
Chronos-Bolt internal scaling. MSE_rv and the plots scale uniformly by A, so
loss ratios and DM results are unaffected. A must merely be CONSISTENT within
a track, because MZ and the ensemble operate on fcst_rv and then convert back
to fcst_var; mixing 252 and 365 across conversion sites would inject a
spurious 365/252 factor into MZ and EW-ENS variance forecasts.

Implementation: AssetData gains a field `ann_factor: float = 252.0`
(additive, default keeps Track A identical). data_crypto sets 365.0. All
seven hardcoded conversion sites switch from the literal 252.0 to the
threaded value (see section 4).

EWMA and GARCH consistency: neither model contains an annualization constant.
EWMA recurses on var_d directly (lambda = 0.94 is a per-observation decay and
stays 0.94 for the 7-day week). GARCH fits daily close-close returns and is
mapped to the var_d scale by the ratio c; annualization never enters. So the
365-vs-252 question reduces to the shared rv_ann / target_rv / fcst_rv
conversion layer, which uses 365 for Track B uniformly. Document this.

### 0.3 Winsorization: unchanged (0.01 / 99.9 percentiles per asset)

True 5-min RV has no GK-negative pathology, but crypto has genuine extreme
days (Mar 2020, May 2021, Nov 2022) and possible near-zero RV on thin early
days. Keeping the identical per-asset winsorization preserves symmetric
treatment across tracks and target stability, and clips only about 1 day per
1000 at each tail. No change unless the universe report shows the clip
binding on more than 1 percent of days for some symbol (then revisit, do not
silently widen).

### 0.4 Missing-interval rule: SPEC-literal, no coverage rescaling

RV_t = sum of squared VALID 5-min returns in the UTC day, where a return is
valid only if both endpoint prices exist (see 1.3). Days with more than 10
percent of the 288 intervals missing are dropped (keep iff n_valid >= 260).
No 288/n rescaling of RV on kept days: SPEC does not ask for it, the downward
bias is bounded at 10 percent on kept days and typically far smaller, and it
avoids a non-spec knob. Document as a stated choice in universe_crypto.md.

### 0.5 Calendar gaps from dropped days

Dropped days leave holes in the DatetimeIndex exactly like Track A holidays.
The whole pipeline is positional (PLAN.md 0.2), so targets, HAR features, and
contexts span dropped days positionally. ret_cc across a dropped day spans
two calendar days, same as a Track A weekend. No special handling.

## 1. src/data_crypto.py (new file)

### 1.1 Constants

```python
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
```

LISTING_MONTH values are best-known listing months; a 404 in the first listed
month is tolerated (bump forward one month and log), so a slightly wrong
entry self-corrects instead of failing (see 1.4).

### 1.2 Functions

```python
def month_range(start_ym: str, end_ym: str) -> list[str]
    # inclusive list of "YYYY-MM"; end = last COMPLETE UTC month
    # (today.replace(day=1) minus one month), never the current month

def download_month(symbol: str, ym: str, cache_dir: Path,
                   force: bool = False) -> Path | None
    # GET {BINANCE_BASE}/{symbol}/1m/{symbol}-1m-{ym}.zip via requests,
    # timeout=(10, 120), 3 retries with backoff; write to *.part then
    # os.replace to data/binance/{symbol}/{symbol}-1m-{ym}.zip;
    # returns None on 404; raises on other HTTP errors after retries;
    # cached zips are immutable, never re-downloaded unless force

def load_month_1m_closes(zip_path: Path) -> pd.Series
    # read the single CSV inside the zip with usecols=[0, 4]
    # (open_time, close), no assumption about header (sniff: if the first
    # field of row 0 is not all digits, skip it); timestamp unit detection:
    # ts >= 1e14 means microseconds (Binance switched from ms to us in
    # 2025-01), else milliseconds; index = bar CLOSE instant =
    # open_time + 1 minute, tz-naive UTC; sort, drop duplicate index keep
    # first; validate zip with zipfile.ZipFile testzip on BadZipFile delete
    # and re-download once

def five_min_closes(closes_1m: pd.Series) -> pd.Series
    # closes_1m.resample("5min", label="right", closed="right").last()
    # NO forward fill; empty 5-min windows stay NaN

def daily_rv(closes_5m: pd.Series) -> pd.DataFrame
    # r = np.log(closes_5m).diff(); a return is VALID iff finite (both
    # endpoint closes present, so gap-bridging returns are excluded);
    # day assignment: return stamped t belongs to the UTC day of
    # (t - 1 microsecond), i.e. stamps 00:05 ... 24:00 belong to that day,
    # exactly 288 slots; output per day: rv = sum(r_valid ** 2),
    # n_valid = count of valid returns

def build_crypto_asset(symbol: str, cache_dir: Path,
                       force: bool = False) -> AssetData | None
def load_crypto_universe(cache_dir: Path, quick: bool = False,
                         force: bool = False) -> list[AssetData]
def write_crypto_universe_md(assets: list[AssetData], out_dir: Path) -> None
    # results/universe_crypto.md: symbol, months downloaded, rows, dropped
    # days (missing rule), span, winsor clip fraction
```

### 1.3 build_crypto_asset assembly (flows into the SAME pipeline)

Per symbol:

1. Download all months from LISTING_MONTH[symbol] to last complete month
   into data/binance/{symbol}/ (zips are the download cache).
2. Parse cache: per-month parquet of 1m closes at
   data/binance/parsed/{symbol}-{ym}.parquet written after first parse;
   subsequent runs read parquet, never re-open zips.
3. Concat months, five_min_closes, daily_rv.
4. Daily frame, index = tz-naive UTC day:
   - daily open/high/low/close aggregated from 1m closes (first/max/min/last;
     informational only, models do not use OHLC for crypto)
   - drop days with n_valid < MIN_VALID_INTERVALS (this also removes the
     partial first listing day and any exchange-outage days)
   - drop days with rv <= 0 or non-finite (mirrors Track A nonpositive rule)
   - `var_d` = rv winsorized at per-asset [0.01, 99.9] percentiles
   - `ret_cc` = ln(close / close.shift(1)) on the surviving daily closes
   - `rv_ann` = sqrt(365 * var_d), `log_rv` = ln(rv_ann)
   - assert np.isfinite(log_rv).all()
5. Apply MIN_ROWS = 2000 (reuse from data.py). All five symbols should pass
   (SOL is the tightest at roughly 2100 days); if a symbol drops, log it and
   proceed, same rule as Track A.
6. Return AssetData(ticker=symbol, asset_class="crypto", df=df,
   rv_estimator="rv5m", ann_factor=365.0, notes={rows, months, dropped_days,
   clip_frac, first_date, last_date}).

Downstream (compute_targets, select_test_origins, all five econ models, both
TSFM wrappers, burn-in calibration, evaluate) consumes this AssetData with
ZERO crypto-specific logic: the models only read var_d, log_rv, ret_cc and
the index, all of which exist with identical semantics.

### 1.4 Listing-date handling

Iterate months from LISTING_MONTH forward. A 404 on the first month advances
the start by one month (up to 3 advances, then raise). A 404 strictly inside
the expected range raises after retries (Binance monthly archives have no
legitimate interior holes). The final month is always the last complete UTC
month; later runs pick up new months automatically because per-month caches
are immutable and month_range extends.

## 2. Forecast store and outputs (fully separate from Track A)

- Store: results/forecasts_crypto.parquet, identical schema to
  results/forecasts.parquet (asset, model, h, origin_date, fcst_var, fcst_rv,
  recalibrated). Same upsert_store / has_cached code, different path.
- Tables: results/tables_crypto/ (pooled_loss, ratio_vs_loghar, dm_summary,
  mz_effect, dm_per_asset), losses at results/losses_crypto.parquet.
- Plots: fig_ratio_bars_crypto.png, fig_ttm_vs_chronos_crypto.png,
  fig_example_BTCUSDT.png in results/.
- Universe report: results/universe_crypto.md.
- Track A and Track B rows are NEVER pooled; separation is enforced by the
  separate store file, not by filtering.

## 3. CLI integration (src/run.py)

Add `--track {a,b,both}` with default "a" so every existing invocation is
unchanged. Main loop:

```python
for track in tracks:          # "a", "b" per the flag
    if track == "a":
        assets = load_universe(data_dir, quick, force)   # unchanged path
        store_path = out_dir / "forecasts.parquet"
        suffix = ""
    else:
        assets = load_crypto_universe(data_dir / "binance", quick, force)
        write_crypto_universe_md(assets, out_dir)
        store_path = out_dir / "forecasts_crypto.parquet"
        suffix = "_crypto"
    run_forecasts(assets, models, n_test, store_path, force)   # REUSED as is
    run_evaluation(assets, store_path, out_dir, suffix=suffix)
```

- run_forecasts is reused untouched except results_to_frame gaining the ann
  parameter (section 4); origin grids, burn-in, caching, and skip logic are
  identical because they are positional.
- --quick with track b: QUICK_SYMBOLS (BTC, ETH) and 200 test origins.
- Example plot asset for Track B falls out of the existing fallback
  (assets[0].ticker == "BTCUSDT"); pass it explicitly for clarity.
- --force-download for Track B re-downloads only the parsed caches, never
  deletes zips (zips are immutable archives).

## 4. Additive edits to existing files (complete list, defaults preserve A)

1. src/data.py: AssetData gains `ann_factor: float = 252.0`; build_asset
   leaves it at default. compute_targets uses `asset.ann_factor` instead of
   literal 252.0 for target_rv. rv_ann in build_asset keeps literal 252.0
   (Track A only; crypto builds its own df).
2. src/models_econ.py: forecast_log_har and forecast_ar1_log use
   `fcst_rv**2 / asset.ann_factor`.
3. src/models_tsfm.py: _emit_results uses `asset.ann_factor`.
4. src/run.py: results_to_frame(results, ann=252.0); run_forecasts passes
   asset.ann_factor. run_evaluation(assets, store_path, out_dir,
   suffix="", ann=252.0) forwards to evaluate.
5. src/evaluate.py: mz_recalibrate(..., ann=252.0), add_ensemble(...,
   ann=252.0), add_mz_rows(..., ann=252.0) replace their literal 252.0;
   build_tables(..., suffix="") writes f"tables{suffix}/" and
   f"losses{suffix}.parquet"; make_plots(..., suffix="") appends suffix to
   figure filenames.

Nothing else in existing files changes. With all-default arguments every
Track A code path is byte-identical, so the concurrent Track A run and its
cached results remain valid.

## 5. Top likely bugs and prescribed handling

1. Binance timestamp unit change (the big one). Monthly kline files from
   2025-01 onward use MICROSECOND timestamps (16 digits); earlier files use
   milliseconds (13 digits). Parsing everything as ms silently places 2025+
   data around year 56000 and resample produces garbage or empty days.
   Handling: per-file unit detection (ts >= 1e14 means us), plus a post-parse
   assert that the month's min/max dates fall inside the expected ym.
2. Header row inconsistency. Some newer CSVs include a header line, older
   ones do not. Handling: sniff the first field of row 0; skip if
   non-numeric. Assert close prices are positive floats.
3. Gap-bridging returns inflating RV. A return computed across a data outage
   (for example 30 minutes) lands in one 5-min slot and overstates that
   day's RV. Handling: no forward fill anywhere; a 5-min return is valid only
   when both endpoint 5-min closes exist; invalid returns are excluded from
   the sum and counted as missing.
4. Midnight off-by-one in day assignment. Assigning the return stamped
   exactly 00:00 to the wrong day yields 287 or 289 slots. Handling: the
   (t - 1 microsecond) day rule in daily_rv; unit test that a fully synthetic
   gap-free day yields exactly 288 valid returns and the correct RV.
5. 252/365 leakage. Missing one conversion site makes run.py write
   fcst_rv = sqrt(365 * fcst_var) while evaluate converts back with
   fcst_rv**2 / 252, silently scaling MZ and EW-ENS variance forecasts by
   365/252 (QLIKE is NOT invariant to that). Handling: the single ann
   parameter of section 4; unit test the round trip fcst_var -> fcst_rv ->
   fcst_var through results_to_frame plus add_ensemble on a toy Track B
   store.
6. Partial trailing month / current month. The current month's zip does not
   exist (404) and must not be treated as an error or a data hole. Handling:
   month_range ends at the last complete UTC month by construction.
7. Early thin months (BNB 2017, XRP 2018). Duplicate or out-of-order
   timestamps and many missing 1m bars. Handling: sort plus duplicate drop in
   load_month_1m_closes; the n_valid >= 260 rule drops thin days; report
   dropped-day counts per symbol in universe_crypto.md and eyeball them.
8. Corrupt or truncated zip from an interrupted download. Handling: download
   to *.part then atomic rename; zipfile integrity check on read; delete and
   re-download once on BadZipFile.
9. Timezone drift. Any tz_localize to local time or mixed tz-aware/naive
   indexes shifts day boundaries off UTC. Handling: everything tz-naive UTC
   from parse onward; assert index.tz is None; never call tz_convert.
10. Memory blowup parsing 1m klines (about 4.5M rows per symbol). Handling:
    usecols=[0, 4] only, float64 close, per-month parsing with immediate
    5-min downsample before concat if needed (a plain concat of closes is
    about 36 MB per symbol, acceptable; do not read all 12 columns).

## 6. Tests (extend tests/ without touching existing ones)

1. daily_rv on a synthetic 2-day 1m series: exact 288 returns per day,
   hand-computed RV, midnight boundary correct.
2. Missing-interval rule: knock out 29 intervals (drop) vs 28 (keep),
   gap-bridging return excluded from both RV and n_valid.
3. Timestamp unit detection: same synthetic month encoded in ms and in us
   parses to identical series; header and no-header variants identical.
4. ann round trip: toy store through results_to_frame(ann=365),
   add_ensemble(ann=365), add_mz_rows(ann=365) preserves fcst_var identity
   and MZ/ENS variance scale.
5. Crypto AssetData contract: build a tiny asset from fixture data and run
   compute_targets, forecast_har, forecast_ewma on it end to end (no
   network, no torch).

## 7. Build order

1. data_crypto.py download plus parse plus daily_rv, with tests 1-3 on
   synthetic fixtures (no network in tests; downloads exercised manually
   once per symbol).
2. Section 4 additive parameters plus test 4; re-run existing Track A unit
   tests to confirm no behavior change.
3. run.py --track wiring, --quick track b smoke (BTC, ETH, 200 origins,
   econ models first, then TSFMs).
4. Full Track B run (5 symbols); tables_crypto plus plots plus
   universe_crypto.md; README Track B findings section.
