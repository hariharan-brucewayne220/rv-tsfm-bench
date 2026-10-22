# Validation of README claims (2026-10-22)

Independent check of this repo's findings against its own artifacts, its code,
and the full text of arXiv:2607.05291v1.

## What was re-run

Fresh data was downloaded (Yahoo for Track A, Binance 1-minute archives for
Track B) and pinned to the original windows with the new `--end-date` option
(2026-07-10 for Track A, 2026-06-30 for Track B).

| Check | Track A (30 assets) | Track B (5 symbols) |
|---|---|---|
| Re-score of cached forecasts vs committed tables | Matches exactly on 28 assets (DM statistic difference 0.000); all-asset ratios within 0.012 | Every committed table matches to 0.0 |
| Econometric forecasts re-run from scratch | HAR, Log-HAR, AR1, EWMA identical to about 1e-14 on 28 assets; GARCH within 0.34% | Identical to about 1e-14; GARCH within 0.02% |
| TSFM forecasts re-run from scratch (Chronos-Bolt tiny/small, TTM 512-96-ft-r2.1) | Median relative difference 2e-8, 99th percentile 5e-7 on 28 assets | Chronos within 0.7%, TTM within 1e-6 |
| All models re-run, re-scored | Per-asset ratios within 0.001 of the cached-forecast re-score on 28 assets; mean ratios equal to 4 decimals | Committed tables reproduced within 3e-5 |

The two Track A exceptions come from data revisions, not code:
- GC=F: the original download was missing about 33 trading days inside the
  test window, which Yahoo has since filled. Re-scoring the *cached* GC=F
  forecasts on the new data is therefore not valid. That alone explains why
  the all-asset Track A tables differ from the committed ones (by up to 0.034
  in the full re-run). `results_rerun/` is the self-consistent reference.
- USDCHF=X: Yahoo now has one fewer row, which shifts its refit schedule by a
  day.

Smaller notes:
- The GARCH differences match an `arch` optimizer version change.
- 296 of 126,000 cached Chronos rows differ by more than 1e-4 (at most 4%).
  All of them fall in one Aug to Sep 2025 block at h=5 and h=22, on the six
  `--quick` tickers, which suggests those cached forecasts came from an
  earlier run. The effect on tables is below 0.001.
- Loading the TTM checkpoint gives exactly 855,100 parameters.
- Chronos-Bolt's returned mean equals its 0.5 quantile bit-for-bit.

Other checks:
- Unit tests: 16 of 16 pass. That includes a fix to the timestamp test,
  which broke under pandas 3, and new MZ tests.
- Cached forecasts: EW-ENS equals 0.5 * (TTM + LogHAR) exactly, units are
  consistent and there are no NaNs.
- Every README number was checked against `results/tables*`, and each claim
  about the paper against its full text.

## README numbers vs this repo's own tables

All headline numbers match the tables (Track A TTM 0.998, Chronos-S 1.000,
Track B TTM 0.876 and Chronos-S 0.922, the DM shares, the MZ before/after pairs,
and EW-ENS 0.984 / 0.903, 1.039 / 1.011, 1.062 / 1.106).

One small mismatch: finding 4 says the TTM checkpoint switch improved QLIKE "by
9 to 22 percent". The tables give reductions of 8.1 / 14.6 / 19.0 percent
(Track A) and 16.9 / 13.1 / 8.0 percent (Track B), so the range is about
8 to 19 percent.

## Claims about the paper

| README statement | Verdict | Paper says |
|---|---|---|
| Headline target is point-in-time; h-day average is the robustness arm (Tab. 15) | True | Sec. 4.2.2, App. B |
| Same ranking under both targets, "a few mid-table models reorder" | Understated | Top of the ranking holds, but some models move a lot (Toto h=22 1.157 to 4.070, HARQ h=5 3.518 to 8.137) |
| TSFMs get raw vol levels, conditional mean as point forecast | True | Sec. 4.2.1, 4.2.2 |
| Context 1000; 512 at or near optimal for Chronos-Bolt | Partly | Tab. 14 numbers support it (within 0.002), but the text says Chronos-Bolt peaks at 128 to 256 |
| TTM is branch 512-96-ft-r2.1, 855,100 parameters | True | App. A footnote |
| Econometric models are iterated multistep, re-estimated daily | Partly | Daily re-estimation is true, but HAR-J/RS/Q are direct |
| Forecasts bounded to in-sample support, binds 0.05 percent | True | The bounds use the full sample, test period included |
| MCS and Giacomini-Rossi tests | True | Sec. 4.4 |
| h=1 after MZ: TimesFM 0.190, Chronos-B 0.192, Log-HAR 0.195, TTM 0.192 to 0.198 | True | Tab. 11 |
| Paper calls Chronos-Bolt output a mean head | True, and the paper contradicts itself | Sec. 4.2.1 says mean head; Tab. 2 says direct quantiles |
| Chronos-Bolt "mean" is the 0.5 quantile | True | chronos-forecasting 2.3.2 `chronos_bolt.py`: "NOTE: the median is returned as the mean here" |

## Paper vs this repo, per-asset mean QLIKE ratio vs Log-HAR

The paper's h-day-average arm (Tab. 15) is the like-for-like comparison. EW-ENS
is only reported under the point-in-time target (Tab. 12).

| Model | Paper (Tab. 15) h=1/5/22 | Track A | Track B |
|---|---|---|---|
| TTM | 0.983 / 0.989 / 0.976 | 0.998 / 1.197 / 1.464 | 0.876 / 1.136 / 1.579 |
| Chronos-Bolt-S | 1.109 / 1.171 / 1.341 | 1.000 / 1.288 / 1.617 | 0.922 / 1.278 / 1.831 |
| HAR | 0.999 / 1.003 / 1.045 | 0.980 / 1.069 / 1.236 | 1.084 / 1.154 / 1.123 |
| EW-ENS (paper: Tab. 12, point-in-time) | 0.977 / 0.984 / 0.982 | 0.984 / 1.039 / 1.062 | 0.903 / 1.011 / 1.106 |

Replicated: TTM is the strongest TSFM, and at h=1 the ensemble beats Log-HAR.
Not replicated: TTM beating Log-HAR at h=5 and h=22. The repo is far worse there.

## Methodology issues that change the interpretation

1. **The MZ arm compares against raw Log-HAR, and the MZ step hurts every model.**
   `mz_effect` divides each model's recalibrated loss by raw Log-HAR loss.
   Log-HAR's own recalibrated ratio is 1.025 / 1.110 / 1.381 (Track A) and
   1.031 / 1.115 / 1.206 (Track B). The rolling 250-day log-log fit makes even
   the benchmark worse. The paper uses an expanding-window fit in levels, which
   improves Log-HAR (0.546 to 0.506 at h=22), and it compares against
   recalibrated Log-HAR.
   **Measured like-for-like result.** These are per-asset mean QLIKE ratios,
   each recalibrated model vs Log-HAR recalibrated the same way, on common
   origins. DM columns give the percent of assets significantly better or
   worse at 5 percent. Source: the full re-run,
   `results_rerun/tables{,_crypto}/mz_like_for_like.md`. The cached-forecast
   re-score in `results_rescore/` agrees within about 0.01.

   Repo scheme (rolling 250-day, log-log):

   | Model | Track A ratio h=1/5/22 | A better / worse | Track B ratio h=1/5/22 | B better / worse |
   |---|---|---|---|---|
   | TTM | 1.003 / 0.999 / 0.976 | 0/3/0 , 7/0/0 | 0.888 / 1.006 / 1.012 | 40/20/20 , 0/0/0 |
   | Chronos-Bolt-S | 0.986 / 1.011 / 0.991 | 7/0/0 , 0/7/3 | 0.902 / 1.003 / 1.013 | 40/0/0 , 0/0/0 |
   | EW-ENS | 0.992 / 0.988 / 0.977 | 10/7/0 , 0/0/0 | 0.909 / 0.984 / 0.997 | 60/40/20 , 0/0/0 |
   | HAR | 1.004 / 1.040 / 1.096 | 7/0/0 , 10/27/13 | 1.088 / 1.180 / 1.269 | 0/0/0 , 40/40/20 |
   | Log-HAR+MZ vs raw Log-HAR | 1.026 / 1.116 / 1.386 | | 1.031 / 1.115 / 1.206 | |

   Paper scheme (expanding window, levels, 252-day warm-up):

   | Model | Track A ratio h=1/5/22 | Track B ratio h=1/5/22 |
   |---|---|---|
   | TTM | 0.986 / 0.963 / 0.933 | 0.908 / 0.973 / 0.954 |
   | Chronos-Bolt-S | 0.965 / 0.974 / 0.933 | 0.909 / 0.977 / 0.964 |
   | EW-ENS | 0.982 / 0.967 / 0.956 | 0.929 / 0.968 / 0.967 |
   | Log-HAR+MZL vs raw Log-HAR | 1.057 / 1.165 / 1.556 | 1.020 / 1.057 / 1.053 |

   What this changes:
   - README finding 2 ("econometrics wins decisively at h=5 and h=22")
     holds only for raw forecasts. Under equal recalibration the gap closes
     to parity (repo scheme), or reverses in the TSFMs' favor (paper scheme,
     0.93 to 0.97 at h=22 on both tracks). The long-horizon TSFM losses are
     mostly a scaling problem, not missing dynamics. That matches the
     paper's "genuine gain at the monthly horizon".
   - README finding 1's "Track A TTM edge is calibration" is not supported.
     TTM had no raw edge to explain (0.996 in the full re-run), and it stays at
     parity after recalibration.
   - The Track B h=1 edges are real. TTM is 0.888 like-for-like, better on
     2 of 5 symbols and worse on none.
   - Caveat: both MZ schemes make Log-HAR itself worse than raw Log-HAR here
     (the last row of each table). The paper's scheme improved Log-HAR in the
     paper (0.546 to 0.506 at h=22). So after recalibration TSFMs beat a
     degraded benchmark, and at h=5 and h=22 raw Log-HAR remains the best
     forecast you can actually use in this setup. A likely reason (an
     inference, not tested): this repo's Log-HAR is already log-space and
     lognormal-corrected, so an extra affine fit adds only estimation noise,
     and a level-space OLS fit is dominated by volatility spikes.

2. **Evaluation targets are winsorized with full-sample percentiles.**
   `var_d` is clipped at the 0.01 and 99.9 percentiles over the whole history
   (`src/data.py:165`, `src/data_crypto.py:237`), and the targets are built from
   the clipped `var_d`. That trims test-period spikes using future information.
   It is small (about 0.15 percent of days), but README says only that the
   *input* series is winsorized.

3. **The README overstates how close the TSFM wrapper is to the paper's MZ arm.**
   It says the wrapper is "functionally close to the paper's MZ recalibrated
   arm". In fact `lognormal_calibrate` adds 0.5 * var(error) after demeaning,
   so it neither removes mean bias nor rescales the slope. It is not an affine
   MZ correction.

## How to reproduce

```
python -m src.run --track a --end-date 2026-07-10 --out results_rerun
python -m src.run --track b --end-date 2026-06-30 --out results_rerun
```

On 4 CPU cores, the Track A TSFM forecasts take about 12 minutes and
evaluation about 5. Note that `granite-tsfm` 0.3.11 pins `torch<2.12`; the
run here used torch 2.14 with `--no-deps` and worked. Downloading the weights
needs the Hugging Face Xet/CDN hosts as well as huggingface.co.
