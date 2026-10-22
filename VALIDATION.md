# Validation of README claims (2026-10-22)

Independent check of this repo's findings against its own artifacts, its code,
and the full text of arXiv:2607.05291v1.

## What could and could not be re-run

The validation container's network policy blocked huggingface.co,
data.binance.vision, query1/query2.finance.yahoo.com and arxiv.org, so neither
the model weights nor the market data could be downloaded. No forecasts were
regenerated and no losses were recomputed. `results/forecasts*.parquet` holds
forecasts only (no realized targets), so the tables cannot be re-scored
without the raw data either.

What was done:

- Unit tests: 11 of 12 pass. The 12th (`test_timestamp_unit_and_header_detection`)
  failed only because pandas 3 builds `date_range` at microsecond resolution, so
  `.asi8` is no longer nanoseconds. Fixed in the test with `.as_unit("ns")`;
  all 12 now pass. The loader code was fine.
- Cached forecasts: EW-ENS equals 0.5 * (TTM + LogHAR) exactly on both tracks;
  `fcst_var` and `fcst_rv` are unit-consistent (252 / 365); no NaNs; 750
  origins per asset, model and horizon (30 assets Track A, 5 symbols Track B).
- Every README number was checked against `results/tables*`.
- The paper text was read in full and each claim about it checked.

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
   Dividing by recalibrated Log-HAR instead (approximate, a ratio of
   cross-asset means of per-asset ratios):

   | After MZ, vs MZ Log-HAR | Track A h=1/5/22 | Track B h=1/5/22 |
   |---|---|---|
   | TTM | 1.002 / 1.000 / 0.975 | 0.886 / 1.005 / 1.010 |
   | Chronos-Bolt-S | 0.985 / 1.015 / 0.998 | 0.901 / 1.002 / 1.013 |
   | EW-ENS | 0.992 / 0.989 / 0.978 | 0.908 / 0.983 / 0.996 |

   Under a like-for-like comparison, the long-horizon losses that README
   finding 2 calls decisive mostly disappear. On Track A, TTM's h=22 result
   (0.975) is the paper's monthly-horizon information gain. README finding 1's
   "Track A TTM edge is calibration" is not supported: TTM had no raw edge
   (0.998), and after recalibration it is still at parity. The Track B h=1
   edges survive, and get stronger.
   This should be confirmed by rerunning with Log-HAR+MZ as the reference.

   **Tooling added for that re-run.** The evaluate stage now builds two MZ
   arms, each compared against Log-HAR recalibrated the same way:
   - `+MZ`: the original method, a rolling 250-pair log-log fit.
   - `+MZL`: the paper's Sec. 7 method, an expanding-window affine fit on the
     volatility level with a 252-pair warm-up. It is floored at half the
     smallest realized target in the fit window, so QLIKE stays defined.

   Outputs:
   - `results/tables{,_crypto}/mz_like_for_like.md`, with columns:
     - `ratio_raw`: raw model vs raw Log-HAR.
     - `ratio_mz`: recalibrated model vs recalibrated Log-HAR.
     - `ref_mz_vs_raw`: whether the scheme helps Log-HAR itself.
     - DM test shares of assets significantly better or worse vs the
       recalibrated Log-HAR.
   - `mz_like_for_like_per_asset.parquet`: the per-asset rows behind it.

   The old `mz_effect` table is unchanged. To fill these tables, run
   `python -m src.run --stage evaluate` and
   `python -m src.run --track b --stage evaluate`. Both need the raw data,
   since no realized targets are cached. Tests: `tests/test_mz.py`. They
   cover no look-ahead under both schemes, exact recovery of an affine bias,
   and a scale-biased copy of Log-HAR scoring exactly 1.0 like-for-like.

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

## To finish the re-run

Allow these hosts in the environment's network settings: huggingface.co (plus
its LFS/xet download hosts), data.binance.vision, query1.finance.yahoo.com,
query2.finance.yahoo.com, fc.yahoo.com. Then run `python -m src.run` and
`python -m src.run --track b`. Note that `granite-tsfm` 0.3.11 pins
`torch<2.12`.
