# Universe and estimator choices

| ticker | class | rows | estimator | zero open-close frac | GK fallback frac | span |
|---|---|---|---|---|---|---|
| SPY | equity | 5413 | gk | 0.0063 | 0.0 | 2005-01-03 to 2026-07-10 |
| QQQ | equity | 5413 | gk | 0.0074 | 0.0 | 2005-01-03 to 2026-07-10 |
| DIA | equity | 5413 | gk | 0.0059 | 0.0 | 2005-01-03 to 2026-07-10 |
| IWM | equity | 5413 | gk | 0.005 | 0.0 | 2005-01-03 to 2026-07-10 |
| XLF | equity | 5413 | gk | 0.0207 | 0.0 | 2005-01-03 to 2026-07-10 |
| XLK | equity | 5413 | gk | 0.0126 | 0.0 | 2005-01-03 to 2026-07-10 |
| XLE | equity | 5413 | gk | 0.0059 | 0.0 | 2005-01-03 to 2026-07-10 |
| XLV | equity | 5413 | gk | 0.0124 | 0.0 | 2005-01-03 to 2026-07-10 |
| XLI | equity | 5413 | gk | 0.0116 | 0.0 | 2005-01-03 to 2026-07-10 |
| XLP | equity | 5413 | gk | 0.015 | 0.0 | 2005-01-03 to 2026-07-10 |
| XLU | equity | 5413 | gk | 0.012 | 0.0 | 2005-01-03 to 2026-07-10 |
| EEM | equity | 5413 | gk | 0.0124 | 0.0 | 2005-01-03 to 2026-07-10 |
| EFA | equity | 5413 | gk | 0.0131 | 0.0 | 2005-01-03 to 2026-07-10 |
| EURUSD=X | fx | 5444 | parkinson | 0.241 | 0.0 | 2005-01-03 to 2026-07-10 |
| GBPUSD=X | fx | 5476 | gk | 0.0875 | 0.0 | 2005-01-03 to 2026-07-10 |
| USDJPY=X | fx | 5319 | parkinson | 0.1918 | 0.0 | 2005-01-03 to 2026-07-10 |
| AUDUSD=X | fx | 5087 | parkinson | 0.1944 | 0.0 | 2006-05-16 to 2026-07-10 |
| USDCAD=X | fx | 5483 | parkinson | 0.2311 | 0.0 | 2005-01-03 to 2026-07-10 |
| USDCHF=X | fx | 5481 | parkinson | 0.2253 | 0.0 | 2005-01-03 to 2026-07-11 |
| NZDUSD=X | fx | 5276 | parkinson | 0.2003 | 0.0 | 2005-01-03 to 2026-07-10 |
| ES=F | futures | 5414 | gk | 0.0096 | 0.0 | 2005-01-03 to 2026-07-10 |
| NQ=F | futures | 5410 | gk | 0.003 | 0.0 | 2005-01-03 to 2026-07-10 |
| YM=F | futures | 5387 | gk | 0.0046 | 0.0 | 2005-01-03 to 2026-07-10 |
| GC=F | futures | 4674 | gk | 0.0385 | 0.0 | 2005-01-07 to 2026-07-10 |
| SI=F | futures | 4023 | gk | 0.0497 | 0.0 | 2005-01-03 to 2026-07-10 |
| CL=F | futures | 5410 | gk | 0.0037 | 0.0 | 2005-01-03 to 2026-07-10 |
| ZN=F | futures | 5391 | gk | 0.0169 | 0.0 | 2005-01-03 to 2026-07-10 |
| ZB=F | futures | 5401 | gk | 0.018 | 0.0 | 2005-01-03 to 2026-07-10 |
| HG=F | futures | 5128 | gk | 0.032 | 0.0 | 2005-01-03 to 2026-07-10 |
| NG=F | futures | 5408 | gk | 0.0033 | 0.0 | 2005-01-03 to 2026-07-10 |

Rules: GK default; per-day Parkinson fallback where GK <= 0; FX assets
switch wholesale to Parkinson when > 10 percent of days have near-zero
open-close moves; H == L rows dropped; var_d winsorized at the 0.01 and
99.9 percentiles per asset.
