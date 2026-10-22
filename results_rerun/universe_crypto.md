# Track B universe: Binance 5-min realized volatility

| symbol | rows | months | dropped days (missing rule) | dropped rv<=0 | winsor clip frac | span |
|---|---|---|---|---|---|---|
| BTCUSDT | 3221 | 107 | 19 | 0 | 0.00155 | 2017-08-18 to 2026-06-30 |
| ETHUSDT | 3221 | 107 | 19 | 0 | 0.00155 | 2017-08-18 to 2026-06-30 |
| BNBUSDT | 3141 | 104 | 18 | 0 | 0.00159 | 2017-11-07 to 2026-06-30 |
| XRPUSDT | 2964 | 98 | 16 | 0 | 0.00135 | 2018-05-05 to 2026-06-30 |
| SOLUSDT | 2145 | 71 | 5 | 0 | 0.00186 | 2020-08-12 to 2026-06-30 |

Rules and stated choices:
- RV_t = sum of squared valid 5-min log returns over the UTC day
  (288 slots); a return is valid only if both endpoint 5-min closes
  exist, so gap-bridging returns are excluded.
- Days with n_valid < 260 (> 10 percent missing) are dropped; kept
  days are NOT rescaled by 288/n (stated choice, bias <= 10 percent
  and typically far smaller).
- Annualization uses 365. var_d winsorized at the 0.01 and 99.9
  percentiles per asset, same as Track A.
- HAR lags and horizons are in observation units (days of a 7-day
  week for crypto): lags 5/22, h = 1, 5, 22, identical to Track A.
