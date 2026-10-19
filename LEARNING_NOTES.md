# Learning notes

A study guide to every modelling choice (`# DECISION:`) and known weakness (`# PITFALL:`) in the code.
The two lists below were extracted directly from the code comments, so the wording matches the code.
If you change a comment, update it here too. Each entry names the file and function: open it next to
this list and read the code around the comment.

Suggested order: `src/data.py` -> `src/portfolio.py` -> `src/var_models.py` (historical, EWMA, GARCH,
ES) -> `src/backtest.py` -> `run_all.py`, then the tests in `tests/`, which show the formulas on tiny
hand-computed examples.

## Decisions: what was chosen and why

### `config.py`

1. **`module level`**: equal weights, rebalanced daily. This keeps the portfolio return a simple weighted sum of daily stock returns with constant weights.

### `src/data.py`

2. **`download_prices()`**: only completed trading days. If no end date is given we use today as the (exclusive) end, so the snapshot stops at yesterday. A download during trading hours would otherwise contain an intraday price for today, and the last "daily" return would be incomplete.
3. **`download_prices()`**: auto_adjust=True makes yfinance return "Close" prices that are already adjusted for dividends and splits. Using adjusted prices means a price return also includes the dividend, i.e. it is a total return. Without adjustment every ex-dividend day would look like a loss.
4. **`load_prices()`**: results are computed from a saved CSV snapshot, not from a live download, so re-running the project gives identical numbers. Yahoo data can be revised over time (e.g. adjustment factors change after new dividends), so a live download is not reproducible.
5. **`align_prices()`**: inner join, no forward-fill. Forward-filling a missing price creates an artificial 0% return on that day and then puts two days of movement into the next return. That understates volatility and distorts the VaR backtest. Dropping the date is cleaner.
6. **`compute_returns()`**: simple returns, not log returns. The portfolio's simple return is exactly the weighted sum of the stocks' simple returns, while log returns do not aggregate across assets. VaR is also quoted as a % loss of portfolio value, which is a simple return.

### `src/portfolio.py`

7. **`portfolio_returns()`**: constant weights applied every day means daily rebalancing back to 20% each. A buy-and-hold portfolio would let weights drift with prices, and then the formula would need yesterday's actual weights.

### `src/var_models.py`

8. **`historical_var()`**: historical simulation makes no distributional assumption. It uses the empirical distribution of the last `window` returns, so fat tails and skew that appeared in the window are included automatically.
9. **`historical_var()`**: quantile method = pandas/numpy default "linear" interpolation. With 250 observations the 1% quantile sits at position 0.01 * 249 = 2.49 in the sorted sample, so the result interpolates between the 3rd and 4th smallest returns. Alternatives ("lower", "nearest", or taking the k-th worst loss) differ slightly, mainly for the 250-day window at 99%, where only ~2.5 observations lie in the tail.
10. **`historical_var()`**: shift(1) is what prevents look-ahead bias. rolling_q at row t uses r_{t-window+1}..r_t. Shifting by one day means the VaR for day t is built from r_{t-window}..r_{t-1}, i.e. only information known at the close of day t-1.
11. **`ewma_variance()`**: initialise with the mean of squared returns over the burn-in window, not the sample variance. The model assumes a zero mean, and the mean of r^2 is the variance estimate under that same assumption. The sample variance would subtract the sample mean, which the recursion itself never does. With lambda = 0.94 the starting value is forgotten quickly anyway (weight 0.94^250 ~ 2e-7 after the burn-in length).
12. **`normal_var()`**: zero mean. The average daily return (~0.05%) is tiny compared with daily volatility (~1.2%), and estimating the mean adds noise. Zero mean is also the RiskMetrics convention and slightly conservative when the true mean is positive.
13. **`student_t_var()`**: arch's Student-t is standardised to unit variance, so the quantile of a standard t(nu) is multiplied by sqrt((nu - 2) / nu). Without this factor VaR would be overstated (a t(5) has variance 5/3).
14. **`garch_forecasts()`**: estimate in percent returns (x = 100 r). The arch optimiser is tuned for numbers of order 1; with decimal returns omega is ~1e-6 and the optimiser can stop early or fail. Results are converted back.
15. **`garch_forecasts()`**: re-estimate every 20 trading days (about one month), not daily. Daily refits cost 20x more computing time and barely change the forecasts, because parameters estimated on ~1000 days move very little from one day to the next. The conditional variance itself still reacts every day through the recursion. The cost is that parameters can be up to 19 days old; refitting much less often (e.g. yearly) would let them drift away from the current regime.
16. **`garch_forecasts()`**: the first fit needs 500 days (expanding until 1000). This is the same burn-in as the longest historical window, so GARCH does not move the common backtest start. 500 daily returns are enough for three GARCH parameters (four with nu), though early estimates are noisier.
17. **`garch_forecasts()`**: rolling 1000-day window (expanding before that). A rolling window lets the parameters adapt to regime changes, while 1000 days (~4 years) keep enough tail events for stable estimates.
18. **`garch_forecasts()`**: if the optimiser fails, keep the previous parameters and simply continue the variance recursion. The failure is logged, never hidden.

### `src/backtest.py`

19. **`kupiec_pof()`**: scipy's xlogy(a, b) computes a * ln(b) but returns 0 when a = 0. That is exactly the convention 0 * ln(0) = 0 needed for the zero-breach case (p_hat = 0 would otherwise give ln(0) = -inf) and for the all-breach case. With x = 0 the statistic becomes -2 n ln(1 - p).
20. **`christoffersen_conditional_coverage()`**: LR_cc is computed as the sum of the Kupiec and independence statistics. Strictly, Kupiec uses all T days and the independence test T-1 transitions (the first day has no "yesterday"); with ~2,500 days the difference is negligible and this is the common textbook form.
21. **`basel_traffic_light()`**: only complete windows are counted (the first window ends on day 250 of the backtest). Partial windows would mechanically look green.
22. **`volatility_regime()`**: a simple rule fixed BEFORE looking at the backtest results: 60-day rolling volatility (about one quarter) against its expanding long-run median. Both use data up to t-1 only, so the regime label is known before the day's return. It is a descriptive split, not tuned: no other windows or thresholds were tried.

### `run_all.py`

23. **`build_forecasts()`**: one common backtest period for all models. It starts on the first day on which EVERY model has a forecast, so differences between models cannot come from testing them on different market periods.

## Pitfalls: known weaknesses

### `src/data.py`

1. **`align_prices()`**: dropping a date means the next return covers two trading days for every stock. With few dropped days this effect is negligible, but it should be checked (the count is logged below).
2. **`replace_return()`**: the Siemens Energy spin-off day (2020-09-28) in the Yahoo data shows SIE.DE +9.22% while the DAX rose 3.22%. Yahoo has no spin-off record and even its unadjusted close jumps, so the pre-spin-off history looks inconsistently rescaled. We keep the raw value in the main results and only measure its impact in a sensitivity run.

### `src/portfolio.py`

3. **`portfolio_returns()`**: daily rebalancing ignores transaction costs. That is fine for measuring risk but would not be a realistic trading strategy.

### `src/var_models.py`

4. **`historical_var()`**: at 99% with a 250-day window the estimate rests on 2-3 data points, so it is noisy and jumps when an extreme day enters or leaves the window.
5. **`historical_var()`**: slow reaction after a shock. All days in the window have equal weight, so a calm window reacts late to a new crisis, and after a crash VaR stays high until the crash days drop out (250 or 500 days later), the "ghost effect".
6. **`historical_es()`**: at 99% with 250 days the "tail" holds only the 2-3 worst returns, so historical ES is even noisier than historical VaR.
7. **`normal_var()`**: the normal distribution has thin tails. Daily equity returns are fat-tailed, so normal VaR at 99% tends to understate the loss quantile even if the volatility forecast is good. The effect is smaller at 95%, where the normal and empirical quantiles are closer.
8. **`student_t_var()`**: the t distribution is symmetric. Equity losses are usually somewhat larger than gains (negative skew), which it cannot capture.
9. **`garch_forecasts()`**: GARCH parameters are estimated by maximum likelihood on a finite window. alpha + beta is often close to 1 (very persistent volatility), and the estimates can jump between refits when a crisis enters or leaves the window. The parameter chart shows this. At alpha + beta = 1 (the optimiser's upper bound) there is no finite long-run variance and the model behaves like EWMA (IGARCH).

### `src/backtest.py`

10. **`kupiec_pof()`**: the chi-squared distribution is only an asymptotic approximation. With ~2,500 observations it is fine, but on a single 250-day window at 99% (2.5 expected breaches) the test has very little power: it often cannot tell a 1% model from a 2% model.
11. **`christoffersen_independence()`**: this test only looks at breaches on CONSECUTIVE days. Breaches that cluster within a few weeks but not on back-to-back days (typical in a crisis) are largely invisible to it, so "not rejected" does not prove that breaches are independent. The per-year table and traffic light show clustering more directly.
12. **`basel_traffic_light()`**: the Basel zones are calibrated for 99% VaR on 250 days. For a correct model, 10+ breaches (red) has a probability of only ~0.03%, but 5+ breaches (yellow or worse) still happens ~11% of the time, so yellow alone is weak evidence of a bad model (binomial, n = 250, p = 1%).
13. **`es_check()`**: ES is much harder to backtest than VaR. A VaR forecast can be checked day by day (breach yes/no), but ES is an average over the tail, so each day gives no clear verdict. (1) Only breach days carry information, about 25 days at 99% over ten years, so the average is very noisy. (2) The check depends on the VaR too: a model with too many breaches includes milder losses and can look better on ES. (3) ES is not "elicitable" on its own: there is no simple loss function that a correct ES forecast minimises (Gneiting 2011), which is why proper ES tests (Acerbi-Szekely 2014, or joint VaR-ES scoring) are more involved. This simple ratio is a plausibility check, not a statistical test.
14. **`volatility_regime()`**: an expanding median moves slowly, so after a long calm (or turbulent) period most days fall into one regime. The split is not 50/50 and with ~1,000 turbulent days a 99% model should only have ~10 breaches there, so regime breach rates are very noisy.

## Data issue: Siemens Energy spin-off (28 Sep 2020)

Found by the data quality check (`src/data.py`, `inspect_window()`); documented in
`replace_return()` and in the README limitations.

- **What the data shows:** SIE.DE +9.22% on 28 Sep 2020, the first trading day after the Siemens
  Energy spin-off. That day the DAX rose 3.22% and the other four stocks rose between 1.06% and 4.62%.
- **Why it looks wrong:** Yahoo lists no spin-off action for Siemens (only the 2020 and 2021
  dividends), and even its unadjusted close jumps by about 9%. The pre-spin-off history seems
  to be rescaled inconsistently. A correct total-return series should move roughly with the market.
- **What we did:** nothing in the main results (raw data kept, nothing silently fixed). A
  sensitivity run replaces that single return with the DAX return (+3.22%), which lowers the
  portfolio return that day from 4.30% to 3.10%, and re-runs every model and backtest.
- **Result of the sensitivity run:** only EWMA 99% changes. It gains one breach (57 -> 58, on
  15 Oct 2020) because without the inflated +9.22% day its variance is slightly lower two weeks later.
  The Kupiec, independence and conditional coverage decisions and the latest Basel zone stay the same.
  65 rolling windows change zone; over the whole period the share of red windows goes from 9.1% to
  10.3%. All other models are unaffected.
- **Lesson:** an error that is a *gain* never causes a breach directly, and historical VaR ignores it
  (it only looks at the loss tail). Volatility models square returns, so gains raise their variance
  forecasts too. Data errors affect different models in different ways.

## Interpreting the results: things to keep in mind

- A model that has *too few* breaches is more suspicious than one with too many. Look-ahead bias
  (using r_t to forecast VaR_t) makes a model look almost perfect. `tests/test_var_models.py`
  checks this for every model by changing r_t and confirming VaR_t does not move.
- Ten model x level combinations are tested at 5% significance each, so about one rejection
  would be expected by chance even if every model were correct (multiple testing).
- Passing a test means "no evidence against the model", not "the model is right". The Kupiec test in
  particular has low power on short samples.
