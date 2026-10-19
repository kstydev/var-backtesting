"""1-day Value-at-Risk models.

Convention used throughout the project:
    VaR_t is a POSITIVE number = the loss (as a fraction of portfolio value)
    that is exceeded with probability (1 - confidence) on day t.
    VaR_t may only use returns up to and including day t-1.
    A breach on day t means r_t < -VaR_t.
"""
import logging

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t as student_t

logger = logging.getLogger(__name__)


def level_label(confidence):
    """0.99 -> '99', 0.975 -> '97.5' (used in column names)."""
    return f"{confidence * 100:g}"


# --- Historical simulation -------------------------------------------------

def historical_var(returns, window, confidence):
    """Historical simulation VaR with a rolling window.

    VaR_t = -quantile(r_{t-window}, ..., r_{t-1}; 1 - confidence)
    """
    # DECISION: historical simulation makes no distributional assumption. It
    # uses the empirical distribution of the last `window` returns, so fat
    # tails and skew that appeared in the window are included automatically.
    # DECISION: quantile method = pandas/numpy default "linear" interpolation.
    # With 250 observations the 1% quantile sits at position 0.01 * 249 = 2.49
    # in the sorted sample, so the result interpolates between the 3rd and 4th
    # smallest returns. Alternatives ("lower", "nearest", or taking the k-th
    # worst loss) differ slightly, mainly for the 250-day window at 99%,
    # where only ~2.5 observations lie in the tail.
    # PITFALL: at 99% with a 250-day window the estimate rests on 2-3 data
    # points, so it is noisy and jumps when an extreme day enters or leaves
    # the window.
    # PITFALL: slow reaction after a shock. All days in the window have equal
    # weight, so a calm window reacts late to a new crisis, and after a crash
    # VaR stays high until the crash days drop out (250 or 500 days later),
    # the "ghost effect".
    rolling_q = returns.rolling(window=window, min_periods=window).quantile(1 - confidence)
    # DECISION: shift(1) is what prevents look-ahead bias. rolling_q at row t
    # uses r_{t-window+1}..r_t. Shifting by one day means the VaR for day t is
    # built from r_{t-window}..r_{t-1}, i.e. only information known at the
    # close of day t-1.
    var = -rolling_q.shift(1)
    var.name = f"HS{window}_{level_label(confidence)}"
    return var


def historical_es(returns, window, confidence):
    """Historical Expected Shortfall: average loss beyond the historical VaR.

    ES_t = -mean(r_s in the window with r_s <= quantile(window; 1 - confidence))
    using the same window r_{t-window}..r_{t-1} as historical_var.
    """
    def tail_mean(w):
        q = np.quantile(w, 1 - confidence)  # same "linear" method as historical_var
        return w[w <= q].mean()

    # PITFALL: at 99% with 250 days the "tail" holds only the 2-3 worst
    # returns, so historical ES is even noisier than historical VaR.
    es = -returns.rolling(window=window, min_periods=window).apply(tail_mean, raw=True).shift(1)
    es.name = f"HS{window}_{level_label(confidence)}"
    return es


# --- EWMA (RiskMetrics) ----------------------------------------------------

def ewma_variance(returns, lam, init_window):
    """EWMA variance forecasts sigma2_t for day t, using returns up to t-1.

    sigma2_t = lam * sigma2_{t-1} + (1 - lam) * r_{t-1}^2

    The first `init_window` returns are a burn-in period: the forecast for
    the first day after it is the mean of their squared returns. Forecasts
    for the burn-in days are NaN.
    """
    r = returns.to_numpy()
    sigma2 = np.full(len(r), np.nan)
    # DECISION: initialise with the mean of squared returns over the burn-in
    # window, not the sample variance. The model assumes a zero mean, and the
    # mean of r^2 is the variance estimate under that same assumption. The
    # sample variance would subtract the sample mean, which the recursion
    # itself never does. With lambda = 0.94 the starting value is forgotten
    # quickly anyway (weight 0.94^250 ~ 2e-7 after the burn-in length).
    sigma2[init_window] = np.mean(r[:init_window] ** 2)
    for t in range(init_window + 1, len(r)):
        sigma2[t] = lam * sigma2[t - 1] + (1 - lam) * r[t - 1] ** 2
    return pd.Series(sigma2, index=returns.index, name="ewma_sigma2")


# --- Parametric VaR and ES from a volatility forecast ----------------------

def normal_var(sigma, confidence):
    """VaR_t = z_alpha * sigma_t for a zero-mean normal distribution."""
    # DECISION: zero mean. The average daily return (~0.05%) is tiny compared
    # with daily volatility (~1.2%), and estimating the mean adds noise. Zero
    # mean is also the RiskMetrics convention and slightly conservative when
    # the true mean is positive.
    # PITFALL: the normal distribution has thin tails. Daily equity returns
    # are fat-tailed, so normal VaR at 99% tends to understate the loss
    # quantile even if the volatility forecast is good. The effect is smaller
    # at 95%, where the normal and empirical quantiles are closer.
    return norm.ppf(confidence) * sigma


def normal_es(sigma, confidence):
    """ES_t = sigma_t * phi(z_alpha) / (1 - alpha) for a zero-mean normal."""
    z = norm.ppf(confidence)
    return sigma * norm.pdf(z) / (1 - confidence)


def student_t_var(sigma, nu, confidence):
    """VaR for zero-mean Student-t innovations scaled to variance sigma^2."""
    # DECISION: arch's Student-t is standardised to unit variance, so the
    # quantile of a standard t(nu) is multiplied by sqrt((nu - 2) / nu).
    # Without this factor VaR would be overstated (a t(5) has variance 5/3).
    # PITFALL: the t distribution is symmetric. Equity losses are usually
    # somewhat larger than gains (negative skew), which it cannot capture.
    scale = np.sqrt((nu - 2) / nu)
    return -student_t.ppf(1 - confidence, nu) * scale * sigma


def student_t_es(sigma, nu, confidence):
    """ES for zero-mean Student-t innovations scaled to variance sigma^2.

    For a standard t(nu) with q = t_nu^{-1}(confidence):
        ES = f_nu(q) / (1 - confidence) * (nu + q^2) / (nu - 1)
    then scaled to unit variance by sqrt((nu - 2) / nu) and multiplied by sigma.
    """
    q = student_t.ppf(confidence, nu)
    es_standard_t = student_t.pdf(q, nu) / (1 - confidence) * (nu + q ** 2) / (nu - 1)
    return es_standard_t * np.sqrt((nu - 2) / nu) * sigma


# --- GARCH(1,1) --------------------------------------------------------------

def garch_forecasts(returns, dist, estimation_window, min_window, refit_every):
    """Rolling GARCH(1,1) volatility forecasts without look-ahead.

    sigma2_t = omega + alpha * r_{t-1}^2 + beta * sigma2_{t-1}

    Parameters are estimated on returns up to t-1 only: on a refit day t the
    model is fitted on r_{t-window}..r_{t-1} (expanding from `min_window`
    days until `estimation_window` days are available). On the following
    days the variance is updated with the same parameters.

    Returns (forecasts, params):
      forecasts: DataFrame with 'sigma' (decimal units) and 'nu' (Student-t
                 degrees of freedom, NaN for the normal model)
      params:    one row per refit with the estimates and convergence flag
    """
    from arch import arch_model  # imported here so the other models work without arch

    # DECISION: estimate in percent returns (x = 100 r). The arch optimiser is
    # tuned for numbers of order 1; with decimal returns omega is ~1e-6 and
    # the optimiser can stop early or fail. Results are converted back.
    x = returns.to_numpy() * 100
    n = len(x)
    sigma2 = np.full(n, np.nan)
    nu = np.full(n, np.nan)
    records = []
    params = None
    # DECISION: re-estimate every 20 trading days (about one month), not daily.
    # Daily refits cost 20x more computing time and barely change the
    # forecasts, because parameters estimated on ~1000 days move very little
    # from one day to the next. The conditional variance itself still reacts
    # every day through the recursion. The cost is that parameters can be up
    # to 19 days old; refitting much less often (e.g. yearly) would let them
    # drift away from the current regime.
    # DECISION: the first fit needs 500 days (expanding until 1000). This is
    # the same burn-in as the longest historical window, so GARCH does not
    # move the common backtest start. 500 daily returns are enough for three
    # GARCH parameters (four with nu), though early estimates are noisier.
    for start in range(min_window, n, refit_every):
        # DECISION: rolling 1000-day window (expanding before that). A rolling
        # window lets the parameters adapt to regime changes, while 1000 days
        # (~4 years) keep enough tail events for stable estimates.
        sample = x[max(0, start - estimation_window):start]  # r up to t-1 only
        model = arch_model(sample, mean="Zero", vol="GARCH", p=1, q=1,
                           dist=dist, rescale=False)
        fit = model.fit(disp="off", show_warning=False)
        converged = fit.convergence_flag == 0
        if converged or params is None:
            params = fit.params
            # Restart the variance path from the last in-sample conditional
            # variance under the new parameters.
            s2_prev = fit.conditional_volatility[-1] ** 2
        else:
            # DECISION: if the optimiser fails, keep the previous parameters
            # and simply continue the variance recursion. The failure is
            # logged, never hidden.
            logger.warning("GARCH-%s refit on %s did not converge (flag %s); "
                           "keeping previous parameters", dist, returns.index[start].date(),
                           fit.convergence_flag)
        omega, alpha, beta = params["omega"], params["alpha[1]"], params["beta[1]"]
        records.append({"date": returns.index[start], "n_obs": len(sample),
                        "omega": omega, "alpha": alpha, "beta": beta,
                        "persistence": alpha + beta, "nu": params.get("nu", np.nan),
                        "converged": converged})
        for t in range(start, min(start + refit_every, n)):
            sigma2[t] = omega + alpha * x[t - 1] ** 2 + beta * s2_prev
            nu[t] = params.get("nu", np.nan)
            s2_prev = sigma2[t]
    # PITFALL: GARCH parameters are estimated by maximum likelihood on a
    # finite window. alpha + beta is often close to 1 (very persistent
    # volatility), and the estimates can jump between refits when a crisis
    # enters or leaves the window. The parameter chart shows this. At
    # alpha + beta = 1 (the optimiser's upper bound) there is no finite
    # long-run variance and the model behaves like EWMA (IGARCH).
    forecasts = pd.DataFrame({"sigma": np.sqrt(sigma2) / 100, "nu": nu}, index=returns.index)
    return forecasts, pd.DataFrame(records).set_index("date")


# --- All models together ---------------------------------------------------

def all_forecasts(returns, hs_windows, ewma_lambda, ewma_init_window, var_levels,
                  es_levels, garch_window, garch_min_window, garch_refit_every):
    """VaR and ES forecasts of every model at every level.

    Returns (var_df, es_df, garch_params):
      var_df, es_df: one column per model and level, e.g. 'HS250_99',
                     'GARCH-t_97.5'. Rows before a model has enough history are NaN.
      garch_params:  {'GARCH-N': params table, 'GARCH-t': params table}
    """
    var, es = {}, {}
    sigma_ewma = np.sqrt(ewma_variance(returns, ewma_lambda, ewma_init_window))
    garch_n, params_n = garch_forecasts(returns, "normal", garch_window,
                                        garch_min_window, garch_refit_every)
    garch_t, params_t = garch_forecasts(returns, "t", garch_window,
                                        garch_min_window, garch_refit_every)
    # VaR is also needed at the ES levels: the ES check uses the VaR breach
    # days at the same level.
    for conf in sorted(set(var_levels) | set(es_levels)):
        label = level_label(conf)
        for window in hs_windows:
            var[f"HS{window}_{label}"] = historical_var(returns, window, conf)
        var[f"EWMA_{label}"] = normal_var(sigma_ewma, conf)
        var[f"GARCH-N_{label}"] = normal_var(garch_n["sigma"], conf)
        var[f"GARCH-t_{label}"] = student_t_var(garch_t["sigma"], garch_t["nu"], conf)
    for conf in es_levels:
        label = level_label(conf)
        for window in hs_windows:
            es[f"HS{window}_{label}"] = historical_es(returns, window, conf)
        es[f"EWMA_{label}"] = normal_es(sigma_ewma, conf)
        es[f"GARCH-N_{label}"] = normal_es(garch_n["sigma"], conf)
        es[f"GARCH-t_{label}"] = student_t_es(garch_t["sigma"], garch_t["nu"], conf)
    return pd.DataFrame(var), pd.DataFrame(es), {"GARCH-N": params_n, "GARCH-t": params_t}
