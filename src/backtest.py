"""VaR backtesting: breach counts, Kupiec POF, Christoffersen tests, Basel traffic light, ES check."""
import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2


def breaches(returns, var):
    """Breach indicator: True on days where the realised return r_t < -VaR_t."""
    return returns < -var


def kupiec_pof(n_obs, n_breaches, p):
    """Kupiec (1995) proportion-of-failures test.

    H0: the true breach probability equals p (e.g. 0.01 for 99% VaR).
    LR = -2 * [ln L(p) - ln L(p_hat)],  p_hat = n_breaches / n_obs
    with ln L(q) = (n - x) ln(1 - q) + x ln(q). Under H0, LR ~ chi2(1).
    """
    x, n = n_breaches, n_obs
    p_hat = x / n
    # DECISION: scipy's xlogy(a, b) computes a * ln(b) but returns 0 when
    # a = 0. That is exactly the convention 0 * ln(0) = 0 needed for the
    # zero-breach case (p_hat = 0 would otherwise give ln(0) = -inf) and for
    # the all-breach case. With x = 0 the statistic becomes -2 n ln(1 - p).
    log_l_null = xlogy(n - x, 1 - p) + xlogy(x, p)
    log_l_alt = xlogy(n - x, 1 - p_hat) + xlogy(x, p_hat)
    lr = -2 * (log_l_null - log_l_alt)
    # PITFALL: the chi-squared distribution is only an asymptotic
    # approximation. With ~2,500 observations it is fine, but on a single
    # 250-day window at 99% (2.5 expected breaches) the test has very little
    # power: it often cannot tell a 1% model from a 2% model.
    return {"LR": lr, "p_value": chi2.sf(lr, df=1)}


def christoffersen_independence(breach_series):
    """Christoffersen (1998) independence test (first-order Markov).

    Counts transitions between consecutive days:
      n00: no breach -> no breach   n01: no breach -> breach
      n10: breach -> no breach      n11: breach -> breach
    H0: tomorrow's breach probability does not depend on today's breach
    (pi01 = pi11). LR_ind = -2 [ln L(pi) - ln L(pi01, pi11)] ~ chi2(1).
    """
    b = breach_series.astype(int).to_numpy()
    prev, curr = b[:-1], b[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    # Same 0 * ln(0) = 0 convention as in the Kupiec test (e.g. n11 = 0,
    # which is common at 99%: two breaches in a row are rare).
    log_l_null = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    log_l_alt = (xlogy(n00, 1 - pi01) + xlogy(n01, pi01)
                 + xlogy(n10, 1 - pi11) + xlogy(n11, pi11))
    lr = -2 * (log_l_null - log_l_alt)
    # PITFALL: this test only looks at breaches on CONSECUTIVE days. Breaches
    # that cluster within a few weeks but not on back-to-back days (typical
    # in a crisis) are largely invisible to it, so "not rejected" does not
    # prove that breaches are independent. The per-year table and traffic
    # light show clustering more directly.
    return {"LR": lr, "p_value": chi2.sf(lr, df=1),
            "n00": n00, "n01": n01, "n10": n10, "n11": n11}


def christoffersen_conditional_coverage(n_obs, n_breaches, p, breach_series):
    """Conditional coverage test: LR_cc = LR_pof + LR_ind ~ chi2(2).

    Tests jointly that the breach rate is right AND breaches are independent.
    """
    # DECISION: LR_cc is computed as the sum of the Kupiec and independence
    # statistics. Strictly, Kupiec uses all T days and the independence test
    # T-1 transitions (the first day has no "yesterday"); with ~2,500 days the
    # difference is negligible and this is the common textbook form.
    lr = kupiec_pof(n_obs, n_breaches, p)["LR"] + christoffersen_independence(breach_series)["LR"]
    return {"LR": lr, "p_value": chi2.sf(lr, df=2)}


def basel_traffic_light(breach_series, window, green_max, yellow_max):
    """Rolling breach count over `window` days and its Basel zone.

    Returns a DataFrame with columns 'count' and 'zone', one row per complete
    window (indexed by the window's last day).
    """
    # DECISION: only complete windows are counted (the first window ends on
    # day 250 of the backtest). Partial windows would mechanically look green.
    counts = breach_series.astype(int).rolling(window).sum().dropna().astype(int)
    zone = pd.Series(np.select([counts <= green_max, counts <= yellow_max],
                               ["green", "yellow"], default="red"),
                     index=counts.index)
    # PITFALL: the Basel zones are calibrated for 99% VaR on 250 days. For a
    # correct model, 10+ breaches (red) has a probability of only ~0.03%, but
    # 5+ breaches (yellow or worse) still happens ~11% of the time, so yellow
    # alone is weak evidence of a bad model (binomial, n = 250, p = 1%).
    return pd.DataFrame({"count": counts, "zone": zone})


def breaches_per_year(breach_df):
    """Number of breaches per calendar year for each model (columns)."""
    table = breach_df.astype(int).groupby(breach_df.index.year).sum()
    table.index.name = "year"
    return table


def backtest_table(returns, var_df, confidence_of, significance,
                   basel_window, green_max, yellow_max, basel_level=0.99):
    """One row per VaR column with breach statistics and test results.

    `confidence_of` maps a column name to its confidence level.
    """
    rows = []
    for name in var_df.columns:
        conf = confidence_of[name]
        p = 1 - conf
        b = breaches(returns, var_df[name])
        n, x = len(b), int(b.sum())
        kup = kupiec_pof(n, x, p)
        row = {
            "model": name.rsplit("_", 1)[0],
            "confidence": conf,
            "n_obs": n,
            "expected_breaches": n * p,
            "actual_breaches": x,
            "breach_rate": x / n,
            "kupiec_LR": kup["LR"],
            "kupiec_p": kup["p_value"],
            "kupiec_reject": kup["p_value"] < significance,
        }
        ind = christoffersen_independence(b)
        cc = christoffersen_conditional_coverage(n, x, p, b)
        row.update({
            "n11": ind["n11"],
            "ind_LR": ind["LR"], "ind_p": ind["p_value"],
            "ind_reject": ind["p_value"] < significance,
            "cc_LR": cc["LR"], "cc_p": cc["p_value"],
            "cc_reject": cc["p_value"] < significance,
        })
        if np.isclose(conf, basel_level):
            tl = basel_traffic_light(b, basel_window, green_max, yellow_max)
            shares = tl["zone"].value_counts(normalize=True)
            row.update({
                "basel_latest_count": int(tl["count"].iloc[-1]),
                "basel_latest_zone": tl["zone"].iloc[-1],
                "share_green": shares.get("green", 0.0),
                "share_yellow": shares.get("yellow", 0.0),
                "share_red": shares.get("red", 0.0),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def es_check(returns, var, es):
    """Simple ES check on the VaR breach days at the same confidence level.

    If the ES forecast is right, the average realised loss on breach days
    should be close to the average ES forecast on those days (ratio ~ 1).
    A ratio above 1 means losses beyond VaR were worse than ES predicted.
    """
    # PITFALL: ES is much harder to backtest than VaR. A VaR forecast can be
    # checked day by day (breach yes/no), but ES is an average over the tail,
    # so each day gives no clear verdict. (1) Only breach days carry
    # information, about 25 days at 99% over ten years, so the average is
    # very noisy. (2) The check depends on the VaR too: a model with too
    # many breaches includes milder losses and can look better on ES. (3) ES
    # is not "elicitable" on its own: there is no simple loss function that
    # a correct ES forecast minimises (Gneiting 2011), which is why proper ES
    # tests (Acerbi-Szekely 2014, or joint VaR-ES scoring) are more involved.
    # This simple ratio is a plausibility check, not a statistical test.
    b = breaches(returns, var)
    n = int(b.sum())
    realised = -returns[b].mean() if n else np.nan
    forecast = es[b].mean() if n else np.nan
    return {"n_breaches": n, "avg_realised_loss": realised, "avg_es_forecast": forecast,
            "ratio": realised / forecast if n else np.nan}


def volatility_regime(returns, window):
    """Label each day 'calm' or 'turbulent' using only past data.

    vol_t       = std(r_{t-window}, ..., r_{t-1})       (known at t-1)
    threshold_t = median of all vol_s with s <= t       (also known at t-1)
    turbulent if vol_t > threshold_t, else calm.
    """
    # DECISION: a simple rule fixed BEFORE looking at the backtest results:
    # 60-day rolling volatility (about one quarter) against its expanding
    # long-run median. Both use data up to t-1 only, so the regime label is
    # known before the day's return. It is a descriptive split, not tuned:
    # no other windows or thresholds were tried.
    # PITFALL: an expanding median moves slowly, so after a long calm (or
    # turbulent) period most days fall into one regime. The split is not 50/50
    # and with ~1,000 turbulent days a 99% model should only have ~10
    # breaches there, so regime breach rates are very noisy.
    vol = returns.rolling(window).std().shift(1)
    threshold = vol.expanding().median()
    regime = pd.Series(np.where(vol > threshold, "turbulent", "calm"), index=returns.index)
    return regime.where(vol.notna())


def regime_breach_table(returns, var_df, confidence_of, regime):
    """Breach rate of every VaR column, separately in each regime."""
    rows = []
    for name in var_df.columns:
        b = breaches(returns, var_df[name])
        for label in ["calm", "turbulent"]:
            mask = regime.loc[b.index] == label
            rows.append({"model": name.rsplit("_", 1)[0], "confidence": confidence_of[name],
                         "regime": label, "n_obs": int(mask.sum()),
                         "breaches": int(b[mask].sum()),
                         "breach_rate": b[mask].mean(),
                         "expected_rate": 1 - confidence_of[name]})
    return pd.DataFrame(rows)


def sanity_checks(table, var_df=None, es_df=None, low=0.5, high=2.0):
    """Flag breach rates far from their target as possible bugs.

    Too FEW breaches is the classic sign of look-ahead bias (the model has
    seen the return it is forecasting), so it is flagged rather than
    celebrated. Too many breaches is more often a genuine model weakness,
    but a rate above 2x target is still worth a second look.
    """
    warnings = []
    for _, row in table.iterrows():
        ratio = row["actual_breaches"] / row["expected_breaches"]
        label = f"{row['model']} at {row['confidence']:.1%}"
        if ratio < low:
            warnings.append(f"POSSIBLE BUG: {label} has only {row['actual_breaches']} breaches "
                            f"vs {row['expected_breaches']:.1f} expected (ratio {ratio:.2f}).")
        elif ratio > high:
            warnings.append(f"CHECK: {label} has {row['actual_breaches']} breaches "
                            f"vs {row['expected_breaches']:.1f} expected (ratio {ratio:.2f}).")
    if var_df is not None:
        if var_df.isna().any().any() or (var_df <= 0).any().any():
            warnings.append("POSSIBLE BUG: missing or non-positive VaR values in the backtest period.")
    if es_df is not None and var_df is not None:
        for name in es_df.columns:
            if (es_df[name] < var_df[name] - 1e-12).any():
                warnings.append(f"POSSIBLE BUG: ES below VaR for {name} on some days.")
    return warnings
