"""Tests for the VaR models: look-ahead bias and the EWMA recursion."""
import numpy as np
import pandas as pd
import pytest

from src import var_models


def fake_returns(n=700, seed=0):
    """Random daily returns with a business-day index."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(rng.normal(0, 0.01, n), index=idx)


def assert_no_lookahead(var_function, returns, t):
    """Changing r_t must leave VaR_t (and every earlier VaR) unchanged."""
    base = var_function(returns)
    shocked = returns.copy()
    shocked.iloc[t] = -0.25  # a huge crash on day t
    after = var_function(shocked)
    pd.testing.assert_series_equal(base.iloc[: t + 1], after.iloc[: t + 1])
    # Sanity check that the test can fail: the crash must affect VaR_{t+1}.
    assert after.iloc[t + 1] != base.iloc[t + 1]


@pytest.mark.parametrize("window", [250, 500])
def test_historical_var_no_lookahead(window):
    r = fake_returns()
    assert_no_lookahead(lambda x: var_models.historical_var(x, window, 0.99), r, t=600)


def test_ewma_var_no_lookahead():
    r = fake_returns()
    f = lambda x: var_models.normal_var(np.sqrt(var_models.ewma_variance(x, 0.94, 250)), 0.99)
    assert_no_lookahead(f, r, t=600)


def test_ewma_recursion_hand_example():
    # Returns:            r0=0.01, r1=-0.02, r2=0.03, r3=0.01, r4=-0.01
    # Burn-in of 3 days -> sigma2_3 = (0.01^2 + 0.02^2 + 0.03^2) / 3
    #                              = (0.0001 + 0.0004 + 0.0009) / 3 = 0.000466667
    # sigma2_4 = 0.94 * 0.000466667 + 0.06 * r3^2
    #          = 0.000438667 + 0.06 * 0.0001 = 0.000444667
    r = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01],
                  index=pd.bdate_range("2024-01-01", periods=5))
    sigma2 = var_models.ewma_variance(r, lam=0.94, init_window=3)
    assert np.isnan(sigma2.iloc[:3]).all()          # burn-in has no forecast
    assert sigma2.iloc[3] == pytest.approx(0.0014 / 3)
    assert sigma2.iloc[4] == pytest.approx(0.94 * 0.0014 / 3 + 0.06 * 0.0001)


def test_normal_var_and_es_known_values():
    # Textbook values: z_0.99 = 2.3263, ES_0.975 / sigma = 2.3378
    assert var_models.normal_var(1.0, 0.99) == pytest.approx(2.3263, abs=1e-4)
    assert var_models.normal_es(1.0, 0.975) == pytest.approx(2.3378, abs=1e-4)


@pytest.mark.parametrize("dist", ["normal", "t"])
@pytest.mark.parametrize("t", [380, 385])  # 380 is a refit day, 385 is between refits
def test_garch_var_no_lookahead(dist, t):
    # min_window 200 and refit every 20 days -> refits on days 200, 220, ..., 380
    r = fake_returns(n=420)

    def f(x):
        fc, _ = var_models.garch_forecasts(x, dist, estimation_window=300,
                                           min_window=200, refit_every=20)
        return var_models.normal_var(fc["sigma"], 0.99)

    assert_no_lookahead(f, r, t=t)


def test_student_t_es_approaches_normal():
    # With very many degrees of freedom the t distribution becomes normal.
    assert var_models.student_t_var(1.0, 1e6, 0.99) == pytest.approx(
        var_models.normal_var(1.0, 0.99), rel=1e-4)
    assert var_models.student_t_es(1.0, 1e6, 0.975) == pytest.approx(
        var_models.normal_es(1.0, 0.975), rel=1e-4)
    # Fat tails: with nu = 4 the 99% ES is larger than the normal ES.
    assert var_models.student_t_es(1.0, 4, 0.99) > var_models.normal_es(1.0, 0.99)


def test_historical_es_hand_example():
    # 10 returns; at 80% confidence the 20% quantile (linear) is
    # sorted[1] + 0.8 * (sorted[2] - sorted[1]) = -0.04 + 0.8 * 0.01 = -0.032,
    # so the tail is {-0.05, -0.04} and ES = 0.045 (forecast for day 10).
    r = pd.Series([0.01, -0.05, 0.02, -0.03, 0.00, 0.03, -0.04, 0.01, 0.02, -0.01, 0.0],
                  index=pd.bdate_range("2024-01-01", periods=11))
    es = var_models.historical_es(r, window=10, confidence=0.80)
    assert es.iloc[10] == pytest.approx(0.045)
