"""Tests for the backtesting statistics."""
import numpy as np
import pandas as pd
import pytest

from src import backtest


def test_kupiec_known_example():
    # T = 250 days, x = 5 breaches, p = 1% -> p_hat = 2%
    # LR = -2 * [245 ln(0.99) + 5 ln(0.01) - 245 ln(0.98) - 5 ln(0.02)] = 1.956810
    result = backtest.kupiec_pof(n_obs=250, n_breaches=5, p=0.01)
    assert result["LR"] == pytest.approx(1.956810, abs=1e-6)
    assert result["p_value"] == pytest.approx(0.1619, abs=1e-4)


def test_kupiec_zero_breaches():
    # x = 0: p_hat = 0 and the alternative likelihood is 1 (ln = 0), so
    # LR = -2 * 250 * ln(0.99) = 5.025168. Must be finite, not NaN or inf.
    result = backtest.kupiec_pof(n_obs=250, n_breaches=0, p=0.01)
    assert result["LR"] == pytest.approx(5.025168, abs=1e-6)
    assert result["p_value"] < 0.05  # zero breaches in 250 days is suspicious


def test_kupiec_exact_rate_gives_zero():
    result = backtest.kupiec_pof(n_obs=1000, n_breaches=10, p=0.01)
    assert result["LR"] == pytest.approx(0.0, abs=1e-12)
    assert result["p_value"] == pytest.approx(1.0)


def test_basel_zones():
    # 260 days with breaches on days 0..5 -> first window (days 0-249) has 6
    # breaches (yellow); once day 0 leaves the window the count falls to 5.
    b = pd.Series(False, index=pd.bdate_range("2024-01-01", periods=260))
    b.iloc[:6] = True
    tl = backtest.basel_traffic_light(b, window=250, green_max=4, yellow_max=9)
    assert len(tl) == 11                       # only complete windows
    assert tl["count"].iloc[0] == 6 and tl["zone"].iloc[0] == "yellow"
    assert tl["count"].iloc[-1] == 0 and tl["zone"].iloc[-1] == "green"


def test_christoffersen_clustered_example():
    # Sequence 0 0 0 0 1 1 1 0 0 0 -> transitions (yesterday -> today):
    #   n00 = 5, n01 = 1, n10 = 1, n11 = 2
    #   pi = 3/9, pi01 = 1/6, pi11 = 2/3
    # ln L(pi)          = 6 ln(2/3) + 3 ln(1/3)
    # ln L(pi01, pi11)  = 5 ln(5/6) + ln(1/6) + ln(1/3) + 2 ln(2/3)
    # LR_ind = -2 * (difference) = 2.231436
    b = pd.Series([0, 0, 0, 0, 1, 1, 1, 0, 0, 0]).astype(bool)
    result = backtest.christoffersen_independence(b)
    assert (result["n00"], result["n01"], result["n10"], result["n11"]) == (5, 1, 1, 2)
    assert result["LR"] == pytest.approx(2.231436, abs=1e-6)


def test_christoffersen_no_consecutive_breaches():
    # Sequence 0 1 0 0 1 0 -> n00 = 1, n01 = 2, n10 = 2, n11 = 0, so pi11 = 0
    # and the term n11 * ln(pi11) = 0 * ln(0) must be treated as 0.
    # ln L(pi) = 3 ln(0.6) + 2 ln(0.4);  ln L(alt) = ln(1/3) + 2 ln(2/3) + 2 ln(1)
    # LR_ind = 2.911032
    b = pd.Series([0, 1, 0, 0, 1, 0]).astype(bool)
    result = backtest.christoffersen_independence(b)
    assert result["n11"] == 0
    assert np.isfinite(result["LR"])
    assert result["LR"] == pytest.approx(2.911032, abs=1e-6)


def test_conditional_coverage_is_sum():
    b = pd.Series([0, 0, 0, 0, 1, 1, 1, 0, 0, 0]).astype(bool)
    cc = backtest.christoffersen_conditional_coverage(10, 3, 0.05, b)
    expected = (backtest.kupiec_pof(10, 3, 0.05)["LR"]
                + backtest.christoffersen_independence(b)["LR"])
    assert cc["LR"] == pytest.approx(expected)
