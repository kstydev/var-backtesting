"""Portfolio return construction."""
import pandas as pd


def portfolio_returns(stock_returns, weights):
    """Daily portfolio return = sum_i w_i * r_{i,t}.

    `weights` is a dict {ticker: weight}. The weights must sum to 1.
    """
    w = pd.Series(weights)[stock_returns.columns]
    if abs(w.sum() - 1.0) > 1e-9:
        raise ValueError(f"Weights must sum to 1, got {w.sum()}")
    # DECISION: constant weights applied every day means daily rebalancing back
    # to 20% each. A buy-and-hold portfolio would let weights drift with prices,
    # and then the formula would need yesterday's actual weights.
    # PITFALL: daily rebalancing ignores transaction costs. That is fine for
    # measuring risk but would not be a realistic trading strategy.
    port = stock_returns.mul(w, axis=1).sum(axis=1)
    port.name = "portfolio"
    return port
