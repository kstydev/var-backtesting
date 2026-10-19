"""Download, snapshot, clean and quality-check the price data."""
import logging
from datetime import date

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def download_prices(tickers, start, end=None):
    """Download daily adjusted close prices (EUR) from Yahoo Finance.

    Returns a DataFrame with one column per ticker, indexed by date.
    """
    # DECISION: only completed trading days. If no end date is given we use
    # today as the (exclusive) end, so the snapshot stops at yesterday. A
    # download during trading hours would otherwise contain an intraday price
    # for today, and the last "daily" return would be incomplete.
    today = pd.Timestamp(date.today())
    if end is None:
        end = today.strftime("%Y-%m-%d")
    # threads=False: parallel downloads can fail with "database is locked" in
    # yfinance's internal cache, silently returning an empty column.
    # DECISION: auto_adjust=True makes yfinance return "Close" prices that are
    # already adjusted for dividends and splits. Using adjusted prices means a
    # price return also includes the dividend, i.e. it is a total return. Without
    # adjustment every ex-dividend day would look like a loss.
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False, group_by="column", threads=False)
    prices = raw["Close"][tickers]
    empty = [t for t in tickers if prices[t].isna().all()]
    if empty:
        raise RuntimeError(f"Download failed for {empty}; no snapshot saved. Try again.")
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.index.name = "Date"
    # Safety net in case the data source ignores the end date.
    return prices[prices.index < today]


def save_snapshot(prices, raw_dir):
    """Save prices as data/raw/prices_<download date>.csv and return the path."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"prices_{date.today().isoformat()}.csv"
    prices.to_csv(path)
    return path


def latest_snapshot(raw_dir):
    """Return the path of the most recent snapshot, or None if there is none."""
    # ISO dates in the filename sort correctly as plain strings.
    snapshots = sorted(raw_dir.glob("prices_*.csv"))
    return snapshots[-1] if snapshots else None


def load_prices(tickers, start, end, raw_dir, refresh=False):
    """Load prices from the latest snapshot, or download a new one.

    `tickers` may include non-portfolio series such as the DAX index; the
    snapshot simply stores one column per ticker.
    """
    # DECISION: results are computed from a saved CSV snapshot, not from a live
    # download, so re-running the project gives identical numbers. Yahoo data
    # can be revised over time (e.g. adjustment factors change after new
    # dividends), so a live download is not reproducible.
    snapshot = latest_snapshot(raw_dir)
    if refresh or snapshot is None:
        prices = download_prices(tickers, start, end)
        snapshot = save_snapshot(prices, raw_dir)
        logger.info("Downloaded new data and saved snapshot %s", snapshot.name)
    else:
        prices = pd.read_csv(snapshot, index_col="Date", parse_dates=True)
        logger.info("Loaded existing snapshot %s", snapshot.name)
    return prices[tickers], snapshot


def align_prices(prices):
    """Keep only dates on which every stock has a price (inner join).

    Returns the aligned prices and the number of dropped dates.
    """
    # DECISION: inner join, no forward-fill. Forward-filling a missing price
    # creates an artificial 0% return on that day and then puts two days of
    # movement into the next return. That understates volatility and distorts
    # the VaR backtest. Dropping the date is cleaner.
    # PITFALL: dropping a date means the next return covers two trading days
    # for every stock. With few dropped days this effect is negligible, but it
    # should be checked (the count is logged below).
    n_before = len(prices)
    aligned = prices.dropna(how="any")
    n_dropped = n_before - len(aligned)
    missing_per_ticker = prices.isna().sum()
    logger.info("Date alignment (inner join): %d of %d dates dropped", n_dropped, n_before)
    logger.info("Missing prices per ticker before alignment: %s", missing_per_ticker.to_dict())
    return aligned, n_dropped


def compute_returns(prices):
    """Daily simple returns r_t = P_t / P_{t-1} - 1."""
    # DECISION: simple returns, not log returns. The portfolio's simple return
    # is exactly the weighted sum of the stocks' simple returns, while log
    # returns do not aggregate across assets. VaR is also quoted as a % loss of
    # portfolio value, which is a simple return.
    return prices.pct_change().dropna(how="all")


def quality_check(returns, threshold):
    """List every single-day stock return with |r| > threshold.

    Nothing is changed. The table is only for manual review.
    """
    stacked = returns.stack()
    flagged = stacked[stacked.abs() > threshold]
    table = flagged.rename("return").reset_index()
    table.columns = ["Date", "Ticker", "return"]
    return table.sort_values(["Date", "Ticker"]).reset_index(drop=True)


def inspect_window(returns, columns, start, end):
    """Show the daily returns of some columns around a given period.

    Used to check specific corporate events, e.g. the Siemens Energy
    spin-off in September 2020, by comparing a stock with the DAX.
    """
    # Returns only, no prices: the outputs are published and Yahoo's terms do
    # not allow redistributing their price data.
    return returns.loc[start:end, columns]


def replace_return(returns, ticker, day, new_value):
    """Return a copy of `returns` with one single value replaced.

    Used only for the Siemens sensitivity run. The main results always use the
    raw data.
    """
    # PITFALL: the Siemens Energy spin-off day (2020-09-28) in the Yahoo data
    # shows SIE.DE +9.22% while the DAX rose 3.22%. Yahoo has no spin-off
    # record and even its unadjusted close jumps, so the pre-spin-off history
    # looks inconsistently rescaled. We keep the raw value in the main results
    # and only measure its impact in a sensitivity run.
    modified = returns.copy()
    modified.loc[pd.Timestamp(day), ticker] = new_value
    return modified
