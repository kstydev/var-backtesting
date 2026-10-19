"""All project parameters in one place.

Change values here, not inside the src/ modules, so every run is fully
described by this file plus the data snapshot.
"""
from pathlib import Path

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_FILE = OUTPUT_DIR / "run_log.txt"

# --- Data ------------------------------------------------------------------
# Five large DAX stocks traded on Xetra (".DE" suffix on Yahoo Finance).
TICKERS = ["SAP.DE", "SIE.DE", "ALV.DE", "BAS.DE", "DTE.DE"]
# DAX performance (total return) index. Not part of the portfolio; only used
# in the Siemens spin-off sensitivity run. Stored in the same snapshot.
BENCHMARK_TICKER = "^GDAXI"
START_DATE = "2015-01-01"
END_DATE = None  # None = up to yesterday (last completed trading day)

# Threshold for the data quality check: flag single-day stock returns whose
# absolute value is above this level for manual review.
QUALITY_RETURN_THRESHOLD = 0.15

# --- Portfolio -------------------------------------------------------------
# DECISION: equal weights, rebalanced daily. This keeps the portfolio return a
# simple weighted sum of daily stock returns with constant weights.
WEIGHTS = {ticker: 0.20 for ticker in TICKERS}

# --- VaR models ------------------------------------------------------------
HS_WINDOWS = [250, 500]        # historical simulation look-back windows (days)
EWMA_LAMBDA = 0.94             # RiskMetrics (1996) decay factor for daily data
EWMA_INIT_WINDOW = 250         # burn-in used to initialise the EWMA variance
CONFIDENCE_LEVELS = [0.99, 0.95]
MAIN_CONFIDENCE = 0.99

# GARCH(1,1), estimated with the `arch` library
GARCH_ESTIMATION_WINDOW = 1000  # rolling estimation window (days)
GARCH_MIN_WINDOW = 500          # expanding window until 1000 days are available
GARCH_REFIT_EVERY = 20          # re-estimate parameters every 20 trading days
# Robustness check only: start GARCH with a full 1000-day window instead.
GARCH_ROBUSTNESS_MIN_WINDOW = 1000

# Expected Shortfall levels (97.5% is the Basel FRTB standard)
ES_LEVELS = [0.975, 0.99]

# Regime split: rolling volatility window (days) for calm vs turbulent
REGIME_VOL_WINDOW = 60

# --- Backtesting -----------------------------------------------------------
KUPIEC_SIGNIFICANCE = 0.05
BASEL_WINDOW = 250             # Basel traffic light uses the last 250 days
BASEL_GREEN_MAX = 4            # 0-4 breaches  -> green
BASEL_YELLOW_MAX = 9           # 5-9 breaches  -> yellow, 10+ -> red

# --- Siemens sensitivity run ----------------------------------------------
SIEMENS_TICKER = "SIE.DE"
SIEMENS_SPINOFF_DAY = "2020-09-28"

# --- Plotting --------------------------------------------------------------
# Stress periods shaded on the charts.
SHADED_PERIODS = {
    "Covid crash": ("2020-02-01", "2020-04-30"),
    "2022": ("2022-01-01", "2022-12-31"),
}
