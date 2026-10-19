"""Run the full pipeline from the command line.

    python run_all.py            # use the latest saved data snapshot
    python run_all.py --refresh  # download fresh data and save a new snapshot
"""
import argparse
import logging

import pandas as pd

import config
from src import backtest, data, plots, portfolio, var_models

log = logging.getLogger("run_all")


def setup_logging():
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(config.LOG_FILE, mode="w")],
        force=True,
    )


def confidence_of(columns):
    """Map 'HS250_97.5' -> 0.975 etc."""
    return {c: float(c.rsplit("_", 1)[1]) / 100 for c in columns}


def build_forecasts(port):
    """Every model's VaR and ES forecasts, restricted to the common backtest period."""
    var_all, es_all, garch_params = var_models.all_forecasts(
        port, config.HS_WINDOWS, config.EWMA_LAMBDA, config.EWMA_INIT_WINDOW,
        config.CONFIDENCE_LEVELS, config.ES_LEVELS, config.GARCH_ESTIMATION_WINDOW,
        config.GARCH_MIN_WINDOW, config.GARCH_REFIT_EVERY)
    # DECISION: one common backtest period for all models. It starts on the
    # first day on which EVERY model has a forecast, so differences between
    # models cannot come from testing them on different market periods.
    common_days = var_all.dropna(how="any").index.intersection(es_all.dropna(how="any").index)
    first_valid = var_all.apply(lambda col: col.first_valid_index())
    return var_all.loc[common_days], es_all.loc[common_days], garch_params, first_valid


def run_backtests(port, var_df):
    """Backtest table, per-year breaches and traffic-light counts."""
    r = port.loc[var_df.index]
    main_cols = [c for c in var_df.columns
                 if confidence_of([c])[c] in config.CONFIDENCE_LEVELS]
    table = backtest.backtest_table(r, var_df[main_cols], confidence_of(main_cols),
                                    config.KUPIEC_SIGNIFICANCE, config.BASEL_WINDOW,
                                    config.BASEL_GREEN_MAX, config.BASEL_YELLOW_MAX,
                                    basel_level=config.MAIN_CONFIDENCE)
    breach_df = pd.DataFrame({c: backtest.breaches(r, var_df[c]) for c in main_cols})
    per_year = backtest.breaches_per_year(breach_df)
    cols_99 = [c for c in main_cols if c.endswith("_99")]
    tl_counts = pd.DataFrame({
        c: backtest.basel_traffic_light(breach_df[c], config.BASEL_WINDOW,
                                        config.BASEL_GREEN_MAX, config.BASEL_YELLOW_MAX)["count"]
        for c in cols_99})
    regime = backtest.volatility_regime(port, config.REGIME_VOL_WINDOW)
    regimes = backtest.regime_breach_table(r, var_df[main_cols], confidence_of(main_cols), regime)
    return {"table": table, "per_year": per_year, "tl_counts": tl_counts, "regimes": regimes}


def compare_sensitivity(base, alt):
    """Which breach counts, test results and traffic-light zones changed?"""
    keys = ["model", "confidence"]
    cols = ["actual_breaches", "kupiec_reject", "ind_reject", "cc_reject",
            "basel_latest_zone", "share_green", "share_yellow", "share_red"]
    merged = base[keys + cols].merge(alt[keys + cols], on=keys, suffixes=("_raw", "_sens"))
    for c in cols:
        a, b = merged[f"{c}_raw"], merged[f"{c}_sens"]
        merged[f"{c}_changed"] = ~((a == b) | (a.isna() & b.isna()))
    return merged


def garch_min_window_check(port, var_df, main_table):
    """Re-run both GARCH models with a longer burn-in and compare test decisions.

    Main specification: first fit after 500 days. Robustness: first fit
    after GARCH_ROBUSTNESS_MIN_WINDOW days, so the backtest starts later.
    Returns (comparison table, robustness period, largest VaR difference
    against the main specification on that same period).
    """
    var_rob = {}
    for name, dist in [("GARCH-N", "normal"), ("GARCH-t", "t")]:
        fc, _ = var_models.garch_forecasts(port, dist, config.GARCH_ESTIMATION_WINDOW,
                                           config.GARCH_ROBUSTNESS_MIN_WINDOW,
                                           config.GARCH_REFIT_EVERY)
        for conf in config.CONFIDENCE_LEVELS:
            col = f"{name}_{var_models.level_label(conf)}"
            var_rob[col] = (var_models.normal_var(fc["sigma"], conf) if dist == "normal"
                            else var_models.student_t_var(fc["sigma"], fc["nu"], conf))
    var_rob = pd.DataFrame(var_rob).dropna()
    max_diff = (var_rob - var_df.loc[var_rob.index, var_rob.columns]).abs().max().max()
    rob_table = backtest.backtest_table(port.loc[var_rob.index], var_rob,
                                        confidence_of(var_rob.columns),
                                        config.KUPIEC_SIGNIFICANCE, config.BASEL_WINDOW,
                                        config.BASEL_GREEN_MAX, config.BASEL_YELLOW_MAX,
                                        basel_level=config.MAIN_CONFIDENCE)
    keys = ["model", "confidence"]
    cols = ["n_obs", "actual_breaches", "breach_rate", "kupiec_reject", "ind_reject",
            "cc_reject", "basel_latest_zone", "share_red"]
    main = main_table[main_table["model"].str.startswith("GARCH")]
    comp = main[keys + cols].merge(rob_table[keys + cols], on=keys, suffixes=("_main", "_rob"))
    comp["decision_changed"] = (
        (comp[["kupiec_reject_main", "ind_reject_main", "cc_reject_main"]].to_numpy()
         != comp[["kupiec_reject_rob", "ind_reject_rob", "cc_reject_rob"]].to_numpy()).any(axis=1)
        | ~((comp["basel_latest_zone_main"] == comp["basel_latest_zone_rob"])
            | comp["basel_latest_zone_main"].isna()))
    return comp, var_rob.index, max_diff


def print_table(table):
    """Readable version of the backtest table."""
    show = table.copy()
    show["confidence"] = show["confidence"].map("{:.0%}".format)
    show["expected_breaches"] = show["expected_breaches"].map("{:.1f}".format)
    show["breach_rate"] = show["breach_rate"].map("{:.2%}".format)
    for c in [c for c in show.columns if c.endswith("_LR")]:
        show[c] = show[c].map("{:.2f}".format)
    for c in [c for c in show.columns if c.endswith("_p")]:
        show[c] = show[c].map("{:.3f}".format)
    if "basel_latest_count" in show:
        show["basel_latest_count"] = show["basel_latest_count"].map(
            lambda v: "" if pd.isna(v) else f"{v:.0f}")
    for c in [c for c in show.columns if c.startswith("share_")]:
        show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v:.0%}")
    return show.fillna("").to_string(index=False)


def main(refresh=False):
    setup_logging()
    pd.set_option("display.width", 200)
    out = config.OUTPUT_DIR

    # 1. Data
    all_tickers = config.TICKERS + [config.BENCHMARK_TICKER]
    raw, snapshot = data.load_prices(all_tickers, config.START_DATE, config.END_DATE,
                                     config.RAW_DATA_DIR, refresh=refresh)
    log.info("Snapshot: %s", snapshot.name)
    prices, n_dropped = data.align_prices(raw[config.TICKERS])
    log.info("Dates dropped by inner join: %d", n_dropped)
    log.info("Price data: %s to %s, %d trading days",
             prices.index[0].date(), prices.index[-1].date(), len(prices))
    stock_returns = data.compute_returns(prices)

    # 2. Data quality check (report only, nothing is fixed)
    flagged = data.quality_check(stock_returns, config.QUALITY_RETURN_THRESHOLD)
    log.info("\nSingle-day returns with |r| > %.0f%% (%d found):\n%s",
             config.QUALITY_RETURN_THRESHOLD * 100, len(flagged),
             flagged.to_string(formatters={"return": "{:.2%}".format}) if len(flagged) else "none")
    dax_returns = data.compute_returns(raw[[config.BENCHMARK_TICKER]].loc[prices.index])
    sie = data.inspect_window(stock_returns.join(dax_returns),
                              [config.SIEMENS_TICKER, config.BENCHMARK_TICKER],
                              "2020-09-21", "2020-10-02")
    log.info("\nDaily returns around the Siemens Energy spin-off (first trading day 28 Sep 2020):\n%s",
             sie.to_string(float_format="{:.2%}".format))

    # 3. Portfolio
    port = portfolio.portfolio_returns(stock_returns, config.WEIGHTS)
    log.info("\nPortfolio returns: %d days, mean %.4f%%, daily std %.4f%%, min %.2f%%, max %.2f%%",
             len(port), port.mean() * 100, port.std() * 100, port.min() * 100, port.max() * 100)

    # 4. VaR forecasts on the common backtest period
    var_df, es_df, garch_params, first_valid = build_forecasts(port)
    start, end = var_df.index[0], var_df.index[-1]
    log.info("\nFirst forecast day per model:\n%s",
             first_valid.map(lambda d: d.date()).to_string())
    log.info("\nCommon backtest period: %s to %s (%d days)", start.date(), end.date(), len(var_df))
    r = port.loc[start:end]

    # 5. Backtests
    res = run_backtests(port, var_df)
    table = res["table"]
    log.info("\nBacktest results:\n%s", print_table(table))
    log.info("\nBreaches per calendar year:\n%s", res["per_year"].to_string())
    table.to_csv(out / "backtest_results.csv", index=False)
    res["per_year"].to_csv(out / "breaches_per_year.csv")
    var_df.to_csv(out / "var_forecasts.csv")
    es_df.to_csv(out / "es_forecasts.csv")

    # Expected Shortfall check
    es_table = pd.DataFrame([{"model": c.rsplit("_", 1)[0], "confidence": confidence_of([c])[c],
                              **backtest.es_check(r, var_df[c], es_df[c])}
                             for c in es_df.columns])
    es_table.to_csv(out / "es_check.csv", index=False)
    log.info("\nExpected Shortfall check (on VaR breach days at the same level):\n%s",
             es_table.to_string(index=False, formatters={
                 "confidence": "{:.1%}".format, "avg_realised_loss": "{:.2%}".format,
                 "avg_es_forecast": "{:.2%}".format, "ratio": "{:.2f}".format}))

    # Regime split (descriptive)
    reg = res["regimes"]
    reg.to_csv(out / "regime_breach_rates.csv", index=False)
    n_days = reg.drop_duplicates("regime").set_index("regime")["n_obs"]
    pivot = reg.pivot_table(index=["confidence", "model"], columns="regime",
                            values="breach_rate", sort=False)
    log.info("\nBreach rates by volatility regime (descriptive, rule fixed in advance; "
             "calm %d days, turbulent %d days):\n%s", n_days["calm"], n_days["turbulent"],
             pivot.to_string(float_format="{:.2%}".format))

    # 6. Charts
    cols_99 = [c for c in var_df.columns if c.endswith("_99")]
    for c in cols_99:
        plots.plot_var_breaches(r, var_df[c], f"1-day VaR backtest: {c}",
                                config.SHADED_PERIODS, out / f"breaches_{c}.png")
    plots.plot_var_comparison(r, var_df[cols_99], "All 99% VaR forecasts",
                              config.SHADED_PERIODS, out / "var_comparison_99.png")
    plots.plot_traffic_light(res["tl_counts"], config.BASEL_GREEN_MAX, config.BASEL_YELLOW_MAX,
                             "Basel traffic light: breaches of 99% VaR in rolling 250-day windows",
                             out / "traffic_light_99.png")

    # GARCH parameter diagnostics
    for name, p in garch_params.items():
        p.to_csv(out / f"garch_params_{name}.csv")
        log.info("\n%s: %d refits, %d convergence failures, alpha+beta range %.3f-%.3f%s",
                 name, len(p), int((~p["converged"]).sum()), p["persistence"].min(),
                 p["persistence"].max(),
                 f", nu range {p['nu'].min():.1f}-{p['nu'].max():.1f}" if p["nu"].notna().any() else "")
    plots.plot_garch_params(garch_params, out / "garch_parameters.png")

    # 7. Siemens spin-off sensitivity run
    # Replace the single SIE.DE return on 2020-09-28 with the DAX return of the
    # same day and re-run every model and backtest. Main results stay raw.
    day = pd.Timestamp(config.SIEMENS_SPINOFF_DAY)
    dax_value = dax_returns.loc[day, config.BENCHMARK_TICKER]
    stock_alt = data.replace_return(stock_returns, config.SIEMENS_TICKER, day, dax_value)
    port_alt = portfolio.portfolio_returns(stock_alt, config.WEIGHTS)
    log.info("\nSiemens sensitivity: SIE.DE return on %s replaced %.2f%% -> %.2f%% (DAX); "
             "portfolio return %.2f%% -> %.2f%%", day.date(),
             stock_returns.loc[day, config.SIEMENS_TICKER] * 100, dax_value * 100,
             port.loc[day] * 100, port_alt.loc[day] * 100)
    var_alt, es_alt, _, _ = build_forecasts(port_alt)
    if not var_alt.index.equals(var_df.index):
        log.warning("Sensitivity run has a different backtest period!")
    res_alt = run_backtests(port_alt, var_alt)
    comp = compare_sensitivity(table, res_alt["table"])
    comp.to_csv(out / "siemens_sensitivity.csv", index=False)
    changed = comp[comp.filter(like="_changed").any(axis=1)]
    zone_raw = res["tl_counts"].apply(lambda c: pd.cut(c, [-1, config.BASEL_GREEN_MAX,
                                                         config.BASEL_YELLOW_MAX, 1e9],
                                                     labels=["green", "yellow", "red"]))
    zone_alt = res_alt["tl_counts"].apply(lambda c: pd.cut(c, [-1, config.BASEL_GREEN_MAX,
                                                             config.BASEL_YELLOW_MAX, 1e9],
                                                         labels=["green", "yellow", "red"]))
    windows_changed = (zone_raw != zone_alt).sum()
    for c in res["per_year"].columns:
        diff = (backtest.breaches(r, var_df[c])
                != backtest.breaches(port_alt.loc[r.index], var_alt[c]))
        if diff.any():
            log.info("%s: breach status changed on %s", c,
                     ", ".join(str(d.date()) for d in diff[diff].index))
    log.info("Traffic-light windows whose zone changed: %s", windows_changed.to_dict())
    if len(changed):
        show = ["model", "confidence"] + [c for c in changed.columns
                                          if c.endswith(("_raw", "_sens"))]
        log.info("Rows with any change in breaches / test result / Basel zone:\n%s",
                 changed[show].T.to_string(header=False))
    else:
        log.info("No breach count, test decision (Kupiec, independence, conditional coverage) "
                 "or traffic-light result changed.")

    # 8. GARCH robustness: longer burn-in (main specification stays at 500 days)
    comp, period, max_diff = garch_min_window_check(port, var_df, table)
    comp.to_csv(out / "garch_min_window_check.csv", index=False)
    log.info("\nGARCH robustness: first fit after %d days instead of %d. Backtest %s to %s "
             "(%d days instead of %d).", config.GARCH_ROBUSTNESS_MIN_WINDOW,
             config.GARCH_MIN_WINDOW, period[0].date(), period[-1].date(), len(period), len(var_df))
    log.info("Largest VaR difference vs the main specification on the same days: %.2e", max_diff)
    log.info("%s", comp.to_string(index=False, formatters={
        "confidence": "{:.0%}".format, "breach_rate_main": "{:.2%}".format,
        "breach_rate_rob": "{:.2%}".format,
        "share_red_main": lambda v: "" if pd.isna(v) else f"{v:.0%}",
        "share_red_rob": lambda v: "" if pd.isna(v) else f"{v:.0%}"}))
    log.info("Any test decision or latest Basel zone changed: %s", bool(comp["decision_changed"].any()))

    # 9. Sanity checks
    warnings = backtest.sanity_checks(table, var_df, es_df)
    log.info("\nSanity checks:\n%s", "\n".join(warnings) if warnings else "All breach rates "
             "within 0.5x-2x of their expected level.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true",
                        help="download fresh data instead of using the saved snapshot")
    main(refresh=parser.parse_args().refresh)
