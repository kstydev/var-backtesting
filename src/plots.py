"""Charts. Kept deliberately simple: matplotlib defaults plus a few labels."""
import matplotlib

matplotlib.use("Agg")  # write files without needing a display
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd


def _shade_periods(ax, periods):
    """Grey shading for stress periods like the Covid crash and 2022."""
    for i, (label, (start, end)) in enumerate(periods.items()):
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="grey",
                   alpha=0.15 + 0.1 * i, label=label, lw=0)


def plot_var_breaches(returns, var, title, periods, path):
    """Daily portfolio returns, the -VaR line, and breaches marked in red."""
    breaches = returns < -var
    fig, ax = plt.subplots(figsize=(12, 5))
    _shade_periods(ax, periods)
    ax.plot(returns.index, returns, color="steelblue", lw=0.6, label="Daily portfolio return")
    ax.plot(var.index, -var, color="black", lw=1.0, label=f"-VaR ({var.name})")
    ax.scatter(returns.index[breaches], returns[breaches], color="red", s=14,
               zorder=3, label=f"Breach ({breaches.sum()})")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title(title)
    ax.set_ylabel("Daily return")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_var_comparison(returns, var_df, title, periods, path):
    """All VaR forecasts (as negative lines) on one chart."""
    fig, ax = plt.subplots(figsize=(12, 5))
    _shade_periods(ax, periods)
    ax.plot(returns.index, returns, color="lightgrey", lw=0.5, label="Daily portfolio return")
    for name in var_df.columns:
        ax.plot(var_df.index, -var_df[name], lw=1.0, label=f"-VaR {name}")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title(title)
    ax.set_ylabel("Daily return / -VaR")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_traffic_light(counts_df, green_max, yellow_max, title, path):
    """Rolling 250-day breach counts per model on green/yellow/red bands."""
    fig, ax = plt.subplots(figsize=(12, 5))
    top = max(counts_df.max().max() + 2, yellow_max + 4)
    ax.axhspan(-0.5, green_max + 0.5, color="green", alpha=0.12, lw=0, label="Green (0-4)")
    ax.axhspan(green_max + 0.5, yellow_max + 0.5, color="gold", alpha=0.18, lw=0,
               label="Yellow (5-9)")
    ax.axhspan(yellow_max + 0.5, top, color="red", alpha=0.12, lw=0, label="Red (10+)")
    for name in counts_df.columns:
        ax.step(counts_df.index, counts_df[name], where="post", lw=1.2, label=name)
    ax.set_ylim(-0.5, top)
    ax.set_title(title)
    ax.set_ylabel("Breaches in last 250 days")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_garch_params(params_by_model, path):
    """alpha, beta and alpha + beta at each refit date, one panel per model."""
    fig, axes = plt.subplots(len(params_by_model), 1, figsize=(12, 3.5 * len(params_by_model)),
                             sharex=True)
    for ax, (name, p) in zip(axes, params_by_model.items()):
        ax.plot(p.index, p["alpha"], marker=".", lw=1, label="alpha")
        ax.plot(p.index, p["beta"], marker=".", lw=1, label="beta")
        ax.plot(p.index, p["persistence"], marker=".", lw=1, label="alpha + beta")
        failed = p[~p["converged"]]
        if len(failed):
            ax.scatter(failed.index, failed["persistence"], color="red", zorder=3,
                       label="no convergence")
        ax.set_title(f"{name}: estimated parameters at each refit")
        ax.legend(loc="center left", fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
