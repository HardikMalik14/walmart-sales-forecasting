"""Shared utilities: stationarity testing, evaluation metrics, and plotting."""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")  # headless-safe backend for CLI / CI runs

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)

# Consistent colour per model across every figure.
MODEL_COLORS = {
    "Actual": "#1f1f1f",
    "Seasonal Naive": "#9e9e9e",
    "SARIMA": "#1f77b4",
    "Prophet": "#2ca02c",
    "XGBoost": "#d62728",
}


# --------------------------------------------------------------------------- #
# Stationarity
# --------------------------------------------------------------------------- #
def run_adf_test(series: pd.Series, significance: float = 0.05, autolag: str = "AIC") -> dict:
    """Run the Augmented Dickey-Fuller test for a unit root.

    H0: the series has a unit root (non-stationary).
    H1: the series is stationary.

    Returns
    -------
    dict with keys ``adf_statistic``, ``p_value``, ``used_lag``, ``n_obs``,
    ``critical_values`` (dict of "1%", "5%", "10%"), ``is_stationary`` (bool)
    and a human-readable ``verdict``.
    """
    clean = pd.Series(series).dropna().astype(float)
    if len(clean) < 10:
        raise ValueError(f"ADF test needs at least 10 observations, got {len(clean)}.")
    if clean.nunique() <= 1:
        raise ValueError("ADF test is undefined for a constant series.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # statsmodels>=0.15 result-object notice
        stat, p_value, used_lag, n_obs, crit, _ = adfuller(clean.values, autolag=autolag)
    is_stationary = bool(p_value < significance)
    verdict = (
        f"Stationary (reject H0 at {significance:.0%}; p={p_value:.4f})"
        if is_stationary
        else f"Non-stationary (fail to reject H0 at {significance:.0%}; p={p_value:.4f})"
    )
    return {
        "adf_statistic": float(stat),
        "p_value": float(p_value),
        "used_lag": int(used_lag),
        "n_obs": int(n_obs),
        "critical_values": {k: float(v) for k, v in crit.items()},
        "is_stationary": is_stationary,
        "verdict": verdict,
    }


def determine_differencing_order(
    series: pd.Series, max_d: int = 2, significance: float = 0.05
) -> tuple[int, list[dict]]:
    """Return the smallest d in [0, max_d] for which the d-th difference passes ADF.

    Also returns the list of ADF results for every order tested, so the
    decision is fully auditable in ``metrics.json``.
    """
    history: list[dict] = []
    current = pd.Series(series).dropna().astype(float)
    for d in range(max_d + 1):
        result = run_adf_test(current, significance=significance)
        result["d"] = d
        history.append(result)
        if result["is_stationary"]:
            return d, history
        current = current.diff().dropna()
    logger.warning("Series still non-stationary after %d differences; using d=%d.", max_d, max_d)
    return max_d, history


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def calculate_metrics(y_true, y_pred) -> dict:
    """Compute RMSE, MAE and MAPE (%) between two aligned arrays.

    MAPE ignores observations where the actual value is exactly zero to avoid
    division by zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("Cannot compute metrics on empty arrays.")
    if np.isnan(y_pred).any():
        raise ValueError("Predictions contain NaN values.")

    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    nonzero = y_true != 0
    mape = float(np.mean(np.abs(err[nonzero] / y_true[nonzero])) * 100) if nonzero.any() else float("nan")
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}


def save_json(obj: Mapping, path: str | Path) -> Path:
    """Write a dict to pretty-printed JSON, creating parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.strftime("%Y-%m-%d")
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serialisable")


def metrics_to_markdown(metrics: Mapping[str, Mapping[str, float]]) -> str:
    """Render ``{model: {RMSE, MAE, MAPE}}`` as a Markdown table sorted by RMSE."""
    rows = sorted(metrics.items(), key=lambda kv: kv[1]["RMSE"])
    best = rows[0][0] if rows else None
    lines = [
        "| Rank | Model | RMSE ($) | MAE ($) | MAPE (%) |",
        "|:---:|:---|---:|---:|---:|",
    ]
    for rank, (name, m) in enumerate(rows, start=1):
        label = f"**{name}**" if name == best else name
        lines.append(f"| {rank} | {label} | {m['RMSE']:,.0f} | {m['MAE']:,.0f} | {m['MAPE']:.2f} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.titleweight": "bold",
            "font.size": 10,
        }
    )


def _millions(x, _pos=None):
    """Axis formatter: $56M for large values, $0.25M for small ones."""
    v = x / 1e6
    if abs(v) >= 10:
        return f"${v:,.0f}M"
    txt = f"{v:,.2f}".rstrip("0").rstrip(".")
    return f"${txt}M"


def plot_actual_vs_predicted(
    train: pd.Series,
    test: pd.Series,
    predictions: pd.Series,
    model_name: str,
    metrics: Mapping[str, float] | None = None,
    save_path: str | Path | None = None,
    history_weeks: int = 60,
):
    """Plot recent training history, the hold-out actuals and one model's forecast."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(12, 4.5))
    hist = train.iloc[-history_weeks:]
    ax.plot(hist.index, hist.values, color="#7f7f7f", lw=1.3, label="Train (history)")
    ax.plot(test.index, test.values, color=MODEL_COLORS["Actual"], lw=2, label="Actual (test)")
    ax.plot(
        predictions.index,
        predictions.values,
        color=MODEL_COLORS.get(model_name, "#ff7f0e"),
        lw=2,
        ls="--",
        marker="o",
        ms=3,
        label=f"{model_name} forecast",
    )
    ax.axvline(test.index[0], color="#bbbbbb", ls=":", lw=1)
    title = f"{model_name}: actual vs. predicted weekly sales"
    if metrics:
        title += (  # "\\$" stops matplotlib parsing "$...$" pairs as mathtext
            f"  |  RMSE \\${metrics['RMSE'] / 1e6:,.2f}M · MAE \\${metrics['MAE'] / 1e6:,.2f}M"
            f" · MAPE {metrics['MAPE']:.2f}%"
        )
    ax.set_title(title, fontsize=11)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_millions))
    ax.set_ylabel("Weekly sales")
    ax.legend(loc="upper left", frameon=False, ncol=3)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_model_comparison(
    test: pd.Series,
    predictions: Mapping[str, pd.Series],
    save_path: str | Path | None = None,
):
    """Overlay every model's forecast against the hold-out actuals."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(test.index, test.values, color=MODEL_COLORS["Actual"], lw=2.4, label="Actual")
    for name, pred in predictions.items():
        ax.plot(
            pred.index,
            pred.values,
            color=MODEL_COLORS.get(name),
            lw=1.8,
            ls="--" if name != "Seasonal Naive" else ":",
            label=name,
        )
    ax.set_title("Hold-out forecast comparison: all models vs. actual")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_millions))
    ax.set_ylabel("Weekly sales")
    ax.legend(frameon=False, ncol=len(predictions) + 1, loc="upper left")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_metrics_bar(metrics: Mapping[str, Mapping[str, float]], save_path: str | Path | None = None):
    """Side-by-side bar charts of RMSE, MAE and MAPE per model."""
    set_plot_style()
    names = sorted(metrics, key=lambda n: metrics[n]["RMSE"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, key in zip(axes, ["RMSE", "MAE", "MAPE"]):
        vals = [metrics[n][key] for n in names]
        bars = ax.bar(names, vals, color=[MODEL_COLORS.get(n, "#888") for n in names])
        ax.set_title(key)
        ax.tick_params(axis="x", rotation=20)
        for b, v in zip(bars, vals):
            txt = f"{v:.2f}%" if key == "MAPE" else f"\\${v / 1e6:.2f}M"
            ax.annotate(txt, (b.get_x() + b.get_width() / 2, b.get_height()), ha="center", va="bottom", fontsize=8)
        if key != "MAPE":
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_millions))
    fig.suptitle("Benchmark error metrics (lower is better)", fontweight="bold")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_residuals(test: pd.Series, predictions: Mapping[str, pd.Series], save_path: str | Path | None = None):
    """Forecast residuals (actual - predicted) over the hold-out window."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(12, 4))
    for name, pred in predictions.items():
        resid = test - pred.reindex(test.index)
        ax.plot(resid.index, resid.values, color=MODEL_COLORS.get(name), lw=1.5, marker="o", ms=2.5, label=name)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Forecast residuals on hold-out (actual − predicted)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_millions))
    ax.legend(frameon=False, ncol=len(predictions))
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig
