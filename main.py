#!/usr/bin/env python
"""End-to-end Walmart weekly sales forecasting pipeline.

Steps
-----
1. Download (kagglehub) and preprocess the data; chronological 80/20 split.
2. ADF stationarity diagnostics on the chain-level series.
3. Leakage-safe ML feature engineering.
4. Fit Seasonal Naive (reference), SARIMA(X), Prophet and XGBoost.
5. Score RMSE / MAE / MAPE -> outputs/metrics.json (+ README results table).
6. Save actual-vs-predicted and diagnostic figures -> outputs/figures/.

Usage
-----
    python main.py                          # full run, data fetched via kagglehub
    python main.py --data-path Walmart.csv  # use a local copy of the CSV
    python main.py --models sarima xgboost  # run a subset of models
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_prep import DATE_COL, KAGGLE_HANDLE, STORE_COL, TARGET, prepare_data
from src.features import DEFAULT_LAGS, DEFAULT_WINDOWS, create_features
from src.models import ProphetForecaster, SARIMAForecaster, SeasonalNaiveForecaster, XGBoostForecaster
from src.utils import (
    calculate_metrics,
    determine_differencing_order,
    metrics_to_markdown,
    plot_actual_vs_predicted,
    plot_metrics_bar,
    plot_model_comparison,
    plot_residuals,
    run_adf_test,
    save_json,
    set_plot_style,
)

ROOT = Path(__file__).resolve().parent
MODEL_REGISTRY = {"naive": "Seasonal Naive", "sarima": "SARIMA", "prophet": "Prophet", "xgboost": "XGBoost"}
README_START, README_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"

logger = logging.getLogger("walmart")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walmart weekly sales forecasting benchmark.")
    p.add_argument("--data-path", type=Path, default=None, help="Local Walmart.csv (skips kagglehub download).")
    p.add_argument("--test-size", type=float, default=0.2, help="Fraction of weeks held out (chronological).")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs", help="Where metrics & figures go.")
    p.add_argument(
        "--models", nargs="+", choices=list(MODEL_REGISTRY), default=list(MODEL_REGISTRY), help="Models to run."
    )
    p.add_argument("--no-readme-update", action="store_true", help="Do not inject results into README.md.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    for noisy in ("cmdstanpy", "prophet", "matplotlib", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def banner(step: int, text: str) -> None:
    logger.info("=" * 70)
    logger.info("STEP %d | %s", step, text)
    logger.info("=" * 70)


def plot_series_with_split(weekly: pd.DataFrame, split_date: pd.Timestamp, path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    set_plot_style()
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(weekly.index, weekly[TARGET], color="#1f1f1f", lw=1.4)
    hol = weekly[weekly["Holiday_Flag"] == 1]
    ax.scatter(hol.index, hol[TARGET], color="#d62728", s=22, zorder=3, label="Holiday week")
    ax.axvspan(split_date, weekly.index.max(), color="#1f77b4", alpha=0.08, label="Test window")
    ax.axvline(split_date, color="#1f77b4", ls="--", lw=1)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x / 1e6:,.0f}M"))
    ax.set_title("Walmart chain-level weekly sales with chronological train/test split")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(imp: pd.Series, path: Path, top: int = 15) -> None:
    import matplotlib.pyplot as plt

    set_plot_style()
    s = imp.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(s.index, s.values / s.values.sum() * 100, color="#d62728")
    ax.set_xlabel("Share of total gain (%)")
    ax.set_title("XGBoost feature importance (gain)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def update_readme(readme: Path, table_md: str, meta: dict) -> bool:
    """Replace the block between RESULTS markers in README.md with fresh numbers."""
    if not readme.exists():
        return False
    text = readme.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        logger.warning("README markers not found; skipping README update.")
        return False
    block = (
        f"{README_START}\n"
        f"_Hold-out: **{meta['test_start']} → {meta['test_end']}** ({meta['horizon']} weeks, multi-step ahead). "
        f"Target: chain-level weekly sales (sum of {meta['n_stores']} stores). "
        f"Generated by `python main.py` on {meta['generated']}._\n\n"
        f"{table_md}\n"
        f"{README_END}"
    )
    new = re.sub(re.escape(README_START) + r".*?" + re.escape(README_END), lambda _m: block, text, flags=re.S)
    readme.write_text(new, encoding="utf-8")
    return True


def main(argv=None) -> dict:
    args = parse_args(argv)
    setup_logging(args.verbose)
    np.random.seed(args.seed)
    out_dir: Path = args.output_dir
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    # ------------------------------------------------------------------ 1
    banner(1, "Data ingestion & chronological split")
    data = prepare_data(args.data_path, test_size=args.test_size)
    weekly, train, test = data.weekly, data.train, data.test
    store_train = data.store_df[data.store_df[DATE_COL] < data.split_date]
    store_test = data.store_df[data.store_df[DATE_COL] >= data.split_date]
    assert train.index.max() < test.index.min(), "Chronological split violated."
    plot_series_with_split(weekly, data.split_date, fig_dir / "00_sales_series_split.png")

    # ------------------------------------------------------------------ 2
    banner(2, "ADF stationarity diagnostics (training data only)")
    adf_level = run_adf_test(train[TARGET])
    adf_diff = run_adf_test(train[TARGET].diff())
    adf_seasonal = run_adf_test(train[TARGET].diff(52)) if len(train) > 52 + 10 else None
    d_sel, d_hist = determine_differencing_order(train[TARGET])
    logger.info("Level series         : %s", adf_level["verdict"])
    logger.info("1st difference       : %s", adf_diff["verdict"])
    if adf_seasonal:
        logger.info("Seasonal diff (m=52) : %s", adf_seasonal["verdict"])
    logger.info("Selected differencing order d = %d", d_sel)

    # ------------------------------------------------------------------ 3
    banner(3, "Feature engineering (lags, past-only rolling stats, calendar)")
    feats = create_features(data.store_df, group_col=STORE_COL, date_col=DATE_COL)
    feats.to_csv(ROOT / "data" / "processed" / "walmart_store_level_features.csv", index=False)
    new_cols = [c for c in feats.columns if c not in data.store_df.columns]
    logger.info("Engineered %d features: %s", len(new_cols), ", ".join(new_cols))

    # ------------------------------------------------------------------ 4
    banner(4, "Model training & hold-out forecasting")
    selected = [MODEL_REGISTRY[m] for m in args.models]
    models = {
        "Seasonal Naive": (SeasonalNaiveForecaster(), train, test),
        "SARIMA": (SARIMAForecaster(), train, test),
        "Prophet": (ProphetForecaster(), train, test),
        "XGBoost": (XGBoostForecaster(random_state=args.seed), store_train, store_test),
    }
    predictions: dict[str, pd.Series] = {}
    fitted = {}
    for name in selected:
        model, tr, te = models[name]
        logger.info("Fitting %s ...", name)
        model.fit(tr)
        pred = model.predict(te).reindex(test.index)
        if pred.isna().any():
            raise RuntimeError(f"{name} produced NaN forecasts for {int(pred.isna().sum())} weeks.")
        predictions[name] = pred
        fitted[name] = model
        logger.info("%s done in %.1fs", name, model.fit_seconds or 0.0)

    # ------------------------------------------------------------------ 5
    banner(5, "Evaluation: RMSE / MAE / MAPE")
    metrics = {name: calculate_metrics(test[TARGET], pred) for name, pred in predictions.items()}
    leaderboard = sorted(metrics.items(), key=lambda kv: kv[1]["RMSE"])
    table_md = metrics_to_markdown(metrics)
    for line in table_md.splitlines():
        logger.info(line)

    xgb_store = None
    if "XGBoost" in fitted:
        sp = fitted["XGBoost"].store_predictions_.merge(
            store_test[[STORE_COL, DATE_COL, TARGET]], on=[STORE_COL, DATE_COL]
        )
        xgb_store = calculate_metrics(sp[TARGET], sp["prediction"])

    report = {
        "project": "Walmart Weekly Sales Forecasting Suite",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": {
            "source": f"kaggle:{KAGGLE_HANDLE}",
            "n_stores": int(data.store_df[STORE_COL].nunique()),
            "n_weeks": int(len(weekly)),
            "start": weekly.index.min(),
            "end": weekly.index.max(),
        },
        "split": {
            "method": "chronological (no shuffling)",
            "test_size": args.test_size,
            "train_weeks": int(len(train)),
            "test_weeks": int(len(test)),
            "train_range": [train.index.min(), train.index.max()],
            "test_range": [test.index.min(), test.index.max()],
        },
        "evaluation": {
            "target": "Chain-level Weekly_Sales (sum across stores)",
            "strategy": "multi-step-ahead forecast of the full hold-out window; exogenous drivers assumed known",
            "metrics_units": {"RMSE": "USD", "MAE": "USD", "MAPE": "percent"},
        },
        "stationarity": {
            "level": adf_level,
            "first_difference": adf_diff,
            "seasonal_difference_52": adf_seasonal,
            "selected_d": d_sel,
            "differencing_search": d_hist,
        },
        "features": {"lags": list(DEFAULT_LAGS), "rolling_windows": list(DEFAULT_WINDOWS), "engineered": new_cols},
        "models": {
            name: {
                **{k: round(v, 4) for k, v in metrics[name].items()},
                "fit_seconds": round(fitted[name].fit_seconds or 0.0, 2),
                "params": fitted[name].get_params_summary(),
            }
            for name in predictions
        },
        "leaderboard": [
            {"rank": i, "model": n, **{k: round(v, 4) for k, v in m.items()}} for i, (n, m) in enumerate(leaderboard, 1)
        ],
        "best_model": leaderboard[0][0],
    }
    if xgb_store:
        report["models"]["XGBoost"]["store_level_metrics"] = {k: round(v, 4) for k, v in xgb_store.items()}
    save_json(report, out_dir / "metrics.json")
    pd.DataFrame({"Actual": test[TARGET], **predictions}).to_csv(out_dir / "predictions.csv", index_label=DATE_COL)
    logger.info("Saved %s and predictions.csv", out_dir / "metrics.json")

    # ------------------------------------------------------------------ 6
    banner(6, "Figures")
    for i, (name, pred) in enumerate(predictions.items(), start=1):
        slug = name.lower().replace(" ", "_")
        plot_actual_vs_predicted(
            train[TARGET], test[TARGET], pred, name, metrics[name], fig_dir / f"{i:02d}_actual_vs_pred_{slug}.png"
        )
    plot_model_comparison(test[TARGET], predictions, fig_dir / "10_model_comparison.png")
    plot_metrics_bar(metrics, fig_dir / "11_metrics_comparison.png")
    plot_residuals(test[TARGET], predictions, fig_dir / "12_residuals.png")
    if "XGBoost" in fitted:
        plot_feature_importance(fitted["XGBoost"].feature_importance(), fig_dir / "13_xgboost_feature_importance.png")
    if "Prophet" in fitted:
        fitted["Prophet"].plot_components(fig_dir / "14_prophet_components.png")
    logger.info("Figures written to %s", fig_dir)

    if not args.no_readme_update:
        meta = {
            "test_start": test.index.min().date(),
            "test_end": test.index.max().date(),
            "horizon": len(test),
            "n_stores": report["dataset"]["n_stores"],
            "generated": datetime.now().strftime("%Y-%m-%d"),
        }
        if update_readme(ROOT / "README.md", table_md, meta):
            logger.info("README.md results table refreshed.")

    logger.info("Best model: %s | total runtime %.1fs", report["best_model"], time.perf_counter() - t_start)
    return report


if __name__ == "__main__":
    main()
