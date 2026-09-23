"""Leakage-safe feature engineering for the supervised (XGBoost) model.

Every target-derived feature for week *t* is computed only from target values
at weeks <= t-1:

* lags use ``shift(k)`` with k >= 1;
* rolling statistics are computed on ``shift(1)`` of the target, so the
  window for week t spans [t-w, t-1] and never includes week t itself.

For panel data (one row per Store x Date) all shifts/rolls are applied
within each store so information never bleeds across series.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

DEFAULT_LAGS: tuple[int, ...] = (1, 4, 52)  # last week, ~last month, same week last year
DEFAULT_WINDOWS: tuple[int, ...] = (4, 12)  # ~monthly and ~quarterly windows
CALENDAR_COLS = ["Month", "Day_of_Week", "Quarter", "Week_of_Year", "Year"]


def _date_series(df: pd.DataFrame, date_col: str) -> pd.Series:
    if date_col in df.columns:
        return pd.to_datetime(df[date_col])
    if isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(df.index, index=df.index)
    raise KeyError(f"'{date_col}' not found as a column and index is not a DatetimeIndex.")


def add_calendar_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """Add Month, Day_of_Week, Quarter, Week_of_Year and Year.

    Note: every Walmart ``Date`` is a Friday, so ``Day_of_Week`` is constant
    (=4) on this dataset; it is kept for schema completeness and generality.
    """
    out = df.copy()
    d = _date_series(out, date_col)
    out["Month"] = d.dt.month.astype(int).values
    out["Day_of_Week"] = d.dt.dayofweek.astype(int).values
    out["Quarter"] = d.dt.quarter.astype(int).values
    out["Week_of_Year"] = d.dt.isocalendar().week.astype(int).values
    out["Year"] = d.dt.year.astype(int).values
    return out


def _grouped(df: pd.DataFrame, target: str, group_col: str | None):
    return df.groupby(group_col, sort=False)[target] if group_col else df[target]


def add_lag_features(
    df: pd.DataFrame,
    target: str = "Weekly_Sales",
    lags: Iterable[int] = DEFAULT_LAGS,
    group_col: str | None = None,
) -> pd.DataFrame:
    """Add ``lag_k`` = target at t-k (within ``group_col`` if given)."""
    out = df.copy()
    for k in lags:
        if k < 1:
            raise ValueError(f"Lag must be >= 1 to avoid leakage, got {k}.")
        out[f"lag_{k}"] = _grouped(out, target, group_col).shift(k)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    target: str = "Weekly_Sales",
    windows: Iterable[int] = DEFAULT_WINDOWS,
    group_col: str | None = None,
) -> pd.DataFrame:
    """Add ``roll_mean_w`` / ``roll_std_w`` over weeks [t-w, t-1] (past only)."""
    out = df.copy()
    for w in windows:
        if w < 2:
            raise ValueError("Rolling window must be >= 2 (std is undefined for w=1).")

        def _mean(s, w=w):
            return s.shift(1).rolling(window=w, min_periods=w).mean()

        def _std(s, w=w):
            return s.shift(1).rolling(window=w, min_periods=w).std()

        if group_col:
            g = out.groupby(group_col, sort=False)[target]
            out[f"roll_mean_{w}"] = g.transform(_mean)
            out[f"roll_std_{w}"] = g.transform(_std)
        else:
            out[f"roll_mean_{w}"] = _mean(out[target])
            out[f"roll_std_{w}"] = _std(out[target])
    return out


def create_features(
    df: pd.DataFrame,
    target: str = "Weekly_Sales",
    lags: Sequence[int] = DEFAULT_LAGS,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    group_col: str | None = None,
    date_col: str = "Date",
) -> pd.DataFrame:
    """Full feature pipeline: calendar + lags + past-only rolling statistics.

    The frame is sorted chronologically (within group) before any shifting.
    Rows are *not* dropped: early rows have NaN lags, which XGBoost handles
    natively; callers may ``dropna`` if their model requires it.
    """
    out = df.copy()
    if date_col in out.columns:
        sort_cols = [group_col, date_col] if group_col else [date_col]
        out = out.sort_values(sort_cols)
    else:
        out = out.sort_index()
    out = add_calendar_features(out, date_col=date_col)
    out = add_lag_features(out, target, lags, group_col)
    out = add_rolling_features(out, target, windows, group_col)
    return out


def get_feature_columns(
    df: pd.DataFrame,
    target: str = "Weekly_Sales",
    exclude: Iterable[str] = ("Date", "n_stores"),
) -> list[str]:
    """All model inputs = every column except the target, date and bookkeeping cols."""
    drop = {target, *exclude}
    return [c for c in df.columns if c not in drop]
