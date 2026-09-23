"""Data ingestion, cleaning, aggregation and chronological train/test splitting.

Dataset: Walmart Sales (Kaggle handle ``mikhail1681/walmart-sales``) - 45 stores,
143 weekly observations per store (2010-02-05 to 2012-10-26) with columns::

    Store, Date, Weekly_Sales, Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment

Each ``Date`` is a Friday marking the week that ends on that day.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

KAGGLE_HANDLE = "mikhail1681/walmart-sales"
RAW_FILENAME = "Walmart.csv"

DATE_COL = "Date"
STORE_COL = "Store"
TARGET = "Weekly_Sales"
EXOG_COLS = ["CPI", "Fuel_Price", "Unemployment", "Holiday_Flag"]
REQUIRED_COLS = [STORE_COL, DATE_COL, TARGET, "Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment"]
WEEKLY_FREQ = "W-FRI"


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
def download_dataset(raw_dir: str | Path = RAW_DIR, force: bool = False) -> Path:
    """Fetch the Walmart CSV with kagglehub and copy it into ``data/raw``.

    kagglehub caches downloads under ``~/.cache/kagglehub``; we copy the CSV
    into the project so every downstream step reads from a stable location.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / RAW_FILENAME
    if target.exists() and not force:
        logger.info("Raw data already present at %s (use force=True to refresh).", target)
        return target

    import kagglehub  # imported lazily so offline unit tests don't need it

    logger.info("Downloading '%s' via kagglehub ...", KAGGLE_HANDLE)
    cache_dir = Path(kagglehub.dataset_download(KAGGLE_HANDLE))
    csvs = sorted(cache_dir.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found in kagglehub download at {cache_dir}")
    source = next((c for c in csvs if c.name.lower() == RAW_FILENAME.lower()), csvs[0])
    shutil.copy2(source, target)
    logger.info("Copied %s -> %s", source, target)
    return target


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Read the raw CSV, validate its schema and parse the ``dd-mm-YYYY`` dates."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Raw data is missing required columns: {missing}")

    # The Kaggle file uses day-first dates (e.g. 05-02-2010 = 5 Feb 2010).
    parsed = pd.to_datetime(df[DATE_COL], format="%d-%m-%Y", errors="coerce")
    if parsed.isna().any():  # fall back for alternative exports (ISO etc.)
        parsed = parsed.fillna(pd.to_datetime(df.loc[parsed.isna(), DATE_COL], dayfirst=True, errors="coerce"))
    if parsed.isna().any():
        bad = df.loc[parsed.isna(), DATE_COL].head().tolist()
        raise ValueError(f"Unparseable dates, e.g. {bad}")
    df[DATE_COL] = parsed
    return df.sort_values([STORE_COL, DATE_COL]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Cleaning & aggregation
# --------------------------------------------------------------------------- #
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce dtypes, drop duplicates and fill small gaps per store.

    * duplicate (Store, Date) rows are dropped (first kept);
    * rows without a target are dropped;
    * exogenous gaps are forward/back-filled within each store.
    """
    out = df.copy()
    n0 = len(out)
    out = out.drop_duplicates(subset=[STORE_COL, DATE_COL], keep="first")
    out = out.dropna(subset=[TARGET])
    out[STORE_COL] = out[STORE_COL].astype(int)
    out["Holiday_Flag"] = out["Holiday_Flag"].fillna(0).astype(int)
    num_cols = ["Temperature", "Fuel_Price", "CPI", "Unemployment", TARGET]
    out[num_cols] = out[num_cols].astype(float)
    exog = ["Temperature", "Fuel_Price", "CPI", "Unemployment"]
    out[exog] = out.groupby(STORE_COL)[exog].transform(lambda s: s.ffill().bfill())
    out = out.sort_values([STORE_COL, DATE_COL]).reset_index(drop=True)
    if len(out) != n0:
        logger.info("clean_data removed %d rows (duplicates / missing target).", n0 - len(out))
    return out


def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate store-level rows into one chain-wide weekly series.

    ``Weekly_Sales`` is summed across stores; macro drivers are averaged;
    ``Holiday_Flag`` is 1 if any store flags the week. The result is indexed
    by a regular ``W-FRI`` DatetimeIndex (gaps interpolated if any).
    """
    agg = (
        df.groupby(DATE_COL)
        .agg(
            Weekly_Sales=(TARGET, "sum"),
            Holiday_Flag=("Holiday_Flag", "max"),
            Temperature=("Temperature", "mean"),
            Fuel_Price=("Fuel_Price", "mean"),
            CPI=("CPI", "mean"),
            Unemployment=("Unemployment", "mean"),
            n_stores=(STORE_COL, "nunique"),
        )
        .sort_index()
    )
    full_index = pd.date_range(agg.index.min(), agg.index.max(), freq=WEEKLY_FREQ)
    if len(full_index) != len(agg):
        logger.warning("Weekly index has %d gaps; interpolating.", len(full_index) - len(agg))
        agg = agg.reindex(full_index).interpolate(limit_direction="both")
        agg["Holiday_Flag"] = agg["Holiday_Flag"].round().astype(int)
    agg.index = pd.DatetimeIndex(agg.index, freq=WEEKLY_FREQ, name=DATE_COL)
    agg["Holiday_Flag"] = agg["Holiday_Flag"].astype(int)
    return agg


# --------------------------------------------------------------------------- #
# Chronological splitting (no shuffling, ever)
# --------------------------------------------------------------------------- #
def get_split_date(dates, test_size: float = 0.2) -> pd.Timestamp:
    """Return the first test date so that ~``test_size`` of unique weeks are held out."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be in (0, 1).")
    unique = pd.DatetimeIndex(sorted(pd.unique(pd.DatetimeIndex(dates))))
    n_test = max(1, int(round(len(unique) * test_size)))
    if n_test >= len(unique):
        raise ValueError("Not enough periods to create a non-empty training set.")
    return unique[len(unique) - n_test]


def chronological_split(
    df: pd.DataFrame, test_size: float = 0.2, date_col: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time: every training date strictly precedes every test date.

    Works on a date-indexed frame (``date_col=None``) or a long/panel frame
    with an explicit date column (all stores share the same cut-off week).
    """
    dates = df.index if date_col is None else df[date_col]
    cutoff = get_split_date(dates, test_size)
    mask = (df.index if date_col is None else df[date_col]) < cutoff
    train, test = df.loc[mask].copy(), df.loc[~mask].copy()
    return train, test


# --------------------------------------------------------------------------- #
# End-to-end helper
# --------------------------------------------------------------------------- #
@dataclass
class PreparedData:
    store_df: pd.DataFrame  # cleaned long/panel data (Store x Date)
    weekly: pd.DataFrame  # chain-level weekly series (Date index)
    train: pd.DataFrame  # chain-level train split
    test: pd.DataFrame  # chain-level test split
    split_date: pd.Timestamp
    raw_path: Path


def prepare_data(
    data_path: str | Path | None = None,
    test_size: float = 0.2,
    processed_dir: str | Path = PROCESSED_DIR,
    save: bool = True,
) -> PreparedData:
    """Download (if needed), clean, aggregate and split the dataset."""
    raw_path = Path(data_path) if data_path else download_dataset()
    store_df = clean_data(load_raw_data(raw_path))
    weekly = aggregate_weekly(store_df)
    train, test = chronological_split(weekly, test_size=test_size)
    split_date = test.index[0]

    if save:
        processed_dir = Path(processed_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)
        store_df.to_csv(processed_dir / "walmart_store_level_clean.csv", index=False)
        weekly.to_csv(processed_dir / "walmart_weekly_aggregate.csv")
        logger.info("Saved processed data to %s", processed_dir)

    logger.info(
        "Data ready: %d stores, %d weeks (%s -> %s). Train %d weeks | Test %d weeks (from %s).",
        store_df[STORE_COL].nunique(),
        len(weekly),
        weekly.index.min().date(),
        weekly.index.max().date(),
        len(train),
        len(test),
        split_date.date(),
    )
    return PreparedData(store_df, weekly, train, test, split_date, raw_path)
