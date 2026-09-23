"""Unit tests for data loading, feature engineering, utilities and models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_prep import (
    REQUIRED_COLS,
    aggregate_weekly,
    chronological_split,
    clean_data,
    get_split_date,
    load_raw_data,
    prepare_data,
)
from src.features import (
    CALENDAR_COLS,
    add_lag_features,
    add_rolling_features,
    create_features,
    get_feature_columns,
)
from src.models import (
    ProphetForecaster,
    SARIMAForecaster,
    SeasonalNaiveForecaster,
    XGBoostForecaster,
    build_us_holiday_calendar,
)
from src.utils import calculate_metrics, determine_differencing_order, metrics_to_markdown, run_adf_test

from .conftest import WALMART_HOLIDAYS


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
class TestDataLoading:
    def test_load_returns_valid_dataframe(self, raw_csv):
        df = load_raw_data(raw_csv)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert set(REQUIRED_COLS).issubset(df.columns)
        assert pd.api.types.is_datetime64_any_dtype(df["Date"])
        assert df[REQUIRED_COLS].isna().sum().sum() == 0

    def test_dates_parsed_day_first(self, raw_csv):
        df = load_raw_data(raw_csv)
        # "05-02-2010" must be 5 Feb 2010, not 2 May 2010.
        assert df["Date"].min() == pd.Timestamp("2010-02-05")
        assert (df["Date"].dt.dayofweek == 4).all()  # every Walmart week ends on a Friday

    def test_sorted_by_store_then_date(self, raw_csv):
        df = load_raw_data(raw_csv)
        assert df.equals(df.sort_values(["Store", "Date"]).reset_index(drop=True))

    def test_missing_columns_raise(self, tmp_path):
        bad = tmp_path / "bad.csv"
        pd.DataFrame({"Store": [1], "Date": ["05-02-2010"]}).to_csv(bad, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            load_raw_data(bad)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_raw_data(tmp_path / "nope.csv")

    def test_clean_drops_duplicates(self, raw_csv):
        df = load_raw_data(raw_csv)
        dup = pd.concat([df, df.head(3)], ignore_index=True)
        cleaned = clean_data(dup)
        assert len(cleaned) == len(df)
        assert not cleaned.duplicated(["Store", "Date"]).any()

    def test_aggregate_weekly(self, raw_csv):
        df = clean_data(load_raw_data(raw_csv))
        weekly = aggregate_weekly(df)
        assert weekly.index.is_monotonic_increasing and weekly.index.is_unique
        assert weekly.index.freqstr == "W-FRI"
        assert len(weekly) == df["Date"].nunique()
        assert np.isclose(weekly["Weekly_Sales"].sum(), df["Weekly_Sales"].sum())
        assert set(weekly["Holiday_Flag"].unique()) <= {0, 1}

    def test_prepare_data_end_to_end(self, raw_csv, tmp_path):
        data = prepare_data(raw_csv, test_size=0.2, processed_dir=tmp_path / "processed")
        assert (tmp_path / "processed" / "walmart_weekly_aggregate.csv").exists()
        assert len(data.train) + len(data.test) == len(data.weekly)
        assert data.train.index.max() < data.test.index.min() == data.split_date


class TestChronologicalSplit:
    def test_no_temporal_overlap_indexed(self, raw_csv):
        weekly = aggregate_weekly(clean_data(load_raw_data(raw_csv)))
        train, test = chronological_split(weekly, test_size=0.2)
        assert train.index.max() < test.index.min()
        assert len(test) == round(len(weekly) * 0.2)
        # zero shuffling: concatenation reproduces the original order exactly
        pd.testing.assert_frame_equal(pd.concat([train, test]), weekly, check_freq=False)

    def test_panel_split_shares_cutoff(self, raw_csv):
        df = clean_data(load_raw_data(raw_csv))
        train, test = chronological_split(df, test_size=0.2, date_col="Date")
        assert train["Date"].max() < test["Date"].min()
        # every store is cut at the same week
        assert test.groupby("Store")["Date"].min().nunique() == 1

    @pytest.mark.parametrize("bad", [0, 1, -0.1, 1.5])
    def test_invalid_test_size(self, bad):
        with pytest.raises(ValueError):
            get_split_date(pd.date_range("2020-01-03", periods=10, freq="W-FRI"), bad)


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
@pytest.fixture()
def store_df(raw_csv):
    return clean_data(load_raw_data(raw_csv))


class TestFeatures:
    def test_expected_columns(self, store_df):
        feats = create_features(store_df, group_col="Store")
        expected = {"lag_1", "lag_4", "lag_52", "roll_mean_4", "roll_std_4", "roll_mean_12", "roll_std_12"}
        assert expected.issubset(feats.columns)
        assert set(CALENDAR_COLS).issubset(feats.columns)
        assert len(feats) == len(store_df)  # no rows silently dropped

    def test_calendar_values(self, store_df):
        feats = create_features(store_df, group_col="Store")
        assert feats["Month"].between(1, 12).all()
        assert feats["Quarter"].between(1, 4).all()
        assert feats["Week_of_Year"].between(1, 53).all()
        assert (feats["Day_of_Week"] == 4).all()  # Fridays

    @pytest.mark.parametrize("k", [1, 4, 52])
    def test_lag_values_are_exact_past_values(self, store_df, k):
        feats = create_features(store_df, group_col="Store")
        for _, g in feats.groupby("Store"):
            g = g.sort_values("Date")
            expected = g["Weekly_Sales"].shift(k)
            pd.testing.assert_series_equal(g[f"lag_{k}"], expected, check_names=False)
            assert g[f"lag_{k}"].iloc[:k].isna().all()  # no cross-store bleed at series start

    def test_rolling_uses_only_past_values(self, store_df):
        feats = create_features(store_df, group_col="Store")
        g = feats[feats["Store"] == 1].sort_values("Date").reset_index(drop=True)
        t = 20
        window = g.loc[t - 4 : t - 1, "Weekly_Sales"]  # weeks t-4 .. t-1
        assert np.isclose(g.loc[t, "roll_mean_4"], window.mean())
        assert np.isclose(g.loc[t, "roll_std_4"], window.std())

    def test_no_future_leakage(self, store_df):
        """Changing the target at week t must not alter any feature at weeks <= t."""
        base = create_features(store_df, group_col="Store")
        dates = np.sort(store_df["Date"].unique())
        t_date = dates[80]
        tampered = store_df.copy()
        tampered.loc[tampered["Date"] >= t_date, "Weekly_Sales"] *= 1000.0
        after = create_features(tampered, group_col="Store")
        cols = [c for c in get_feature_columns(base) if c.startswith(("lag_", "roll_"))]
        mask = base["Date"] <= t_date
        pd.testing.assert_frame_equal(base.loc[mask, cols], after.loc[mask, cols])
        # ... but the week right after t *should* see the change via lag_1
        nxt = base["Date"] == dates[81]
        assert not np.allclose(base.loc[nxt, "lag_1"], after.loc[nxt, "lag_1"])

    def test_invalid_lag_or_window(self, store_df):
        with pytest.raises(ValueError):
            add_lag_features(store_df, lags=[0])
        with pytest.raises(ValueError):
            add_rolling_features(store_df, windows=[1])

    def test_feature_columns_exclude_target_and_date(self, store_df):
        cols = get_feature_columns(create_features(store_df, group_col="Store"))
        assert "Weekly_Sales" not in cols and "Date" not in cols


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
class TestADF:
    def test_structure(self):
        res = run_adf_test(pd.Series(np.random.default_rng(1).normal(size=200)))
        for key in ("adf_statistic", "p_value", "used_lag", "n_obs", "critical_values", "is_stationary", "verdict"):
            assert key in res
        assert set(res["critical_values"]) == {"1%", "5%", "10%"}
        assert isinstance(res["is_stationary"], bool)
        assert 0.0 <= res["p_value"] <= 1.0
        assert isinstance(res["verdict"], str)

    def test_white_noise_is_stationary(self):
        res = run_adf_test(pd.Series(np.random.default_rng(2).normal(size=300)))
        assert res["is_stationary"]

    def test_random_walk_is_not_stationary_and_needs_one_difference(self):
        rw = pd.Series(np.cumsum(np.random.default_rng(3).normal(size=300)))
        assert not run_adf_test(rw)["is_stationary"]
        d, history = determine_differencing_order(rw)
        assert d == 1 and len(history) == 2 and history[-1]["is_stationary"]

    def test_bad_inputs(self):
        with pytest.raises(ValueError):
            run_adf_test(pd.Series([1.0] * 50))
        with pytest.raises(ValueError):
            run_adf_test(pd.Series([1.0, 2.0, 3.0]))


class TestMetrics:
    def test_known_values(self):
        m = calculate_metrics([100, 200, 300], [110, 190, 330])
        assert set(m) == {"RMSE", "MAE", "MAPE"}
        assert np.isclose(m["MAE"], (10 + 10 + 30) / 3)
        assert np.isclose(m["RMSE"], np.sqrt((100 + 100 + 900) / 3))
        assert np.isclose(m["MAPE"], np.mean([10 / 100, 10 / 200, 30 / 300]) * 100)

    def test_perfect_forecast(self):
        m = calculate_metrics([1, 2, 3], [1, 2, 3])
        assert m == {"RMSE": 0.0, "MAE": 0.0, "MAPE": 0.0}

    def test_errors(self):
        with pytest.raises(ValueError):
            calculate_metrics([1, 2], [1])
        with pytest.raises(ValueError):
            calculate_metrics([1, 2], [1, np.nan])

    def test_markdown_table(self):
        md = metrics_to_markdown({"A": {"RMSE": 2, "MAE": 1, "MAPE": 1}, "B": {"RMSE": 1, "MAE": 1, "MAPE": 1}})
        assert md.splitlines()[2].startswith("| 1 | **B**")


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
@pytest.fixture()
def split(raw_csv, tmp_path):
    return prepare_data(raw_csv, test_size=0.2, processed_dir=tmp_path / "p", save=False)


class TestModels:
    def test_holiday_calendar_matches_walmart_flags(self):
        cal = build_us_holiday_calendar(pd.Timestamp("2010-02-05"), pd.Timestamp("2012-10-26"))
        mapped = set(cal["ds"].dt.strftime("%Y-%m-%d"))
        assert WALMART_HOLIDAYS <= mapped
        assert (cal["ds"].dt.dayofweek == 4).all()

    def test_seasonal_naive(self, split):
        pred = SeasonalNaiveForecaster().fit(split.train).predict(split.test)
        assert pred.index.equals(split.test.index)
        assert np.allclose(pred.values[:5], split.train["Weekly_Sales"].values[-52:-47])

    def test_xgboost_recursive_forecast_never_sees_test_actuals(self, split):
        sd = split.store_df
        tr, te = sd[sd["Date"] < split.split_date], sd[sd["Date"] >= split.split_date]
        model = XGBoostForecaster(params={"n_estimators": 200}, validation_weeks=8).fit(tr)
        pred = model.predict(te)
        assert len(pred) == len(split.test) and not pred.isna().any()
        # Corrupt the test targets: forecasts must be byte-identical.
        te_bad = te.assign(Weekly_Sales=-1.0)
        pd.testing.assert_series_equal(pred, model.predict(te_bad))
        assert {"lag_1", "lag_4", "lag_52", "Store"} <= set(model.feature_cols_)

    def test_xgboost_store_scaling_returns_usd_and_excludes_trend_covariates(self, split):
        sd = split.store_df
        tr, te = sd[sd["Date"] < split.split_date], sd[sd["Date"] >= split.split_date]
        model = XGBoostForecaster(target_scaling="store_mean", params={"n_estimators": 100}, validation_weeks=8).fit(tr)
        pred = model.predict(te)
        actual = te.groupby("Date")["Weekly_Sales"].sum()
        # predictions are rescaled back to dollars (same order of magnitude as actuals)
        assert 0.5 < pred.sum() / actual.sum() < 2.0
        assert not {"Year", "CPI", "Fuel_Price", "Unemployment"} & set(model.feature_cols_)
        with pytest.raises(ValueError):
            XGBoostForecaster(target_scaling="zscore")

    def test_sarima_fits_and_forecasts(self, split):
        model = SARIMAForecaster(seasonal=False, max_p=1, max_q=1).fit(split.train)
        pred = model.predict(split.test)
        assert len(pred) == len(split.test) and np.isfinite(pred).all()
        assert model.get_params_summary()["d_from_adf"] in (0, 1, 2)

    def test_prophet_fits_and_forecasts(self, split):
        pred = ProphetForecaster().fit(split.train).predict(split.test)
        assert pred.index.equals(split.test.index) and np.isfinite(pred).all()

    def test_predict_before_fit_raises(self, split):
        with pytest.raises(RuntimeError):
            SeasonalNaiveForecaster().predict(split.test)
