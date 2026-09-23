"""Unified forecasting wrappers: Seasonal Naive, SARIMA(X), Prophet and XGBoost.

All forecasters share one interface::

    model.fit(train_df)                -> self
    model.predict(test_df)             -> pd.Series indexed by the test dates
    model.get_params_summary()         -> dict (logged to metrics.json)

Evaluation protocol (identical for every model)
------------------------------------------------
* Models are fitted on the training weeks only.
* They forecast the whole hold-out window in one go (multi-step ahead) and
  never see an actual test-period sales value. XGBoost achieves this by
  recursive forecasting: its own predictions feed the lag / rolling features.
* Exogenous drivers (CPI, Fuel_Price, Unemployment, Holiday_Flag) for the test
  weeks are treated as known, the standard "ex-post" setting for regression
  forecasters. Holiday dates are calendar facts; the macro series are assumed
  to be available from external forecasts at prediction time.
"""

from __future__ import annotations

import logging
import time
import warnings
from abc import ABC, abstractmethod
from datetime import date, timedelta
from typing import Sequence

import numpy as np
import pandas as pd

from .data_prep import DATE_COL, EXOG_COLS, STORE_COL, TARGET
from .features import DEFAULT_LAGS, DEFAULT_WINDOWS, create_features, get_feature_columns
from .utils import determine_differencing_order

logger = logging.getLogger(__name__)


class BaseForecaster(ABC):
    name: str = "Base"

    def __init__(self) -> None:
        self.fit_seconds: float | None = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> "BaseForecaster": ...

    @abstractmethod
    def predict(self, test: pd.DataFrame) -> pd.Series: ...

    def get_params_summary(self) -> dict:
        return {}

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before calling predict().")


# --------------------------------------------------------------------------- #
# 0. Seasonal naive reference
# --------------------------------------------------------------------------- #
class SeasonalNaiveForecaster(BaseForecaster):
    """y_hat(t) = y(t - m). The benchmark every model must beat."""

    name = "Seasonal Naive"

    def __init__(self, season_length: int = 52, target: str = TARGET) -> None:
        super().__init__()
        self.m = season_length
        self.target = target

    def fit(self, train: pd.DataFrame) -> "SeasonalNaiveForecaster":
        if len(train) < self.m:
            raise ValueError(f"Need at least {self.m} training weeks, got {len(train)}.")
        self.history_ = train[self.target].astype(float).copy()
        self.is_fitted = True
        self.fit_seconds = 0.0
        return self

    def predict(self, test: pd.DataFrame) -> pd.Series:
        self._check_fitted()
        values = list(self.history_.values)
        preds = []
        for _ in range(len(test)):
            nxt = values[-self.m]
            preds.append(nxt)
            values.append(nxt)
        return pd.Series(preds, index=test.index, name=self.name)

    def get_params_summary(self) -> dict:
        return {"season_length": self.m}


# --------------------------------------------------------------------------- #
# 1. SARIMA / SARIMAX (statistical baseline)
# --------------------------------------------------------------------------- #
class SARIMAForecaster(BaseForecaster):
    """SARIMAX with ADF-driven differencing and exogenous regressors.

    * ``d`` is the smallest order whose differenced training series passes
      the ADF test (p < ``significance``).
    * ``D`` (seasonal differencing, m=52) is chosen with the OCSB test via
      ``pmdarima.arima.nsdiffs`` when the sample is long enough (> 2m).
    * (p, q, P, Q) are searched stepwise by AIC with ``pmdarima.auto_arima``.
    * Exogenous regressors are z-scored with training statistics only.
    """

    name = "SARIMA"

    def __init__(
        self,
        exog_cols: Sequence[str] | None = tuple(EXOG_COLS),
        seasonal_period: int = 52,
        seasonal: bool = True,
        max_p: int = 3,
        max_q: int = 3,
        max_P: int = 1,
        max_Q: int = 1,
        significance: float = 0.05,
        target: str = TARGET,
    ) -> None:
        super().__init__()
        self.exog_cols = list(exog_cols) if exog_cols else []
        self.m = seasonal_period
        self.seasonal = seasonal
        self.max_p, self.max_q, self.max_P, self.max_Q = max_p, max_q, max_P, max_Q
        self.significance = significance
        self.target = target
        self.adf_history_: list[dict] = []

    # -- helpers ---------------------------------------------------------- #
    def _exog(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray | None:
        if not self.exog_cols:
            return None
        X = df[self.exog_cols].astype(float)
        if fit:
            self.exog_mean_ = X.mean()
            self.exog_std_ = X.std(ddof=0).replace(0, 1.0)
        return ((X - self.exog_mean_) / self.exog_std_).values

    def _seasonal_D(self, y: pd.Series) -> int:
        if not self.seasonal or len(y) <= 2 * self.m:
            return 0
        from pmdarima.arima import nsdiffs

        try:
            return int(min(1, nsdiffs(y.values, m=self.m, max_D=1, test="ocsb")))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("nsdiffs failed (%s); defaulting to D=0.", exc)
            return 0

    # -- API ------------------------------------------------------------- #
    def fit(self, train: pd.DataFrame) -> "SARIMAForecaster":
        import pmdarima as pm

        t0 = time.perf_counter()
        y = train[self.target].astype(float)
        self.d_, self.adf_history_ = determine_differencing_order(y, max_d=2, significance=self.significance)
        self.D_ = self._seasonal_D(y)
        use_seasonal = self.seasonal and len(y) > self.m + 10
        X = self._exog(train, fit=True)
        logger.info("SARIMA: ADF selected d=%d; OCSB selected D=%d (m=%d).", self.d_, self.D_, self.m)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.model_ = pm.auto_arima(
                    y.values,
                    X=X,
                    d=self.d_,
                    D=self.D_ if use_seasonal else 0,
                    seasonal=use_seasonal,
                    m=self.m if use_seasonal else 1,
                    start_p=0,
                    start_q=0,
                    max_p=self.max_p,
                    max_q=self.max_q,
                    start_P=0,
                    start_Q=0,
                    max_P=self.max_P,
                    max_Q=self.max_Q,
                    information_criterion="aic",
                    stepwise=True,
                    suppress_warnings=True,
                    error_action="ignore",
                    trace=False,
                )
            except Exception as exc:
                # Fallback: a compact, well-behaved specification.
                logger.warning("auto_arima failed (%s); falling back to SARIMAX(1,d,1)x(0,D,0,m).", exc)
                self.model_ = pm.ARIMA(
                    order=(1, self.d_, 1),
                    seasonal_order=(0, self.D_, 0, self.m) if use_seasonal else (0, 0, 0, 0),
                    suppress_warnings=True,
                ).fit(y.values, X=X)

        self.order_ = tuple(int(v) for v in self.model_.order)
        self.seasonal_order_ = tuple(int(v) for v in self.model_.seasonal_order)
        self.fit_seconds = time.perf_counter() - t0
        self.is_fitted = True
        logger.info("SARIMA: selected SARIMAX%s x %s  (AIC=%.1f)", self.order_, self.seasonal_order_, self.model_.aic())
        return self

    def predict(self, test: pd.DataFrame) -> pd.Series:
        self._check_fitted()
        X = self._exog(test)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds = self.model_.predict(n_periods=len(test), X=X)
        return pd.Series(np.asarray(preds, dtype=float), index=test.index, name=self.name)

    def get_params_summary(self) -> dict:
        return {
            "order": list(self.order_),
            "seasonal_order": list(self.seasonal_order_),
            "d_from_adf": self.d_,
            "D_from_ocsb": self.D_,
            "aic": float(self.model_.aic()),
            "exogenous": self.exog_cols,
        }


# --------------------------------------------------------------------------- #
# 2. Prophet (additive seasonal model)
# --------------------------------------------------------------------------- #
def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-th ``weekday`` (Mon=0) of ``month``; n=1 is the first."""
    d = date(year, month, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _week_ending_friday(d: date) -> pd.Timestamp:
    """Map a calendar day to the Walmart week (Sat..Fri) that contains it."""
    return pd.Timestamp(d + timedelta(days=(4 - d.weekday()) % 7))


def build_us_holiday_calendar(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """US retail holidays mapped onto Walmart's week-ending-Friday grid.

    Covers the four events Walmart flags (Super Bowl, Labor Day, Thanksgiving,
    Christmas). Windows add the pre-Christmas shopping weeks, the real peak,
    which fall *before* the flagged Christmas week.
    """
    rows = []
    for yr in range(start.year - 1, end.year + 2):
        super_bowl = _nth_weekday(yr, 2, 6, 1 if yr < 2022 else 2)  # Sunday
        labor_day = _nth_weekday(yr, 9, 0, 1)  # 1st Monday of Sep
        thanksgiving = _nth_weekday(yr, 11, 3, 4)  # 4th Thursday of Nov
        christmas = date(yr, 12, 25)
        rows += [
            ("super_bowl", _week_ending_friday(super_bowl), 0, 0),
            ("labor_day", _week_ending_friday(labor_day), 0, 0),
            ("thanksgiving", _week_ending_friday(thanksgiving), 0, 0),
            ("christmas", _week_ending_friday(christmas), -14, 0),
        ]
    hol = pd.DataFrame(rows, columns=["holiday", "ds", "lower_window", "upper_window"])
    pad = pd.Timedelta(weeks=60)
    return hol[(hol.ds >= start - pad) & (hol.ds <= end + pad)].reset_index(drop=True)


class ProphetForecaster(BaseForecaster):
    """Additive trend + yearly/monthly seasonality + US holidays + regressors."""

    name = "Prophet"

    def __init__(
        self,
        regressors: Sequence[str] | None = ("CPI", "Fuel_Price", "Unemployment"),
        seasonality_mode: str = "additive",
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
        yearly_fourier_order: int = 10,
        monthly_fourier_order: int = 3,
        target: str = TARGET,
    ) -> None:
        super().__init__()
        self.regressors = list(regressors) if regressors else []
        self.seasonality_mode = seasonality_mode
        self.cps = changepoint_prior_scale
        self.sps = seasonality_prior_scale
        self.hps = holidays_prior_scale
        self.yearly_fourier_order = yearly_fourier_order
        self.monthly_fourier_order = monthly_fourier_order
        self.target = target

    def _to_prophet(self, df: pd.DataFrame, with_y: bool) -> pd.DataFrame:
        out = pd.DataFrame({"ds": pd.DatetimeIndex(df.index)})
        if with_y:
            out["y"] = df[self.target].astype(float).values
        for r in self.regressors:
            out[r] = df[r].astype(float).values
        return out

    def fit(self, train: pd.DataFrame) -> "ProphetForecaster":
        logging.getLogger("prophet.plot").setLevel(logging.CRITICAL)  # optional plotly not needed
        from prophet import Prophet

        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
        logging.getLogger("prophet").setLevel(logging.WARNING)
        t0 = time.perf_counter()
        self.holidays_ = build_us_holiday_calendar(train.index.min(), train.index.max())
        self.model_ = Prophet(
            growth="linear",
            yearly_seasonality=self.yearly_fourier_order,
            weekly_seasonality=False,  # data are weekly totals
            daily_seasonality=False,
            holidays=self.holidays_,
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=self.cps,
            seasonality_prior_scale=self.sps,
            holidays_prior_scale=self.hps,
        )
        if self.monthly_fourier_order:
            self.model_.add_seasonality(name="monthly", period=30.5, fourier_order=self.monthly_fourier_order)
        for r in self.regressors:
            self.model_.add_regressor(r, standardize=True)
        self.model_.fit(self._to_prophet(train, with_y=True))
        self.fit_seconds = time.perf_counter() - t0
        self.is_fitted = True
        return self

    def predict(self, test: pd.DataFrame) -> pd.Series:
        self._check_fitted()
        self.forecast_ = self.model_.predict(self._to_prophet(test, with_y=False))
        return pd.Series(self.forecast_["yhat"].values, index=test.index, name=self.name)

    def plot_components(self, save_path=None):
        """Save Prophet's trend / seasonality / holiday decomposition."""
        import matplotlib.pyplot as plt

        self._check_fitted()
        full = self.model_.predict(self.model_.history.drop(columns=["y"], errors="ignore"))
        fig = self.model_.plot_components(full)
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=130)
            plt.close(fig)
        return fig

    def get_params_summary(self) -> dict:
        return {
            "seasonality_mode": self.seasonality_mode,
            "yearly_fourier_order": self.yearly_fourier_order,
            "monthly_fourier_order": self.monthly_fourier_order,
            "changepoint_prior_scale": self.cps,
            "holidays": sorted(self.holidays_["holiday"].unique().tolist()),
            "regressors": self.regressors,
        }


# --------------------------------------------------------------------------- #
# 3. XGBoost (global supervised model over all stores)
# --------------------------------------------------------------------------- #
class XGBoostForecaster(BaseForecaster):
    """Global gradient-boosted model trained on every store's history.

    One model learns from all 45 store series (Store is a categorical
    feature), which gives it ~45x more training rows than a single aggregate
    series. Forecasts are produced recursively per store and summed to the
    chain level for the benchmark.

    Tree ensembles cannot extrapolate: a covariate value above the training
    maximum lands in the last leaf learned on training data. On this dataset
    the trending level covariates (``Year``, ``CPI``, ``Fuel_Price``,
    ``Unemployment``) all move outside their training range during the test
    window, which biased one-step forecasts by roughly +10%. The model
    therefore uses only bounded, repeating inputs: lags, rolling statistics,
    calendar fields (without ``Year``), ``Holiday_Flag`` and ``Temperature``.
    Level drift is carried by the lag features themselves. This setup also
    scored best on an inner validation fold carved from the training data
    (see README, "Model selection notes").

    ``target_scaling="store_mean"`` (divide by each store's training mean) is
    available as an option but scored worse on the inner fold, so it is off
    by default.
    """

    name = "XGBoost"

    def __init__(
        self,
        lags: Sequence[int] = DEFAULT_LAGS,
        windows: Sequence[int] = DEFAULT_WINDOWS,
        exog_cols: Sequence[str] = ("Holiday_Flag", "Temperature"),
        validation_weeks: int = 12,
        target_scaling: str | None = None,
        exclude_features: Sequence[str] = ("Year",),
        params: dict | None = None,
        random_state: int = 42,
        target: str = TARGET,
    ) -> None:
        super().__init__()
        self.lags = tuple(lags)
        self.windows = tuple(windows)
        self.exog_cols = list(exog_cols)
        self.validation_weeks = validation_weeks
        if target_scaling not in (None, "store_mean"):
            raise ValueError("target_scaling must be None or 'store_mean'.")
        self.target_scaling = target_scaling
        self.exclude_features = tuple(exclude_features)
        self.target = target
        self.random_state = random_state
        self.params = {
            "n_estimators": 3000,
            "learning_rate": 0.03,
            "max_depth": 5,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "enable_categorical": True,
            "objective": "reg:squarederror",
        }
        if params:
            self.params.update(params)

    # -- feature assembly -------------------------------------------------- #
    def _base_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [STORE_COL, DATE_COL, self.target, *self.exog_cols]
        return df[cols].copy()

    def _featurize(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = create_features(
            df, target=self.target, lags=self.lags, windows=self.windows, group_col=STORE_COL, date_col=DATE_COL
        )
        feats[STORE_COL] = pd.Categorical(feats[STORE_COL], categories=self.store_categories_)
        return feats

    def _make_model(self, n_estimators: int | None = None, early_stopping: bool = False):
        from xgboost import XGBRegressor

        p = dict(self.params)
        if n_estimators is not None:
            p["n_estimators"] = n_estimators
        if early_stopping:
            p["early_stopping_rounds"] = 100
            p["eval_metric"] = "rmse"
        return XGBRegressor(random_state=self.random_state, n_jobs=-1, **p)

    # -- API ---------------------------------------------------------------- #
    def fit(self, train: pd.DataFrame) -> "XGBoostForecaster":
        """``train`` is the store-level (long) frame restricted to training weeks."""
        t0 = time.perf_counter()
        base = self._base_frame(train)
        self.store_categories_ = sorted(base[STORE_COL].unique().tolist())
        if self.target_scaling == "store_mean":
            # Training-period statistics only, so nothing leaks from the test weeks.
            self.store_scale_ = base.groupby(STORE_COL)[self.target].mean()
        else:
            self.store_scale_ = pd.Series(1.0, index=self.store_categories_)
        base[self.target] = base[self.target] / base[STORE_COL].map(self.store_scale_)
        feats = self._featurize(base)
        self.feature_cols_ = [
            c for c in get_feature_columns(feats, target=self.target) if c not in self.exclude_features
        ]
        feats = feats.dropna(subset=[f"lag_{min(self.lags)}"])  # first week per store has no history

        # Chronological inner validation for early stopping (last N train weeks).
        dates = np.sort(feats[DATE_COL].unique())
        cut = dates[-self.validation_weeks] if len(dates) > self.validation_weeks * 2 else dates[-1]
        tr, va = feats[feats[DATE_COL] < cut], feats[feats[DATE_COL] >= cut]
        probe = self._make_model(early_stopping=True)
        probe.fit(tr[self.feature_cols_], tr[self.target], eval_set=[(va[self.feature_cols_], va[self.target])], verbose=False)
        self.best_iteration_ = int(getattr(probe, "best_iteration", self.params["n_estimators"] - 1) or 0) + 1
        # Refit on *all* training weeks with the tuned number of trees.
        n_final = max(50, int(self.best_iteration_ * 1.1))
        self.model_ = self._make_model(n_estimators=n_final)
        self.model_.fit(feats[self.feature_cols_], feats[self.target], verbose=False)
        self.n_estimators_ = n_final
        self.history_ = base
        self.fit_seconds = time.perf_counter() - t0
        self.is_fitted = True
        logger.info("XGBoost: early-stopped at %d trees; refit with %d trees on %d rows.", self.best_iteration_, n_final, len(feats))
        return self

    def predict_store_level(self, test: pd.DataFrame) -> pd.DataFrame:
        """Recursive multi-step forecast per store. Returns Store, Date, prediction."""
        self._check_fitted()
        future = self._base_frame(test)
        future[self.target] = np.nan
        full = pd.concat([self.history_, future], ignore_index=True).sort_values([STORE_COL, DATE_COL])
        full = full.reset_index(drop=True)
        for d in np.sort(future[DATE_COL].unique()):
            feats = self._featurize(full)
            step = feats[feats[DATE_COL] == d]
            yhat = self.model_.predict(step[self.feature_cols_])
            full.loc[step.index, self.target] = yhat  # feed prediction back as history
        out = full[full[DATE_COL].isin(future[DATE_COL].unique())][[STORE_COL, DATE_COL, self.target]].copy()
        unknown = set(out[STORE_COL]) - set(self.store_scale_.index)
        if unknown:
            raise ValueError(f"Stores not seen in training: {sorted(unknown)}")
        out[self.target] = out[self.target] * out[STORE_COL].map(self.store_scale_)  # back to USD
        return out.rename(columns={self.target: "prediction"}).reset_index(drop=True)

    def predict(self, test: pd.DataFrame) -> pd.Series:
        """Chain-level forecast = sum of per-store recursive forecasts."""
        self.store_predictions_ = self.predict_store_level(test)
        agg = self.store_predictions_.groupby(DATE_COL)["prediction"].sum().sort_index()
        agg.index = pd.DatetimeIndex(agg.index, name=DATE_COL)
        return agg.rename(self.name)

    def feature_importance(self) -> pd.Series:
        self._check_fitted()
        imp = self.model_.get_booster().get_score(importance_type="gain")
        return pd.Series(imp, name="gain").reindex(self.feature_cols_).fillna(0).sort_values(ascending=False)

    def get_params_summary(self) -> dict:
        keep = ("learning_rate", "max_depth", "min_child_weight", "subsample", "colsample_bytree", "reg_lambda")
        return {
            "strategy": "global panel model over all stores, recursive multi-step forecast",
            "target_scaling": self.target_scaling,
            "excluded_features": list(self.exclude_features),
            "lags": list(self.lags),
            "rolling_windows": list(self.windows),
            "n_features": len(self.feature_cols_),
            "features": self.feature_cols_,
            "early_stopping_best_iteration": self.best_iteration_,
            "n_estimators_final": self.n_estimators_,
            **{k: self.params[k] for k in keep},
        }
