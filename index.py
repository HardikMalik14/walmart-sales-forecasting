"""Sales Pred Engine: archived benchmark explorer and bounded live forecast lab."""
from __future__ import annotations
import base64
import binascii
import csv
import io
import json
import logging
import os
import tempfile
import threading
import time
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Literal

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'sales-pred-matplotlib'))
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from src.data_prep import aggregate_weekly, chronological_split, clean_data
from src.models import SeasonalNaiveForecaster, XGBoostForecaster, build_us_holiday_calendar
from src.utils import calculate_metrics
from web_sample import make_synthetic_walmart

ROOT = Path(__file__).resolve().parent
app = FastAPI(title='Sales Pred Engine', version='1.0.0')
WORKER = threading.BoundedSemaphore(1)
log = logging.getLogger(__name__)
MODELS = ['Seasonal Naive', 'SARIMA', 'Prophet', 'XGBoost']


class BenchmarkOptions(BaseModel):
    model_config = ConfigDict(extra='forbid')
    start: str | None = Field(None, pattern=r'^\d{4}-\d{2}-\d{2}$')
    end: str | None = Field(None, pattern=r'^\d{4}-\d{2}-\d{2}$')


class LabOptions(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    source: Literal['synthetic', 'upload'] = 'synthetic'
    seed: int = Field(21, ge=0, le=10000)
    stores: int = Field(5, ge=1, le=10)
    weeks: int = Field(143, ge=80, le=180)
    horizon: int = Field(8, ge=1, le=16)
    test_size: float = Field(.2, ge=.15, le=.3)
    file_name: str = Field('', max_length=200)
    file_data: str = Field('', max_length=2000000)
    temperature_shift: float = Field(0, ge=-20, le=20)


class WebXGBoost(XGBoostForecaster):
    """Same forecasting wrapper, with single-threaded models for bounded web requests."""
    def _make_model(self, n_estimators=None, early_stopping=False):
        return super()._make_model(n_estimators, early_stopping).set_params(n_jobs=1)


def records(frame, index=False):
    return json.loads((frame.reset_index() if index else frame).to_json(orient='records', date_format='iso'))


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


@lru_cache(maxsize=1)
def metadata():
    return json.loads((ROOT / 'outputs/metrics.json').read_text())


@lru_cache(maxsize=1)
def archived_predictions():
    return pd.read_csv(ROOT / 'outputs/predictions.csv', parse_dates=['Date'])


def metrics(actual, predicted):
    out = calculate_metrics(actual, predicted)
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    total = float(actual.sum())
    out.update({'bias_pct': float((predicted-actual).sum()/total*100) if total else None,
                'actual_total': total, 'predicted_total': float(predicted.sum()),
                'max_abs_error': float(np.max(np.abs(predicted-actual))), 'weeks': len(actual)})
    return safe(out)


@app.middleware('http')
async def body_limit(request: Request, call_next):
    if request.method == 'POST':
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > 2100000:
                return JSONResponse({'detail': 'Use a UTF-8 CSV under 1.5 MB.'}, status_code=413)
        request._body = bytes(data)
    return await call_next(request)


@app.get('/', include_in_schema=False)
def home():
    return FileResponse(ROOT / 'public/index.html')


@app.get('/api/health')
def health():
    return {'status': 'ok', 'name': 'Sales Pred Engine'}


@app.get('/api/metadata')
def meta():
    return metadata()


@app.post('/api/benchmark')
def benchmark(o: BenchmarkOptions):
    frame = archived_predictions()
    try:
        for bound in [o.start, o.end]:
            if bound and pd.isna(pd.to_datetime(bound, format='%Y-%m-%d', errors='coerce')):
                raise ValueError('Use valid calendar dates.')
        if o.start and o.end and o.start > o.end:
            raise ValueError('Start must not be after end.')
        if o.start:
            frame = frame[frame['Date'] >= pd.Timestamp(o.start)]
        if o.end:
            frame = frame[frame['Date'] <= pd.Timestamp(o.end)]
        if frame.empty:
            raise ValueError('No archived forecast weeks in the selected range.')
        scores = [{'model': model, **metrics(frame['Actual'], frame[model])} for model in MODELS]
        scores.sort(key=lambda r: r['MAPE'] if r['MAPE'] is not None else float('inf'))
        return {'source': 'Original saved Walmart benchmark', 'predictions': records(frame),
                'scores': scores, 'start': frame['Date'].min().strftime('%Y-%m-%d'),
                'end': frame['Date'].max().strftime('%Y-%m-%d'), 'weeks': len(frame),
                'note': 'Metrics are recalculated from saved holdout predictions. Selecting a date window does not refit the archived models.'}
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, str(exc)) from exc


def read_upload(o):
    try:
        raw = base64.b64decode(o.file_data, validate=True)
        text = raw.decode('utf-8-sig')
    except (ValueError, binascii.Error, UnicodeDecodeError) as exc:
        raise ValueError('Upload a valid UTF-8 CSV.') from exc
    if len(raw) > 1500000:
        raise ValueError('Use a CSV under 1.5 MB.')
    header = next(csv.reader(io.StringIO(text)), [])
    if not header or len(set(header)) != len(header):
        raise ValueError('CSV headers must be unique.')
    frame = pd.read_csv(io.StringIO(text), nrows=8001)
    if not {'Date', 'Weekly_Sales'}.issubset(frame):
        raise ValueError('CSV needs Date and Weekly_Sales columns. Store, Holiday_Flag and Temperature are optional.')
    return frame


def load_history(o):
    frame = read_upload(o) if o.source == 'upload' else make_synthetic_walmart(o.stores, o.weeks, o.seed)
    if not 80 <= len(frame) <= 8000 or len(frame.columns) > 20:
        raise ValueError('Use at most 8,000 rows and 20 columns, with at least 80 consecutive weeks per store.')
    parsed = pd.to_datetime(frame['Date'], format='%Y-%m-%d', errors='coerce')
    bad = parsed.isna()
    parsed.loc[bad] = pd.to_datetime(frame.loc[bad, 'Date'], format='%d-%m-%Y', errors='coerce')
    if parsed.isna().any() or (parsed.dt.dayofweek != 4).any():
        raise ValueError('Dates must be weekly Fridays, using YYYY-MM-DD or DD-MM-YYYY.')
    frame = frame.copy()
    frame['Date'] = parsed
    assumptions = []
    if 'Store' not in frame:
        frame['Store'] = 1
        assumptions.append('No Store column: treated as one aggregate series.')
    frame['Store'] = pd.to_numeric(frame['Store'], errors='coerce')
    if frame['Store'].isna().any() or (frame['Store'] < 1).any() or (frame['Store'] > 1000000).any() or (frame['Store'] % 1 != 0).any():
        raise ValueError('Store identifiers must be positive integers up to 1,000,000.')
    frame['Store'] = frame['Store'].astype(int)
    if frame['Store'].nunique() > 45:
        raise ValueError('Use at most 45 stores.')
    frame['Weekly_Sales'] = pd.to_numeric(frame['Weekly_Sales'], errors='coerce')
    sales = frame['Weekly_Sales'].to_numpy(dtype=float)
    if not np.isfinite(sales).all() or (np.abs(sales) > 1e10).any():
        raise ValueError('Weekly_Sales must contain finite numeric values within ±10 billion.')
    if frame.duplicated(['Store', 'Date']).any():
        raise ValueError('Each store must have one row per week; duplicate Store/Date pairs are not aggregated.')
    dates = pd.DatetimeIndex(sorted(frame['Date'].unique()))
    if not 80 <= len(dates) <= 200:
        raise ValueError('Use 80–200 historical weeks per store.')
    expected = pd.date_range(dates.min(), dates.max(), freq='W-FRI')
    if not dates.equals(expected) or any(not pd.DatetimeIndex(sorted(g['Date'])).equals(dates) for _, g in frame.groupby('Store')):
        raise ValueError('Every store must have the same consecutive Friday dates, with no gaps.')
    if 'Holiday_Flag' not in frame:
        calendar = build_us_holiday_calendar(dates.min(), dates.max())
        frame['Holiday_Flag'] = frame['Date'].isin(calendar['ds']).astype(int)
        assumptions.append('Holiday_Flag missing: mapped from the original US holiday calendar.')
    if 'Temperature' not in frame:
        frame['Temperature'] = 60.0
        assumptions.append('Temperature missing: assumed constant 60°F. This is an input assumption, not weather data.')
    for col in ['Temperature', 'Holiday_Flag']:
        frame[col] = pd.to_numeric(frame[col], errors='coerce')
        if not np.isfinite(frame[col].to_numpy(dtype=float)).all():
            raise ValueError(f'{col} must contain finite numbers.')
    if not frame['Holiday_Flag'].isin([0, 1]).all():
        raise ValueError('Holiday_Flag must be 0 or 1.')
    if not frame['Temperature'].between(-100, 150).all():
        raise ValueError('Temperature must be between -100 and 150°F.')
    # These columns are part of the original loader schema but are not features
    # of this repository’s XGBoost configuration or the seasonal-naive baseline.
    for col in ['Fuel_Price', 'CPI', 'Unemployment']:
        if col not in frame:
            frame[col] = 0.0
        frame[col] = pd.to_numeric(frame[col], errors='coerce')
        if not np.isfinite(frame[col].to_numpy(dtype=float)).all():
            raise ValueError(f'{col} must contain finite numbers when supplied.')
    return clean_data(frame), assumptions


def future_frame(history, horizon, temperature_shift):
    dates = pd.date_range(history['Date'].max() + pd.Timedelta(days=7), periods=horizon, freq='W-FRI')
    calendar = build_us_holiday_calendar(dates.min(), dates.max())
    rows = []
    for store, g in history.groupby('Store'):
        last = g.sort_values('Date').iloc[-1]
        for date in dates:
            rows.append({'Store': int(store), 'Date': date, 'Weekly_Sales': np.nan,
                         'Holiday_Flag': int(date in set(calendar['ds'])),
                         'Temperature': float(last['Temperature'] + temperature_shift),
                         'Fuel_Price': float(last['Fuel_Price']), 'CPI': float(last['CPI']),
                         'Unemployment': float(last['Unemployment'])})
    return pd.DataFrame(rows)


def fit_web_model(train, seed):
    return WebXGBoost(params={'n_estimators': 300}, validation_weeks=8, random_state=seed).fit(train)


def run_lab(o):
    started = time.perf_counter()
    history, assumptions = load_history(o)
    weekly = aggregate_weekly(history)
    train, test = chronological_split(history, o.test_size, date_col='Date')
    train_weekly, test_weekly = chronological_split(weekly, o.test_size)
    xgb = fit_web_model(train, o.seed)
    heldout = pd.DataFrame({'Actual': test_weekly['Weekly_Sales'],
                           'Seasonal Naive': SeasonalNaiveForecaster().fit(train_weekly).predict(test_weekly),
                           'XGBoost': xgb.predict(test)})
    if not np.isfinite(heldout.to_numpy(dtype=float)).all():
        raise ValueError('The heldout forecast contained invalid values.')
    scores = [{'model': m, **metrics(heldout['Actual'], heldout[m])} for m in ['Seasonal Naive', 'XGBoost']]
    scores.sort(key=lambda r: r['RMSE'])
    # Refit on all available history only AFTER scoring the chronological holdout.
    fitted = fit_web_model(history, o.seed)
    future = future_frame(history, o.horizon, o.temperature_shift)
    future_weekly = aggregate_weekly(future.fillna({'Weekly_Sales': 0}))
    predictions = pd.DataFrame({'Seasonal Naive': SeasonalNaiveForecaster().fit(weekly).predict(future_weekly),
                                'XGBoost': fitted.predict(future)})
    if not np.isfinite(predictions.to_numpy(dtype=float)).all():
        raise ValueError('The future forecast contained invalid values.')
    importance = xgb.feature_importance()
    total_gain = float(importance.sum())
    result = safe({'source': o.file_name if o.source == 'upload' else 'Synthetic Walmart-shaped sample',
        'synthetic': o.source == 'synthetic', 'stores': int(history['Store'].nunique()), 'weeks': len(weekly),
        'train_weeks': len(train_weekly), 'test_weeks': len(test_weekly),
        'split_date': test_weekly.index[0].strftime('%Y-%m-%d'), 'horizon': o.horizon,
        'settings': o.model_dump(exclude={'file_data'}), 'history': records(weekly[['Weekly_Sales', 'Holiday_Flag']], True),
        'holdout': records(heldout, True), 'scores': scores, 'future': records(predictions, True),
        'feature_importance': [{'feature': str(k), 'gain_share': float(v/total_gain) if total_gain else 0} for k, v in importance.head(12).items()],
        'evaluation_params': xgb.get_params_summary(), 'refit_params': fitted.get_params_summary(),
        'elapsed_seconds': round(time.perf_counter()-started, 3), 'assumptions': assumptions + [
            'Future temperature is the last observed value per store plus the chosen temperature shift; future US holiday flags follow the original calendar.',
            'Holdout evaluation uses known holdout temperature and holiday covariates (ex-post). Future forecasts use assumptions, not observed future weather.',
            'Web XGBoost uses at most 300 early-stopping probe trees and an 8-week inner validation window, then refits with the original wrapper’s tree-count rule.',
            'No prediction intervals are estimated. Future forecasts have no observed actuals and therefore no measured accuracy.'],
        'store_totals': records(history.groupby('Store', as_index=False)['Weekly_Sales'].sum().sort_values('Weekly_Sales', ascending=False))})
    data = io.BytesIO()
    with zipfile.ZipFile(data, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('future_predictions.csv', predictions.to_csv())
        z.writestr('holdout_predictions.csv', heldout.to_csv())
        z.writestr('sales_history.csv', history.to_csv(index=False))
        z.writestr('forecast_report.json', json.dumps(result, indent=2, allow_nan=False))
    result['download_base64'] = base64.b64encode(data.getvalue()).decode()
    return result


@app.post('/api/profile')
def profile(o: LabOptions):
    try:
        frame, assumptions = load_history(o)
        return {'rows': len(frame), 'stores': int(frame['Store'].nunique()), 'weeks': int(frame['Date'].nunique()),
                'start': frame['Date'].min().strftime('%Y-%m-%d'), 'end': frame['Date'].max().strftime('%Y-%m-%d'),
                'assumptions': assumptions}
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post('/api/forecast')
def forecast(o: LabOptions):
    if not WORKER.acquire(blocking=False):
        raise HTTPException(429, 'Another forecast is running. Try again shortly.')
    try:
        return run_lab(o)
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        log.exception('Forecast failed')
        raise HTTPException(500, 'Forecast failed. Check deployment logs for details.') from exc
    finally:
        WORKER.release()


@app.get('/api/sample.csv')
def sample(seed: int = Query(21, ge=0, le=10000), stores: int = Query(5, ge=1, le=10)):
    return Response(make_synthetic_walmart(stores, 143, seed).to_csv(index=False), media_type='text/csv',
                    headers={'Content-Disposition': 'attachment; filename="Sales_Pred_Synthetic_Sample.csv"'})
