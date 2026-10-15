import base64
import io
import zipfile
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from index import app, archived_predictions, load_history, LabOptions, run_lab
from web_sample import make_synthetic_walmart

client = TestClient(app)


def upload(frame, **kw):
    return {'source': 'upload', 'file_name': 'test.csv', 'file_data': base64.b64encode(frame.to_csv(index=False).encode()).decode(), **kw}


def test_brand_and_original_benchmark_metrics():
    assert 'Sales Pred Engine' in client.get('/').text
    assert client.get('/api/health').json()['name'] == 'Sales Pred Engine'
    d = client.post('/api/benchmark', json={}).json()
    assert d['weeks'] == 29
    saved = client.get('/api/metadata').json()['models']
    for row in d['scores']:
        for key in ['RMSE', 'MAE', 'MAPE']:
            np.testing.assert_allclose(row[key], saved[row['model']][key], rtol=1e-4)


def test_benchmark_date_window_recalculates_from_selected_rows():
    r = client.post('/api/benchmark', json={'start': '2012-05-04', 'end': '2012-05-25'})
    assert r.status_code == 200
    d = r.json()
    assert d['weeks'] == 4
    sample = archived_predictions()
    sample = sample[(sample.Date >= '2012-05-04') & (sample.Date <= '2012-05-25')]
    expected = np.abs((sample.Actual - sample.XGBoost) / sample.Actual).mean()*100
    actual = next(row['MAPE'] for row in d['scores'] if row['model'] == 'XGBoost')
    np.testing.assert_allclose(actual, expected)
    assert client.post('/api/benchmark', json={'start': '2012-99-99'}).status_code == 400


def test_live_lab_fits_and_forecasts_complete_future_weeks():
    r = client.post('/api/forecast', json={'stores': 2, 'weeks': 100, 'horizon': 4})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d['train_weeks'] == 80 and d['test_weeks'] == 20
    assert len(d['future']) == 4
    assert pd.Timestamp(d['future'][0]['Date']) > pd.Timestamp(d['history'][-1]['Date'])
    assert all(np.isfinite(row['XGBoost']) for row in d['future'])
    assert d['evaluation_params']['n_features'] >= 10
    archive = zipfile.ZipFile(io.BytesIO(base64.b64decode(d['download_base64'])))
    forecast = pd.read_csv(archive.open('future_predictions.csv'))
    assert len(forecast) == 4 and 'XGBoost' in forecast
    assert 'forecast_report.json' in archive.namelist()


def test_uploaded_holdout_actuals_do_not_change_heldout_predictions():
    frame = make_synthetic_walmart(2, 100, 15)
    a = run_lab(LabOptions(**upload(frame, horizon=2)))
    # Same dates and exogenous data; only sales after the shared split are changed.
    changed = frame.copy()
    parsed = pd.to_datetime(changed.Date, format='%d-%m-%Y')
    changed.loc[parsed >= pd.Timestamp(a['split_date']), 'Weekly_Sales'] *= 1.25
    b = run_lab(LabOptions(**upload(changed, horizon=2)))
    np.testing.assert_allclose([r['XGBoost'] for r in a['holdout']], [r['XGBoost'] for r in b['holdout']], rtol=0, atol=0)
    assert a['scores'] != b['scores']
    assert a['future'] != b['future']  # the full-history refit legitimately sees changed history


def test_minimal_csv_has_explicit_assumptions_and_iso_dates_parse_correctly():
    frame = make_synthetic_walmart(1, 100, 3)[['Date', 'Weekly_Sales']]
    frame.Date = pd.to_datetime(frame.Date, format='%d-%m-%Y').dt.strftime('%Y-%m-%d')
    history, assumptions = load_history(LabOptions(**upload(frame)))
    assert history.Date.min() == pd.Timestamp('2010-02-05')
    assert assumptions and (history.Temperature == 60).all()
    response = client.post('/api/profile', json=upload(frame))
    assert response.status_code == 200 and response.json()['stores'] == 1


def test_invalid_dates_duplicates_gaps_and_numeric_values_are_rejected():
    original = make_synthetic_walmart(2, 100, 1)
    invalid = [original.iloc[1:].copy(), pd.concat([original, original.head(1)])]
    weekend = original.copy(); weekend.loc[0, 'Date'] = '06-02-2010'; invalid.append(weekend)
    bad = original.copy(); bad.loc[0, 'Weekly_Sales'] = np.inf; invalid.append(bad)
    for frame in invalid:
        assert client.post('/api/profile', json=upload(frame)).status_code == 400
    assert client.post('/api/forecast', json={'horizon': 100}).status_code == 422
    assert client.post('/api/forecast', content=b'x'*2100001).status_code == 413


def test_sample_download_is_valid_for_upload():
    r = client.get('/api/sample.csv?seed=73&stores=2')
    assert r.status_code == 200
    frame = pd.read_csv(io.BytesIO(r.content))
    assert len(frame) == 286
    assert client.post('/api/profile', json=upload(frame)).status_code == 200
