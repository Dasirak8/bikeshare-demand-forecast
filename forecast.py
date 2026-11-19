"""Download, prepare and backtest a one-day-ahead bike rental forecast."""
import hashlib
from io import BytesIO
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parent
DATA_URL = 'https://archive.ics.uci.edu/static/public/275/bike%2Bsharing%2Bdataset.zip'
DAY_SHA256 = 'a6bcf826782d3c0fbfdcbeead17cd0884185a0dafe8ff10cd48a874ee7ba18be'
HOLDOUT_DAYS = 120
FOLDS = 4
VALIDATION_DAYS = 84
SIMPLICITY_TOLERANCE = 0.02
BASELINES = {'Yesterday': 'lag_1', 'Same weekday': 'lag_7', '7-day mean': 'mean_7'}
MODEL_ORDER = [*BASELINES, 'Ridge', 'Shallow tree']
FEATURES = ['lag_1', 'lag_7', 'mean_7', 'temp_1', 'hum_1', 'workingday', 'year_sin', 'year_cos']


def load_snapshot(path=ROOT / 'data' / 'day.csv'):
    """Read just the named member; verify its bytes before using or caching it."""
    cached = path.exists()
    if cached:
        raw = path.read_bytes()
    else:
        try:
            with urlopen(DATA_URL, timeout=45) as response:
                archive = response.read()
            with ZipFile(BytesIO(archive)) as z:
                raw = z.read('day.csv')
        except (URLError, TimeoutError, BadZipFile, KeyError) as exc:
            raise RuntimeError('Could not download day.csv from UCI. Retry or place the official day.csv under data/.') from exc
    if hashlib.sha256(raw).hexdigest() != DAY_SHA256:
        raise ValueError('day.csv SHA-256 mismatch. Restore the pinned UCI snapshot.')
    daily = pd.read_csv(BytesIO(raw), parse_dates=['dteday'])
    if len(daily) != 731 or daily.isna().any().any():
        raise ValueError('Expected 731 complete daily records.')
    if not daily['cnt'].eq(daily['casual'] + daily['registered']).all():
        raise ValueError('Rental totals do not reconcile to casual plus registered.')
    if not cached:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return daily


def make_features(daily):
    """On row t, use the calendar for t and observations through t-1 only."""
    data = daily.set_index('dteday').sort_index()
    calendar = pd.date_range(data.index.min(), data.index.max(), freq='D')
    if not data.index.equals(calendar):
        raise ValueError('Expected unique, consecutive calendar dates; gaps invalidate row shifts.')
    d = data[['cnt', 'temp', 'hum', 'workingday']]
    if not np.isfinite(d.to_numpy()).all() or (d['cnt'] < 0).any():
        raise ValueError('Missing, non-finite or invalid source values.')
    feats = d[['cnt', 'workingday']].copy()
    feats['lag_1'] = d['cnt'].shift(1)
    feats['lag_7'] = d['cnt'].shift(7)
    feats['mean_7'] = feats['lag_1'].rolling(7).mean()
    feats['temp_1'] = d['temp'].shift(1)
    feats['hum_1'] = d['hum'].shift(1)
    phase = 2 * np.pi * (d.index.dayofyear - 1) / np.where(d.index.is_leap_year, 366, 365)
    feats['year_sin'] = np.sin(phase)
    feats['year_cos'] = np.cos(phase)
    return feats.iloc[7:].copy()


def choose_model(average_mae):
    """Take the simplest candidate within 2% of the lowest validation MAE."""
    threshold = average_mae.min() * (1 + SIMPLICITY_TOLERANCE)
    return next(name for name in MODEL_ORDER if average_mae.loc[name] <= threshold)


def predict(name, train, scoring):
    if name in BASELINES:
        return scoring[BASELINES[name]].to_numpy()
    estimators = {
        'Ridge': make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        'Shallow tree': DecisionTreeRegressor(max_depth=4, min_samples_leaf=15, random_state=42),
    }
    model = estimators[name].fit(train[FEATURES], train['cnt'])
    return np.maximum(0, model.predict(scoring[FEATURES]))


def evaluate(frame):
    """Use only development folds to choose the winner and benchmark it once."""
    dev = frame.iloc[:-HOLDOUT_DAYS]
    splitter = TimeSeriesSplit(n_splits=FOLDS, test_size=VALIDATION_DAYS)
    fold_rows, scores = [], []
    for i, (a, b) in enumerate(splitter.split(dev), start=1):
        train, validation = dev.iloc[a], dev.iloc[b]
        fold_rows.append({'fold': i, 'train_start': train.index[0], 'train_end': train.index[-1],
                          'validation_start': validation.index[0], 'validation_end': validation.index[-1],
                          'train_days': len(train), 'validation_days': len(validation)})
        for name in MODEL_ORDER:
            pred = predict(name, train, validation)
            scores.append({'fold': i, 'model': name, 'mae': mean_absolute_error(validation['cnt'], pred)})
    cv = pd.DataFrame(scores)
    average = cv.groupby('model')['mae'].mean().reindex(MODEL_ORDER)
    winner = choose_model(average)
    other_rules = [name for name in BASELINES if name != winner]
    benchmark = average.loc[other_rules].idxmin()

    # Both choices are locked before scoring the final 120 dates.
    test = frame.iloc[-HOLDOUT_DAYS:]
    rows = test[['cnt', 'workingday']].copy()
    rows['forecast'] = predict(winner, dev, test)
    rows['baseline'] = predict(benchmark, dev, test)
    rows['error'] = rows['forecast'] - rows['cnt']  # positive = overprediction
    return {'cv': cv, 'average': average, 'folds': pd.DataFrame(fold_rows),
            'winner': winner, 'benchmark': benchmark, 'predictions': rows,
            'fit_start': dev.index[0], 'fit_end': dev.index[-1], 'fit_days': len(dev),
            'test_mae': mean_absolute_error(rows['cnt'], rows['forecast']),
            'baseline_mae': mean_absolute_error(rows['cnt'], rows['baseline'])}
