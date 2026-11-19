import unittest
from io import BytesIO
from unittest.mock import patch

import numpy as np
import pandas as pd

import forecast as f


def fixture(days=500):
    i = np.arange(days)
    return pd.DataFrame({
        'dteday': pd.date_range('2011-01-01', periods=days),
        'cnt': 100 + i + i % 7,
        'temp': i / days,
        'hum': 0.5,
        'workingday': (i % 7 < 5).astype(int),
    })


class ForecastTests(unittest.TestCase):
    def test_feature_cutoff_uses_only_data_known_the_night_before(self):
        daily = fixture(14)
        day = daily.dteday[7]
        before = f.make_features(daily).loc[day]
        self.assertEqual(before['lag_1'], daily.cnt[6])
        self.assertEqual(before['lag_7'], daily.cnt[0])
        self.assertAlmostEqual(before['mean_7'], daily.cnt[:7].mean())
        self.assertEqual(before['temp_1'], daily.temp[6])
        self.assertEqual(before['workingday'], daily.workingday[7])
        daily.loc[7:, ['cnt', 'temp', 'hum']] = 99
        pd.testing.assert_series_equal(before[f.FEATURES], f.make_features(daily).loc[day, f.FEATURES])

    def test_missing_calendar_day_cannot_be_treated_as_yesterday(self):
        with self.assertRaisesRegex(ValueError, 'consecutive'):
            f.make_features(fixture(14).drop(index=4))

    def test_input_is_sorted_and_warmup_is_exactly_seven_days(self):
        features = f.make_features(fixture(14).iloc[::-1])
        self.assertEqual(len(features), 7)
        self.assertEqual(features.index[0], pd.Timestamp('2011-01-08'))
        self.assertTrue(features.index.is_monotonic_increasing)

    def test_held_out_outcomes_do_not_change_selection(self):
        df = f.make_features(fixture())
        result = f.evaluate(df)
        df.loc[df.index[-f.HOLDOUT_DAYS:], 'cnt'] += 200
        changed = f.evaluate(df)
        pd.testing.assert_frame_equal(result['cv'], changed['cv'])
        self.assertEqual(result['winner'], changed['winner'])
        np.testing.assert_allclose(result['predictions'].forecast, changed['predictions'].forecast)

    def test_each_validation_block_follows_training_and_precedes_test(self):
        df = f.make_features(fixture())
        result = f.evaluate(df)
        folds, predictions = result['folds'], result['predictions']
        self.assertEqual(len(folds), 4)
        self.assertTrue((folds.train_end < folds.validation_start).all())
        self.assertTrue((folds.validation_end < predictions.index.min()).all())
        self.assertEqual(len(predictions), 120)
        self.assertEqual(result['fit_start'], df.index[0])
        self.assertEqual(result['fit_end'] + pd.Timedelta(days=1), predictions.index.min())
        self.assertAlmostEqual(result['test_mae'], (predictions.cnt - predictions.forecast).abs().mean())

    def test_test_benchmark_is_a_different_simple_rule_chosen_by_validation(self):
        result = f.evaluate(f.make_features(fixture()))
        candidates = [name for name in f.BASELINES if name != result['winner']]
        expected = result['average'].loc[candidates].idxmin()
        self.assertNotEqual(result['winner'], result['benchmark'])
        self.assertEqual(expected, result['benchmark'])

    def test_simpler_candidate_can_win_only_within_validation_tolerance(self):
        averaged = pd.Series({'Yesterday': 120, 'Same weekday': 110, '7-day mean': 101.9, 'Ridge': 100, 'Shallow tree': 99.99})
        self.assertEqual(f.choose_model(averaged), '7-day mean')
        averaged['7-day mean'] = 102
        self.assertEqual(f.choose_model(averaged), 'Ridge')

    def test_rejects_snapshot_that_does_not_match_pinned_digest(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        import zipfile
        archive = BytesIO()
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('day.csv', 'modified')
        archive.seek(0)
        with TemporaryDirectory() as tmp, patch.object(f, 'urlopen', return_value=archive):
            path = Path(tmp) / 'day.csv'
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                f.load_snapshot(path)
            self.assertFalse(path.exists())
