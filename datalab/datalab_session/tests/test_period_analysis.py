from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
from astropy.timeseries import LombScargle

from datalab.datalab_session.utils.period_analysis import (
    MINIMUM_PERIOD_POINTS,
    analyze_period,
    period_output_from_light_curve_rows,
)


def _sinusoid(period_days: float, n: int = 40, amplitude: float = 0.2, noise: float = 0.005, seed: int = 0):
    rng = np.random.default_rng(seed)
    times = np.sort(rng.uniform(0.0, 8.0 * period_days, n))
    mags = 15.0 + amplitude * np.sin(2.0 * np.pi * times / period_days) + rng.normal(0.0, noise, n)
    errors = np.full(n, noise if noise > 0 else 0.01)
    return times, mags, errors


class TestAnalyzePeriod(unittest.TestCase):
    def test_recovers_an_injected_period(self) -> None:
        times, mags, errors = _sinusoid(0.3)
        analysis = analyze_period(times, mags, errors)
        self.assertAlmostEqual(analysis.period, 0.3, delta=0.01)

    def test_default_path_matches_the_original_calculate_period_logic(self) -> None:
        """The default (no alias exclusion) must reproduce autopower + argmax + analytic FAP exactly."""
        times, mags, errors = _sinusoid(0.3)
        lomb_scargle = LombScargle(times, mags, errors)
        frequency, power = lomb_scargle.autopower()
        expected_period = 1.0 / frequency[np.argmax(power)]
        expected_fap = float(lomb_scargle.false_alarm_probability(power.max()))

        analysis = analyze_period(times, mags, errors)
        self.assertTrue(np.array_equal(analysis.frequency, frequency))
        self.assertTrue(np.array_equal(analysis.power, power))
        self.assertEqual(analysis.period, expected_period)
        self.assertEqual(analysis.false_alarm_probability, expected_fap)

    def test_offers_the_doubled_period_as_a_candidate(self) -> None:
        times, mags, errors = _sinusoid(0.3)
        analysis = analyze_period(times, mags, errors)
        kinds = {candidate.kind for candidate in analysis.candidates}
        self.assertEqual(kinds, {"peak", "double"})
        peak = next(c for c in analysis.candidates if c.kind == "peak")
        double = next(c for c in analysis.candidates if c.kind == "double")
        # The double is exactly twice the peak -- the rotation period for a two-maxima source.
        self.assertAlmostEqual(double.period, 2.0 * peak.period, places=9)

    def test_returns_the_window_function_on_the_same_grid(self) -> None:
        times, mags, errors = _sinusoid(0.3)
        analysis = analyze_period(times, mags, errors)
        self.assertEqual(analysis.window_power.shape, analysis.frequency.shape)

class TestPeriodOutputFromRows(unittest.TestCase):
    def _rows(self, times, mags, error=0.01):
        return [
            SimpleNamespace(
                date_obs=datetime(2025, 3, 1, tzinfo=timezone.utc) + timedelta(days=float(t)),
                target_calibrated_apparent_magnitude=float(m),
                target_calibrated_apparent_magnitude_uncertainty=error,
            )
            for t, m in zip(times, mags)
        ]

    def test_emits_variable_star_keys_plus_additive(self) -> None:
        times, mags, _ = _sinusoid(0.3)
        output = period_output_from_light_curve_rows(self._rows(times, mags))
        self.assertIsNotNone(output)
        # The four keys VariableStar emits, so the same frontend rendering drives both.
        self.assertLessEqual({"period", "fap", "frequency", "power"}, set(output))
        self.assertIn("period_candidates", output)
        self.assertIn("window_power", output)

    def test_returns_none_below_the_minimum_point_count(self) -> None:
        times, mags, _ = _sinusoid(0.3, n=MINIMUM_PERIOD_POINTS - 1)
        self.assertIsNone(period_output_from_light_curve_rows(self._rows(times, mags)))

    def test_skips_non_finite_and_non_positive_error_points(self) -> None:
        times, mags, _ = _sinusoid(0.3, n=MINIMUM_PERIOD_POINTS + 3)
        rows = self._rows(times, mags)
        rows[0].target_calibrated_apparent_magnitude = float("nan")
        rows[1].target_calibrated_apparent_magnitude_uncertainty = 0.0
        # Two points dropped still leaves >= minimum, so a period is still produced.
        self.assertIsNotNone(period_output_from_light_curve_rows(rows))
        # Dropping enough good points falls under the minimum and returns None.
        for row in rows[2:]:
            row.target_calibrated_apparent_magnitude = float("nan")
        self.assertIsNone(period_output_from_light_curve_rows(rows))


if __name__ == "__main__":
    unittest.main()
