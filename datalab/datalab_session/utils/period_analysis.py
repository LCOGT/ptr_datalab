import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np
from astropy.timeseries import LombScargle


log = logging.getLogger()
log.setLevel(logging.INFO)


# A light curve needs at least this many points before a period is worth reporting. Matches the
# VariableStar operation's own minimum, below which Lomb-Scargle on sparse data is noise.
MINIMUM_PERIOD_POINTS = 8


@dataclass(frozen=True)
class PeriodCandidate:
    """
        One period worth offering the user to fold on.

        A single-sinusoid Lomb-Scargle locks onto the strongest Fourier component, which for a
        double-peaked source (an elongated, tumbling asteroid) is *half* the rotation period. So the
        strongest peak and its double are both offered: "peak" is what the periodogram maximises,
        "double" is the physical rotation period when the light curve has two maxima per cycle.
    """
    period: float
    power: float
    false_alarm_probability: float
    kind: str  # "peak" or "double"
    description: str


@dataclass(frozen=True)
class PeriodAnalysis:
    """
        Lomb-Scargle period analysis of a light curve.

        frequency/power/period/false_alarm_probability reproduce the existing VariableStar result
        exactly (autopower, the strongest peak, and its analytic FAP). The rest is additive: ranked
        candidates including the doubled period, and the sampling window function on the same grid so
        a plot can show which peaks are cadence artifacts rather than signal.
    """
    frequency: np.ndarray = field(repr=False)
    power: np.ndarray = field(repr=False)
    period: float
    false_alarm_probability: float
    candidates: list[PeriodCandidate]
    window_power: np.ndarray = field(repr=False)


def analyze_period(
    times: Sequence[float],
    magnitudes: Sequence[float],
    magnitude_errors: Sequence[float],
) -> PeriodAnalysis:
    """
        Runs a Lomb-Scargle period search over a light curve.

        times are in days (only their differences matter); magnitudes and magnitude_errors are the
        light curve and its per-point uncertainties. The strongest periodogram peak is taken as the
        best period; the returned window function shows which peaks are sampling-cadence artifacts
        rather than signal.
    """
    times_arr = np.asarray(times, dtype=float)
    magnitudes_arr = np.asarray(magnitudes, dtype=float)
    errors_arr = np.asarray(magnitude_errors, dtype=float)

    lomb_scargle = LombScargle(times_arr, magnitudes_arr, errors_arr)
    frequency, power = lomb_scargle.autopower()

    best_index = int(np.argmax(power))
    best_frequency = float(frequency[best_index])
    best_power = float(power[best_index])
    period = 1.0 / best_frequency
    false_alarm_probability = float(lomb_scargle.false_alarm_probability(best_power))

    candidates = _candidates(lomb_scargle, best_frequency, best_power, false_alarm_probability)
    window_power = _window_function(times_arr, frequency)

    log.info(
        "Period analysis: best period %.6f d (FAP %.3g) from %d points; %d candidate period(s)",
        period, false_alarm_probability, len(times_arr), len(candidates),
    )
    return PeriodAnalysis(
        frequency=frequency,
        power=power,
        period=period,
        false_alarm_probability=false_alarm_probability,
        candidates=candidates,
        window_power=window_power,
    )


def period_output_from_light_curve_rows(
    light_curve_rows: Sequence[Any],
    *,
    minimum_points: int = MINIMUM_PERIOD_POINTS,
) -> dict[str, Any] | None:
    """
        Builds the period-analysis output keys for a photometry operation, or None if too sparse.

        Reads the calibrated magnitude and its uncertainty from each light curve row, skipping
        non-finite points (frames the pipeline could not measure), and returns None when fewer than
        minimum_points remain so the caller can note it rather than report a meaningless period. The
        emitted keys match the VariableStar operation (period, false_alarm_probability, frequency,
        power) so the same frontend rendering drives both, plus the additive candidates and window
        function.
    """
    times: list[float] = []
    magnitudes: list[float] = []
    magnitude_errors: list[float] = []
    for row in light_curve_rows:
        magnitude = row.target_calibrated_apparent_magnitude
        magnitude_error = row.target_calibrated_apparent_magnitude_uncertainty
        if not (math.isfinite(magnitude) and math.isfinite(magnitude_error) and magnitude_error > 0.0):
            continue
        times.append(row.date_obs.timestamp() / 86400.0)
        magnitudes.append(magnitude)
        magnitude_errors.append(magnitude_error)

    if len(times) < minimum_points:
        return None

    analysis = analyze_period(times, magnitudes, magnitude_errors)
    # 'period'/'fap'/'frequency'/'power' match the VariableStar operation's existing output keys, so
    # one frontend periodogram-and-fold renderer drives both operations.
    return {
        'period': analysis.period,
        'fap': analysis.false_alarm_probability,
        'frequency': analysis.frequency,
        'power': analysis.power,
        'period_candidates': [asdict(candidate) for candidate in analysis.candidates],
        'window_power': analysis.window_power,
    }


def _candidates(
    lomb_scargle: LombScargle,
    best_frequency: float,
    best_power: float,
    best_false_alarm_probability: float,
) -> list[PeriodCandidate]:
    """The strongest peak and its double, the latter being the rotation period of a two-maxima source."""
    candidates = [
        PeriodCandidate(
            period=1.0 / best_frequency,
            power=best_power,
            false_alarm_probability=best_false_alarm_probability,
            kind="peak",
            description="Strongest periodogram peak.",
        )
    ]
    double_frequency = best_frequency / 2.0
    double_power = float(lomb_scargle.power(np.atleast_1d(double_frequency))[0])
    candidates.append(
        PeriodCandidate(
            period=1.0 / double_frequency,
            power=double_power,
            false_alarm_probability=float(lomb_scargle.false_alarm_probability(double_power)),
            kind="double",
            description="Twice the strongest peak -- the rotation period if the light curve is double-peaked.",
        )
    )
    return candidates


def _window_function(times: np.ndarray, frequency: np.ndarray) -> np.ndarray:
    """
        The observing window's own Lomb-Scargle power on the same frequency grid.

        Computed from a constant signal at the real observation times, so its peaks are purely the
        sampling cadence. Overlaid on the data periodogram it shows the user which peaks come from the
        schedule rather than the sky.
    """
    return LombScargle(times, np.ones_like(times), fit_mean=False, center_data=False).power(frequency)
