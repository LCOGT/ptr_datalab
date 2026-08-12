import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np
from astropy.timeseries import LombScargle


log = logging.getLogger(__name__)


# Below this, Lomb-Scargle on sparse data is noise. Matches VariableStar's own minimum.
MINIMUM_PERIOD_POINTS = 8


@dataclass(frozen=True)
class PeriodCandidate:
    """
        A period worth offering the user to fold on. Lomb-Scargle locks onto the strongest Fourier
        component, which for a double-peaked source (a tumbling asteroid) is half the rotation
        period, so both the peak and its double are offered.
    """
    period: float
    power: float
    false_alarm_probability: float
    kind: str  # "peak" or "double"
    description: str


@dataclass(frozen=True)
class PeriodAnalysis:
    """
        Lomb-Scargle period analysis. candidates and window_power are empty under
        include_extras=False; the rest reproduces the VariableStar result exactly.
    """
    frequency: np.ndarray = field(repr=False)
    power: np.ndarray = field(repr=False)
    period: float
    false_alarm_probability: float
    candidates: list[PeriodCandidate]
    window_power: np.ndarray = field(repr=False)

    @classmethod
    def from_light_curve(
        cls,
        times: Sequence[float],
        magnitudes: Sequence[float],
        magnitude_errors: Sequence[float],
        *,
        include_extras: bool = True,
    ) -> "PeriodAnalysis":
        """
            Lomb-Scargle search over a light curve. times are in days, only differences matter.

            include_extras=False skips the doubled period and the window function, the latter a
            second full periodogram over the same grid.
        """
        times_arr = np.asarray(times, dtype=float)
        lomb_scargle = LombScargle(times_arr, np.asarray(magnitudes, dtype=float),
                                   np.asarray(magnitude_errors, dtype=float))
        frequency, power = lomb_scargle.autopower()

        best_index = int(np.argmax(power))
        best_frequency = float(frequency[best_index])
        best_power = float(power[best_index])
        false_alarm_probability = float(lomb_scargle.false_alarm_probability(best_power))

        if include_extras:
            candidates = _candidates(lomb_scargle, best_frequency, best_power, false_alarm_probability)
            window_power = LombScargle(
                times_arr, np.ones_like(times_arr), fit_mean=False, center_data=False
            ).power(frequency)
        else:
            candidates = []
            window_power = np.empty(0)

        log.info(
            "Period analysis: best period %.6f d (FAP %.3g) from %d points; %d candidate period(s)",
            1.0 / best_frequency, false_alarm_probability, len(times_arr), len(candidates),
        )
        return cls(
            frequency=frequency,
            power=power,
            period=1.0 / best_frequency,
            false_alarm_probability=false_alarm_probability,
            candidates=candidates,
            window_power=window_power,
        )

    @classmethod
    def from_light_curve_rows(
        cls,
        light_curve_rows: Sequence[Any],
        *,
        minimum_points: int = MINIMUM_PERIOD_POINTS,
    ) -> "PeriodAnalysis | None":
        """
            None when fewer than minimum_points frames were measurable, so the caller can say so
            rather than report a meaningless period.
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
        return cls.from_light_curve(times, magnitudes, magnitude_errors)


def _candidates(
    lomb_scargle: LombScargle,
    best_frequency: float,
    best_power: float,
    best_false_alarm_probability: float,
) -> list[PeriodCandidate]:
    """The strongest peak and its double."""
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
