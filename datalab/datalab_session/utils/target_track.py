import logging
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from datalab.datalab_session.utils.geometry import angular_distance_arcsec, unit_vectors


log = logging.getLogger(__name__)


MINIMUM_TRACK_SAMPLES = 2
# Two samples give a line, three or more a quadratic. Higher degrees would oscillate between samples.
MAX_TRACK_FIT_ORDER = 2
# Beyond this arc a two-sample line drifts off the object: measured against JPL Horizons for 216
# Kleopatra, 0.4" at 12 h, 1.1" at 24 h, 4.4" at 48 h. Faster movers curve sooner.
LINEAR_TRACK_MAX_SPAN_HOURS = 12.0

@dataclass(frozen=True)
class TrackSample:
    """One position of the target, at the MJD (UTC) of the exposure midpoint it was marked on."""
    mjd: float
    ra_deg: float
    dec_deg: float

    @classmethod
    def from_input(
        cls,
        raw_samples: Any,
        *,
        minimum: int = MINIMUM_TRACK_SAMPLES,
        require_mjd: bool = True,
    ) -> tuple["TrackSample", ...]:
        """
            Parses submitted {mjd, ra, dec} positions into samples, sorted by time. require_mjd=False
            accepts an untimed position, for a fixed target, and records NaN.
        """
        if not isinstance(raw_samples, Sequence) or isinstance(raw_samples, (str, bytes)):
            raise ValueError("Target positions must be a list of {ra, dec} entries.")
        if len(raw_samples) < minimum:
            raise ValueError(
                f"A target needs at least {minimum} position(s), got {len(raw_samples)}."
            )

        samples: list[TrackSample] = []
        for index, raw_sample in enumerate(raw_samples):
            if not isinstance(raw_sample, Mapping):
                raise ValueError(f"Target position {index} must be a mapping with ra/dec.")
            try:
                mjd = float(raw_sample["mjd"]) if require_mjd or "mjd" in raw_sample else math.nan
                ra_deg = float(raw_sample["ra"])
                dec_deg = float(raw_sample["dec"])
            except KeyError as exc:
                raise ValueError(f"Target position {index} is missing {exc.args[0]!r}.") from exc
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Target position {index} has a non-numeric mjd/ra/dec.") from exc
            if (require_mjd and not math.isfinite(mjd)) or not (math.isfinite(ra_deg) and math.isfinite(dec_deg)):
                raise ValueError(f"Target position {index} has non-finite values.")
            if not -90.0 <= dec_deg <= 90.0:
                raise ValueError(f"Target position {index} has dec {dec_deg} outside [-90, 90].")
            samples.append(cls(mjd=mjd, ra_deg=ra_deg, dec_deg=dec_deg))

        # Untimed samples sort last, keeping submitted order: comparing NaN would leave it undefined.
        samples.sort(key=lambda sample: (math.isnan(sample.mjd), 0.0 if math.isnan(sample.mjd) else sample.mjd))
        if require_mjd and len({sample.mjd for sample in samples}) < len(samples):
            raise ValueError("Each target position must be at a distinct time.")
        return tuple(samples)


# eq=False because the coefficient and basis fields are numpy arrays: the generated __eq__ would
# compare them elementwise and raise on the ambiguous truth value of the resulting array. Identity
# comparison is all a track is ever needed for.
@dataclass(frozen=True, eq=False)
class TargetTrack:
    """
        A polynomial track through the sample positions, fitted in a gnomonic tangent plane about
        their mean direction: that removes the RA wrap at 0h, the cos(dec) compression and the pole
        degeneracy in one step.
    """
    samples: tuple[TrackSample, ...]
    order: int
    reference_mjd: float
    # Orthonormal tangent-plane basis at the mean sample direction: outward (line of sight), east, north.
    radial_axis: np.ndarray = field(repr=False)
    east_axis: np.ndarray = field(repr=False)
    north_axis: np.ndarray = field(repr=False)
    # Polynomial coefficients (numpy order, highest power first) for the projected coordinates.
    xi_coefficients: np.ndarray = field(repr=False)
    eta_coefficients: np.ndarray = field(repr=False)

    @property
    def sample_mjd_span(self) -> tuple[float, float]:
        """Earliest and latest sample time."""
        times = [sample.mjd for sample in self.samples]
        return min(times), max(times)

    @property
    def sample_span_hours(self) -> float:
        first, last = self.sample_mjd_span
        return (last - first) * 24.0

    def position_at(self, mjd: float) -> tuple[float, float]:
        """Target RA/Dec (degrees) at an arbitrary time."""
        elapsed = float(mjd) - self.reference_mjd
        xi = float(np.polyval(self.xi_coefficients, elapsed))
        eta = float(np.polyval(self.eta_coefficients, elapsed))
        return _deproject(self.radial_axis, self.east_axis, self.north_axis, xi, eta)

    def covers(self, mjd: float) -> bool:
        """Whether a time is inside the sample span, so interpolated rather than extrapolated."""
        first, last = self.sample_mjd_span
        return first <= float(mjd) <= last


def fit_target_track(samples: Sequence[TrackSample]) -> TargetTrack:
    """
        Fits a track through the sample positions. Degree follows the number of distinct sample
        times, capped at MAX_TRACK_FIT_ORDER; over-determined fits are solved by least squares.
    """
    if len(samples) < MINIMUM_TRACK_SAMPLES:
        raise ValueError(
            f"A target track needs at least {MINIMUM_TRACK_SAMPLES} sample positions, got {len(samples)}."
        )
    ordered = tuple(sorted(samples, key=lambda sample: sample.mjd))
    distinct_times = len({sample.mjd for sample in ordered})
    if distinct_times < MINIMUM_TRACK_SAMPLES:
        raise ValueError("Track samples must be at two or more distinct times.")
    order = min(distinct_times - 1, MAX_TRACK_FIT_ORDER)

    directions = unit_vectors([sample.ra_deg for sample in ordered], [sample.dec_deg for sample in ordered])
    radial_axis, east_axis, north_axis = _tangent_basis(directions.mean(axis=0))

    # Samples more than 90 degrees from the plane's centre project behind the observer, so they
    # cannot be one short arc of the same object.
    along_radial = directions @ radial_axis
    if np.any(along_radial <= 0.0):
        raise ValueError("Track samples span more than 90 degrees on the sky; they cannot be one short arc.")
    xi = (directions @ east_axis) / along_radial
    eta = (directions @ north_axis) / along_radial

    reference_mjd = float(np.mean([sample.mjd for sample in ordered]))
    elapsed = np.array([sample.mjd - reference_mjd for sample in ordered])
    xi_coefficients = np.polyfit(elapsed, xi, order)
    eta_coefficients = np.polyfit(elapsed, eta, order)

    track = TargetTrack(
        samples=ordered,
        order=order,
        reference_mjd=reference_mjd,
        radial_axis=radial_axis,
        east_axis=east_axis,
        north_axis=north_axis,
        xi_coefficients=xi_coefficients,
        eta_coefficients=eta_coefficients,
    )
    log.info(
        "Aperture Photometry target track fitted: "
        f"samples={len(ordered)}, order={order}, span_hours={track.sample_span_hours:.3f}, "
        f"rate_arcsec_per_min={track_rate_arcsec_per_minute(track):.4f}"
    )
    return track


def track_rate_arcsec_per_minute(track: TargetTrack) -> float:
    """Mean apparent rate of motion along the fitted track."""
    first, last = track.sample_mjd_span
    if last <= first:
        return 0.0
    start = track.position_at(first)
    end = track.position_at(last)
    minutes = (last - first) * 24.0 * 60.0
    return angular_distance_arcsec(*start, *end) / minutes


def _tangent_basis(mean_direction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        Right-handed (radial, east, north) orthonormal basis at a direction on the sky. At a pole the
        construction is degenerate, so an arbitrary perpendicular pair is chosen; the fit is
        unaffected, only the labelling of the in-plane axes.
    """
    radial_axis = mean_direction / np.linalg.norm(mean_direction)
    pole = np.array([0.0, 0.0, 1.0])
    east_axis = np.cross(pole, radial_axis)
    norm = np.linalg.norm(east_axis)
    if norm < 1e-8:
        east_axis = np.cross(np.array([1.0, 0.0, 0.0]), radial_axis)
        norm = np.linalg.norm(east_axis)
    east_axis = east_axis / norm
    north_axis = np.cross(radial_axis, east_axis)
    return radial_axis, east_axis, north_axis


def _deproject(
    radial_axis: np.ndarray,
    east_axis: np.ndarray,
    north_axis: np.ndarray,
    xi: float,
    eta: float,
) -> tuple[float, float]:
    """Maps a tangent-plane offset back onto the sky, returning RA/Dec in degrees."""
    direction = radial_axis + xi * east_axis + eta * north_axis
    direction = direction / np.linalg.norm(direction)
    ra_deg = math.degrees(math.atan2(float(direction[1]), float(direction[0]))) % 360.0
    dec_deg = math.degrees(math.asin(float(np.clip(direction[2], -1.0, 1.0))))
    return ra_deg, dec_deg
