import logging
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


log = logging.getLogger()
log.setLevel(logging.INFO)


MINIMUM_TRACK_SAMPLES = 2
# A user-supplied track is fitted with a polynomial of degree min(distinct_sample_times - 1, this). Two
# samples give a straight line, three or more a quadratic. Higher degrees are deliberately not offered:
# extra samples sharpen the quadratic by least squares instead of raising the degree, which would let
# the fit oscillate between samples and wander off the object in the gaps.
MAX_TRACK_FIT_ORDER = 2
# Apparent tracks curve, so a straight line drifts off the object as the arc lengthens. Measured
# against the JPL Horizons ephemeris for a main-belt asteroid (216 Kleopatra, 0.55 arcsec/min), the
# worst-case deviation of a two-sample line is 0.4" over 12 h, 1.1" at 24 h and 4.4" at 48 h -- by two
# days comparable to the whole aperture radius. A three-sample quadratic holds to ~0.5" out to four
# days. Past this span with only two samples we emit a diagnostic recommending a third. Faster movers
# (near-Earth objects) curve harder and go bad sooner, so this is guidance, not a guarantee.
LINEAR_TRACK_MAX_SPAN_HOURS = 12.0

SAMPLE_MJD_KEY = "mjd"
SAMPLE_RA_KEY = "ra"
SAMPLE_DEC_KEY = "dec"


@dataclass(frozen=True)
class TrackSample:
    """
        One user-supplied sample of the moving target: where it was, and when.

        mjd is the MJD (UTC) of the *exposure midpoint* of the frame the user marked it on, which is
        the position the object's trail is centred on. Callers sending exposure start times instead
        bias every predicted position by (exposure_time / 2) x apparent_rate -- sub-arcsecond for
        short exposures on a slow mover, but several arcseconds for long exposures on a fast one.
    """
    mjd: float
    ra_deg: float
    dec_deg: float


# eq=False because the coefficient and basis fields are numpy arrays: the generated __eq__ would
# compare them elementwise and raise on the ambiguous truth value of the resulting array. Identity
# comparison is all a track is ever needed for.
@dataclass(frozen=True, eq=False)
class TargetTrack:
    """
        A polynomial track through user-supplied sample positions, evaluated per frame.

        The fit is done in a gnomonic tangent plane about the mean sample direction rather than
        directly in RA/Dec, which removes the RA wrap at 0h, the cos(dec) compression of RA, and the
        degeneracy at the poles in one step. Over the hours-to-days arcs this mode targets the
        tangent plane is very nearly flat, so a low-order polynomial in the projected coordinates
        tracks the real apparent motion closely.
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
        """Earliest and latest sample time; frames outside this range are extrapolated, not interpolated."""
        times = [sample.mjd for sample in self.samples]
        return min(times), max(times)

    @property
    def sample_span_hours(self) -> float:
        first, last = self.sample_mjd_span
        return (last - first) * 24.0

    def position_at(self, mjd: float) -> tuple[float, float]:
        """
            Target RA/Dec (degrees) at an arbitrary time, by evaluating the fitted polynomial in the
            tangent plane and projecting back onto the sky.
        """
        elapsed = float(mjd) - self.reference_mjd
        xi = float(np.polyval(self.xi_coefficients, elapsed))
        eta = float(np.polyval(self.eta_coefficients, elapsed))
        return _deproject(self.radial_axis, self.east_axis, self.north_axis, xi, eta)

    def covers(self, mjd: float) -> bool:
        """Whether a time falls inside the sample span (an interpolation rather than an extrapolation)."""
        first, last = self.sample_mjd_span
        return first <= float(mjd) <= last


def track_samples_from_input(raw_samples: Any, *, minimum: int = MINIMUM_TRACK_SAMPLES) -> tuple[TrackSample, ...]:
    """
        Parses the user-supplied track samples into TrackSample records, sorted by time.

        Each sample is a mapping with "mjd", "ra" and "dec" -- decimal degrees, and MJD (UTC) of the
        exposure midpoint. Deliberately carries no frame identity: samples are just positions on the
        sky, so they need not come from the submitted frames at all. Raises ValueError on anything
        malformed; callers wrap it in their own error type.

        minimum is how many samples the caller needs: fitting a track needs the default two, but a
        fixed-target operation reuses this parser for a single {mjd, ra, dec} position (minimum=1),
        where the mjd is carried but unused. The distinct-times check only applies once more than one
        sample is required -- a lone sample has nothing to be distinct from.
    """
    if not isinstance(raw_samples, Sequence) or isinstance(raw_samples, (str, bytes)):
        raise ValueError("Track samples must be a list of {mjd, ra, dec} entries.")
    if len(raw_samples) < minimum:
        raise ValueError(
            f"A target track needs at least {minimum} sample position(s), got {len(raw_samples)}."
        )

    samples: list[TrackSample] = []
    for index, raw_sample in enumerate(raw_samples):
        if not isinstance(raw_sample, Mapping):
            raise ValueError(f"Track sample {index} must be a mapping with {SAMPLE_MJD_KEY}/{SAMPLE_RA_KEY}/{SAMPLE_DEC_KEY}.")
        try:
            mjd = float(raw_sample[SAMPLE_MJD_KEY])
            ra_deg = float(raw_sample[SAMPLE_RA_KEY])
            dec_deg = float(raw_sample[SAMPLE_DEC_KEY])
        except KeyError as exc:
            raise ValueError(f"Track sample {index} is missing {exc.args[0]!r}.") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Track sample {index} has a non-numeric {SAMPLE_MJD_KEY}/{SAMPLE_RA_KEY}/{SAMPLE_DEC_KEY}.") from exc
        if not (math.isfinite(mjd) and math.isfinite(ra_deg) and math.isfinite(dec_deg)):
            raise ValueError(f"Track sample {index} has non-finite values.")
        if not -90.0 <= dec_deg <= 90.0:
            raise ValueError(f"Track sample {index} has dec {dec_deg} outside [-90, 90].")
        samples.append(TrackSample(mjd=mjd, ra_deg=ra_deg, dec_deg=dec_deg))

    samples.sort(key=lambda sample: sample.mjd)
    if minimum >= MINIMUM_TRACK_SAMPLES and len({sample.mjd for sample in samples}) < MINIMUM_TRACK_SAMPLES:
        raise ValueError("Track samples must be at two or more distinct times.")
    return tuple(samples)


def fit_target_track(samples: Sequence[TrackSample]) -> TargetTrack:
    """
        Fits a track through the sample positions, for predicting where the target is on each frame.

        The polynomial degree follows the number of *distinct* sample times: two give a line, three or
        more a quadratic, capped at MAX_TRACK_FIT_ORDER. Over-determined fits are solved by least
        squares, so extra samples reduce the influence of an imprecise click rather than forcing the
        curve through every one of them.
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

    directions = np.array([_unit_vector(sample.ra_deg, sample.dec_deg) for sample in ordered])
    radial_axis, east_axis, north_axis = _tangent_basis(directions.mean(axis=0))

    # Project each sample into the tangent plane. The denominator is the cosine of the angle from the
    # plane's centre; samples more than 90 degrees away would project behind the observer, which for a
    # short-arc track means the samples are not of the same object.
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
    """
        Mean apparent rate of motion along the fitted track, for diagnostics and trail-length
        guidance (a target moving fast enough to streak within one exposure loses flux from a
        circular aperture).
    """
    first, last = track.sample_mjd_span
    if last <= first:
        return 0.0
    start = track.position_at(first)
    end = track.position_at(last)
    minutes = (last - first) * 24.0 * 60.0
    return _angular_separation_arcsec(start, end) / minutes


def _unit_vector(ra_deg: float, dec_deg: float) -> np.ndarray:
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    return np.array([math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)])


def _tangent_basis(mean_direction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        Right-handed (radial, east, north) orthonormal basis at a direction on the sky.

        East is the direction of increasing RA, taken perpendicular to both the pole and the line of
        sight. Exactly at a celestial pole that construction is degenerate and any perpendicular pair
        will do, so an arbitrary one is chosen -- the track is still fitted correctly, only the
        labelling of the two in-plane axes is arbitrary there.
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


def _angular_separation_arcsec(first: tuple[float, float], second: tuple[float, float]) -> float:
    a = _unit_vector(*first)
    b = _unit_vector(*second)
    return math.degrees(math.atan2(float(np.linalg.norm(np.cross(a, b))), float(np.dot(a, b)))) * 3600.0
