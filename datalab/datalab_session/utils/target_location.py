"""
Where the target is on each frame.

Aperture photometry needs one RA/Dec per frame before it touches a pixel, and there are three ways
to know it, decided by how the observation was taken rather than by anything the pipeline can infer:

- the target does not move, so one position serves the whole series (sidereal);
- the mount tracked the target, so each frame's header records where it was pointed;
- the mount tracked the stars while the target moved through the field, so nothing recorded its
  position and it has to be interpolated from positions a user marked, then looked for in the
  frame's own source catalog.

Each is a TargetLocator. The pipeline holds one and asks it for positions, so adding a fourth way
(an ephemeris service, say) is a new class here rather than another branch in the pipeline.
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

from datalab.datalab_session.utils.fits_metadata import frame_midpoint_mjd, target_radec_from_header
from datalab.datalab_session.utils.light_curve_errors import LightCurveError
from datalab.datalab_session.utils.moving_target_search import (
    DEFAULT_TRACK_SEARCH_RADIUS_ARCSEC,
    refine_positions_from_catalog,
)
from datalab.datalab_session.utils.target_track import (
    LINEAR_TRACK_MAX_SPAN_HOURS,
    MAX_TRACK_FIT_ORDER,
    TrackSample,
    fit_target_track,
    track_rate_arcsec_per_minute,
)

if TYPE_CHECKING:  # avoid a runtime import cycle with aperture_light_curve
    from datalab.datalab_session.utils.aperture_light_curve import FrameContext

log = logging.getLogger()
log.setLevel(logging.INFO)


@dataclass(frozen=True)
class TargetPositions:
    """
        The target's RA/Dec (degrees) on every frame, and what is worth saying about how they were
        arrived at: which frames were extrapolated, whether a catalog search confirmed them, and so
        on. A locator that simply knows the answer returns no diagnostics.
    """
    by_frame: dict[str, tuple[float, float]]
    diagnostics: list[str] = field(default_factory=list)


class TargetLocator(ABC):
    """Knows where the target is on each frame. See the module docstring for the three kinds."""

    @abstractmethod
    def locate(self, frames: Sequence["FrameContext"]) -> TargetPositions:
        """
            RA/Dec per frame, keyed by fits_path, for every frame given.

            Raises LightCurveError when the target cannot be located at all, which is fatal: there is
            nothing to measure. A locator that is merely unsure returns its best positions and says
            so in diagnostics, because a faint target that was not detected is still presumed present.
        """


@dataclass(frozen=True)
class FixedPosition(TargetLocator):
    """A target that does not move against the stars: one position, every frame."""

    ra_deg: float
    dec_deg: float

    def locate(self, frames: Sequence["FrameContext"]) -> TargetPositions:
        position = (float(self.ra_deg), float(self.dec_deg))
        return TargetPositions(by_frame={frame.fits_path: position for frame in frames})


@dataclass(frozen=True)
class EphemerisHeaders(TargetLocator):
    """
        A moving target whose frames were tracked non-sidereally, so the mount was following it and
        each frame's ephemeris keywords (CAT-RA/CAT-DEC) record where it was.

        A frame missing those keywords is fatal rather than skippable: without them there is no way
        to know where the object was, and guessing from neighbouring frames would invent an ephemeris.
    """

    def locate(self, frames: Sequence["FrameContext"]) -> TargetPositions:
        positions: dict[str, tuple[float, float]] = {}
        for frame in frames:
            try:
                positions[frame.fits_path] = target_radec_from_header(frame.header)
            except ValueError as exc:
                raise LightCurveError(
                    f"Cannot read moving-target position for {frame.fits_path}: {exc}"
                ) from exc
            log.info(
                "Aperture Photometry moving-target position: "
                f"frame={frame.fits_path}, ra={positions[frame.fits_path][0]:.8f}, "
                f"dec={positions[frame.fits_path][1]:.8f}"
            )
        return TargetPositions(by_frame=positions)


@dataclass(frozen=True)
class FittedTrack(TargetLocator):
    """
        A moving target on sidereally-tracked frames, where nothing recorded its position.

        A polynomial is fitted through the positions a user marked and evaluated at each frame's
        exposure midpoint, not its start, because the midpoint is what the target's trail is centred
        on. That prediction is then used as a search position: the frame's own source catalog is
        checked for a detection near it that is not a field star, and the track refitted through the
        detections that move consistently. Where that succeeds the aperture lands on a real
        detection; where it does not the interpolated position stands, so a faint or uncatalogued
        target still yields a measurement.

        Frames outside the sample span are extrapolated rather than dropped, since the fit remains
        the best information available, but extrapolation and a long arc carried by only two samples
        are both surfaced: they are the two ways a predicted position quietly drifts off the object.
    """

    samples: tuple[TrackSample, ...]
    search_radius_arcsec: float = DEFAULT_TRACK_SEARCH_RADIUS_ARCSEC

    def locate(self, frames: Sequence["FrameContext"]) -> TargetPositions:
        if not self.samples:
            raise LightCurveError(
                "Locating a moving target on sidereally-tracked frames needs the positions it was "
                "identified at on two or more frames."
            )
        try:
            track = fit_target_track(self.samples)
        except ValueError as exc:
            raise LightCurveError(f"Cannot fit a target track from the supplied samples: {exc}") from exc

        diagnostics = [
            f"Fitted a degree-{track.order} target track from {len(track.samples)} sample(s) over a "
            f"{track.sample_span_hours:.2f} h arc, mean rate "
            f"{track_rate_arcsec_per_minute(track):.3f} arcsec/min."
        ]

        frame_times: list[tuple[str, float]] = []
        extrapolated: list[str] = []
        for frame in frames:
            try:
                midpoint_mjd = frame_midpoint_mjd(frame.header, fallback_start=frame.date_obs)
            except ValueError as exc:
                raise LightCurveError(
                    f"Cannot determine an observation time for {frame.fits_path}: {exc}"
                ) from exc
            frame_times.append((frame.fits_path, midpoint_mjd))
            if not track.covers(midpoint_mjd):
                extrapolated.append(os.path.basename(frame.fits_path))

        refinement = refine_positions_from_catalog(
            frame_times=frame_times,
            catalog_rows_by_frame={frame.fits_path: frame.second_hdu_rows for frame in frames},
            track=track,
            samples=track.samples,
            search_radius_arcsec=self.search_radius_arcsec,
        )
        positions = refinement.positions
        diagnostics.extend(refinement.diagnostics)
        for fits_path, midpoint_mjd in frame_times:
            log.info(
                "Aperture Photometry tracked-target position: "
                f"frame={fits_path}, midpoint_mjd={midpoint_mjd:.8f}, "
                f"ra={positions[fits_path][0]:.8f}, dec={positions[fits_path][1]:.8f}"
            )

        if extrapolated:
            diagnostics.append(
                f"Extrapolated {len(extrapolated)} frame(s) outside the sample time span: "
                f"{', '.join(extrapolated)}."
            )
        if track.order < MAX_TRACK_FIT_ORDER and track.sample_span_hours > LINEAR_TRACK_MAX_SPAN_HOURS:
            diagnostics.append(
                f"Track is a straight line from {len(track.samples)} samples over a "
                f"{track.sample_span_hours:.1f} h arc. Tracks curve beyond about "
                f"{LINEAR_TRACK_MAX_SPAN_HOURS:.0f} h -- identify the target on a third, mid-series "
                "frame to fit a curve."
            )
        return TargetPositions(by_frame=positions, diagnostics=diagnostics)
