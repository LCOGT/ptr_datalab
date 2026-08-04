"""Where the target is on each frame: one TargetLocator per way of knowing."""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

from datalab.datalab_session.utils.fits_metadata import frame_midpoint_mjd, target_radec_from_header
from datalab.datalab_session.exceptions import LightCurveError
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

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TargetPositions:
    """The target's RA/Dec (degrees) per frame, with anything worth saying about how they were found."""
    by_frame: dict[str, tuple[float, float]]
    diagnostics: list[str] = field(default_factory=list)


class TargetLocator(ABC):
    @abstractmethod
    def locate(self, frames: Sequence["FrameContext"]) -> TargetPositions:
        """
            RA/Dec per frame, keyed by fits_path. Raises LightCurveError only when the target cannot
            be located at all; a locator that is merely unsure returns its best positions and says so.
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
        A target whose mount tracked it, so each frame's CAT-RA/CAT-DEC record where it was. A frame
        missing them is fatal: interpolating from neighbours would invent an ephemeris.
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

        diagnostics: list[str] = []
        # CAT-RA/CAT-DEC carry the requested target position on every frame, not just non-sidereal
        # ones, and stay fixed when the mount tracked the stars. Measuring at a fixed position is
        # then silently wrong for anything that moves, so say so rather than produce a light curve.
        if len(positions) > 1 and len(set(positions.values())) == 1:
            diagnostics.append(
                f"CAT-RA/CAT-DEC are identical on all {len(positions)} frames, so the mount was not "
                "tracking a moving target. Measured at that one position; if the object moves, use "
                "Moving Target Aperture Photometry and mark it on two or more frames instead."
            )
        return TargetPositions(by_frame=positions, diagnostics=diagnostics)


@dataclass(frozen=True)
class FittedTrack(TargetLocator):
    """
        A moving target on sidereally-tracked frames, where nothing recorded its position.

        A track fitted through the positions a user marked predicts each frame's, and the frame's own
        catalog is then searched near that prediction. Where the search fails the prediction stands,
        so a faint or uncatalogued target still yields a measurement. Extrapolated frames and a long
        arc carried by two samples are surfaced as diagnostics.
    """

    samples: tuple[TrackSample, ...]
    search_radius_arcsec: float = DEFAULT_TRACK_SEARCH_RADIUS_ARCSEC

    def locate(self, frames: Sequence["FrameContext"]) -> TargetPositions:
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
