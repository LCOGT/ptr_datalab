import logging
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from datalab.datalab_session.utils.target_track import (
    TargetTrack,
    TrackSeed,
    fit_target_track,
)


log = logging.getLogger()
log.setLevel(logging.INFO)


# How far from the predicted position to look for the target. An interpolated track is only as good
# as the seeds and the arc it spans, so the search has to be wider than the aperture -- but every
# extra arcsecond admits more field stars to be confused with, so this is a deliberate compromise
# rather than a generous default. Exposed as a parameter for fast movers and long arcs.
DEFAULT_TRACK_SEARCH_RADIUS_ARCSEC = 10.0

# Two catalog detections within this angular distance, on different frames, are the same source.
# Comfortably larger than the astrometric scatter between frames and much smaller than the motion a
# moving target accumulates between them.
STATIC_SOURCE_MATCH_ARCSEC = 1.5
# A source found at the same sky position on at least this many frames is stationary -- a field star,
# not the target. Two frames is too weak (a slow mover can sit inside the match radius across a pair
# of frames taken minutes apart); three separates them reliably.
STATIC_SOURCE_MIN_FRAMES = 3

# Motion-consistency clipping. Picks are fitted against a track and those lying more than this many
# sigma off are rejected as mis-identifications.
TRACK_RESIDUAL_CLIP_SIGMA = 3.0
# Floor on the clipping scale, so a run whose picks happen to fit tightly does not start rejecting
# perfectly good picks over sub-arcsecond residuals.
MIN_TRACK_RESIDUAL_ARCSEC = 1.0
TRACK_CLIP_ITERATIONS = 3
# Below this many surviving picks there is nothing to cross-check, so the refinement is abandoned in
# favour of the user's own seeds rather than trusted on the strength of one or two detections.
MIN_ACCEPTED_PICKS = 3
# If every pick sits within this distance of every other, the "track" is a stationary source: the
# search locked onto one field star on every frame. Refuse it however well it fits.
STATIONARY_PICK_SPREAD_ARCSEC = 2.0


@dataclass(frozen=True)
class TargetPick:
    """A catalog source selected as the moving target on one frame."""
    fits_path: str
    mjd: float
    ra_deg: float
    dec_deg: float
    source_id: str
    offset_from_prediction_arcsec: float


@dataclass
class TrackRefinement:
    """
        Outcome of searching for the target near the predicted track and cross-checking the picks.

        positions is what the pipeline measures at: the refined track where the refinement held, and
        the seed track everywhere it did not. Frames with no pick still get a position -- a predicted
        one -- because the target is presumed present and simply undetected, which is a routine
        outcome for a faint object rather than a reason to drop the frame.
    """
    positions: dict[str, tuple[float, float]]
    picks: list[TargetPick] = field(default_factory=list)
    rejected_picks: list[TargetPick] = field(default_factory=list)
    frames_without_pick: list[str] = field(default_factory=list)
    refined_track: TargetTrack | None = None
    residual_rms_arcsec: float = math.nan
    diagnostics: list[str] = field(default_factory=list)


def refine_positions_from_catalog(
    *,
    frame_times: Sequence[tuple[str, float]],
    catalog_rows_by_frame: Mapping[str, Sequence[Mapping[str, Any]]],
    track: TargetTrack,
    seeds: Sequence[TrackSeed],
    search_radius_arcsec: float = DEFAULT_TRACK_SEARCH_RADIUS_ARCSEC,
) -> TrackRefinement:
    """
        Finds the moving target near its predicted position on each frame and refines the track.

        Works entirely from the frames' source catalogs, which the pipeline has already read, so it
        costs no extra image I/O and runs before any pixel is touched. On each frame the catalog
        sources within search_radius_arcsec of the prediction are collected, any that appear at the
        same sky position on several frames are discarded as field stars, and the nearest survivor is
        taken as the target. Those picks are then fitted against a track and sigma-clipped, so a
        single frame where a star slipped through does not drag the whole series.

        frame_times pairs each frame path with the MJD its position should be evaluated at.
    """
    predictions = {path: track.position_at(mjd) for path, mjd in frame_times}
    refinement = TrackRefinement(positions=dict(predictions))

    catalog_arrays = _catalog_arrays(catalog_rows_by_frame)
    picks: list[TargetPick] = []
    for fits_path, mjd in frame_times:
        predicted_ra, predicted_dec = predictions[fits_path]
        pick = _select_target_on_frame(
            fits_path=fits_path,
            mjd=mjd,
            predicted_ra_deg=predicted_ra,
            predicted_dec_deg=predicted_dec,
            catalog_arrays=catalog_arrays,
            search_radius_arcsec=search_radius_arcsec,
        )
        if pick is None:
            refinement.frames_without_pick.append(fits_path)
        else:
            picks.append(pick)

    if len(picks) < MIN_ACCEPTED_PICKS:
        refinement.diagnostics.append(
            f"Only {len(picks)} of {len(frame_times)} frames yielded a moving-target candidate within "
            f"{search_radius_arcsec:.1f} arcsec of the predicted track, fewer than the "
            f"{MIN_ACCEPTED_PICKS} needed to cross-check them, so the target was measured at the "
            "positions interpolated from your own sightings instead."
        )
        return refinement

    accepted, rejected = _clip_inconsistent_picks(picks)
    refinement.rejected_picks = rejected
    if len(accepted) < MIN_ACCEPTED_PICKS:
        refinement.diagnostics.append(
            f"Of {len(picks)} moving-target candidates, only {len(accepted)} moved consistently with "
            "a single track, so none were trusted and the target was measured at the positions "
            "interpolated from your own sightings instead."
        )
        return refinement

    spread = _maximum_separation_arcsec(accepted)
    if spread < STATIONARY_PICK_SPREAD_ARCSEC:
        refinement.diagnostics.append(
            f"The {len(accepted)} candidates selected as the target span only {spread:.2f} arcsec "
            "across the whole series, so they are a stationary source rather than a moving one -- "
            "most likely a field star sitting near the predicted track. They were discarded and the "
            "target was measured at the positions interpolated from your own sightings."
        )
        return refinement

    # Refit against the accepted picks together with the user's own sightings: the picks are precise
    # but automatic, the seeds are coarse but human-verified, and keeping both means a run where the
    # search drifted onto a companion is still anchored to the positions the user confirmed.
    refined_track = fit_target_track(
        [TrackSeed(mjd=pick.mjd, ra_deg=pick.ra_deg, dec_deg=pick.dec_deg) for pick in accepted]
        + list(seeds)
    )
    refinement.refined_track = refined_track
    refinement.picks = accepted
    refinement.residual_rms_arcsec = _residual_rms_arcsec(accepted, refined_track)

    for fits_path, mjd in frame_times:
        refinement.positions[fits_path] = refined_track.position_at(mjd)

    moved_arcsec = [
        _angular_distance_arcsec(predictions[path], refinement.positions[path])
        for path, _ in frame_times
    ]
    refinement.diagnostics.append(
        f"Located the moving target in the source catalog of {len(accepted)} of {len(frame_times)} "
        f"frames and refined the track through them (residual RMS {refinement.residual_rms_arcsec:.2f} "
        f"arcsec). Measured positions moved a median {float(np.median(moved_arcsec)):.2f} arcsec from "
        "the positions interpolated from your sightings alone."
    )
    if rejected:
        refinement.diagnostics.append(
            f"{len(rejected)} candidate(s) did not move consistently with the others and were "
            f"rejected as mis-identifications: {', '.join(pick.source_id for pick in rejected)}."
        )
    if refinement.frames_without_pick:
        refinement.diagnostics.append(
            f"{len(refinement.frames_without_pick)} frame(s) had no catalogued source near the "
            "predicted track -- the target is likely below the detection limit there -- so they were "
            "measured at the predicted position."
        )
    return refinement


def _catalog_arrays(
    catalog_rows_by_frame: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, np.ndarray]]:
    """
        Repacks each frame's catalog into numpy arrays once, so the stationarity test can sweep every
        frame's full catalog per candidate without paying Python-level iteration each time.
    """
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for fits_path, rows in catalog_rows_by_frame.items():
        ra = np.array([_as_float(row.get("ra")) for row in rows], dtype=float)
        dec = np.array([_as_float(row.get("dec")) for row in rows], dtype=float)
        identifiers = np.array(
            [str(row.get("id") or row.get("name") or index) for index, row in enumerate(rows)],
            dtype=object,
        )
        finite = np.isfinite(ra) & np.isfinite(dec)
        arrays[fits_path] = {"ra": ra[finite], "dec": dec[finite], "id": identifiers[finite]}
    return arrays


def _select_target_on_frame(
    *,
    fits_path: str,
    mjd: float,
    predicted_ra_deg: float,
    predicted_dec_deg: float,
    catalog_arrays: Mapping[str, Mapping[str, np.ndarray]],
    search_radius_arcsec: float,
) -> TargetPick | None:
    """
        Picks the most likely moving-target detection on one frame.

        Sources within the search radius are ranked by distance from the prediction, and the first
        one that is *not* also present on several other frames at the same sky position is taken.
        Stationarity is the discriminating feature: at the search radii involved a field star is far
        more likely to fall near the predicted track than the target is to be missing, so proximity
        alone would confidently return the wrong source.
    """
    frame_catalog = catalog_arrays.get(fits_path)
    if frame_catalog is None or frame_catalog["ra"].size == 0:
        return None

    separations = _angular_distances_arcsec(
        predicted_ra_deg, predicted_dec_deg, frame_catalog["ra"], frame_catalog["dec"]
    )
    within = np.flatnonzero(separations <= search_radius_arcsec)
    if within.size == 0:
        return None

    for index in within[np.argsort(separations[within])]:
        candidate_ra = float(frame_catalog["ra"][index])
        candidate_dec = float(frame_catalog["dec"][index])
        if _is_stationary(candidate_ra, candidate_dec, fits_path, catalog_arrays):
            continue
        return TargetPick(
            fits_path=fits_path,
            mjd=mjd,
            ra_deg=candidate_ra,
            dec_deg=candidate_dec,
            source_id=str(frame_catalog["id"][index]),
            offset_from_prediction_arcsec=float(separations[index]),
        )
    return None


def _is_stationary(
    ra_deg: float,
    dec_deg: float,
    own_fits_path: str,
    catalog_arrays: Mapping[str, Mapping[str, np.ndarray]],
) -> bool:
    """Whether a source appears at this same sky position on enough other frames to be a field star."""
    matches = 0
    for fits_path, frame_catalog in catalog_arrays.items():
        if fits_path == own_fits_path or frame_catalog["ra"].size == 0:
            continue
        # Declination alone bounds the angular separation, so this rejects almost every row without
        # evaluating the full spherical distance.
        nearby = np.flatnonzero(np.abs(frame_catalog["dec"] - dec_deg) <= STATIC_SOURCE_MATCH_ARCSEC / 3600.0)
        if nearby.size == 0:
            continue
        separations = _angular_distances_arcsec(
            ra_deg, dec_deg, frame_catalog["ra"][nearby], frame_catalog["dec"][nearby]
        )
        if np.any(separations <= STATIC_SOURCE_MATCH_ARCSEC):
            matches += 1
            if matches >= STATIC_SOURCE_MIN_FRAMES:
                return True
    return False


def _clip_inconsistent_picks(picks: Sequence[TargetPick]) -> tuple[list[TargetPick], list[TargetPick]]:
    """
        Rejects picks that do not lie on a common track with the rest.

        A star that slipped past the stationarity test appears as one position wildly off the line
        the other picks trace, so fitting a track through the picks and clipping on the residuals
        isolates it. The clip scale is floored so a tightly-fitting series does not start rejecting
        sound picks over sub-arcsecond scatter.
    """
    accepted = list(picks)
    rejected: list[TargetPick] = []
    for _ in range(TRACK_CLIP_ITERATIONS):
        if len(accepted) < MIN_ACCEPTED_PICKS:
            break
        try:
            candidate_track = fit_target_track(
                [TrackSeed(mjd=pick.mjd, ra_deg=pick.ra_deg, dec_deg=pick.dec_deg) for pick in accepted]
            )
        except ValueError:
            break
        residuals = np.array([
            _angular_distance_arcsec(
                (pick.ra_deg, pick.dec_deg), candidate_track.position_at(pick.mjd)
            )
            for pick in accepted
        ])
        scale = max(float(np.median(residuals)) * 1.4826, MIN_TRACK_RESIDUAL_ARCSEC)
        keep = residuals <= TRACK_RESIDUAL_CLIP_SIGMA * scale
        if bool(np.all(keep)):
            break
        rejected.extend(pick for pick, keeping in zip(accepted, keep) if not keeping)
        accepted = [pick for pick, keeping in zip(accepted, keep) if keeping]
    return accepted, rejected


def _residual_rms_arcsec(picks: Sequence[TargetPick], track: TargetTrack) -> float:
    if not picks:
        return math.nan
    residuals = [
        _angular_distance_arcsec((pick.ra_deg, pick.dec_deg), track.position_at(pick.mjd))
        for pick in picks
    ]
    return float(np.sqrt(np.mean(np.square(residuals))))


def _maximum_separation_arcsec(picks: Sequence[TargetPick]) -> float:
    """Largest angular distance between any two picks -- how far the supposed target moved at all."""
    largest = 0.0
    for index, first in enumerate(picks):
        for second in picks[index + 1:]:
            largest = max(
                largest,
                _angular_distance_arcsec((first.ra_deg, first.dec_deg), (second.ra_deg, second.dec_deg)),
            )
    return largest


def _as_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result


def _angular_distance_arcsec(first: tuple[float, float], second: tuple[float, float]) -> float:
    return float(_angular_distances_arcsec(first[0], first[1], np.array([second[0]]), np.array([second[1]]))[0])


def _angular_distances_arcsec(
    ra_deg: float,
    dec_deg: float,
    other_ra_deg: np.ndarray,
    other_dec_deg: np.ndarray,
) -> np.ndarray:
    """Angular distance from one position to an array of positions, in arcseconds."""
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    other_ra = np.radians(other_ra_deg)
    other_dec = np.radians(other_dec_deg)
    cos_angle = np.sin(dec) * np.sin(other_dec) + math.cos(dec) * np.cos(other_dec) * np.cos(ra - other_ra)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))) * 3600.0
