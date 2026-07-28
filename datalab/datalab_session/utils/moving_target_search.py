import logging
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from datalab.datalab_session.utils.fits_metadata import optional_float
from datalab.datalab_session.utils.geometry import (
    angular_distance_arcsec,
    angular_distances_arcsec,
    unit_vectors,
)
from datalab.datalab_session.utils.target_track import (
    TargetTrack,
    TrackSample,
    fit_target_track,
)


log = logging.getLogger()
log.setLevel(logging.INFO)


# How far from the predicted position to look for the target. An interpolated track is only as good
# as the samples and the arc it spans, so the search has to be wider than the aperture -- but every
# extra arcsecond admits more field stars to be confused with, so this is a deliberate compromise
# rather than a generous default. Exposed as a parameter for fast movers and long arcs.
DEFAULT_TRACK_SEARCH_RADIUS_ARCSEC = 10.0

# Two catalog detections within this angular distance, on different frames, are the same source.
# Comfortably larger than the astrometric scatter between frames and much smaller than the motion a
# moving target accumulates between them.
STATIC_SOURCE_MATCH_ARCSEC = 1.5
# A source found at the same sky position on at least this many *other* frames is stationary -- a
# field star, not the target. Two frames is too weak (a slow mover can sit inside the match radius
# across a pair of frames taken minutes apart); three separates them reliably. On a series with too
# few discriminating frames to reach it the threshold drops to what those frames can supply, so the
# test still fires rather than silently passing every field star.
STATIC_SOURCE_MIN_FRAMES = 3
# Below this many frames there are not enough independent looks to call anything stationary at all,
# so the refinement is abandoned rather than run with a test that cannot fail.
MIN_FRAMES_FOR_STATIONARITY_TEST = 3
# A frame only tells you something about a candidate's stationarity if the target is predicted to
# have moved away from that candidate's position by then. Below this many such frames the test has
# too little to go on and is skipped for that frame rather than run on one comparison.
MIN_DISCRIMINATING_FRAMES = 2

# Motion-consistency clipping. Picks are fitted against a track and those lying more than this many
# sigma off are rejected as mis-identifications.
TRACK_RESIDUAL_CLIP_SIGMA = 3.0
# Floor on the clipping scale, so a run whose picks happen to fit tightly does not start rejecting
# perfectly good picks over sub-arcsecond residuals.
MIN_TRACK_RESIDUAL_ARCSEC = 1.0
TRACK_CLIP_ITERATIONS = 3
# Below this many surviving picks there is nothing to cross-check, so the refinement is abandoned in
# favour of the user's own samples rather than trusted on the strength of one or two detections.
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


@dataclass(frozen=True)
class _SearchContext:
    """
        Everything the per-frame search reads: where the target is predicted to be, what each frame
        catalogued, how far to look, and which pairs of frames the target is predicted to have moved
        between (see _discriminating_frames).

        One object because the whole set is fixed for the search and every helper needs most of it.
        `discriminating` is held as the boolean matrix rather than per-frame path lists: at 999
        frames the lists are ~9 MB of duplicated strings for something only ever counted and
        iterated.
    """
    frame_times: Sequence[tuple[str, float]]
    predictions: Mapping[str, tuple[float, float]]
    catalog_arrays: Mapping[str, Mapping[str, np.ndarray]]
    search_radius_arcsec: float
    discriminating: np.ndarray

    @classmethod
    def build(
        cls,
        *,
        frame_times: Sequence[tuple[str, float]],
        catalog_rows_by_frame: Mapping[str, Sequence[Mapping[str, Any]]],
        track: TargetTrack,
        search_radius_arcsec: float,
    ) -> "_SearchContext":
        predictions = {path: track.position_at(mjd) for path, mjd in frame_times}
        return cls(
            frame_times=frame_times,
            predictions=predictions,
            catalog_arrays=_catalog_arrays(catalog_rows_by_frame),
            search_radius_arcsec=search_radius_arcsec,
            discriminating=_discriminating_frames(frame_times, predictions),
        )

    @property
    def frame_paths(self) -> list[str]:
        return [path for path, _ in self.frame_times]

    @property
    def frame_count(self) -> int:
        return len(self.frame_times)

    def predicted(self, fits_path: str) -> tuple[float, float]:
        return self.predictions[fits_path]

    def catalog(self, fits_path: str) -> Mapping[str, np.ndarray] | None:
        return self.catalog_arrays.get(fits_path)

    def discriminating_paths(self, frame_index: int) -> list[str]:
        """The frames the target is predicted to have moved away from by, for one frame."""
        return [
            path
            for path, has_moved in zip(self.frame_paths, self.discriminating[frame_index])
            if has_moved
        ]

    def inconclusive_frames(self) -> list[str]:
        """Frames with too few discriminating partners for the field-star test to mean anything."""
        return [
            path
            for path, row in zip(self.frame_paths, self.discriminating)
            if int(row.sum()) < MIN_DISCRIMINATING_FRAMES
        ]


@dataclass
class TrackRefinement:
    """
        Outcome of searching for the target near the predicted track and cross-checking the picks.

        positions is what the pipeline measures at: the refined track where the refinement held, and
        the sample track everywhere it did not. Frames with no pick still get a position -- a predicted
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
    samples: Sequence[TrackSample],
    search_radius_arcsec: float = DEFAULT_TRACK_SEARCH_RADIUS_ARCSEC,
) -> TrackRefinement:
    """
        Finds the moving target near its predicted position on each frame and refines the track.

        Works from the source catalogs the pipeline has already read, so it costs no extra image I/O.
        Per frame: collect catalog sources within search_radius_arcsec of the prediction, discard
        those that appear at the same sky position on several frames as field stars, take the nearest
        survivor. The picks are then fitted to a track and sigma-clipped, so one frame where a star
        slipped through does not drag the series.

        frame_times pairs each frame path with the MJD its position is evaluated at.
    """
    predictions = {path: track.position_at(mjd) for path, mjd in frame_times}
    if len(frame_times) < MIN_FRAMES_FOR_STATIONARITY_TEST:
        return TrackRefinement(
            positions=dict(predictions),
            diagnostics=[
                f"Skipped the catalog search: {len(frame_times)} frame(s) is too few to identify a "
                "moving target. Measured at the interpolated positions."
            ],
        )

    context = _SearchContext.build(
        frame_times=frame_times,
        catalog_rows_by_frame=catalog_rows_by_frame,
        track=track,
        search_radius_arcsec=search_radius_arcsec,
    )
    diagnostics: list[str] = []
    inconclusive = context.inconclusive_frames()
    if inconclusive:
        diagnostics.append(
            f"Predicted motion is under {STATIC_SOURCE_MATCH_ARCSEC:.1f} arcsec between "
            f"{len(inconclusive)} of {context.frame_count} frame(s) and the rest of the series, so a "
            "field star cannot be told apart from the target there; the nearest catalogued source "
            "to the predicted position was used."
        )

    picks, frames_without_pick = _collect_picks(context)
    accepted, rejected, rejection = _trustworthy_picks(picks, context)
    if rejection is not None:
        return TrackRefinement(
            positions=dict(predictions),
            rejected_picks=rejected,
            frames_without_pick=frames_without_pick,
            diagnostics=diagnostics + [rejection],
        )

    # Refit against the accepted picks together with the user's own samples: the picks are precise
    # but automatic, the samples are coarse but human-verified, and keeping both means a run where the
    # search drifted onto a companion is still anchored to the positions the user confirmed.
    try:
        refined_track = fit_target_track(
            [TrackSample(mjd=pick.mjd, ra_deg=pick.ra_deg, dec_deg=pick.dec_deg) for pick in accepted]
            + list(samples)
        )
    except ValueError as exc:
        # Same graceful fallback as every other failure here: the interpolated positions still stand,
        # so the run continues on the user's own samples rather than aborting the whole operation.
        return TrackRefinement(
            positions=dict(predictions),
            rejected_picks=rejected,
            frames_without_pick=frames_without_pick,
            diagnostics=diagnostics + [
                f"Could not refit a track through the {len(accepted)} located candidate(s) and the "
                f"supplied samples ({exc}). Measured at the interpolated positions."
            ],
        )

    positions = {fits_path: refined_track.position_at(mjd) for fits_path, mjd in frame_times}
    residual_rms_arcsec = _residual_rms_arcsec(accepted, refined_track)
    moved_arcsec = [
        angular_distance_arcsec(*predictions[path], *positions[path]) for path, _ in frame_times
    ]
    diagnostics.append(
        f"Located the target in the catalog of {len(accepted)} of {context.frame_count} frames and "
        f"refined the track (residual RMS {residual_rms_arcsec:.2f} arcsec, positions moved "
        f"a median {float(np.median(moved_arcsec)):.2f} arcsec)."
    )
    if rejected:
        diagnostics.append(
            f"Rejected {len(rejected)} inconsistently-moving candidate(s): "
            f"{', '.join(pick.source_id for pick in rejected)}."
        )
    if frames_without_pick:
        diagnostics.append(
            f"No catalogued source near the predicted track on {len(frames_without_pick)} "
            "frame(s); measured at the predicted position."
        )
    return TrackRefinement(
        positions=positions,
        picks=accepted,
        rejected_picks=rejected,
        frames_without_pick=frames_without_pick,
        refined_track=refined_track,
        residual_rms_arcsec=residual_rms_arcsec,
        diagnostics=diagnostics,
    )


def _collect_picks(context: _SearchContext) -> tuple[list[TargetPick], list[str]]:
    """Selects the target on each frame; returns the picks and the frames where nothing was found."""
    picks: list[TargetPick] = []
    frames_without_pick: list[str] = []
    for frame_index, (fits_path, mjd) in enumerate(context.frame_times):
        pick = _select_target_on_frame(
            context,
            fits_path=fits_path,
            mjd=mjd,
            discriminating_paths=context.discriminating_paths(frame_index),
        )
        if pick is None:
            frames_without_pick.append(fits_path)
        else:
            picks.append(pick)
    return picks, frames_without_pick


def _trustworthy_picks(
    picks: Sequence[TargetPick],
    context: _SearchContext,
) -> tuple[list[TargetPick], list[TargetPick], str | None]:
    """
        Decides whether the picks can be trusted as the moving target.

        Returns the accepted picks, those clipped as inconsistent, and a diagnostic explaining a
        refusal (None when sound). Three ways to fail: too few candidates to cross-check, too few
        surviving the motion-consistency clip, or a set that barely moves while the track says it
        should have. That last is a field star no residual test would catch, so the observed spread
        is judged against the spread the track predicts rather than a fixed distance: a set that
        barely moves because the target barely moves is not a failure.
    """
    if len(picks) < MIN_ACCEPTED_PICKS:
        return [], [], (
            f"Found candidates on only {len(picks)} of {context.frame_count} frames within "
            f"{context.search_radius_arcsec:.1f} arcsec of the predicted track, below the {MIN_ACCEPTED_PICKS} "
            "needed to cross-check them. Measured at the interpolated positions."
        )

    accepted, rejected = _clip_inconsistent_picks(picks)
    if len(accepted) < MIN_ACCEPTED_PICKS:
        return [], rejected, (
            f"Only {len(accepted)} of {len(picks)} candidates moved consistently with a single track. "
            "Measured at the interpolated positions."
        )

    spread = _maximum_separation_arcsec([(pick.ra_deg, pick.dec_deg) for pick in accepted])
    predicted_spread = _maximum_separation_arcsec([context.predicted(pick.fits_path) for pick in accepted])
    if spread < STATIONARY_PICK_SPREAD_ARCSEC <= predicted_spread:
        return [], rejected, (
            f"Discarded {len(accepted)} candidates spanning only {spread:.2f} arcsec across the "
            f"series while the track predicts {predicted_spread:.2f} arcsec of motion: a stationary "
            "source, not a moving target. Measured at the interpolated positions."
        )
    return accepted, rejected, None


def _catalog_arrays(
    catalog_rows_by_frame: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, np.ndarray]]:
    """
        Repacks each frame's catalog into numpy arrays once, so the stationarity test can sweep every
        frame's full catalog per candidate without paying Python-level iteration each time.
    """
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for fits_path, rows in catalog_rows_by_frame.items():
        ra = np.array([optional_float(row.get("ra")) for row in rows], dtype=float)
        dec = np.array([optional_float(row.get("dec")) for row in rows], dtype=float)
        identifiers = np.array(
            [str(row.get("id") or row.get("name") or index) for index, row in enumerate(rows)],
            dtype=object,
        )
        finite = np.isfinite(ra) & np.isfinite(dec)
        arrays[fits_path] = {"ra": ra[finite], "dec": dec[finite], "id": identifiers[finite]}
    return arrays


def _select_target_on_frame(
    context: _SearchContext,
    *,
    fits_path: str,
    mjd: float,
    discriminating_paths: Sequence[str],
) -> TargetPick | None:
    """
        Picks the most likely moving-target detection on one frame.

        Sources within the search radius are ranked by distance from the prediction, and the first
        that is not also present at the same sky position on several other frames is taken. At these
        search radii a field star is far likelier to fall near the track than the target is to be
        missing, so proximity alone would confidently return the wrong source. Where the target
        moves too little for that test to mean anything it is skipped and the nearest source taken.
    """
    frame_catalog = context.catalog(fits_path)
    if frame_catalog is None or frame_catalog["ra"].size == 0:
        return None

    predicted_ra_deg, predicted_dec_deg = context.predicted(fits_path)
    separations = angular_distances_arcsec(
        predicted_ra_deg, predicted_dec_deg, frame_catalog["ra"], frame_catalog["dec"]
    )
    within = np.flatnonzero(separations <= context.search_radius_arcsec)
    if within.size == 0:
        return None

    for index in within[np.argsort(separations[within])]:
        candidate_ra = float(frame_catalog["ra"][index])
        candidate_dec = float(frame_catalog["dec"][index])
        if _is_stationary(context, candidate_ra, candidate_dec, discriminating_paths):
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


def _static_source_min_frames(discriminating_count: int) -> int:
    """
        How many discriminating frames a source must appear on to count as stationary.

        The constant assumes there are always at least three frames the target has moved away from
        by. Where fewer exist that threshold can never be reached, which would quietly turn the
        field-star test into a no-op, so it falls back to every discriminating frame there is.
    """
    return min(STATIC_SOURCE_MIN_FRAMES, max(discriminating_count, 1))


def _discriminating_frames(
    frame_times: Sequence[tuple[str, float]],
    predictions: Mapping[str, tuple[float, float]],
) -> np.ndarray:
    """
        Boolean matrix: [i, j] is True where the target is predicted to have moved far enough
        between frames i and j for a same-position match between them to mean anything.

        A source reappearing at one sky position is evidence of a field star only where the target
        would have moved off that position by then. Near a stationary point the target matches its
        own detections everywhere, so an unconditional test rejects what it is looking for.
    """
    paths = [path for path, _ in frame_times]
    directions = unit_vectors(
        [predictions[path][0] for path in paths], [predictions[path][1] for path in paths]
    )
    separations = np.degrees(np.arccos(np.clip(directions @ directions.T, -1.0, 1.0))) * 3600.0
    # The diagonal is zero, so a frame is never discriminating against itself.
    return separations > STATIC_SOURCE_MATCH_ARCSEC


def _is_stationary(
    context: _SearchContext,
    ra_deg: float,
    dec_deg: float,
    discriminating_paths: Sequence[str],
) -> bool:
    """
        Whether a source appears at this same sky position on enough other frames to be a field star.

        Only frames the target is predicted to have moved away from are consulted; below
        MIN_DISCRIMINATING_FRAMES of them the question cannot be answered and the source is not
        rejected, since a slow target is far likelier than a series where every source is static.
    """
    if len(discriminating_paths) < MIN_DISCRIMINATING_FRAMES:
        return False
    # Never more than the constant, and never more than the discriminating frames can supply.
    required_matches = min(STATIC_SOURCE_MIN_FRAMES, len(discriminating_paths))
    matches = 0
    for fits_path in discriminating_paths:
        frame_catalog = context.catalog(fits_path)
        if frame_catalog is None or frame_catalog["ra"].size == 0:
            continue
        # Declination alone bounds the angular separation, so this rejects almost every row without
        # evaluating the full spherical distance.
        nearby = np.flatnonzero(np.abs(frame_catalog["dec"] - dec_deg) <= STATIC_SOURCE_MATCH_ARCSEC / 3600.0)
        if nearby.size == 0:
            continue
        separations = angular_distances_arcsec(
            ra_deg, dec_deg, frame_catalog["ra"][nearby], frame_catalog["dec"][nearby]
        )
        if np.any(separations <= STATIC_SOURCE_MATCH_ARCSEC):
            matches += 1
            if matches >= required_matches:
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
                [TrackSample(mjd=pick.mjd, ra_deg=pick.ra_deg, dec_deg=pick.dec_deg) for pick in accepted]
            )
        except ValueError:
            break
        residuals = np.array([
            angular_distance_arcsec(pick.ra_deg, pick.dec_deg, *candidate_track.position_at(pick.mjd))
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
        angular_distance_arcsec(pick.ra_deg, pick.dec_deg, *track.position_at(pick.mjd))
        for pick in picks
    ]
    return float(np.sqrt(np.mean(np.square(residuals))))


def _maximum_separation_arcsec(positions: Sequence[tuple[float, float]]) -> float:
    """
        Largest angular distance between any two of these RA/Dec positions: how far the supposed
        target moved at all, whether those positions are the picks or the track's predictions.

        Every pair still counts, but as one dot product of the unit-vector matrix with itself rather
        than a Python loop calling a distance function per pair.
    """
    if len(positions) < 2:
        return 0.0
    directions = unit_vectors([ra for ra, _ in positions], [dec for _, dec in positions])
    smallest_cosine = float(np.min(np.clip(directions @ directions.T, -1.0, 1.0)))
    return math.degrees(math.acos(smallest_cosine)) * 3600.0
