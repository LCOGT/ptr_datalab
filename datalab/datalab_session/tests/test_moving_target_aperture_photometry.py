from __future__ import annotations

import math
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from datalab.datalab_session.tests.test_aperture_photometry import gaussian_star
from datalab.datalab_session.utils.aperture_light_curve import (
    FrameContext,
    LightCurveError,
    TARGET_POSITION_TRACK,
    _track_target_positions,
    generate_light_curve,
)
from datalab.datalab_session.utils.comparison_calibration import COMPARISON_AUTO
from datalab.datalab_session.utils.fits_metadata import frame_midpoint_mjd
from datalab.datalab_session.utils.moving_target_search import (
    MIN_ACCEPTED_PICKS,
    refine_positions_from_catalog,
)
from datalab.datalab_session.utils.target_track import (
    LINEAR_TRACK_MAX_SPAN_HOURS,
    TrackSeed,
    fit_target_track,
    track_rate_arcsec_per_minute,
    track_seeds_from_input,
)


DEG_PER_PIXEL = 1.0 / 3600.0  # 1 arcsec/pixel
FLUX_ZERO_POINT = 25.0        # counts = 10 ** (-0.4 * (mag - FLUX_ZERO_POINT))
MJD_EPOCH = datetime(1858, 11, 17, tzinfo=timezone.utc)

BASE_RA = 100.0
BASE_DEC = 20.0


def _flux_for_mag(mag: float) -> float:
    return 10.0 ** (-0.4 * (mag - FLUX_ZERO_POINT))


def _to_mjd(moment: datetime) -> float:
    return (moment - MJD_EPOCH).total_seconds() / 86400.0


def _ra_offset_deg(arcsec: float, dec_deg: float = BASE_DEC) -> float:
    """RA offset in degrees that subtends `arcsec` on the sky at this declination."""
    return arcsec / 3600.0 / math.cos(math.radians(dec_deg))


def _header(width: int, height: int, date_obs: datetime, exposure_seconds: float) -> dict[str, Any]:
    """
        Header for a sidereally-tracked frame: the field center is fixed on the sky, so the stars sit
        at the same pixels every frame and the moving target is the only thing that shifts. This is
        the inverse of the non-sidereal harness, where the mount follows the target instead.
    """
    return {
        "DATE-OBS": date_obs.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
        "MJD-OBS": _to_mjd(date_obs),
        "EXPTIME": exposure_seconds,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "CRVAL1": BASE_RA,
        "CRVAL2": BASE_DEC,
        # CRPIX is 1-based (astropy world_to_pixel returns 0-based), so +1 puts the reference sky
        # position at 0-based pixel (width/2, height/2).
        "CRPIX1": width / 2.0 + 1.0,
        "CRPIX2": height / 2.0 + 1.0,
        "CD1_1": DEG_PER_PIXEL,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": DEG_PER_PIXEL,
        "GAIN": 1.7,
        "RDNOISE": 3.2,
    }


def build_tracked_frame_set(
    *,
    frame_count: int = 8,
    width: int = 140,
    height: int = 140,
    cadence_hours: float = 1.0,
    exposure_seconds: float = 60.0,
    target_rate_arcsec_per_hour: float = 6.0,
    target_curvature_arcsec: float = 0.0,
    target_flux_by_frame: list[float] | None = None,
    star_ra_offsets_arcsec: tuple[float, ...] = tuple(range(-55, 60, 10)),
    star_dec_offsets_arcsec: tuple[float, ...] = (-45.0, -25.0, 25.0, 45.0),
    star_mag: float = 15.0,
    include_target_in_catalog: bool = False,
) -> tuple[dict[str, dict[str, Any]], list[tuple[float, float, float]]]:
    """
        Builds a synthetic sidereally-tracked frame set with a target drifting across a fixed field.

        The target moves in RA at target_rate_arcsec_per_hour, optionally with a quadratic term in
        Dec (target_curvature_arcsec is the peak departure from a straight line, at mid-series), so
        the curved case can be told apart from the straight one. Returns the frames dict and the
        truth track as (mjd, ra_deg, dec_deg) at each frame's exposure midpoint.
    """
    start = datetime(2025, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
    if target_flux_by_frame is None:
        target_flux_by_frame = [_flux_for_mag(15.0)] * frame_count

    frames: dict[str, dict[str, Any]] = {}
    truth: list[tuple[float, float, float]] = []
    total_hours = cadence_hours * max(frame_count - 1, 1)

    for frame_index in range(frame_count):
        date_obs = start + timedelta(hours=cadence_hours * frame_index)
        header = _header(width, height, date_obs, exposure_seconds)
        wcs = WCS(header)

        # The target's position is defined at the exposure *midpoint*, which is where its light is
        # centred; the track fit has to agree or every prediction is half an exposure of motion off.
        midpoint = date_obs + timedelta(seconds=exposure_seconds / 2.0)
        elapsed_hours = (midpoint - start).total_seconds() / 3600.0
        target_ra = BASE_RA + _ra_offset_deg(target_rate_arcsec_per_hour * elapsed_hours)
        # A parabola through zero at both ends of the series, peaking at target_curvature_arcsec.
        fraction = elapsed_hours / total_hours if total_hours else 0.0
        target_dec = BASE_DEC + target_curvature_arcsec * 4.0 * fraction * (1.0 - fraction) / 3600.0
        truth.append((_to_mjd(midpoint), target_ra, target_dec))

        image = np.full((height, width), 100.0, dtype=float)
        target_x, target_y = (float(v) for v in wcs.world_to_pixel_values(target_ra, target_dec))
        gaussian_star(image, target_x, target_y, flux=target_flux_by_frame[frame_index], sigma=1.05)

        second_hdu: list[dict[str, Any]] = []
        star_index = 0
        for ra_off in star_ra_offsets_arcsec:
            for dec_off in star_dec_offsets_arcsec:
                star_index += 1
                star_ra = BASE_RA + _ra_offset_deg(ra_off)
                star_dec = BASE_DEC + dec_off / 3600.0
                x, y = (float(v) for v in wcs.world_to_pixel_values(star_ra, star_dec))
                if not (6.0 <= x <= width - 6.0 and 6.0 <= y <= height - 6.0):
                    continue
                flux = _flux_for_mag(star_mag)
                gaussian_star(image, x, y, flux=flux, sigma=1.05)
                second_hdu.append(
                    {
                        "id": f"star-{star_index:03d}",
                        "ra": star_ra,
                        "dec": star_dec,
                        "mag": star_mag,
                        "flux": flux,
                    }
                )

        if include_target_in_catalog:
            second_hdu.append(
                {
                    "id": "target-shadow",
                    "ra": target_ra,
                    "dec": target_dec,
                    "mag": 15.0,
                    "flux": target_flux_by_frame[frame_index],
                }
            )

        frames[f"frame_{frame_index + 1:02d}.fits"] = {
            "image": image,
            "header": header,
            "second_hdu": second_hdu,
        }

    return frames, truth


def _seeds_from_truth(truth: list[tuple[float, float, float]], indices: tuple[int, ...]) -> list[TrackSeed]:
    return [TrackSeed(mjd=truth[i][0], ra_deg=truth[i][1], dec_deg=truth[i][2]) for i in indices]


def _separation_arcsec(first: tuple[float, float], second: tuple[float, float]) -> float:
    ra1, dec1 = math.radians(first[0]), math.radians(first[1])
    ra2, dec2 = math.radians(second[0]), math.radians(second[1])
    cos_sep = (
        math.sin(dec1) * math.sin(dec2)
        + math.cos(dec1) * math.cos(dec2) * math.cos(ra1 - ra2)
    )
    return math.degrees(math.acos(min(1.0, max(-1.0, cos_sep)))) * 3600.0


class TestTargetTrackFitting(unittest.TestCase):
    """The track fit on its own, independent of any frames or photometry."""

    def test_two_seeds_fit_a_line_and_recover_a_linear_track(self) -> None:
        _, truth = build_tracked_frame_set(frame_count=9)
        track = fit_target_track(_seeds_from_truth(truth, (0, 8)))
        self.assertEqual(track.order, 1)
        for mjd, ra, dec in truth:
            self.assertLess(_separation_arcsec(track.position_at(mjd), (ra, dec)), 0.01)

    def test_three_seeds_fit_a_quadratic_and_beat_a_line_on_a_curved_track(self) -> None:
        _, truth = build_tracked_frame_set(frame_count=9, target_curvature_arcsec=12.0)
        linear = fit_target_track(_seeds_from_truth(truth, (0, 8)))
        quadratic = fit_target_track(_seeds_from_truth(truth, (0, 4, 8)))
        self.assertEqual(linear.order, 1)
        self.assertEqual(quadratic.order, 2)

        worst_linear = max(_separation_arcsec(linear.position_at(m), (r, d)) for m, r, d in truth)
        worst_quadratic = max(_separation_arcsec(quadratic.position_at(m), (r, d)) for m, r, d in truth)
        # The line must miss by roughly the injected curvature, and the curve must recover it.
        self.assertGreater(worst_linear, 8.0)
        self.assertLess(worst_quadratic, 0.05)

    def test_fit_order_is_capped_at_quadratic_and_uses_least_squares(self) -> None:
        _, truth = build_tracked_frame_set(frame_count=9, target_curvature_arcsec=6.0)
        track = fit_target_track(_seeds_from_truth(truth, (0, 2, 4, 6, 8)))
        self.assertEqual(track.order, 2)
        self.assertEqual(len(track.seeds), 5)

    def test_track_handles_the_ra_wrap_at_zero_hours(self) -> None:
        start = _to_mjd(datetime(2025, 3, 1, tzinfo=timezone.utc))
        seeds = [
            TrackSeed(mjd=start, ra_deg=359.99, dec_deg=5.0),
            TrackSeed(mjd=start + 0.25, ra_deg=0.01, dec_deg=5.0),
        ]
        track = fit_target_track(seeds)
        midpoint = track.position_at(start + 0.125)
        # Halfway between 359.99 and 0.01 is 0.0, not the 180.0 a naive average of the two would give.
        self.assertLess(min(midpoint[0], 360.0 - midpoint[0]), 0.001)
        self.assertAlmostEqual(midpoint[1], 5.0, places=4)

    def test_track_handles_a_seed_at_the_celestial_pole(self) -> None:
        start = _to_mjd(datetime(2025, 3, 1, tzinfo=timezone.utc))
        seeds = [
            TrackSeed(mjd=start, ra_deg=0.0, dec_deg=89.99),
            TrackSeed(mjd=start + 0.25, ra_deg=180.0, dec_deg=89.99),
        ]
        track = fit_target_track(seeds)
        midpoint = track.position_at(start + 0.125)
        self.assertGreater(midpoint[1], 89.9)

    def test_apparent_rate_is_reported(self) -> None:
        _, truth = build_tracked_frame_set(frame_count=9, target_rate_arcsec_per_hour=60.0)
        track = fit_target_track(_seeds_from_truth(truth, (0, 8)))
        self.assertAlmostEqual(track_rate_arcsec_per_minute(track), 1.0, delta=0.01)

    def test_seeds_are_sorted_by_time(self) -> None:
        _, truth = build_tracked_frame_set(frame_count=5)
        track = fit_target_track(_seeds_from_truth(truth, (4, 0, 2)))
        self.assertEqual([seed.mjd for seed in track.seeds], sorted(seed.mjd for seed in track.seeds))


class TestTrackSeedParsing(unittest.TestCase):
    def test_parses_and_sorts_valid_seeds(self) -> None:
        seeds = track_seeds_from_input(
            [
                {"mjd": 60002.0, "ra": 100.2, "dec": 20.0},
                {"mjd": 60000.0, "ra": 100.0, "dec": 20.0},
            ]
        )
        self.assertEqual(len(seeds), 2)
        self.assertEqual(seeds[0].mjd, 60000.0)
        self.assertEqual(seeds[1].ra_deg, 100.2)

    def test_rejects_fewer_than_two_seeds(self) -> None:
        with self.assertRaises(ValueError):
            track_seeds_from_input([{"mjd": 60000.0, "ra": 100.0, "dec": 20.0}])

    def test_rejects_missing_key(self) -> None:
        with self.assertRaises(ValueError):
            track_seeds_from_input([{"mjd": 60000.0, "ra": 100.0}, {"mjd": 60001.0, "ra": 100.1, "dec": 20.0}])

    def test_rejects_non_numeric_and_non_finite(self) -> None:
        with self.assertRaises(ValueError):
            track_seeds_from_input([{"mjd": 60000.0, "ra": "abc", "dec": 20.0}, {"mjd": 60001.0, "ra": 1.0, "dec": 2.0}])
        with self.assertRaises(ValueError):
            track_seeds_from_input([{"mjd": 60000.0, "ra": float("nan"), "dec": 20.0}, {"mjd": 60001.0, "ra": 1.0, "dec": 2.0}])

    def test_rejects_out_of_range_declination(self) -> None:
        with self.assertRaises(ValueError):
            track_seeds_from_input([{"mjd": 60000.0, "ra": 100.0, "dec": 120.0}, {"mjd": 60001.0, "ra": 1.0, "dec": 2.0}])

    def test_rejects_seeds_all_at_the_same_time(self) -> None:
        with self.assertRaises(ValueError):
            track_seeds_from_input([{"mjd": 60000.0, "ra": 100.0, "dec": 20.0}, {"mjd": 60000.0, "ra": 100.1, "dec": 20.0}])

    def test_rejects_non_list_input(self) -> None:
        with self.assertRaises(ValueError):
            track_seeds_from_input("60000,100,20")


class TestExposureMidpoint(unittest.TestCase):
    def test_midpoint_is_half_an_exposure_after_the_start(self) -> None:
        start = _to_mjd(datetime(2025, 3, 1, tzinfo=timezone.utc))
        midpoint = frame_midpoint_mjd({"MJD-OBS": start, "EXPTIME": 600.0})
        self.assertAlmostEqual((midpoint - start) * 86400.0, 300.0, places=6)

    def test_falls_back_to_date_obs_when_mjd_obs_is_absent(self) -> None:
        moment = datetime(2025, 3, 1, tzinfo=timezone.utc)
        midpoint = frame_midpoint_mjd({"EXPTIME": 0.0}, fallback_start=moment)
        self.assertAlmostEqual(midpoint, _to_mjd(moment), places=9)

    def test_raises_without_any_time(self) -> None:
        with self.assertRaises(ValueError):
            frame_midpoint_mjd({"EXPTIME": 10.0})

    def test_track_is_evaluated_at_the_midpoint_not_the_start(self) -> None:
        """A long exposure on a fast mover: using the start time would bias the position visibly."""
        start_moment = datetime(2025, 3, 1, tzinfo=timezone.utc)
        exposure_seconds = 1200.0
        rate_arcsec_per_hour = 3600.0  # 1 arcsec/sec, a deliberately fast mover
        seeds = [
            TrackSeed(mjd=_to_mjd(start_moment), ra_deg=BASE_RA, dec_deg=BASE_DEC),
            TrackSeed(
                mjd=_to_mjd(start_moment + timedelta(hours=2)),
                ra_deg=BASE_RA + _ra_offset_deg(rate_arcsec_per_hour * 2.0),
                dec_deg=BASE_DEC,
            ),
        ]
        frame = FrameContext(
            fits_path="frame.fits",
            date_obs=start_moment,
            header={"MJD-OBS": _to_mjd(start_moment), "EXPTIME": exposure_seconds},
            second_hdu_rows=(),
            width=100,
            height=100,
        )
        positions = _track_target_positions(frames=[frame], target_track_seeds=seeds, diagnostics=[])
        predicted = positions["frame.fits"]
        start_position = (BASE_RA, BASE_DEC)
        # Half of a 1200 s exposure at 1 arcsec/s is 600 arcsec of motion past the start position.
        self.assertAlmostEqual(_separation_arcsec(predicted, start_position), 600.0, delta=1.0)


def _linear_truth(frame_count: int, cadence_hours: float = 1.0, rate_arcsec_per_hour: float = 6.0):
    """Truth track for the catalog-level tests: steady motion in RA from a fixed start."""
    start = _to_mjd(datetime(2025, 3, 1, tzinfo=timezone.utc))
    return [
        (
            start + cadence_hours * index / 24.0,
            BASE_RA + _ra_offset_deg(rate_arcsec_per_hour * cadence_hours * index),
            BASE_DEC,
        )
        for index in range(frame_count)
    ]


def _catalog_row(source_id: str, ra_deg: float, dec_deg: float) -> dict[str, Any]:
    return {"id": source_id, "ra": ra_deg, "dec": dec_deg, "mag": 15.0, "flux": 1000.0}


class TestMovingTargetSearch(unittest.TestCase):
    """
        Candidate identification near the predicted track, and the motion-consistency cross-check,
        exercised directly on synthetic source catalogs.
    """

    def _frames_and_catalogs(
        self,
        truth,
        *,
        static_stars: tuple[tuple[float, float], ...] = (),
        include_target_on: set[int] | None = None,
        extra_rows_by_frame: dict[int, list[dict[str, Any]]] | None = None,
    ):
        frame_times = []
        catalogs: dict[str, list[dict[str, Any]]] = {}
        for index, (mjd, ra, dec) in enumerate(truth):
            path = f"frame_{index:02d}.fits"
            frame_times.append((path, mjd))
            rows = [
                _catalog_row(f"star-{star_index}", BASE_RA + _ra_offset_deg(ra_off), BASE_DEC + dec_off / 3600.0)
                for star_index, (ra_off, dec_off) in enumerate(static_stars)
            ]
            if include_target_on is None or index in include_target_on:
                rows.append(_catalog_row("moving-target", ra, dec))
            for extra in (extra_rows_by_frame or {}).get(index, []):
                rows.append(extra)
            catalogs[path] = rows
        return frame_times, catalogs

    def test_finds_the_moving_target_in_the_catalog(self) -> None:
        truth = _linear_truth(6)
        frame_times, catalogs = self._frames_and_catalogs(truth)
        track = fit_target_track(_seeds_from_truth(truth, (0, 5)))
        result = refine_positions_from_catalog(
            frame_times=frame_times, catalog_rows_by_frame=catalogs, track=track, seeds=track.seeds
        )
        self.assertEqual(len(result.picks), 6)
        self.assertTrue(all(pick.source_id == "moving-target" for pick in result.picks))
        for (path, _), (_, ra, dec) in zip(frame_times, truth):
            self.assertLess(_separation_arcsec(result.positions[path], (ra, dec)), 0.5)

    def test_prefers_the_moving_target_over_a_closer_field_star(self) -> None:
        """A stationary source nearer the prediction must still lose: stationarity is disqualifying."""
        truth = _linear_truth(6, rate_arcsec_per_hour=6.0)
        # Sits right on the middle of the track, so on some frames it is nearer than the target.
        star_ra_offset = 6.0 * 2.5
        frame_times, catalogs = self._frames_and_catalogs(truth, static_stars=((star_ra_offset, 0.0),))
        track = fit_target_track(_seeds_from_truth(truth, (0, 5)))
        result = refine_positions_from_catalog(
            frame_times=frame_times, catalog_rows_by_frame=catalogs, track=track, seeds=track.seeds
        )
        self.assertTrue(result.picks)
        self.assertTrue(all(pick.source_id == "moving-target" for pick in result.picks))

    def test_a_persistent_field_star_is_rejected_as_stationary(self) -> None:
        """With the target absent, a star sitting on the track is caught by the stationarity test."""
        truth = _linear_truth(6)
        fixed_ra, fixed_dec = truth[2][1], truth[2][2]
        extra = {index: [_catalog_row(f"ghost-{index}", fixed_ra, fixed_dec)] for index in range(6)}
        frame_times, catalogs = self._frames_and_catalogs(
            truth, include_target_on=set(), extra_rows_by_frame=extra
        )
        track = fit_target_track(_seeds_from_truth(truth, (0, 5)))
        result = refine_positions_from_catalog(
            frame_times=frame_times, catalog_rows_by_frame=catalogs, track=track, seeds=track.seeds
        )
        self.assertIsNone(result.refined_track)
        self.assertEqual(len(result.picks), 0)

    def test_rejects_a_stationary_source_that_slips_past_the_stationarity_test(self) -> None:
        """
            A source at one fixed position, catalogued on just enough frames to be picked but too few
            to be counted stationary, produces picks that fit a track perfectly -- they simply do not
            move. The spread guard is the only thing that catches this; a residual test never would,
            because a stationary set of points is a flawless fit to a zero-motion track.
        """
        truth = _linear_truth(8)
        fixed_ra, fixed_dec = truth[3][1], truth[3][2]
        # On exactly three frames: each one sees only two others carrying it, one short of the
        # stationarity threshold, while three picks is exactly the minimum to be cross-checked.
        extra = {index: [_catalog_row(f"ghost-{index}", fixed_ra, fixed_dec)] for index in (2, 3, 4)}
        frame_times, catalogs = self._frames_and_catalogs(
            truth, include_target_on=set(), extra_rows_by_frame=extra
        )
        track = fit_target_track(_seeds_from_truth(truth, (0, 7)))
        result = refine_positions_from_catalog(
            frame_times=frame_times, catalog_rows_by_frame=catalogs, track=track, seeds=track.seeds
        )
        self.assertIsNone(result.refined_track)
        self.assertTrue(any("stationary source" in message for message in result.diagnostics))

    def test_clips_a_single_inconsistent_pick(self) -> None:
        truth = _linear_truth(7)
        # Frame 3 has no target but does have a one-off source well off the track, which passes the
        # stationarity test (it appears once) and so is picked -- then must be clipped.
        rogue_ra = truth[3][1]
        rogue_dec = truth[3][2] + 8.0 / 3600.0
        frame_times, catalogs = self._frames_and_catalogs(
            truth,
            include_target_on={0, 1, 2, 4, 5, 6},
            extra_rows_by_frame={3: [_catalog_row("rogue", rogue_ra, rogue_dec)]},
        )
        track = fit_target_track(_seeds_from_truth(truth, (0, 6)))
        result = refine_positions_from_catalog(
            frame_times=frame_times, catalog_rows_by_frame=catalogs, track=track, seeds=track.seeds
        )
        self.assertIn("rogue", {pick.source_id for pick in result.rejected_picks})
        self.assertNotIn("rogue", {pick.source_id for pick in result.picks})
        self.assertTrue(any("did not move consistently" in message for message in result.diagnostics))

    def test_falls_back_to_the_seed_track_when_too_few_candidates(self) -> None:
        truth = _linear_truth(6)
        frame_times, catalogs = self._frames_and_catalogs(truth, include_target_on={0})
        track = fit_target_track(_seeds_from_truth(truth, (0, 5)))
        result = refine_positions_from_catalog(
            frame_times=frame_times, catalog_rows_by_frame=catalogs, track=track, seeds=track.seeds
        )
        self.assertIsNone(result.refined_track)
        self.assertLess(len(result.picks), MIN_ACCEPTED_PICKS)
        # Every frame still gets a position, from the user's own sightings.
        self.assertEqual(len(result.positions), len(frame_times))
        self.assertTrue(any("your own sightings" in message for message in result.diagnostics))

    def test_frames_without_a_detection_keep_the_predicted_position(self) -> None:
        truth = _linear_truth(7)
        frame_times, catalogs = self._frames_and_catalogs(truth, include_target_on={0, 1, 2, 4, 5, 6})
        track = fit_target_track(_seeds_from_truth(truth, (0, 6)))
        result = refine_positions_from_catalog(
            frame_times=frame_times, catalog_rows_by_frame=catalogs, track=track, seeds=track.seeds
        )
        self.assertEqual(result.frames_without_pick, ["frame_03.fits"])
        self.assertIn("frame_03.fits", result.positions)
        self.assertTrue(any("no catalogued source" in message for message in result.diagnostics))

    def test_search_radius_bounds_the_search(self) -> None:
        truth = _linear_truth(6)
        frame_times, catalogs = self._frames_and_catalogs(truth)
        # A track offset in declination by more than the search radius finds nothing.
        offset_seeds = [
            TrackSeed(mjd=mjd, ra_deg=ra, dec_deg=dec + 30.0 / 3600.0) for mjd, ra, dec in (truth[0], truth[5])
        ]
        track = fit_target_track(offset_seeds)
        result = refine_positions_from_catalog(
            frame_times=frame_times,
            catalog_rows_by_frame=catalogs,
            track=track,
            seeds=track.seeds,
            search_radius_arcsec=10.0,
        )
        self.assertEqual(len(result.picks), 0)

        widened = refine_positions_from_catalog(
            frame_times=frame_times,
            catalog_rows_by_frame=catalogs,
            track=track,
            seeds=track.seeds,
            search_radius_arcsec=45.0,
        )
        self.assertEqual(len(widened.picks), 6)


class TestMovingTargetLightCurve(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_frames(self, frames: dict[str, dict[str, Any]]) -> list[str]:
        fits_paths: list[str] = []
        for name, frame in frames.items():
            path = os.path.join(self.temp_dir.name, name)
            header = fits.Header()
            for key, value in frame["header"].items():
                header[key] = value
            hdus = [
                fits.PrimaryHDU(),
                fits.ImageHDU(data=frame["image"], header=header, name="SCI"),
                self._cat_hdu(frame["second_hdu"]),
            ]
            fits.HDUList(hdus).writeto(path, overwrite=True)
            fits_paths.append(path)
        return fits_paths

    def _cat_hdu(self, rows: list[dict[str, Any]]) -> fits.BinTableHDU:
        if not rows:
            return fits.BinTableHDU.from_columns([], name="CAT")
        columns = []
        for key in rows[0]:
            values = [row.get(key, np.nan) for row in rows]
            if isinstance(values[0], str):
                columns.append(fits.Column(name=key, format="32A", array=np.asarray(values)))
            else:
                columns.append(fits.Column(name=key, format="D", array=np.asarray(values, dtype=float)))
        return fits.BinTableHDU.from_columns(columns, name="CAT")

    def _run(self, fits_paths: list[str], seeds: list[TrackSeed], **kwargs: Any) -> Any:
        return generate_light_curve(
            fits_paths,
            aperture_radius=5.0,
            annulus_inner_radius=8.0,
            annulus_outer_radius=12.0,
            min_comparisons=3,
            max_comparisons=10,
            target_position_mode=TARGET_POSITION_TRACK,
            comparison_mode=COMPARISON_AUTO,
            target_track_seeds=seeds,
            **kwargs,
        )

    def _finite_rows(self, result: Any) -> list[Any]:
        return [
            row for row in result.light_curve_rows
            if math.isfinite(row.target_calibrated_apparent_magnitude)
        ]

    def test_two_seeds_measure_the_target_on_every_frame(self) -> None:
        frames, truth = build_tracked_frame_set(frame_count=8)
        paths = self.write_frames(frames)
        result = self._run(paths, _seeds_from_truth(truth, (0, 7)))
        self.assertEqual(len(self._finite_rows(result)), 8)

    def test_recovers_injected_brightness_variation(self) -> None:
        frame_count = 8
        fluxes = [_flux_for_mag(15.0 + 0.2 * math.sin(2 * math.pi * i / frame_count)) for i in range(frame_count)]
        frames, truth = build_tracked_frame_set(frame_count=frame_count, target_flux_by_frame=fluxes)
        paths = self.write_frames(frames)
        result = self._run(paths, _seeds_from_truth(truth, (0, frame_count - 1)))

        rows = self._finite_rows(result)
        self.assertEqual(len(rows), frame_count)
        expected = [15.0 + 0.2 * math.sin(2 * math.pi * i / frame_count) for i in range(frame_count)]
        measured = [row.target_calibrated_apparent_magnitude for row in rows]
        offset = float(np.mean(np.array(measured) - np.array(expected)))
        for want, got in zip(expected, measured):
            self.assertAlmostEqual(got - offset, want, delta=0.02)

    def test_three_seeds_needed_for_a_curved_track(self) -> None:
        """A line through the endpoints of a curved track walks the aperture off the target."""
        frames, truth = build_tracked_frame_set(frame_count=9, target_curvature_arcsec=25.0)
        paths = self.write_frames(frames)

        quadratic = self._run(paths, _seeds_from_truth(truth, (0, 4, 8)))
        self.assertEqual(len(self._finite_rows(quadratic)), 9)

        linear = self._run(paths, _seeds_from_truth(truth, (0, 8)))
        mid_linear = [row for row in self._finite_rows(linear) if "frame_05" in row.fits_path]
        mid_quadratic = [row for row in self._finite_rows(quadratic) if "frame_05" in row.fits_path]
        self.assertTrue(mid_quadratic)
        # Mid-series is where a straight line departs furthest from the true curved track, so the
        # target either falls outside the aperture entirely or is measured much too faint.
        if mid_linear:
            self.assertGreater(
                mid_linear[0].target_calibrated_apparent_magnitude
                - mid_quadratic[0].target_calibrated_apparent_magnitude,
                0.3,
            )

    def test_catalog_search_rescues_a_curved_track_seeded_with_only_two_points(self) -> None:
        """
            The interpolated position is a search position, not the final answer. With curvature
            inside the search radius, a two-seed straight line mispredicts by several arcseconds --
            far beyond the centroid recenter cap -- and the catalog search still recovers the target
            on every frame. Without the target in the catalog there is nothing to find, and the same
            run degrades.
        """
        curvature_arcsec = 8.0
        found, truth = build_tracked_frame_set(
            frame_count=9, target_curvature_arcsec=curvature_arcsec, include_target_in_catalog=True
        )
        result = self._run(self.write_frames(found), _seeds_from_truth(truth, (0, 8)))
        self.assertEqual(len(self._finite_rows(result)), 9)
        self.assertTrue(any("Located the moving target" in message for message in result.diagnostics))

        # Same geometry, but the target is not catalogued, so the search has nothing to lock onto.
        self.tearDown()
        self.setUp()
        missing, truth = build_tracked_frame_set(
            frame_count=9, target_curvature_arcsec=curvature_arcsec, include_target_in_catalog=False
        )
        degraded = self._run(self.write_frames(missing), _seeds_from_truth(truth, (0, 8)))
        self.assertTrue(any("your own sightings" in message for message in degraded.diagnostics))

    def test_extrapolated_frames_are_flagged(self) -> None:
        frames, truth = build_tracked_frame_set(frame_count=8)
        paths = self.write_frames(frames)
        # Seed only the middle of the series, so the first and last frames are extrapolated.
        result = self._run(paths, _seeds_from_truth(truth, (2, 5)))
        self.assertTrue(any("extrapolated" in message for message in result.diagnostics))

    def test_long_arc_with_two_seeds_recommends_a_third(self) -> None:
        span_hours = LINEAR_TRACK_MAX_SPAN_HOURS * 3.0
        frames, truth = build_tracked_frame_set(
            frame_count=6,
            cadence_hours=span_hours / 5.0,
            target_rate_arcsec_per_hour=0.3,
        )
        paths = self.write_frames(frames)
        result = self._run(paths, _seeds_from_truth(truth, (0, 5)))
        self.assertTrue(any("third frame" in message for message in result.diagnostics))

    def test_short_arc_with_two_seeds_is_not_warned_about(self) -> None:
        frames, truth = build_tracked_frame_set(frame_count=6, cadence_hours=0.5)
        paths = self.write_frames(frames)
        result = self._run(paths, _seeds_from_truth(truth, (0, 5)))
        self.assertFalse(any("third frame" in message for message in result.diagnostics))

    def test_track_fit_is_reported_in_diagnostics(self) -> None:
        frames, truth = build_tracked_frame_set(frame_count=6)
        paths = self.write_frames(frames)
        result = self._run(paths, _seeds_from_truth(truth, (0, 5)))
        self.assertTrue(any("Target track fitted" in message for message in result.diagnostics))

    def test_missing_seeds_raises(self) -> None:
        frames, _ = build_tracked_frame_set(frame_count=3)
        paths = self.write_frames(frames)
        with self.assertRaises(LightCurveError):
            self._run(paths, [])

    def test_target_own_catalog_entry_is_not_used_as_a_comparison(self) -> None:
        frames, truth = build_tracked_frame_set(frame_count=6, include_target_in_catalog=True)
        paths = self.write_frames(frames)
        result = self._run(paths, _seeds_from_truth(truth, (0, 5)))
        self.assertNotIn(
            "target-shadow",
            {star.candidate_id for star in result.selected_comparison_stars},
        )


if __name__ == "__main__":
    unittest.main()
