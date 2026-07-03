from __future__ import annotations

import math
import os
import tempfile
import unittest
import base64
from io import BytesIO
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from astropy.io import fits
from PIL import Image

from datalab.datalab_session.utils.comparison_stars import (
    ComparisonStar,
    _reject_zero_point_outliers,
    _source_catalog_sort_key,
)
from datalab.datalab_session.utils.aperture_light_curve import (
    LightCurveError,
    _extract_candidate_row,
    _validated_frame_contexts,
    generate_light_curve,
)
from datalab.datalab_session.utils.geometry import angular_distance_arcsec


APERTURE_PHOTOMETRY_TEST_DIR = Path(__file__).resolve().parent / "test_files" / "aperture_photometry"

def print_nearest_fits_catalog_target_matches(
    input_handlers: list[Any],
    target_ra_deg: float,
    target_dec_deg: float,
) -> None:
    frames = _validated_frame_contexts(input_handlers)
    print("\nNearest FITS catalog rows to target:")
    for frame in frames:
        candidates = [
            _extract_candidate_row(row, frame.fits_path)
            for row in frame.second_hdu_rows
        ]
        nearest = min(
            candidates,
            key=lambda candidate: angular_distance_arcsec(
                target_ra_deg,
                target_dec_deg,
                candidate["ra_deg"],
                candidate["dec_deg"],
            ),
        )
        distance_arcsec = angular_distance_arcsec(
            target_ra_deg,
            target_dec_deg,
            nearest["ra_deg"],
            nearest["dec_deg"],
        )
        print(
            "  - "
            f"{Path(frame.fits_path).name}: "
            f"source={nearest['source_label']}, "
            f"distance_arcsec={distance_arcsec:.3f}, "
            f"catalog_mag={nearest['mag']:.6f}, "
            f"catalog_flux={nearest['flux']:.6f}"
        )

def gaussian_star(
    image: np.ndarray,
    x: float,
    y: float,
    flux: float,
    sigma: float = 1.0,
) -> None:
    radius = int(math.ceil(5 * sigma))
    norm = flux / (2.0 * math.pi * sigma * sigma)
    for j in range(max(0, int(y) - radius), min(image.shape[0], int(y) + radius + 1)):
        for i in range(max(0, int(x) - radius), min(image.shape[1], int(x) + radius + 1)):
            dx = (i + 0.5) - x
            dy = (j + 0.5) - y
            image[j, i] += norm * math.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))


def build_frame_set(
    *,
    frame_count: int = 3,
    include_target_in_second_hdu: bool = False,
    variable_candidate_index: int | None = None,
) -> tuple[dict[str, dict[str, Any]], tuple[float, float]]:
    width = 80
    height = 80
    target_xy = (30.3, 28.8)
    target_ra = 100.0 + target_xy[0] * 1.0e-5
    target_dec = 20.0 + target_xy[1] * 1.0e-5
    comparison_positions = [
        (12.5, 12.5),
        (18.0, 48.0),
        (26.0, 60.0),
        (42.0, 18.5),
        (48.0, 46.0),
        (55.0, 22.0),
        (60.0, 58.0),
        (67.0, 35.0),
        (35.0, 68.0),
        (24.0, 34.5),
        (63.5, 14.0),
        (51.0, 63.0),
    ]
    comparison_mags = [11.9, 12.0, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9, 13.0]
    comparison_fluxes = [12000, 11000, 11500, 11800, 12500, 12700, 12200, 11900, 11700, 11600, 11300, 12100]

    frames: dict[str, dict[str, Any]] = {}
    for frame_index in range(frame_count):
        header = {
            "DATE-OBS": f"2025-01-01T00:00:{frame_count - frame_index:02d}",
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CUNIT1": "deg",
            "CUNIT2": "deg",
            "CRVAL1": 100.0,
            "CRVAL2": 20.0,
            "CRPIX1": 0.0,
            "CRPIX2": 0.0,
            "CD1_1": 1.0e-5,
            "CD1_2": 0.0,
            "CD2_1": 0.0,
            "CD2_2": 1.0e-5,
            "GAIN": 1.7,
            "RDNOISE": 3.2,
        }
        image = np.full((height, width), 100.0, dtype=float)
        target_flux = 8000.0 + 200.0 * frame_index
        gaussian_star(image, *target_xy, flux=target_flux, sigma=1.05)
        second_hdu: list[dict[str, Any]] = []
        for idx, ((x, y), mag, flux) in enumerate(zip(comparison_positions, comparison_mags, comparison_fluxes)):
            effective_flux = flux
            if variable_candidate_index is not None and idx == variable_candidate_index:
                effective_flux = flux * (1.0 + 0.35 * frame_index)
            gaussian_star(image, x, y, flux=effective_flux, sigma=1.05)
            ra = header["CRVAL1"] + x * header["CD1_1"]
            dec = header["CRVAL2"] + y * header["CD2_2"]
            second_hdu.append(
                {
                    "id": f"comp-{idx + 1:02d}",
                    "ra": ra,
                    "dec": dec,
                    "mag": mag,
                    "flux": effective_flux,
                }
            )

        if include_target_in_second_hdu:
            second_hdu.append(
                {
                    "id": "target-shadow",
                    "ra": target_ra,
                    "dec": target_dec,
                    "mag": 12.2,
                    "flux": target_flux,
                }
            )

        frames[f"frame_{frame_index + 1}.fits"] = {
            "image": image,
            "header": header,
            "second_hdu": second_hdu,
        }

    return frames, (target_ra, target_dec)

class TestAperturePhotometry(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_frames(self, frames: dict[str, dict[str, Any]]) -> list[Any]:
        input_handlers: list[Any] = []
        for name, frame in frames.items():
            path = os.path.join(self.temp_dir.name, name)
            header = fits.Header()
            for key, value in frame["header"].items():
                header[key] = value
            hdus = [
                fits.PrimaryHDU(),
                fits.ImageHDU(data=frame["image"], header=header, name="SCI"),
            ]
            hdus.append(self._cat_hdu(frame["second_hdu"]))
            fits.HDUList(hdus).writeto(path, overwrite=True)
            input_handlers.append(self.input_handler_for_path(path))
        return input_handlers

    def input_handler_for_path(self, path: str) -> Any:
        with fits.open(path) as hdul:
            hdus = {hdu.name: hdu.copy() for hdu in hdul}
            hdus["PRIMARY"] = hdul[0].copy()
        return SimpleNamespace(
            fits_file=path,
            sci_hdu=hdus["SCI"],
            sci_data=hdus["SCI"].data,
            get_hdu=lambda extension=None, hdus=hdus: hdus[extension or "SCI"],
        )

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

    def row_names(self, result: Any) -> list[str]:
        return [Path(row.fits_path).name for row in result.light_curve_rows]

    def star_by_source_label(self, result: Any) -> dict[str, ComparisonStar]:
        stars: dict[str, ComparisonStar] = {}
        for star in result.selected_comparison_stars:
            labels = {
                str(row["source_label"])
                for row in star.source_catalog_by_frame.values()
                if "source_label" in row
            }
            for label in labels:
                stars[label] = star
        return stars

    def test_sorting_by_date_obs(self) -> None:
        frames, (target_ra, target_dec) = build_frame_set()
        fits_paths = self.write_frames(frames)

        result = generate_light_curve(
            fits_paths,
            target_ra_deg=target_ra,
            target_dec_deg=target_dec,
            aperture_radius_px=4.0,
            annulus_inner_radius_px=6.0,
            annulus_outer_radius_px=9.0,
        )

        self.assertEqual(self.row_names(result), ["frame_3.fits", "frame_2.fits", "frame_1.fits"])

    def test_ignores_frame_with_missing_date_obs(self) -> None:
        frames, (target_ra, target_dec) = build_frame_set()
        del frames["frame_1.fits"]["header"]["DATE-OBS"]
        fits_paths = self.write_frames(frames)

        result = generate_light_curve(fits_paths, target_ra, target_dec, 4.0, 6.0, 9.0)

        self.assertEqual(self.row_names(result), ["frame_3.fits", "frame_2.fits"])

    def test_ignores_frame_with_malformed_date_obs(self) -> None:
        frames, (target_ra, target_dec) = build_frame_set()
        frames["frame_1.fits"]["header"]["DATE-OBS"] = "not-a-date"
        fits_paths = self.write_frames(frames)

        result = generate_light_curve(fits_paths, target_ra, target_dec, 4.0, 6.0, 9.0)

        self.assertEqual(self.row_names(result), ["frame_3.fits", "frame_2.fits"])

    def test_ignores_frame_with_missing_wcs(self) -> None:
        frames, (target_ra, target_dec) = build_frame_set()
        for key in ("CTYPE1", "CTYPE2", "CUNIT1", "CUNIT2", "CD1_1", "CD1_2", "CD2_1", "CD2_2"):
            del frames["frame_1.fits"]["header"][key]
        fits_paths = self.write_frames(frames)

        result = generate_light_curve(fits_paths, target_ra, target_dec, 4.0, 6.0, 9.0)

        self.assertEqual(self.row_names(result), ["frame_3.fits", "frame_2.fits"])

    def test_ignores_frame_with_missing_second_hdu(self) -> None:
        frames, (target_ra, target_dec) = build_frame_set()
        frames["frame_1.fits"]["second_hdu"] = []
        fits_paths = self.write_frames(frames)

        result = generate_light_curve(fits_paths, target_ra, target_dec, 4.0, 6.0, 9.0)

        self.assertEqual(self.row_names(result), ["frame_3.fits", "frame_2.fits"])

    def test_ignores_frame_with_missing_second_hdu_columns(self) -> None:
        frames, (target_ra, target_dec) = build_frame_set()
        del frames["frame_1.fits"]["second_hdu"][0]["ra"]
        fits_paths = self.write_frames(frames)

        result = generate_light_curve(fits_paths, target_ra, target_dec, 4.0, 6.0, 9.0)

        self.assertEqual(self.row_names(result), ["frame_3.fits", "frame_2.fits"])

    def test_failure_when_ignored_frames_leave_too_few_valid_inputs(self) -> None:
        frames, (target_ra, target_dec) = build_frame_set(frame_count=1)
        del frames["frame_1.fits"]["header"]["DATE-OBS"]
        fits_paths = self.write_frames(frames)

        with self.assertRaisesRegex(LightCurveError, "requires at least 1 valid input file"):
            generate_light_curve(fits_paths, target_ra, target_dec, 4.0, 6.0, 9.0)

    def test_target_photometry_succeeds_when_target_absent_from_second_hdu(self) -> None:
        frames, (target_ra, target_dec) = build_frame_set(include_target_in_second_hdu=False)
        fits_paths = self.write_frames(frames)

        result = generate_light_curve(fits_paths, target_ra, target_dec, 4.0, 6.0, 9.0)

        self.assertEqual(len(result.light_curve_rows), 3)
        self.assertTrue(all(row.target_net_source_counts > 0.0 for row in result.light_curve_rows))

    def test_target_centroiding_converges_on_synthetic_images(self) -> None:
        frames, (target_ra, target_dec) = build_frame_set()
        fits_paths = self.write_frames(frames)

        result = generate_light_curve(fits_paths, target_ra, target_dec, 4.0, 6.0, 9.0)

        first = result.light_curve_rows[0]
        self.assertAlmostEqual(first.target_centroid_x, 30.3, delta=0.2)
        self.assertAlmostEqual(first.target_centroid_y, 28.8, delta=0.2)

    def test_target_recenter_falls_back_to_wcs_position(self) -> None:
        # Point the target at an empty patch of sky (~16px from any source) so centroiding cannot
        # lock on. _measure_target must never raise or drift onto a neighbour: it measures at the
        # authoritative WCS position instead of the real source at (30.3, 28.8).
        from datalab.datalab_session.utils.aperture_light_curve import _measure_target
        from datalab.datalab_session.utils.fits_metadata import world_to_pixel

        frames, _ = build_frame_set()
        header = frames["frame_1.fits"]["header"]
        empty_ra = header["CRVAL1"] + 35.0 * header["CD1_1"]
        empty_dec = header["CRVAL2"] + 45.0 * header["CD2_2"]
        frame = _validated_frame_contexts(self.write_frames(frames))[0]
        initial_x, initial_y = world_to_pixel(frame.header, empty_ra, empty_dec)

        measurement = _measure_target(
            frame=frame,
            target_ra_deg=empty_ra,
            target_dec_deg=empty_dec,
            aperture_radius_px=4.0,
            annulus_inner_radius_px=6.0,
            annulus_outer_radius_px=9.0,
        )

        self.assertAlmostEqual(measurement.x, initial_x, delta=1.0e-6)
        self.assertAlmostEqual(measurement.y, initial_y, delta=1.0e-6)
        self.assertGreater(math.hypot(measurement.x - 30.3, measurement.y - 28.8), 10.0)

    def test_arcsec_aperture_matches_equivalent_pixel_aperture(self) -> None:
        # arcsec radii equal to the pixel radii times the frame plate scale must convert
        # back to the same pixels per frame, yielding identical measurements.
        frames, (target_ra, target_dec) = build_frame_set()
        fits_paths = self.write_frames(frames)
        px_result = generate_light_curve(fits_paths, target_ra, target_dec, 4.0, 6.0, 9.0)

        arcsec_per_px = abs(frames["frame_1.fits"]["header"]["CD1_1"]) * 3600.0
        arcsec_result = generate_light_curve(
            fits_paths,
            target_ra,
            target_dec,
            4.0 * arcsec_per_px,
            6.0 * arcsec_per_px,
            9.0 * arcsec_per_px,
            aperture_unit="arcsec",
        )

        self.assertEqual(len(arcsec_result.light_curve_rows), len(px_result.light_curve_rows))
        for px_row, arcsec_row in zip(px_result.light_curve_rows, arcsec_result.light_curve_rows):
            self.assertAlmostEqual(arcsec_row.target_centroid_x, px_row.target_centroid_x, delta=1.0e-4)
            self.assertAlmostEqual(arcsec_row.target_centroid_y, px_row.target_centroid_y, delta=1.0e-4)
            self.assertAlmostEqual(
                arcsec_row.target_net_source_counts, px_row.target_net_source_counts, delta=1.0e-3
            )

    def test_invalid_aperture_unit_fails_fast(self) -> None:
        frames, (target_ra, target_dec) = build_frame_set()
        fits_paths = self.write_frames(frames)
        with self.assertRaisesRegex(LightCurveError, "aperture_unit"):
            generate_light_curve(
                fits_paths, target_ra, target_dec, 4.0, 6.0, 9.0, aperture_unit="degrees"
            )

    def test_zero_target_counts_does_not_crash_calibration(self) -> None:
        # Zero the target aperture+annulus on one frame so its net counts are exactly 0.0 (as a
        # blank/masked region would give). Dividing by net_source_counts**2 in the relative-flux
        # error would raise ZeroDivisionError; the frame must instead be retained with a NaN
        # magnitude and NaN uncertainty.
        frames, (target_ra, target_dec) = build_frame_set()
        frames["frame_1.fits"]["image"][18:40, 20:42] = 0.0   # aperture + background annulus -> 0
        fits_paths = self.write_frames(frames)

        result = generate_light_curve(fits_paths, target_ra, target_dec, 4.0, 6.0, 9.0)

        self.assertEqual(len(result.light_curve_rows), 3)
        zeroed = next(row for row in result.light_curve_rows if row.fits_path.endswith("frame_1.fits"))
        self.assertEqual(zeroed.target_net_source_counts, 0.0)
        self.assertTrue(math.isnan(zeroed.target_calibrated_apparent_magnitude))
        self.assertTrue(math.isnan(zeroed.target_differential_flux_uncertainty))

    def test_second_hdu_radec_cross_match_across_frames(self) -> None:
        frames, (target_ra, target_dec) = build_frame_set()
        for frame in frames.values():
            for row in frame["second_hdu"]:
                row["ra"] += 1.0e-8
                row["dec"] -= 1.0e-8
        fits_paths = self.write_frames(frames)

        result = generate_light_curve(fits_paths, target_ra, target_dec, 4.0, 6.0, 9.0)
        self.assertGreaterEqual(len(result.selected_comparison_stars), 5)

    def test_variability_score_reflects_variable_candidate(self) -> None:
        frames, (target_ra, target_dec) = build_frame_set(variable_candidate_index=3)
        fits_paths = self.write_frames(frames)

        result = generate_light_curve(
            fits_paths,
            target_ra,
            target_dec,
            4.0,
            6.0,
            9.0,
            min_comparisons=5,
            max_comparisons=12,
        )

        stars = self.star_by_source_label(result)
        self.assertIn("comp-04", stars)
        variability_by_label = {label: star.variability_score for label, star in stars.items()}
        self.assertEqual(max(variability_by_label, key=variability_by_label.get), "comp-04")

    def test_target_exclusion_from_candidate_pool(self) -> None:
        frames, (target_ra, target_dec) = build_frame_set(include_target_in_second_hdu=True)
        fits_paths = self.write_frames(frames)

        result = generate_light_curve(fits_paths, target_ra, target_dec, 4.0, 6.0, 9.0)

        self.assertTrue(all(star.target_separation_px > 9.0 for star in result.selected_comparison_stars))
        self.assertFalse(any("too close to target" in diagnostic for diagnostic in result.diagnostics))
        self.assertTrue(all("comparison" in diagnostic for diagnostic in result.diagnostics))

    def test_comparison_selection_ignores_catalog_magnitude_outliers(self) -> None:
        frames, (target_ra, target_dec) = build_frame_set()
        for frame in frames.values():
            frame["second_hdu"][0]["mag"] = 30.0
        fits_paths = self.write_frames(frames)

        result = generate_light_curve(
            fits_paths,
            target_ra,
            target_dec,
            4.0,
            6.0,
            9.0,
            min_comparisons=5,
            max_comparisons=5,
        )

        self.assertEqual(len(result.selected_comparison_stars), 5)

    def test_failure_when_fewer_than_five_valid_comparisons_remain(self) -> None:
        frames, (target_ra, target_dec) = build_frame_set()
        for frame in frames.values():
            frame["second_hdu"] = frame["second_hdu"][:4]
        fits_paths = self.write_frames(frames)

        with self.assertRaisesRegex(LightCurveError, "minimum comparison ensemble"):
            generate_light_curve(fits_paths, target_ra, target_dec, 4.0, 6.0, 9.0)

    def test_invalid_radius_and_comparison_limits_fail_fast(self) -> None:
        frames, (target_ra, target_dec) = build_frame_set()
        fits_paths = self.write_frames(frames)

        with self.assertRaisesRegex(LightCurveError, "aperture_radius_px must be > 0"):
            generate_light_curve(fits_paths, target_ra, target_dec, 0.0, 6.0, 9.0)
        with self.assertRaisesRegex(LightCurveError, "annulus_inner_radius_px"):
            generate_light_curve(fits_paths, target_ra, target_dec, 4.0, 4.0, 9.0)
        with self.assertRaisesRegex(LightCurveError, "annulus_outer_radius_px"):
            generate_light_curve(fits_paths, target_ra, target_dec, 4.0, 6.0, 6.0)
        with self.assertRaisesRegex(LightCurveError, "min_comparisons and max_comparisons"):
            generate_light_curve(fits_paths, target_ra, target_dec, 4.0, 6.0, 9.0, min_comparisons=6, max_comparisons=5)

    def test_aperture_photometry_and_calibration_outputs_are_finite(self) -> None:
        frames, (target_ra, target_dec) = build_frame_set()
        fits_paths = self.write_frames(frames)

        result = generate_light_curve(fits_paths, target_ra, target_dec, 4.0, 6.0, 9.0)
        first = result.light_curve_rows[0]

        self.assertGreater(first.target_net_source_counts, 0.0)
        self.assertGreater(first.comparison_ensemble_total_counts, 0.0)
        self.assertTrue(math.isfinite(first.target_calibrated_apparent_magnitude))
        self.assertTrue(math.isfinite(first.target_calibrated_apparent_magnitude_uncertainty))

    def test_aiJ_style_serror_numeric_regression(self) -> None:
        src = 1234.5
        bck = 12.3
        s_cnt = 48.0
        src_cnt = 48.0
        bck_cnt = 166.0
        gain = 1.7
        dark = 4.0
        ron = 3.2
        expected = math.sqrt((src * gain) + s_cnt * (1.0 + src_cnt / bck_cnt) * (bck * gain + dark + ron * ron + gain * gain * 0.083521)) / gain
        self.assertAlmostEqual(expected, 38.522232252565516, places=10)

    def test_differential_flux_and_uncertainty_propagation(self) -> None:
        frames, (target_ra, target_dec) = build_frame_set()
        fits_paths = self.write_frames(frames)

        result = generate_light_curve(fits_paths, target_ra, target_dec, 4.0, 6.0, 9.0)
        row = result.light_curve_rows[0]

        recomputed_rel_flux = row.target_net_source_counts / row.comparison_ensemble_total_counts
        self.assertAlmostEqual(row.target_differential_flux, recomputed_rel_flux, places=12)
        self.assertGreater(row.target_differential_flux_uncertainty, 0.0)

    def test_comparison_validation_diagnostics_include_candidate_ra_dec(self) -> None:
        frames, (target_ra, target_dec) = build_frame_set()
        fits_paths = self.write_frames(frames)

        result = generate_light_curve(fits_paths, target_ra, target_dec, 4.0, 6.0, 9.0)

        self.assertIn("comparison star identifier | RA | Dec | calculated flux", result.diagnostics[0])
        first_star = result.selected_comparison_stars[0]
        first_row = next(
            diagnostic
            for diagnostic in result.diagnostics
            if diagnostic.startswith(f"comparison-star validation row: {first_star.candidate_id} |")
        )
        self.assertIn(f"{first_star.ra_deg:.4f}", first_row)
        self.assertIn(f"{first_star.dec_deg:.4f}", first_row)
        fields = [field.strip() for field in first_row.replace("comparison-star validation row:", "").split("|")]
        self.assertRegex(fields[1], r"^-?\d+\.\d{4}$")
        self.assertRegex(fields[2], r"^-?\d+\.\d{4}$")
        self.assertRegex(fields[3], r"^-?\d+$")
        self.assertRegex(fields[4], r"^-?\d+$")
        self.assertRegex(fields[5], r"^-?\d+\.\d{3}$")
        self.assertRegex(fields[6], r"^-?\d+\.\d{3}$")

    @staticmethod
    def _comparison_star(candidate_id: str, reference_magnitude: float, measured_magnitude: float, isolation_px: float = 10.0) -> ComparisonStar:
        return ComparisonStar(
            candidate_id=candidate_id,
            ra_deg=0.0,
            dec_deg=0.0,
            reference_magnitude=reference_magnitude,
            reference_magnitude_source="second_hdu",
            source_catalog_by_frame={},
            variability_score=0.01,
            isolation_px=isolation_px,
            target_separation_px=50.0,
            measured_instrumental_magnitude=measured_magnitude,
        )

    def test_comparison_ranking_prefers_measured_brightness_near_target(self) -> None:
        # Ranking is on each candidate's own measured instrumental magnitude versus the target's
        # (same zero point), so the candidate closest in measured brightness sorts first -- not the
        # brightest catalog star.
        target_mag_proxy = -9.0
        near = self._comparison_star("near", reference_magnitude=12.0, measured_magnitude=-9.1)
        far = self._comparison_star("far", reference_magnitude=12.0, measured_magnitude=-12.0)
        self.assertLess(
            _source_catalog_sort_key(near, target_mag_proxy),
            _source_catalog_sort_key(far, target_mag_proxy),
        )

    def test_zero_point_guard_drops_blended_catalog_magnitude_outlier(self) -> None:
        # Good comparisons share a (catalog_mag - measured_mag) zero point (~20 here). A blended
        # star with a corrupt catalog magnitude has a wildly different residual and must be dropped
        # before it biases the ensemble zero point.
        good = [
            self._comparison_star(f"cand-{i}", reference_magnitude=12.0 + i * 0.1, measured_magnitude=-8.0 + i * 0.1)
            for i in range(5)
        ]
        blended = self._comparison_star("cand-blended", reference_magnitude=12.0, measured_magnitude=-13.0)

        kept = _reject_zero_point_outliers(good + [blended])
        kept_ids = {candidate.candidate_id for candidate in kept}

        self.assertNotIn("cand-blended", kept_ids)
        self.assertEqual(len(kept), 5)

    def test_diagnostic_images_include_labeled_candidate_overlay_jpegs(self) -> None:
        frames, (target_ra, target_dec) = build_frame_set()
        fits_paths = self.write_frames(frames)

        result = generate_light_curve(fits_paths, target_ra, target_dec, 4.0, 6.0, 9.0)

        self.assertEqual(
            set(result.diagnostic_images_by_fits_basename),
            {"frame_1.fits", "frame_2.fits", "frame_3.fits"},
        )
        image_data = base64.b64decode(result.diagnostic_images_by_fits_basename["frame_1.fits"])
        image = Image.open(BytesIO(image_data))
        self.assertEqual(image.format, "JPEG")
        self.assertEqual(image.size, (80, 80))
        rgb = np.asarray(image.convert("RGB"))
        blue_overlay_pixels = np.count_nonzero(
            (rgb[:, :, 0] < 80) &
            (rgb[:, :, 1] > 130) &
            (rgb[:, :, 2] > 180)
        )
        orange_target_pixels = np.count_nonzero(
            (rgb[:, :, 0] > 180) &
            (rgb[:, :, 1] > 80) &
            (rgb[:, :, 1] < 170) &
            (rgb[:, :, 2] < 90)
        )
        self.assertGreater(blue_overlay_pixels, 50)
        self.assertGreater(orange_target_pixels, 10)
        orange_rows = np.argwhere(
            (rgb[:, :, 0] > 180) &
            (rgb[:, :, 1] > 80) &
            (rgb[:, :, 1] < 170) &
            (rgb[:, :, 2] < 90)
        )[:, 0]
        self.assertGreater(float(np.mean(orange_rows)), 40.0)

    def test_determinism(self) -> None:
        frames, (target_ra, target_dec) = build_frame_set()
        fits_paths = self.write_frames(frames)
        result1 = generate_light_curve(fits_paths, target_ra, target_dec, 4.0, 6.0, 9.0)
        result2 = generate_light_curve(fits_paths, target_ra, target_dec, 4.0, 6.0, 9.0)

        self.assertEqual(
            [star.candidate_id for star in result1.selected_comparison_stars],
            [star.candidate_id for star in result2.selected_comparison_stars],
        )
        self.assertEqual(
            [row.target_calibrated_apparent_magnitude for row in result1.light_curve_rows],
            [row.target_calibrated_apparent_magnitude for row in result2.light_curve_rows],
        )

    def test_default_fits_dependencies_read_sci_header_and_cat_rows(self) -> None:
        frames, (target_ra, target_dec) = build_frame_set(frame_count=1)
        handle = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
        path = handle.name
        handle.close()
        frame = frames["frame_1.fits"]
        header = fits.Header()
        for key, value in frame["header"].items():
            header[key] = value
        header["CTYPE1"] = "RA---TAN"
        header["CTYPE2"] = "DEC--TAN"
        header["CUNIT1"] = "deg"
        header["CUNIT2"] = "deg"

        cat_columns = [
            fits.Column(name="id", format="16A", array=np.asarray([row["id"] for row in frame["second_hdu"]])),
            fits.Column(name="ra", format="D", array=np.asarray([row["ra"] for row in frame["second_hdu"]])),
            fits.Column(name="dec", format="D", array=np.asarray([row["dec"] for row in frame["second_hdu"]])),
            fits.Column(name="mag", format="D", array=np.asarray([row["mag"] for row in frame["second_hdu"]])),
            fits.Column(name="flux", format="D", array=np.asarray([row["flux"] for row in frame["second_hdu"]])),
        ]
        hdul = fits.HDUList([
            fits.PrimaryHDU(),
            fits.ImageHDU(data=frame["image"], header=header, name="SCI"),
            fits.BinTableHDU.from_columns(cat_columns, name="CAT"),
        ])
        hdul.writeto(path, overwrite=True)
        try:
            input_handler = self.input_handler_for_path(path)
            result = generate_light_curve(
                [input_handler],
                target_ra_deg=target_ra,
                target_dec_deg=target_dec,
                aperture_radius_px=4.0,
                annulus_inner_radius_px=6.0,
                annulus_outer_radius_px=9.0,
            )
        finally:
            os.remove(path)

        self.assertEqual(len(result.light_curve_rows), 1)
        self.assertGreaterEqual(len(result.selected_comparison_stars), 5)

    def test_second_hdu_requires_flux_column_for_candidate_comparisons(self) -> None:
        frames, (target_ra, target_dec) = build_frame_set()
        for frame in frames.values():
            for row in frame["second_hdu"]:
                row.pop("flux")
        fits_paths = self.write_frames(frames)

        with self.assertRaisesRegex(LightCurveError, "requires at least 1 valid input file"):
            generate_light_curve(fits_paths, target_ra, target_dec, 4.0, 6.0, 9.0)

    def test_second_hdu_requires_finite_flux_for_candidate_comparisons(self) -> None:
        frames, (target_ra, target_dec) = build_frame_set()
        for frame in frames.values():
            frame["second_hdu"][0]["flux"] = math.nan
        fits_paths = self.write_frames(frames)

        result = generate_light_curve(fits_paths, target_ra, target_dec, 4.0, 6.0, 9.0)

        self.assertNotIn("comp-01", self.star_by_source_label(result))
        self.assertFalse(any("magnitude/flux values" in diagnostic for diagnostic in result.diagnostics))
        self.assertTrue(all("comparison" in diagnostic for diagnostic in result.diagnostics))

    def test_real_compressed_fits_aperture_photometry_prints_diagnostics_and_results(self) -> None:
        fits_paths = sorted(str(path) for path in APERTURE_PHOTOMETRY_TEST_DIR.glob("*.fits.fz"))
        self.assertEqual(len(fits_paths), 3)
        input_handlers = [self.input_handler_for_path(path) for path in fits_paths]
        target_ra_deg = 199.150264
        target_dec_deg = 42.093592

        print_nearest_fits_catalog_target_matches(
            input_handlers,
            target_ra_deg=target_ra_deg,
            target_dec_deg=target_dec_deg,
        )

        result = generate_light_curve(
            input_handlers,
            target_ra_deg=target_ra_deg,
            target_dec_deg=target_dec_deg,
            aperture_radius_px=7.64,
            annulus_inner_radius_px=12.73,
            annulus_outer_radius_px=19.10,
        )

        print("\nLight curve diagnostics:")
        for diagnostic in result.diagnostics:
            print(f"  - {diagnostic}")

        print("\nSelected comparison stars:")
        for star in result.selected_comparison_stars:
            print(f"  - {asdict(star)}")

        print("\nLight curve results:")
        for row in result.light_curve_rows:
            print(f"  - {asdict(row)}")

    def test_default_fits_dependencies_reject_missing_cat(self) -> None:
        handle = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
        path = handle.name
        handle.close()
        header = fits.Header()
        header["DATE-OBS"] = "2025-01-01T00:00:00"
        header["CTYPE1"] = "RA---TAN"
        header["CTYPE2"] = "DEC--TAN"
        header["CRVAL1"] = 100.0
        header["CRVAL2"] = 20.0
        header["CRPIX1"] = 1.0
        header["CRPIX2"] = 1.0
        header["CD1_1"] = 1.0e-5
        header["CD1_2"] = 0.0
        header["CD2_1"] = 0.0
        header["CD2_2"] = 1.0e-5
        fits.HDUList([
            fits.PrimaryHDU(),
            fits.ImageHDU(data=np.zeros((20, 20), dtype=float), header=header, name="SCI"),
        ]).writeto(path, overwrite=True)
        try:
            input_handler = self.input_handler_for_path(path)
            with self.assertRaisesRegex(LightCurveError, "requires at least 1 valid input file"):
                generate_light_curve([input_handler], 100.0, 20.0, 2.0, 3.0, 5.0)
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
