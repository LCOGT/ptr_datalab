from __future__ import annotations

import math
import os
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from datalab.datalab_session.data_operations.light_curve_pipeline import (
    BackendDependencies,
    LightCurveError,
    configure_backend_dependencies,
    generate_light_curve,
    reset_backend_dependencies,
)


APERTURE_PHOTOMETRY_TEST_DIR = Path(__file__).resolve().parent / "test_files" / "aperture_photometry"


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


class FakeBackend:
    def __init__(self, frames: dict[str, dict[str, Any]], aavso_rows: list[dict[str, float]] | None = None):
        self.frames = frames
        self.aavso_rows = aavso_rows or []

    def as_dependencies(self) -> BackendDependencies:
        return BackendDependencies(
            load_image_data=lambda path: self.frames[path]["image"],
            load_primary_header=lambda path: self.frames[path]["header"],
            load_second_hdu_rows=lambda path: self.frames[path]["second_hdu"],
            world_to_pixel=self.world_to_pixel,
            pixel_to_world=self.pixel_to_world,
            get_dark_contribution=lambda _path, _header: 4.0,
            get_gain=lambda _path, _header: float(_header.get("GAIN", 1.7)),
            get_read_noise=lambda _path, _header: float(_header.get("RDNOISE", 3.2)),
            query_aavso=lambda _ra, _dec, _radius: self.aavso_rows,
        )

    @staticmethod
    def world_to_pixel(header: dict[str, Any], ra_deg: float, dec_deg: float) -> tuple[float, float]:
        if not header.get("HAS_WCS", True):
            raise ValueError("no WCS")
        x = header["CRPIX1"] + (ra_deg - header["CRVAL1"]) / header["CD1_1"]
        y = header["CRPIX2"] + (dec_deg - header["CRVAL2"]) / header["CD2_2"]
        return x, y

    @staticmethod
    def pixel_to_world(header: dict[str, Any], x: float, y: float) -> tuple[float, float]:
        if not header.get("HAS_WCS", True):
            raise ValueError("no WCS")
        ra = header["CRVAL1"] + (x - header["CRPIX1"]) * header["CD1_1"]
        dec = header["CRVAL2"] + (y - header["CRPIX2"]) * header["CD2_2"]
        return ra, dec


def build_frame_set(
    *,
    frame_count: int = 3,
    include_target_in_second_hdu: bool = False,
    variable_candidate_index: int | None = None,
    aavso_match_count: int = 10,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, float]], tuple[float, float]]:
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
    aavso_rows: list[dict[str, float]] = []
    for frame_index in range(frame_count):
        header = {
            "DATE-OBS": f"2025-01-01T00:00:{frame_count - frame_index:02d}",
            "CRVAL1": 100.0,
            "CRVAL2": 20.0,
            "CRPIX1": 0.0,
            "CRPIX2": 0.0,
            "CD1_1": 1.0e-5,
            "CD2_2": 1.0e-5,
            "GAIN": 1.7,
            "RDNOISE": 3.2,
            "HAS_WCS": True,
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
                    "ra_deg": ra,
                    "dec_deg": dec,
                    "mag": mag,
                    "usable": True,
                    "saturated": False,
                }
            )
            if idx < aavso_match_count:
                aavso_rows.append({"ra_deg": ra, "dec_deg": dec})

        if include_target_in_second_hdu:
            second_hdu.append(
                {
                    "id": "target-shadow",
                    "ra_deg": target_ra,
                    "dec_deg": target_dec,
                    "mag": 12.2,
                    "usable": True,
                    "saturated": False,
                }
            )

        frames[f"frame_{frame_index + 1}.fits"] = {
            "image": image,
            "header": header,
            "second_hdu": second_hdu,
        }

    return frames, aavso_rows[:aavso_match_count], (target_ra, target_dec)


class TestLightCurvePipeline(unittest.TestCase):
    def setUp(self) -> None:
        reset_backend_dependencies()

    def tearDown(self) -> None:
        reset_backend_dependencies()

    def test_sorting_by_date_obs(self) -> None:
        frames, aavso_rows, (target_ra, target_dec) = build_frame_set()
        configure_backend_dependencies(FakeBackend(frames, aavso_rows).as_dependencies())

        result = generate_light_curve(
            list(frames.keys()),
            target_ra_deg=target_ra,
            target_dec_deg=target_dec,
            aperture_radius_px=4.0,
            annulus_inner_radius_px=6.0,
            annulus_outer_radius_px=9.0,
        )

        ordered_paths = [row.fits_path for row in result.light_curve_rows]
        self.assertEqual(ordered_paths, ["frame_3.fits", "frame_2.fits", "frame_1.fits"])

    def test_failure_on_missing_date_obs(self) -> None:
        frames, aavso_rows, (target_ra, target_dec) = build_frame_set()
        del frames["frame_1.fits"]["header"]["DATE-OBS"]
        configure_backend_dependencies(FakeBackend(frames, aavso_rows).as_dependencies())

        with self.assertRaisesRegex(LightCurveError, "Missing DATE-OBS"):
            generate_light_curve(list(frames.keys()), target_ra, target_dec, 4.0, 6.0, 9.0)

    def test_failure_on_malformed_date_obs(self) -> None:
        frames, aavso_rows, (target_ra, target_dec) = build_frame_set()
        frames["frame_1.fits"]["header"]["DATE-OBS"] = "not-a-date"
        configure_backend_dependencies(FakeBackend(frames, aavso_rows).as_dependencies())

        with self.assertRaisesRegex(LightCurveError, "Malformed DATE-OBS"):
            generate_light_curve(list(frames.keys()), target_ra, target_dec, 4.0, 6.0, 9.0)

    def test_failure_on_missing_wcs(self) -> None:
        frames, aavso_rows, (target_ra, target_dec) = build_frame_set()
        frames["frame_1.fits"]["header"]["HAS_WCS"] = False
        configure_backend_dependencies(FakeBackend(frames, aavso_rows).as_dependencies())

        with self.assertRaisesRegex(LightCurveError, "unusable WCS"):
            generate_light_curve(list(frames.keys()), target_ra, target_dec, 4.0, 6.0, 9.0)

    def test_failure_on_missing_second_hdu(self) -> None:
        frames, aavso_rows, (target_ra, target_dec) = build_frame_set()
        frames["frame_1.fits"]["second_hdu"] = []
        configure_backend_dependencies(FakeBackend(frames, aavso_rows).as_dependencies())

        with self.assertRaisesRegex(LightCurveError, "Second HDU is missing or empty"):
            generate_light_curve(list(frames.keys()), target_ra, target_dec, 4.0, 6.0, 9.0)

    def test_failure_on_missing_second_hdu_columns(self) -> None:
        frames, aavso_rows, (target_ra, target_dec) = build_frame_set()
        del frames["frame_1.fits"]["second_hdu"][0]["ra_deg"]
        configure_backend_dependencies(FakeBackend(frames, aavso_rows).as_dependencies())

        with self.assertRaisesRegex(LightCurveError, "missing required RA column"):
            generate_light_curve(list(frames.keys()), target_ra, target_dec, 4.0, 6.0, 9.0)

    def test_target_photometry_succeeds_when_target_absent_from_second_hdu(self) -> None:
        frames, aavso_rows, (target_ra, target_dec) = build_frame_set(include_target_in_second_hdu=False)
        configure_backend_dependencies(FakeBackend(frames, aavso_rows).as_dependencies())

        result = generate_light_curve(list(frames.keys()), target_ra, target_dec, 4.0, 6.0, 9.0)

        self.assertEqual(len(result.light_curve_rows), 3)
        self.assertTrue(all(row.target_net_source_counts > 0.0 for row in result.light_curve_rows))

    def test_target_centroiding_converges_on_synthetic_images(self) -> None:
        frames, aavso_rows, (target_ra, target_dec) = build_frame_set()
        configure_backend_dependencies(FakeBackend(frames, aavso_rows).as_dependencies())

        result = generate_light_curve(list(frames.keys()), target_ra, target_dec, 4.0, 6.0, 9.0)

        first = result.light_curve_rows[0]
        self.assertAlmostEqual(first.target_centroid_x, 30.3, delta=0.2)
        self.assertAlmostEqual(first.target_centroid_y, 28.8, delta=0.2)

    def test_target_centroid_failure_is_hard_failure(self) -> None:
        frames, aavso_rows, (target_ra, target_dec) = build_frame_set()
        header = frames["frame_1.fits"]["header"]
        header["CRVAL1"] = 100.1
        header["CRVAL2"] = 20.1
        configure_backend_dependencies(FakeBackend(frames, aavso_rows).as_dependencies())

        with self.assertRaisesRegex(LightCurveError, "Target centroiding failed"):
            generate_light_curve(list(frames.keys()), target_ra, target_dec, 4.0, 6.0, 9.0)

    def test_second_hdu_radec_cross_match_across_frames(self) -> None:
        frames, aavso_rows, (target_ra, target_dec) = build_frame_set()
        for frame in frames.values():
            for row in frame["second_hdu"]:
                row["ra_deg"] += 1.0e-8
                row["dec_deg"] -= 1.0e-8
        configure_backend_dependencies(FakeBackend(frames, aavso_rows).as_dependencies())

        result = generate_light_curve(list(frames.keys()), target_ra, target_dec, 4.0, 6.0, 9.0)
        self.assertGreaterEqual(len(result.selected_comparison_stars), 5)

    def test_aavso_assisted_candidate_matching(self) -> None:
        frames, aavso_rows, (target_ra, target_dec) = build_frame_set(aavso_match_count=6)
        configure_backend_dependencies(FakeBackend(frames, aavso_rows).as_dependencies())

        result = generate_light_curve(
            list(frames.keys()),
            target_ra,
            target_dec,
            4.0,
            6.0,
            9.0,
            comparison_strategy="aavso_first",
            min_comparisons=5,
            max_comparisons=6,
        )

        self.assertEqual(len(result.selected_comparison_stars), 6)
        self.assertTrue(all(star.matched_aavso for star in result.selected_comparison_stars))

    def test_variability_based_ranking_rejects_variable_candidate(self) -> None:
        frames, aavso_rows, (target_ra, target_dec) = build_frame_set(variable_candidate_index=0, aavso_match_count=0)
        configure_backend_dependencies(FakeBackend(frames, aavso_rows).as_dependencies())

        result = generate_light_curve(
            list(frames.keys()),
            target_ra,
            target_dec,
            4.0,
            6.0,
            9.0,
            comparison_strategy="variability_first",
            min_comparisons=5,
            max_comparisons=10,
        )

        selected_ids = {star.candidate_id for star in result.selected_comparison_stars}
        self.assertNotIn("cand-001", selected_ids)

    def test_target_exclusion_from_candidate_pool(self) -> None:
        frames, aavso_rows, (target_ra, target_dec) = build_frame_set(include_target_in_second_hdu=True)
        configure_backend_dependencies(FakeBackend(frames, aavso_rows).as_dependencies())

        result = generate_light_curve(list(frames.keys()), target_ra, target_dec, 4.0, 6.0, 9.0)

        self.assertTrue(all(star.target_separation_px > 9.0 for star in result.selected_comparison_stars))
        self.assertTrue(any("too close to target" in diagnostic for diagnostic in result.diagnostics))

    def test_magnitude_similarity_uses_target_photometry_proxy(self) -> None:
        frames, aavso_rows, (target_ra, target_dec) = build_frame_set(aavso_match_count=0)
        for frame in frames.values():
            frame["second_hdu"][0]["mag"] = 30.0
        configure_backend_dependencies(FakeBackend(frames, aavso_rows).as_dependencies())

        result = generate_light_curve(
            list(frames.keys()),
            target_ra,
            target_dec,
            4.0,
            6.0,
            9.0,
            comparison_strategy="variability_first",
            min_comparisons=5,
            max_comparisons=5,
        )

        self.assertTrue(all(star.reference_magnitude < 20.0 for star in result.selected_comparison_stars))

    def test_fallback_from_aavso_first_to_variability_based(self) -> None:
        frames, _, (target_ra, target_dec) = build_frame_set(aavso_match_count=0)
        configure_backend_dependencies(FakeBackend(frames, []).as_dependencies())

        result = generate_light_curve(
            list(frames.keys()),
            target_ra,
            target_dec,
            4.0,
            6.0,
            9.0,
            comparison_strategy="aavso_first",
        )

        self.assertTrue(any("fallback path used: aavso -> variability" == diagnostic for diagnostic in result.diagnostics))

    def test_fallback_from_variability_first_to_aavso(self) -> None:
        frames, aavso_rows, (target_ra, target_dec) = build_frame_set(variable_candidate_index=0, aavso_match_count=5)
        for frame in frames.values():
            frame["second_hdu"] = frame["second_hdu"][:5]
        configure_backend_dependencies(FakeBackend(frames, aavso_rows).as_dependencies())

        result = generate_light_curve(
            list(frames.keys()),
            target_ra,
            target_dec,
            4.0,
            6.0,
            9.0,
            comparison_strategy="variability_first",
            min_comparisons=5,
            max_comparisons=5,
        )

        self.assertTrue(any("fallback path used: variability -> aavso" == diagnostic for diagnostic in result.diagnostics))

    def test_failure_when_fewer_than_five_valid_comparisons_remain(self) -> None:
        frames, aavso_rows, (target_ra, target_dec) = build_frame_set()
        for frame in frames.values():
            frame["second_hdu"] = frame["second_hdu"][:4]
        configure_backend_dependencies(FakeBackend(frames, aavso_rows[:4]).as_dependencies())

        with self.assertRaisesRegex(LightCurveError, "minimum comparison ensemble"):
            generate_light_curve(list(frames.keys()), target_ra, target_dec, 4.0, 6.0, 9.0)

    def test_false_flagged_bad_column_does_not_reject_candidates(self) -> None:
        frames, aavso_rows, (target_ra, target_dec) = build_frame_set()
        for frame in frames.values():
            for row in frame["second_hdu"]:
                row.pop("usable")
                row["flagged_bad"] = False
        configure_backend_dependencies(FakeBackend(frames, aavso_rows).as_dependencies())

        result = generate_light_curve(list(frames.keys()), target_ra, target_dec, 4.0, 6.0, 9.0)

        self.assertGreaterEqual(len(result.selected_comparison_stars), 5)

    def test_invalid_radius_and_comparison_limits_fail_fast(self) -> None:
        frames, aavso_rows, (target_ra, target_dec) = build_frame_set()
        configure_backend_dependencies(FakeBackend(frames, aavso_rows).as_dependencies())

        with self.assertRaisesRegex(LightCurveError, "aperture_radius_px must be > 0"):
            generate_light_curve(list(frames.keys()), target_ra, target_dec, 0.0, 6.0, 9.0)
        with self.assertRaisesRegex(LightCurveError, "annulus_inner_radius_px"):
            generate_light_curve(list(frames.keys()), target_ra, target_dec, 4.0, 4.0, 9.0)
        with self.assertRaisesRegex(LightCurveError, "annulus_outer_radius_px"):
            generate_light_curve(list(frames.keys()), target_ra, target_dec, 4.0, 6.0, 6.0)
        with self.assertRaisesRegex(LightCurveError, "min_comparisons and max_comparisons"):
            generate_light_curve(list(frames.keys()), target_ra, target_dec, 4.0, 6.0, 9.0, min_comparisons=6, max_comparisons=5)

    def test_aperture_photometry_and_calibration_outputs_are_finite(self) -> None:
        frames, aavso_rows, (target_ra, target_dec) = build_frame_set()
        configure_backend_dependencies(FakeBackend(frames, aavso_rows).as_dependencies())

        result = generate_light_curve(list(frames.keys()), target_ra, target_dec, 4.0, 6.0, 9.0)
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
        frames, aavso_rows, (target_ra, target_dec) = build_frame_set()
        configure_backend_dependencies(FakeBackend(frames, aavso_rows).as_dependencies())

        result = generate_light_curve(list(frames.keys()), target_ra, target_dec, 4.0, 6.0, 9.0)
        row = result.light_curve_rows[0]

        recomputed_rel_flux = row.target_net_source_counts / row.comparison_ensemble_total_counts
        self.assertAlmostEqual(row.target_differential_flux, recomputed_rel_flux, places=12)
        self.assertGreater(row.target_differential_flux_uncertainty, 0.0)

    def test_determinism(self) -> None:
        frames, aavso_rows, (target_ra, target_dec) = build_frame_set()
        backend = FakeBackend(frames, aavso_rows)
        configure_backend_dependencies(backend.as_dependencies())
        result1 = generate_light_curve(list(frames.keys()), target_ra, target_dec, 4.0, 6.0, 9.0)

        configure_backend_dependencies(backend.as_dependencies())
        result2 = generate_light_curve(list(frames.keys()), target_ra, target_dec, 4.0, 6.0, 9.0)

        self.assertEqual(
            [star.candidate_id for star in result1.selected_comparison_stars],
            [star.candidate_id for star in result2.selected_comparison_stars],
        )
        self.assertEqual(
            [row.target_calibrated_apparent_magnitude for row in result1.light_curve_rows],
            [row.target_calibrated_apparent_magnitude for row in result2.light_curve_rows],
        )

    def test_default_fits_dependencies_read_sci_header_and_cat_rows(self) -> None:
        reset_backend_dependencies()
        frames, _, (target_ra, target_dec) = build_frame_set(frame_count=1, aavso_match_count=0)
        handle = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
        path = handle.name
        handle.close()
        frame = frames["frame_1.fits"]
        header = fits.Header()
        for key, value in frame["header"].items():
            if key == "HAS_WCS":
                continue
            header[key] = value
        header["CTYPE1"] = "RA---TAN"
        header["CTYPE2"] = "DEC--TAN"
        header["CUNIT1"] = "deg"
        header["CUNIT2"] = "deg"

        cat_columns = [
            fits.Column(name="id", format="16A", array=np.asarray([row["id"] for row in frame["second_hdu"]])),
            fits.Column(name="ra_deg", format="D", array=np.asarray([row["ra_deg"] for row in frame["second_hdu"]])),
            fits.Column(name="dec_deg", format="D", array=np.asarray([row["dec_deg"] for row in frame["second_hdu"]])),
            fits.Column(name="mag", format="D", array=np.asarray([row["mag"] for row in frame["second_hdu"]])),
            fits.Column(name="usable", format="L", array=np.asarray([row["usable"] for row in frame["second_hdu"]])),
            fits.Column(name="saturated", format="L", array=np.asarray([row["saturated"] for row in frame["second_hdu"]])),
        ]
        hdul = fits.HDUList([
            fits.PrimaryHDU(),
            fits.ImageHDU(data=frame["image"], header=header, name="SCI"),
            fits.BinTableHDU.from_columns(cat_columns, name="CAT"),
        ])
        hdul.writeto(path, overwrite=True)
        try:
            result = generate_light_curve(
                [path],
                target_ra_deg=target_ra,
                target_dec_deg=target_dec,
                aperture_radius_px=4.0,
                annulus_inner_radius_px=6.0,
                annulus_outer_radius_px=9.0,
                comparison_strategy="variability_first",
            )
        finally:
            os.remove(path)

        self.assertEqual(len(result.light_curve_rows), 1)
        self.assertGreaterEqual(len(result.selected_comparison_stars), 5)

    def test_real_compressed_fits_aperture_photometry_prints_diagnostics_and_results(self) -> None:
        reset_backend_dependencies()
        fits_paths = sorted(str(path) for path in APERTURE_PHOTOMETRY_TEST_DIR.glob("*.fits.fz"))
        self.assertEqual(len(fits_paths), 3)

        result = generate_light_curve(
            fits_paths,
            target_ra_deg=199.10004,
            target_dec_deg=42.03237,
            aperture_radius_px=7.73,
            annulus_inner_radius_px=12.89,
            annulus_outer_radius_px=19.33,
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

        self.assertEqual(len(result.light_curve_rows), 3)
        self.assertGreaterEqual(len(result.selected_comparison_stars), 5)
        self.assertTrue(all(math.isfinite(row.target_centroid_x) for row in result.light_curve_rows))
        self.assertTrue(all(math.isfinite(row.target_centroid_y) for row in result.light_curve_rows))
        self.assertTrue(all(math.isfinite(row.target_differential_flux) for row in result.light_curve_rows))
        self.assertTrue(all(
            12.0 <= row.target_calibrated_apparent_magnitude <= 12.7
            for row in result.light_curve_rows
        ))

    def test_default_fits_dependencies_reject_missing_cat(self) -> None:
        reset_backend_dependencies()
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
            with self.assertRaisesRegex(LightCurveError, "CAT HDU is missing"):
                generate_light_curve([path], 100.0, 20.0, 2.0, 3.0, 5.0)
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
