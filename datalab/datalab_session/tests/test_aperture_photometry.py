"""Characterization tests for the aperture photometry core.

These lock in the behavior of the photometry components. Centroiding and the
background estimator have been consolidated onto the shared
analysis/centroiding_core.py algorithm, and the aperture pipeline uses AIJ's
photometer background convergence (100 iterations / 1e-4 tolerance). They are
intentionally golden/characterization tests: the real-frame values were captured
from the implementation, so any change that moves them will surface as an
explicit, reviewable diff.

Fixtures are the three well-behaved tfn0m4 frames (rp filter, 2400x2400, GAIN 1.0,
RDNOISE ~3.18) in tests/test_files/aperture_photometry/.
"""

import glob
import math
import os
from datetime import datetime, timezone
from unittest import mock

import numpy as np

from datalab.datalab_session.analysis.centroiding_core import sigma_clipped_annulus_background
from datalab.datalab_session.data_operations import aperture_photometry as ap
from datalab.datalab_session.tests.test_files.file_extended_test_case import FileExtendedTestCase

APERTURE_FIXTURE_DIR = "datalab/datalab_session/tests/test_files/aperture_photometry"

# Default aperture geometry shared by the operation wizard and these tests.
APERTURE_RADIUS_PX = 7.64
ANNULUS_INNER_RADIUS_PX = 12.73
ANNULUS_OUTER_RADIUS_PX = 19.10

# A bright, interior source in the tfn0m4 field used as the differential target.
# Catalog magnitude here is 12.712, which the pipeline should recover closely.
TARGET_RA_DEG = 199.121476
TARGET_DEC_DEG = 41.940252

# Golden light-curve values captured from the current implementation. Keyed by
# FITS basename. Centroids are deterministic; counts/mags asserted with tolerance.
GOLDEN_LIGHT_CURVE = {
    "tfn0m419-sq32-20260504-0061-e91.fits.fz": {
        "centroid": (613.9212, 788.3587),
        "net_source_counts": 736719.981,
        "comparison_ensemble_total_counts": 7324317.249,
        "calibrated_mag": 12.68863,
    },
    "tfn0m419-sq32-20260504-0077-e91.fits.fz": {
        "centroid": (604.3355, 776.3546),
        "net_source_counts": 729667.929,
        "comparison_ensemble_total_counts": 7363224.233,
        "calibrated_mag": 12.70483,
    },
    "tfn0m436-sq33-20260504-0083-e91.fits.fz": {
        "centroid": (576.8525, 776.8594),
        "net_source_counts": 727216.109,
        "comparison_ensemble_total_counts": 7335032.049,
        "calibrated_mag": 12.70432,
    },
}


def _fixture_frame_paths():
    paths = sorted(glob.glob(os.path.join(APERTURE_FIXTURE_DIR, "tfn0m4*.fits.fz")))
    return paths


class TestMeasureApertureSynthetic(FileExtendedTestCase):
    """_measure_aperture on a synthetic frame with analytically known answers."""

    def test_flat_sky_with_core_block(self):
        # Flat sky of 200 counts plus a 3x3 block of +1000 fully inside the
        # aperture core. Net source = 9 * 1000 = 9000; background = 200.
        image = np.full((60, 60), 200.0)
        image[28:31, 28:31] += 1000.0

        result = ap._measure_aperture(
            image=image,
            x_center=30.0,
            y_center=30.0,
            aperture_radius_px=APERTURE_RADIUS_PX,
            annulus_inner_radius_px=ANNULUS_INNER_RADIUS_PX,
            annulus_outer_radius_px=ANNULUS_OUTER_RADIUS_PX,
            gain=1.0,
            read_noise=3.18,
            dark=0.0,
        )

        self.assertAlmostEqual(result["net_source_counts"], 9000.0, places=4)
        self.assertAlmostEqual(result["mean_background_per_pixel"], 200.0, places=4)
        self.assertAlmostEqual(result["peak_pixel_value"], 1200.0, places=4)
        # Effective source pixels ~= pi * r^2 sampled on a 5x5 sub-pixel grid.
        self.assertAlmostEqual(result["effective_source_pixels"], 182.88, places=2)
        # Regression value for the AIJ noise formula (gain=1, rdnoise=3.18).
        self.assertAlmostEqual(result["source_uncertainty"], 241.2942, places=3)

    def test_empty_annulus_raises(self):
        # Image smaller than the inner annulus radius: no pixel can fall in the
        # background annulus, so measurement fails.
        image = np.full((10, 10), 100.0)
        with self.assertRaises(ap.LightCurveError):
            ap._measure_aperture(
                image=image,
                x_center=5.0,
                y_center=5.0,
                aperture_radius_px=APERTURE_RADIUS_PX,
                annulus_inner_radius_px=ANNULUS_INNER_RADIUS_PX,
                annulus_outer_radius_px=ANNULUS_OUTER_RADIUS_PX,
                gain=1.0,
                read_noise=3.18,
                dark=0.0,
            )


class TestEstimateBackgroundSynthetic(FileExtendedTestCase):
    """The shared background estimator sigma-clips bright outliers out of the annulus."""

    def test_sigma_clip_rejects_outliers(self):
        # Uniform sky of 500 with a handful of star-like outliers in the annulus.
        image = np.full((60, 60), 500.0)
        for i, j in [(45, 30), (15, 30), (30, 45), (30, 15), (40, 40)]:
            image[j, i] = 8000.0

        result = sigma_clipped_annulus_background(
            image,
            30.0,
            30.0,
            ANNULUS_INNER_RADIUS_PX,
            ANNULUS_OUTER_RADIUS_PX,
            remove_stars=True,
        )

        self.assertIsNotNone(result)
        background, kept = result
        # The five injected outliers are rejected, leaving the uniform sky.
        self.assertAlmostEqual(background, 500.0, places=3)


class TestCentroidCharacterization(FileExtendedTestCase):
    """Pin the _iterative_centroid result (now the shared AIJ core) per frame."""

    def test_target_centroid_per_frame(self):
        from astropy.io import fits

        paths = _fixture_frame_paths()
        self.assertEqual(len(paths), 3)
        for path in paths:
            basename = os.path.basename(path)
            with fits.open(path) as hdul:
                image = np.asarray(hdul["SCI"].data, dtype=float)
                header = dict(hdul["SCI"].header)
            x0, y0 = ap._world_to_pixel(header, TARGET_RA_DEG, TARGET_DEC_DEG)
            centroid = ap._iterative_centroid(
                image=image,
                x_start=x0,
                y_start=y0,
                aperture_radius_px=APERTURE_RADIUS_PX,
                annulus_inner_radius_px=ANNULUS_INNER_RADIUS_PX,
                annulus_outer_radius_px=ANNULUS_OUTER_RADIUS_PX,
            )
            self.assertIsNotNone(centroid, f"centroid failed for {basename}")
            expected_x, expected_y = GOLDEN_LIGHT_CURVE[basename]["centroid"]
            self.assertAlmostEqual(centroid[0], expected_x, places=2)
            self.assertAlmostEqual(centroid[1], expected_y, places=2)


class TestGenerateLightCurveGolden(FileExtendedTestCase):
    """End-to-end generate_light_curve on the three well-behaved frames.

    The pipeline is expensive (pure-Python pixel loops over every catalog source
    on every frame), so it is run once for the whole class.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.result = ap.generate_light_curve(
            fits_paths=_fixture_frame_paths(),
            target_ra_deg=TARGET_RA_DEG,
            target_dec_deg=TARGET_DEC_DEG,
            aperture_radius_px=APERTURE_RADIUS_PX,
            annulus_inner_radius_px=ANNULUS_INNER_RADIUS_PX,
            annulus_outer_radius_px=ANNULUS_OUTER_RADIUS_PX,
            min_comparisons=5,
            max_comparisons=10,
        )

    def test_selects_full_comparison_ensemble(self):
        self.assertEqual(len(self.result.selected_comparison_stars), 10)

    def test_one_row_per_frame_in_time_order(self):
        rows = self.result.light_curve_rows
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows, sorted(rows, key=lambda r: r.date_obs))

    def test_light_curve_matches_golden(self):
        for row in self.result.light_curve_rows:
            basename = os.path.basename(row.fits_path)
            golden = GOLDEN_LIGHT_CURVE[basename]
            self.assertAlmostEqual(row.target_centroid_x, golden["centroid"][0], places=2)
            self.assertAlmostEqual(row.target_centroid_y, golden["centroid"][1], places=2)
            self.assertAlmostEqual(
                row.target_net_source_counts,
                golden["net_source_counts"],
                delta=abs(golden["net_source_counts"]) * 1e-4,
            )
            self.assertAlmostEqual(
                row.comparison_ensemble_total_counts,
                golden["comparison_ensemble_total_counts"],
                delta=abs(golden["comparison_ensemble_total_counts"]) * 1e-4,
            )
            self.assertTrue(math.isfinite(row.target_calibrated_apparent_magnitude))
            self.assertAlmostEqual(
                row.target_calibrated_apparent_magnitude,
                golden["calibrated_mag"],
                delta=0.001,
            )

    def test_recovers_catalog_magnitude(self):
        # Sanity: the calibrated magnitude should land near the catalog mag (12.712).
        mags = [r.target_calibrated_apparent_magnitude for r in self.result.light_curve_rows]
        self.assertTrue(all(12.5 < m < 12.9 for m in mags))


# ---------------------------------------------------------------------------
# Regression tests for the NGC 7331 supernova robustness behaviours, using the
# BackendDependencies injection hook so no real FITS frames are needed.
# ---------------------------------------------------------------------------

def _synthetic_backend(**overrides):
    """A BackendDependencies whose I/O is backed by in-memory dicts.

    Sensible defaults (gain=1, read_noise=0, dark=0, identity-ish WCS); pass
    callables to override any field.
    """
    defaults = dict(
        load_image_data=lambda path: np.zeros((1, 1)),
        load_primary_header=lambda path: {},
        load_second_hdu_rows=lambda path: [],
        world_to_pixel=lambda header, ra, dec: (float(ra), float(dec)),
        pixel_to_world=lambda header, x, y: (float(x), float(y)),
        get_dark_contribution=lambda path, header: 0.0,
        get_gain=lambda path, header: 1.0,
        get_read_noise=lambda path, header: 0.0,
    )
    defaults.update(overrides)
    return ap.BackendDependencies(**defaults)


def _frame(fits_path, image, rows, *, header=None):
    height, width = image.shape
    return ap.FrameContext(
        fits_path=fits_path,
        date_obs=datetime(2026, 1, 1, tzinfo=timezone.utc),
        header=header or {},
        image=image,
        second_hdu_rows=tuple(rows),
        width=width,
        height=height,
    )


class TestTargetRecenterFallback(FileExtendedTestCase):
    """A target recenter beyond the cap (or a failed centroid) falls back to WCS."""

    def setUp(self):
        # WCS always places the target at pixel (30, 30) on a flat frame.
        ap.configure_backend_dependencies(
            _synthetic_backend(world_to_pixel=lambda header, ra, dec: (30.0, 30.0))
        )
        self.addCleanup(ap.reset_backend_dependencies)
        self.frame = _frame("t.fits", np.full((60, 60), 100.0), [{"ra": 1.0, "dec": 2.0}])

    def _measure(self):
        diagnostics: list[str] = []
        measurement = ap._measure_target(
            frame=self.frame,
            target_ra_deg=1.0,
            target_dec_deg=2.0,
            aperture_radius_px=7.64,
            annulus_inner_radius_px=12.73,
            annulus_outer_radius_px=19.10,
            diagnostics=diagnostics,
        )
        return measurement, diagnostics

    def test_recenter_within_cap_is_kept(self):
        # Centroid 2px from WCS (< 6px cap) -> use the centroid, no fallback.
        with mock.patch.object(ap, "_iterative_centroid", return_value=(32.0, 30.0)):
            measurement, diagnostics = self._measure()
        self.assertEqual((measurement.x, measurement.y), (32.0, 30.0))
        self.assertFalse(any("recenter rejected" in m for m in diagnostics))

    def test_recenter_beyond_cap_falls_back_to_wcs(self):
        # Centroid 10px from WCS (> 6px cap) -> fall back to the WCS position.
        with mock.patch.object(ap, "_iterative_centroid", return_value=(40.0, 30.0)):
            measurement, diagnostics = self._measure()
        self.assertEqual((measurement.x, measurement.y), (30.0, 30.0))
        self.assertTrue(any("recenter rejected" in m and "exceeded" in m for m in diagnostics))

    def test_failed_centroid_falls_back_to_wcs(self):
        with mock.patch.object(ap, "_iterative_centroid", return_value=None):
            measurement, diagnostics = self._measure()
        self.assertEqual((measurement.x, measurement.y), (30.0, 30.0))
        self.assertTrue(any("centroiding failed" in m for m in diagnostics))


class TestDegenerateCatalogSkip(FileExtendedTestCase):
    """Frames whose catalog lacks RA/Dec/mag are skipped, not fatal."""

    def _store_backend(self, store):
        ap.configure_backend_dependencies(
            _synthetic_backend(
                load_image_data=lambda path: store[path]["image"],
                load_primary_header=lambda path: store[path]["header"],
                load_second_hdu_rows=lambda path: store[path]["rows"],
            )
        )
        self.addCleanup(ap.reset_backend_dependencies)

    def test_skips_frame_missing_ra_and_keeps_the_rest(self):
        header = {"DATE-OBS": "2026-01-01T00:00:00", "CRVAL1": 1.0, "CRVAL2": 2.0}
        good_rows = [{"ra": 1.0, "dec": 2.0, "mag": 15.0, "flux": 1000.0}]
        bad_rows = [{"dec": 2.0, "mag": 15.0, "flux": 1000.0}]  # no RA column
        image = np.zeros((50, 50))
        self._store_backend({
            "good1.fits": {"image": image, "header": header, "rows": good_rows},
            "bad.fits": {"image": image, "header": header, "rows": bad_rows},
            "good2.fits": {"image": image, "header": header, "rows": good_rows},
        })

        frames, skipped = ap._load_and_validate_frames(["good1.fits", "bad.fits", "good2.fits"])

        self.assertEqual([f.fits_path for f in frames], ["good1.fits", "good2.fits"])
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0][0], "bad.fits")
        self.assertIn("RA", skipped[0][1])

    def test_all_bad_frames_yield_no_usable_frames(self):
        # When every frame is degenerate, none survive (generate_light_curve then
        # raises the "no usable source catalog" error on the empty list).
        header = {"DATE-OBS": "2026-01-01T00:00:00", "CRVAL1": 1.0, "CRVAL2": 2.0}
        bad_rows = [{"dec": 2.0, "mag": 15.0, "flux": 1000.0}]
        self._store_backend({
            "bad.fits": {"image": np.zeros((50, 50)), "header": header, "rows": bad_rows},
        })
        frames, skipped = ap._load_and_validate_frames(["bad.fits"])
        self.assertEqual(frames, [])
        self.assertEqual(len(skipped), 1)


class TestComparisonCoverageRule(FileExtendedTestCase):
    """A candidate is kept only if cross-matched in >= 80% of frames."""

    def setUp(self):
        # Map RA/Dec (small offsets from 10,20) to interior pixels of a 200x200 frame.
        ap.configure_backend_dependencies(
            _synthetic_backend(
                world_to_pixel=lambda header, ra, dec: (
                    (ra - 10.0) * 5000.0 + 50.0,
                    (dec - 20.0) * 5000.0 + 50.0,
                )
            )
        )
        self.addCleanup(ap.reset_backend_dependencies)

    def test_candidate_below_coverage_fraction_is_rejected(self):
        star_a = {"ra": 10.01, "dec": 20.01, "mag": 16.0, "flux": 500.0}  # -> (100,100)
        star_b = {"ra": 10.02, "dec": 20.00, "mag": 16.5, "flux": 400.0}  # -> (150, 50)
        star_c = {"ra": 10.00, "dec": 20.02, "mag": 17.0, "flux": 300.0}  # -> ( 50,150)
        image = np.zeros((200, 200))
        # 5 frames: A in 4 (>=ceil(0.8*5)=4 -> kept), B in 3 (rejected), C in 1 (rejected).
        rows_per_frame = [[star_a, star_b], [star_a, star_b], [star_a, star_b], [star_a], [star_c]]
        frames = [_frame(f"f{i}.fits", image, rows) for i, rows in enumerate(rows_per_frame)]

        catalog = ap._build_field_star_catalog(
            frames=frames,
            target_ra_deg=10.0,
            target_dec_deg=20.0,
            aperture_radius_px=7.64,
            annulus_outer_radius_px=19.10,
        )

        self.assertEqual(len(catalog), 1)
        self.assertAlmostEqual(catalog[0]["ra_deg"], 10.01, places=6)
