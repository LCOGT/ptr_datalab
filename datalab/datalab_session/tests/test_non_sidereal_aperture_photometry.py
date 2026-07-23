from __future__ import annotations

import math
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import astropy.units as u
from astropy.coordinates import Angle
from astropy.io import fits
from astropy.wcs import WCS

from datalab.datalab_session.tests.test_aperture_photometry import gaussian_star
from datalab.datalab_session.utils.aperture_light_curve import (
    LightCurveError,
    TARGET_POSITION_HEADER,
    REFINEMENT_FORCED,
    generate_light_curve,
)
from datalab.datalab_session.utils.comparison_calibration import (
    COMPARISON_AUTO,
    COMPARISON_EVOLVING,
    COMPARISON_SHARED,
    FrameCalibration,
    _empirical_error_floor,
)
from datalab.datalab_session.utils.comparison_stars import ComparisonMeasurement, ComparisonStar
from datalab.datalab_session.utils.fits_metadata import target_radec_from_header


DEG_PER_PIXEL = 1.0 / 3600.0  # 1 arcsec/pixel
FLUX_ZERO_POINT = 25.0        # counts = 10 ** (-0.4 * (mag - FLUX_ZERO_POINT))


def _flux_for_mag(mag: float) -> float:
    return 10.0 ** (-0.4 * (mag - FLUX_ZERO_POINT))


def sexagesimal(ra_deg: float, dec_deg: float) -> tuple[str, str]:
    ra = Angle(ra_deg, unit=u.deg).to_string(unit=u.hourangle, sep=":", pad=True, precision=4)
    dec = Angle(dec_deg, unit=u.deg).to_string(unit=u.deg, sep=":", pad=True, alwayssign=True, precision=3)
    return ra, dec


def _header(target_ra: float, target_dec: float, width: int, height: int, date_obs: str) -> dict[str, Any]:
    ra_str, dec_str = sexagesimal(target_ra, target_dec)
    return {
        "DATE-OBS": date_obs,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        # The mount tracks the moving target, so the field center follows the ephemeris and the
        # target sits at frame center every frame (as on real LCO MINORPLANET frames).
        "CRVAL1": target_ra,
        "CRVAL2": target_dec,
        # CRPIX is 1-based (astropy world_to_pixel returns 0-based), so +1 puts the reference sky
        # position -- the target -- at 0-based pixel (width/2, height/2), where it is drawn.
        "CRPIX1": width / 2.0 + 1.0,
        "CRPIX2": height / 2.0 + 1.0,
        "CD1_1": DEG_PER_PIXEL,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": DEG_PER_PIXEL,
        "GAIN": 1.7,
        "RDNOISE": 3.2,
        "CAT-RA": ra_str,
        "CAT-DEC": dec_str,
    }


def build_non_sidereal_frame_set(
    *,
    frame_count: int = 12,
    width: int = 120,
    height: int = 120,
    ra_drift_arcsec_per_frame: float = 12.0,
    field_center_ra_offsets_arcsec: list[float] | None = None,
    star_ra_offsets_arcsec: tuple[float, ...] = tuple(range(-90, 210, 15)),
    star_dec_offsets_arcsec: tuple[float, ...] = (-40.0, -20.0, 20.0, 40.0),
    target_flux_by_frame: list[float] | None = None,
    star_mag: float = 15.0,
    variable_star_offset: tuple[float, float] | None = None,
    include_target_in_catalog: bool = False,
) -> tuple[dict[str, dict[str, Any]], float]:
    """
        Builds a synthetic non-sidereal frame set: a target tracked at frame center whose ephemeris
        (CAT-RA/CAT-DEC) marches in RA, over a fixed grid of field stars. As the field center drifts,
        stars leave one edge and enter the other, so the comparison cast turns over across the series.

        Star positions are fixed in RA/Dec; each frame draws only the stars that fall in-frame and
        catalogs them. Returns the frames dict (for TestCase.write_frames) and the base target RA.
    """
    base_ra = 100.0
    base_dec = 20.0
    if field_center_ra_offsets_arcsec is None:
        field_center_ra_offsets_arcsec = [ra_drift_arcsec_per_frame * index for index in range(frame_count)]
    if target_flux_by_frame is None:
        target_flux_by_frame = [_flux_for_mag(15.0)] * frame_count

    frames: dict[str, dict[str, Any]] = {}
    for frame_index, center_offset_arcsec in enumerate(field_center_ra_offsets_arcsec):
        target_ra = base_ra + center_offset_arcsec / 3600.0 / math.cos(math.radians(base_dec))
        target_dec = base_dec
        header = _header(target_ra, target_dec, width, height, f"2025-03-0{1 + frame_index // 6}T0{frame_index % 6}:00:00")
        wcs = WCS(header)

        image = np.full((height, width), 100.0, dtype=float)
        gaussian_star(image, width / 2.0, height / 2.0, flux=target_flux_by_frame[frame_index], sigma=1.05)

        second_hdu: list[dict[str, Any]] = []
        star_index = 0
        for ra_off in star_ra_offsets_arcsec:
            for dec_off in star_dec_offsets_arcsec:
                star_index += 1
                star_ra = base_ra + ra_off / 3600.0 / math.cos(math.radians(base_dec))
                star_dec = base_dec + dec_off / 3600.0
                x, y = (float(v) for v in wcs.world_to_pixel_values(star_ra, star_dec))
                if not (6.0 <= x <= width - 6.0 and 6.0 <= y <= height - 6.0):
                    continue
                this_mag = star_mag
                flux = _flux_for_mag(this_mag)
                if variable_star_offset is not None and (ra_off, dec_off) == variable_star_offset:
                    flux = flux * (1.0 + 0.4 * frame_index)
                gaussian_star(image, x, y, flux=flux, sigma=1.05)
                second_hdu.append(
                    {
                        "id": f"star-{star_index:03d}",
                        "ra": star_ra,
                        "dec": star_dec,
                        "mag": this_mag,
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

    return frames, base_ra


class TestNonSiderealAperturePhotometry(unittest.TestCase):
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

    def _finite_rows(self, result: Any) -> list[Any]:
        return [
            row for row in result.light_curve_rows
            if math.isfinite(row.target_calibrated_apparent_magnitude)
        ]

    # ------------------------------------------------------------------ helpers

    def test_target_radec_from_header_parses_sexagesimal(self) -> None:
        ra_deg, dec_deg = target_radec_from_header({"CAT-RA": "18:07:23.824", "CAT-DEC": "-07:28:36.39"})
        self.assertAlmostEqual(ra_deg, 271.84927, places=4)
        self.assertAlmostEqual(dec_deg, -7.47677, places=4)

    def test_target_radec_from_header_raises_when_absent(self) -> None:
        with self.assertRaises(ValueError):
            target_radec_from_header({"RA": "10:00:00", "OBJECT": "x"})

    def test_missing_cat_radec_header_raises(self) -> None:
        frames, _ = build_non_sidereal_frame_set(frame_count=3)
        for frame in frames.values():
            del frame["header"]["CAT-RA"]
            del frame["header"]["CAT-DEC"]
        fits_paths = self.write_frames(frames)
        with self.assertRaises(LightCurveError):
            generate_light_curve(
                fits_paths, aperture_radius=4.0, annulus_inner_radius=6.0, annulus_outer_radius=9.0,
                target_position_mode=TARGET_POSITION_HEADER, comparison_mode=COMPARISON_AUTO,
            )

    # ------------------------------------------------------------ target locating

    def test_header_target_is_measured_at_moving_position(self) -> None:
        frames, _ = build_non_sidereal_frame_set(frame_count=6, ra_drift_arcsec_per_frame=8.0)
        fits_paths = self.write_frames(frames)
        result = generate_light_curve(
            fits_paths, aperture_radius=4.0, annulus_inner_radius=6.0, annulus_outer_radius=9.0,
            target_position_mode=TARGET_POSITION_HEADER, comparison_mode=COMPARISON_AUTO,
        )
        self.assertEqual(len(result.light_curve_rows), 6)
        # Target tracked at frame center (60, 60) on every frame regardless of the moving ephemeris.
        for row in result.light_curve_rows:
            self.assertAlmostEqual(row.target_centroid_x, 60.0, delta=1.5)
            self.assertAlmostEqual(row.target_centroid_y, 60.0, delta=1.5)
        self.assertEqual(len(self._finite_rows(result)), 6)

    def test_forced_refinement_measures_at_ephemeris_pixel(self) -> None:
        frames, _ = build_non_sidereal_frame_set(frame_count=4, ra_drift_arcsec_per_frame=8.0)
        fits_paths = self.write_frames(frames)
        result = generate_light_curve(
            fits_paths, aperture_radius=4.0, annulus_inner_radius=6.0, annulus_outer_radius=9.0,
            target_position_mode=TARGET_POSITION_HEADER, comparison_mode=COMPARISON_AUTO,
            refinement_mode=REFINEMENT_FORCED,
        )
        # Forced photometry never recenters: it measures at the ephemeris (frame-center) pixel, to
        # projection precision, rather than chasing a light centroid.
        for row in result.light_curve_rows:
            self.assertAlmostEqual(row.target_centroid_x, 60.0, delta=0.01)
            self.assertAlmostEqual(row.target_centroid_y, 60.0, delta=0.01)

    # ------------------------------------------------------------- evolving path

    def test_shared_fails_but_evolving_carries_turned_over_field(self) -> None:
        # Drift a full field width so no comparison star is present on every frame.
        frames, _ = build_non_sidereal_frame_set(frame_count=12, ra_drift_arcsec_per_frame=12.0)
        fits_paths = self.write_frames(frames)
        common = dict(
            aperture_radius=4.0, annulus_inner_radius=6.0, annulus_outer_radius=9.0,
            target_position_mode=TARGET_POSITION_HEADER,
        )
        with self.assertRaises(LightCurveError):
            generate_light_curve(fits_paths, comparison_mode=COMPARISON_SHARED, **common)

        auto = generate_light_curve(fits_paths, comparison_mode=COMPARISON_AUTO, **common)
        self.assertEqual(len(self._finite_rows(auto)), 12)
        # The fallback is announced in diagnostics.
        self.assertTrue(any("evolving per-frame calibration" in d for d in auto.diagnostics))

        evolving = generate_light_curve(fits_paths, comparison_mode=COMPARISON_EVOLVING, **common)
        self.assertEqual(len(self._finite_rows(evolving)), 12)

    def test_evolving_is_stepless_for_constant_target(self) -> None:
        # Constant target through a fully turning-over, non-variable comparison cast: the recovered
        # light curve must stay flat across membership changes (no step artifacts at swaps).
        frames, _ = build_non_sidereal_frame_set(
            frame_count=12,
            ra_drift_arcsec_per_frame=12.0,
            target_flux_by_frame=[_flux_for_mag(15.0)] * 12,
        )
        fits_paths = self.write_frames(frames)
        result = generate_light_curve(
            fits_paths, aperture_radius=4.0, annulus_inner_radius=6.0, annulus_outer_radius=9.0,
            target_position_mode=TARGET_POSITION_HEADER, comparison_mode=COMPARISON_EVOLVING,
        )
        mags = np.asarray([row.target_calibrated_apparent_magnitude for row in self._finite_rows(result)])
        self.assertEqual(mags.size, 12)
        # A step at a membership swap would show up as scatter well above the mmag photon floor.
        self.assertLess(float(np.std(mags)), 0.02)
        self.assertLess(float(np.ptp(mags)), 0.05)

    def test_evolving_recovers_injected_variation(self) -> None:
        # Sinusoidal target brightness; evolving calibration should track it (period-use sanity).
        amplitude = 0.3
        fluxes = [_flux_for_mag(15.0 + amplitude * math.sin(2 * math.pi * i / 6.0)) for i in range(12)]
        frames, _ = build_non_sidereal_frame_set(
            frame_count=12, ra_drift_arcsec_per_frame=12.0, target_flux_by_frame=fluxes,
        )
        fits_paths = self.write_frames(frames)
        result = generate_light_curve(
            fits_paths, aperture_radius=4.0, annulus_inner_radius=6.0, annulus_outer_radius=9.0,
            target_position_mode=TARGET_POSITION_HEADER, comparison_mode=COMPARISON_EVOLVING,
        )
        rows = self._finite_rows(result)
        self.assertEqual(len(rows), 12)
        recovered = np.asarray([row.target_calibrated_apparent_magnitude for row in rows])
        injected = np.asarray([15.0 + amplitude * math.sin(2 * math.pi * i / 6.0) for i in range(12)])
        # Recovered variation tracks the injected signal (shape, not absolute level).
        self.assertGreater(float(np.corrcoef(recovered - recovered.mean(), injected - injected.mean())[0, 1]), 0.97)

    def test_evolving_flags_connectivity_break(self) -> None:
        # Two blocks of frames whose fields do not overlap -> disjoint comparison groups.
        offsets = [0.0, 10.0, 20.0, 600.0, 610.0, 620.0]
        frames, _ = build_non_sidereal_frame_set(
            frame_count=6,
            field_center_ra_offsets_arcsec=offsets,
            star_ra_offsets_arcsec=tuple(range(-40, 60, 15)) + tuple(range(560, 660, 15)),
        )
        fits_paths = self.write_frames(frames)
        result = generate_light_curve(
            fits_paths, aperture_radius=4.0, annulus_inner_radius=6.0, annulus_outer_radius=9.0,
            target_position_mode=TARGET_POSITION_HEADER, comparison_mode=COMPARISON_EVOLVING,
        )
        self.assertTrue(any("disconnected groups" in d for d in result.diagnostics))
        # Still produces a light curve from each block despite the break.
        self.assertGreaterEqual(len(self._finite_rows(result)), 4)

    def test_empirical_error_floor_recovers_injected_scatter(self) -> None:
        # Comparison stars with a known frame-to-frame reproducibility scatter; the empirical floor
        # (median per-star RMS about own mean) must recover it, independent of static catalog offsets.
        rng = np.random.default_rng(0)
        zp = 25.0
        star_base = {"cand-001": 14.6, "cand-002": 15.1, "cand-003": 15.4, "cand-004": 15.8}
        injected = 0.02
        n_frames = 40

        def cal_for_frame(fits_path: str) -> FrameCalibration:
            stars, meas = [], []
            for cid, base in star_base.items():
                mag = base + rng.normal(0.0, injected)  # per-star, per-frame independent scatter
                counts = 10.0 ** (-0.4 * (mag - zp))
                stars.append(ComparisonStar(
                    candidate_id=cid, ra_deg=0.0, dec_deg=0.0, reference_magnitude=base,
                    reference_magnitude_source="x", source_catalog_by_frame={},
                    variability_score=0.0, isolation_arcsec=10.0, target_separation_px=100.0))
                meas.append(ComparisonMeasurement(
                    candidate_id=cid, fits_path=fits_path, x=0.0, y=0.0, net_source_counts=counts,
                    source_uncertainty=1.0, mean_background_per_pixel=0.0, peak_pixel_value=0.0,
                    effective_source_pixels=0.0, effective_background_pixels=0.0))
            return FrameCalibration(
                stars=tuple(stars), measurements=tuple(meas), ensemble_flux=1.0, ensemble_variance=1.0,
                target_rel_flux=1.0, target_rel_flux_sigma=0.001, frame_zero_point=zp,
                calibrated_mag=15.0, calibrated_mag_sigma=0.003)

        order = [f"f{i:02d}" for i in range(n_frames)]
        floor = _empirical_error_floor({fp: cal_for_frame(fp) for fp in order}, order)
        self.assertAlmostEqual(floor, injected, delta=0.005)

    def test_error_floor_applied_to_uncertainties(self) -> None:
        frames, _ = build_non_sidereal_frame_set(frame_count=8, ra_drift_arcsec_per_frame=8.0)
        fits_paths = self.write_frames(frames)
        result = generate_light_curve(
            fits_paths, aperture_radius=4.0, annulus_inner_radius=6.0, annulus_outer_radius=9.0,
            target_position_mode=TARGET_POSITION_HEADER, comparison_mode=COMPARISON_AUTO,
        )
        for row in result.light_curve_rows:
            if math.isfinite(row.target_calibrated_apparent_magnitude):
                self.assertTrue(math.isfinite(row.target_calibrated_apparent_magnitude_uncertainty))
                self.assertGreater(row.target_calibrated_apparent_magnitude_uncertainty, 0.0)

    def test_moving_target_excluded_from_comparisons(self) -> None:
        frames, _ = build_non_sidereal_frame_set(
            frame_count=6, ra_drift_arcsec_per_frame=8.0, include_target_in_catalog=True,
        )
        fits_paths = self.write_frames(frames)
        result = generate_light_curve(
            fits_paths, aperture_radius=4.0, annulus_inner_radius=6.0, annulus_outer_radius=9.0,
            target_position_mode=TARGET_POSITION_HEADER, comparison_mode=COMPARISON_AUTO,
        )
        for star in result.selected_comparison_stars:
            labels = {
                str(row.get("source_label", ""))
                for row in star.source_catalog_by_frame.values()
            }
            self.assertNotIn("target-shadow", labels)


class TestNonSiderealRealData(unittest.TestCase):
    KLEOPATRA_DIR = (
        Path(__file__).resolve().parent / "test_files" / "aperture_photometry" / "Kleopatra"
    )

    @unittest.skipUnless(KLEOPATRA_DIR.is_dir(), "Kleopatra sample frames not present")
    def test_kleopatra_header_mode_smoke(self) -> None:
        # A few real LCO MINORPLANET frames end to end: header target locating + auto calibration.
        fits_paths = sorted(str(p) for p in self.KLEOPATRA_DIR.glob("*.fits.fz"))[:4]
        self.assertGreaterEqual(len(fits_paths), 3)
        result = generate_light_curve(
            fits_paths, aperture_radius=5.0, annulus_inner_radius=8.0, annulus_outer_radius=12.0,
            min_comparisons=5, max_comparisons=15,
            target_position_mode=TARGET_POSITION_HEADER, comparison_mode=COMPARISON_AUTO,
        )
        finite = [r for r in result.light_curve_rows if math.isfinite(r.target_calibrated_apparent_magnitude)]
        self.assertEqual(len(finite), len(fits_paths))
        # The asteroid sits near the 2400x2400 frame center on every frame (non-sidereal tracking).
        for row in finite:
            self.assertAlmostEqual(row.target_centroid_x, 1200.0, delta=60.0)
            self.assertAlmostEqual(row.target_centroid_y, 1200.0, delta=60.0)
        # The empirical error floor is applied: per-point uncertainties reflect the real ~10 mmag
        # comparison-star reproducibility, not the ~3 mmag formal photon+ZP error.
        self.assertTrue(any("error floor" in d for d in result.diagnostics))
        median_err = float(np.median([r.target_calibrated_apparent_magnitude_uncertainty for r in finite]))
        self.assertGreater(median_err, 0.006)


if __name__ == "__main__":
    unittest.main()
