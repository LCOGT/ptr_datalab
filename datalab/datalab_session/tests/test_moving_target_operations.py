"""
Operation-layer checks for the two moving-target operations, needing no Redis, S3 or pixels.

The mode suites (test_moving_target_track_mode, test_moving_target_header_mode) call
generate_light_curve directly, so nothing there exercises the operations' own wiring, which is
where a stale result-field name or a broken wizard contract sits unnoticed until a real run.
"""
import re
import unittest


class TestMovingTargetOperations(unittest.TestCase):
    def test_both_operations_are_registered_under_their_names(self) -> None:
        from datalab.datalab_session.data_operations.utils import available_operations
        operations = available_operations()
        self.assertIn("Moving Target Aperture Photometry", operations)
        self.assertIn("Non-Sidereal Aperture Photometry", operations)
        # An intermediate base class would register itself with no name; nothing should be unnamed.
        self.assertNotIn(None, operations)

    def test_wizard_descriptions_declare_the_expected_inputs(self) -> None:
        from datalab.datalab_session.data_operations.aperture_photometry import (
            MovingTargetAperturePhotometry,
        )
        from datalab.datalab_session.data_operations.aperture_photometry import (
            NonSiderealAperturePhotometry,
        )
        shared = {"input_files", "aperture_radius", "annulus_inner_radius", "annulus_outer_radius"}

        tracked = MovingTargetAperturePhotometry.wizard_description()["inputs"]
        self.assertTrue(shared <= set(tracked))
        self.assertIn("target_track", tracked)
        self.assertIn("track_search_radius", tracked)

        header_mode = NonSiderealAperturePhotometry.wizard_description()["inputs"]
        self.assertTrue(shared <= set(header_mode))
        # The ephemeris-header operation takes no source and no track: that is the whole point of it.
        self.assertNotIn("target_track", header_mode)
        self.assertNotIn("source", header_mode)

    def test_operations_only_read_light_curve_result_fields_that_exist(self) -> None:
        """Guards the failure the rebase introduced: an operation naming a renamed result field."""
        import dataclasses
        import inspect
        from datalab.datalab_session.data_operations import aperture_photometry
        from datalab.datalab_session.utils.aperture_light_curve import LightCurveResult

        available = {f.name for f in dataclasses.fields(LightCurveResult)}
        source = inspect.getsource(aperture_photometry)
        referenced = set(re.findall(r"\bresult\.([a-zA-Z_][a-zA-Z0-9_]*)", source))
        self.assertTrue(referenced, "expected the shared runner to read fields off the result")
        self.assertEqual(referenced - available, set())

    def test_operation_output_carries_both_diagnostic_scopes(self) -> None:
        """
            Both scopes must reach the frontend. Emitting only the per-frame dict is what made a run
            whose catalog search failed entirely indistinguishable from a clean one.
        """
        from types import SimpleNamespace
        from unittest import mock
        from datalab.datalab_session.data_operations.aperture_photometry import (
            NonSiderealAperturePhotometry,
        )

        operation = NonSiderealAperturePhotometry({
            "input_files": [{"basename": "frame_1", "source": "local", "filter": "rp"}],
            "aperture_radius": 5.0,
            "annulus_inner_radius": 8.0,
            "annulus_outer_radius": 12.0,
        })
        with mock.patch(
            "datalab.datalab_session.data_operations.aperture_photometry.generate_light_curve"
        ) as mock_generate, mock.patch(
            "datalab.datalab_session.data_operations.aperture_photometry.FileCache"
        ) as mock_file_cache, mock.patch(
            "datalab.datalab_session.data_operations.aperture_photometry.save_diagnostic_images_to_s3",
            return_value={},
        ), mock.patch.object(
            NonSiderealAperturePhotometry, "set_output"
        ) as mock_set_output, mock.patch.object(
            NonSiderealAperturePhotometry, "set_operation_progress"
        ), mock.patch.object(NonSiderealAperturePhotometry, "set_status"):
            mock_file_cache.return_value.get_fits.return_value = "/tmp/frame_1.fits"
            mock_generate.return_value = SimpleNamespace(
                light_curve_rows=[],
                selected_comparison_stars=[],
                diagnostics=["no ensemble spans every frame", "frame_1.fits: 2 usable stars"],
                pipeline_diagnostics=["no ensemble spans every frame"],
                diagnostics_by_fits_basename={"frame_1.fits": ["frame_1.fits: 2 usable stars"]},
                diagnostic_image_jpegs_by_fits_basename={},
            )
            operation.operate(submitter=None)

        output = mock_set_output.call_args.args[0]["output_data"][0]
        self.assertEqual(output["pipeline_diagnostics"], ["no ensemble spans every frame"])
        self.assertEqual(output["diagnostics"], {"frame_1.fits": ["frame_1.fits: 2 usable stars"]})

    def test_track_operation_rejects_missing_samples(self) -> None:
        from datalab.datalab_session.data_operations.aperture_photometry import (
            MovingTargetAperturePhotometry,
        )
        from datalab.datalab_session.exceptions import ClientAlertException
        operation = MovingTargetAperturePhotometry({"input_files": [], "aperture_radius": 5.0})
        with self.assertRaises(ClientAlertException):
            operation.operate(submitter=None)

    def test_track_operation_rejects_malformed_samples(self) -> None:
        from datalab.datalab_session.data_operations.aperture_photometry import (
            MovingTargetAperturePhotometry,
        )
        from datalab.datalab_session.exceptions import ClientAlertException
        operation = MovingTargetAperturePhotometry(
            {"input_files": [], "aperture_radius": 5.0, "target_track": [{"mjd": 1.0, "ra": 2.0}]}
        )
        with self.assertRaises(ClientAlertException):
            operation.operate(submitter=None)


if __name__ == "__main__":
    unittest.main()
