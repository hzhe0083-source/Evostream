import copy
import json
import tempfile
import unittest
from pathlib import Path

from fabri_cache_acceptance.check import (
    is_strict_bool,
    is_strict_number,
    main,
    validate_report,
    validate_report_file,
)


def get_valid_report_dict():
    return {
        "official_single_frame": {
            "passed": True,
            "max_abs_error": 1.2e-5,
        },
        "cached_full": {
            "passed": True,
            "max_abs_error": 0.0,
        },
        "window_rebuild": {
            "passed": True,
            "max_abs_error": 3.4e-6,
        },
        "reset": {
            "passed": True,
        },
        "vision_reuse": {
            "passed": True,
        },
        "strict_weight_load": {
            "passed": True,
        },
        "training": {
            "loss": 0.1234,
            "backbone_has_grad": False,
            "action_head_updated": True,
        },
        "padded_actions_zero": True,
        "extra_info": {
            "device": "cuda:0",
            "runtime_seconds": 42.5,
        },
    }


class TestAcceptanceReportChecker(unittest.TestCase):
    def test_type_guards(self):
        self.assertTrue(is_strict_bool(True))
        self.assertTrue(is_strict_bool(False))
        self.assertFalse(is_strict_bool(1))
        self.assertFalse(is_strict_bool(0))
        self.assertFalse(is_strict_bool("True"))

        self.assertTrue(is_strict_number(0))
        self.assertTrue(is_strict_number(1.5))
        self.assertTrue(is_strict_number(0.0))
        self.assertFalse(is_strict_number(True))
        self.assertFalse(is_strict_number(False))
        self.assertFalse(is_strict_number("1.5"))
        self.assertFalse(is_strict_number(None))

    def test_valid_report(self):
        report = get_valid_report_dict()
        errors = validate_report(report)
        self.assertEqual(errors, [], f"Expected no errors, got: {errors}")

    def test_missing_fields(self):
        # Missing root field
        report = get_valid_report_dict()
        del report["padded_actions_zero"]
        errors = validate_report(report)
        self.assertTrue(any("padded_actions_zero" in e for e in errors))

        # Missing sub-field
        report = get_valid_report_dict()
        del report["training"]["loss"]
        errors = validate_report(report)
        self.assertTrue(any("training.loss" in e for e in errors))

        # Missing section
        report = get_valid_report_dict()
        del report["official_single_frame"]
        errors = validate_report(report)
        self.assertTrue(any("official_single_frame" in e for e in errors))

    def test_passed_must_be_true(self):
        sections = [
            "official_single_frame",
            "cached_full",
            "window_rebuild",
            "reset",
            "vision_reuse",
            "strict_weight_load",
        ]
        for sec in sections:
            with self.subTest(section=sec):
                report = get_valid_report_dict()
                report[sec]["passed"] = False
                errors = validate_report(report)
                self.assertTrue(any(f"{sec}.passed" in e and "must be True" in e for e in errors))

    def test_nan_and_infinity_in_numbers(self):
        report = get_valid_report_dict()
        report["training"]["loss"] = float("nan")
        errors = validate_report(report)
        self.assertTrue(any("training.loss" in e and "finite" in e for e in errors))

        report = get_valid_report_dict()
        report["cached_full"]["max_abs_error"] = float("inf")
        errors = validate_report(report)
        self.assertTrue(any("cached_full.max_abs_error" in e and "finite" in e for e in errors))

        report = get_valid_report_dict()
        report["cached_full"]["max_abs_error"] = float("-inf")
        errors = validate_report(report)
        self.assertTrue(any("cached_full.max_abs_error" in e and "finite" in e for e in errors))

    def test_negative_numbers(self):
        report = get_valid_report_dict()
        report["training"]["loss"] = -0.001
        errors = validate_report(report)
        self.assertTrue(any("training.loss" in e and "non-negative" in e for e in errors))

        report = get_valid_report_dict()
        report["window_rebuild"]["max_abs_error"] = -1e-5
        errors = validate_report(report)
        self.assertTrue(any("window_rebuild.max_abs_error" in e and "non-negative" in e for e in errors))

    def test_boolean_posing_as_number(self):
        report = get_valid_report_dict()
        report["training"]["loss"] = True
        errors = validate_report(report)
        self.assertTrue(any("training.loss" in e and "finite number" in e for e in errors))

        report = get_valid_report_dict()
        report["official_single_frame"]["max_abs_error"] = False
        errors = validate_report(report)
        self.assertTrue(any("official_single_frame.max_abs_error" in e and "finite number" in e for e in errors))

    def test_number_or_string_posing_as_boolean(self):
        report = get_valid_report_dict()
        report["padded_actions_zero"] = 1
        errors = validate_report(report)
        self.assertTrue(any("padded_actions_zero" in e and "boolean" in e for e in errors))

        report = get_valid_report_dict()
        report["padded_actions_zero"] = "True"
        errors = validate_report(report)
        self.assertTrue(any("padded_actions_zero" in e and "boolean" in e for e in errors))

    def test_gradient_contract(self):
        # backbone_has_grad must be False
        report = get_valid_report_dict()
        report["training"]["backbone_has_grad"] = True
        errors = validate_report(report)
        self.assertTrue(any("training.backbone_has_grad" in e and "must be False" in e for e in errors))

    def test_action_head_not_updated(self):
        # action_head_updated must be True
        report = get_valid_report_dict()
        report["training"]["action_head_updated"] = False
        errors = validate_report(report)
        self.assertTrue(any("training.action_head_updated" in e and "must be True" in e for e in errors))

    def test_padded_actions_zero_false(self):
        report = get_valid_report_dict()
        report["padded_actions_zero"] = False
        errors = validate_report(report)
        self.assertTrue(any("padded_actions_zero" in e and "must be True" in e for e in errors))

    def test_cli_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            valid_file = Path(tmpdir) / "valid.json"
            valid_file.write_text(json.dumps(get_valid_report_dict()), encoding="utf-8")
            self.assertEqual(main([str(valid_file)]), 0)

            invalid_data = get_valid_report_dict()
            invalid_data["training"]["backbone_has_grad"] = True
            invalid_file = Path(tmpdir) / "invalid.json"
            invalid_file.write_text(json.dumps(invalid_data), encoding="utf-8")
            self.assertEqual(main([str(invalid_file)]), 2)

            missing_file = Path(tmpdir) / "non_existent.json"
            self.assertEqual(main([str(missing_file)]), 2)

            empty_call = main([])
            self.assertEqual(empty_call, 2)


if __name__ == "__main__":
    unittest.main()
