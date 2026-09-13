#!/usr/bin/env python3
"""Acceptance report contract checker for fabri_cache.

Validates the JSON report contract produced by cache and training verification runs.
Note: Validating this report confirms schema and constraint compliance, but cannot
independently prove that execution took place on actual hardware.
"""

import json
import math
import sys
from typing import Any, Dict, List


def is_strict_bool(val: Any) -> bool:
    """Return True if val is strictly a boolean."""
    return isinstance(val, bool)


def is_strict_number(val: Any) -> bool:
    """Return True if val is int or float, but NOT bool."""
    return isinstance(val, (int, float)) and not isinstance(val, bool)


def _check_bool(container: Dict[str, Any], key: str, path: str, expected_val: bool, errors: List[str]) -> None:
    if key not in container:
        errors.append(f"Missing required field: '{path}'")
        return
    val = container[key]
    if not is_strict_bool(val):
        errors.append(
            f"Field '{path}' must be a boolean (expected {expected_val}), got {type(val).__name__} ({val!r})"
        )
    elif val is not expected_val:
        errors.append(
            f"Field '{path}' must be {expected_val}, got {val}"
        )


def _check_non_negative_number(container: Dict[str, Any], key: str, path: str, errors: List[str]) -> None:
    if key not in container:
        errors.append(f"Missing required field: '{path}'")
        return
    val = container[key]
    if not is_strict_number(val):
        errors.append(
            f"Field '{path}' must be a finite number, got {type(val).__name__} ({val!r})"
        )
        return
    if not math.isfinite(val):
        errors.append(
            f"Field '{path}' must be a finite number, got {val}"
        )
        return
    if val < 0:
        errors.append(
            f"Field '{path}' must be non-negative (>= 0), got {val}"
        )


def _check_dict(container: Dict[str, Any], key: str, path: str, errors: List[str]) -> bool:
    if key not in container:
        errors.append(f"Missing required field: '{path}'")
        return False
    val = container[key]
    if not isinstance(val, dict):
        errors.append(
            f"Field '{path}' must be an object (dict), got {type(val).__name__} ({val!r})"
        )
        return False
    return True


def validate_report(data: Any) -> List[str]:
    """Validate report dictionary against the fabri_cache acceptance contract.

    Returns a list of error descriptions. An empty list means validation succeeded.
    """
    errors: List[str] = []

    if not isinstance(data, dict):
        return [f"Root of report must be a JSON object (dict), got {type(data).__name__}"]

    # 1. official_single_frame: {passed: bool, max_abs_error: number}
    if _check_dict(data, "official_single_frame", "official_single_frame", errors):
        sub = data["official_single_frame"]
        _check_bool(sub, "passed", "official_single_frame.passed", True, errors)
        _check_non_negative_number(sub, "max_abs_error", "official_single_frame.max_abs_error", errors)

    # 2. cached_full: {passed: bool, max_abs_error: number}
    if _check_dict(data, "cached_full", "cached_full", errors):
        sub = data["cached_full"]
        _check_bool(sub, "passed", "cached_full.passed", True, errors)
        _check_non_negative_number(sub, "max_abs_error", "cached_full.max_abs_error", errors)

    # 3. window_rebuild: {passed: bool, max_abs_error: number}
    if _check_dict(data, "window_rebuild", "window_rebuild", errors):
        sub = data["window_rebuild"]
        _check_bool(sub, "passed", "window_rebuild.passed", True, errors)
        _check_non_negative_number(sub, "max_abs_error", "window_rebuild.max_abs_error", errors)

    # 4. reset: {passed: bool}
    if _check_dict(data, "reset", "reset", errors):
        sub = data["reset"]
        _check_bool(sub, "passed", "reset.passed", True, errors)

    # 5. vision_reuse: {passed: bool}
    if _check_dict(data, "vision_reuse", "vision_reuse", errors):
        sub = data["vision_reuse"]
        _check_bool(sub, "passed", "vision_reuse.passed", True, errors)

    # 6. strict_weight_load: {passed: bool}
    if _check_dict(data, "strict_weight_load", "strict_weight_load", errors):
        sub = data["strict_weight_load"]
        _check_bool(sub, "passed", "strict_weight_load.passed", True, errors)

    # 7. training: {loss: number, backbone_has_grad: bool, action_head_updated: bool}
    if _check_dict(data, "training", "training", errors):
        sub = data["training"]
        _check_non_negative_number(sub, "loss", "training.loss", errors)
        _check_bool(sub, "backbone_has_grad", "training.backbone_has_grad", False, errors)
        _check_bool(sub, "action_head_updated", "training.action_head_updated", True, errors)

    # 8. padded_actions_zero: bool
    _check_bool(data, "padded_actions_zero", "padded_actions_zero", True, errors)

    # Note: extra fields are explicitly allowed and ignored.
    return errors


def validate_report_file(file_path: str) -> List[str]:
    """Load and validate a JSON report file.

    Returns a list of error descriptions.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return [f"File not found: {file_path}"]
    except PermissionError:
        return [f"Permission denied: {file_path}"]
    except json.JSONDecodeError as exc:
        return [f"Invalid JSON in {file_path}: {exc}"]
    except Exception as exc:
        return [f"Failed to read {file_path}: {exc}"]

    return validate_report(data)


def main(argv: List[str] = None) -> int:
    """CLI entry point."""
    if argv is None:
        argv = sys.argv[1:]

    if not argv or argv[0] in ("-h", "--help"):
        print("Usage: python -m fabri_cache_acceptance.check REPORT.json", file=sys.stderr)
        return 2 if not argv else 0

    report_path = argv[0]
    errors = validate_report_file(report_path)

    if errors:
        print(f"ACCEPTANCE REPORT FAILED ({len(errors)} error(s)):", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 2

    print(f"ACCEPTANCE REPORT PASSED: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
