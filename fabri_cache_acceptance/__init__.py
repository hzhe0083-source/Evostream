"""fabri_cache_acceptance - Standalone acceptance report checker for fabri_cache.

This package provides a strict contract validator for JSON reports generated
by cache and training runs.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .check import validate_report, validate_report_file

__all__ = ["validate_report", "validate_report_file"]


def __getattr__(name: str):
    if name in ("validate_report", "validate_report_file"):
        from . import check
        return getattr(check, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
