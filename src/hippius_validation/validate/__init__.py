"""validate — Hippius repo file-manifest check + universal (spec-driven) architecture check."""

from hippius_validation.validate.architecture import check as check_architecture
from hippius_validation.validate.repo import check as check_repo

__all__ = ["check_repo", "check_architecture"]
