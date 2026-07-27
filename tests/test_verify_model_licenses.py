from __future__ import annotations

import pytest

from scripts.verify_model_licenses import (
    ModelLicenseVerificationError,
    validate_model_metadata,
)

EXPECTED = {
    "revision": "a" * 40,
    "license": "openrail++",
    "role": "base",
}


def _observed() -> dict:
    return {
        "revision": "a" * 40,
        "license": "openrail++",
        "private": False,
        "gated": False,
        "disabled": False,
    }


def test_validate_model_metadata_accepts_public_exact_revision() -> None:
    result = validate_model_metadata("owner/model", observed=_observed(), expected=EXPECTED)
    assert result["license"] == "openrail++"
    assert result["revision"] == "a" * 40


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("private", True, "private"),
        ("gated", "auto", "gated"),
        ("disabled", True, "disabled"),
        ("revision", "b" * 40, "revision"),
        ("license", "mit", "license"),
    ],
)
def test_validate_model_metadata_fails_closed(
    field: str,
    value: object,
    message: str,
) -> None:
    observed = _observed()
    observed[field] = value
    with pytest.raises(ModelLicenseVerificationError, match=message):
        validate_model_metadata("owner/model", observed=observed, expected=EXPECTED)
