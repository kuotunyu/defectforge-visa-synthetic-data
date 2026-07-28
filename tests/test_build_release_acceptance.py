from __future__ import annotations

from scripts.build_release_acceptance import release_blockers, render_acceptance


def test_release_blockers_ignore_only_self_referential_acceptance_checks() -> None:
    audit = {
        "checks": {
            "required_paths": False,
            "release_acceptance_complete": False,
            "readme_final": True,
            "demo_gif_valid": True,
        },
        "required": {"missing": ["reports/release_acceptance.md"]},
    }
    assert release_blockers(audit) == []

    audit["checks"]["demo_gif_valid"] = False
    audit["required"]["missing"].append("assets/demo.gif")
    assert release_blockers(audit) == [
        "check failed: demo_gif_valid",
        "required path missing: assets/demo.gif",
    ]


def test_render_acceptance_has_required_sections_and_no_unchecked_items() -> None:
    audit = {
        "checks": {
            "required_paths": False,
            "release_acceptance_complete": False,
            "readme_final": True,
            "demo_gif_valid": True,
        }
    }

    report = render_acceptance(
        audit,
        classification_sha256="a" * 64,
        segmentation_sha256="b" * 64,
    )

    assert "## Passed" in report
    assert "## Fixed" in report
    assert "## Residual risks" in report
    assert "- [ ]" not in report
    assert "explicit user approval" in report
