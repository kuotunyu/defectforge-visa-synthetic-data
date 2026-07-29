"""Run the fail-closed public Repository audit without publishing anything."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

EXPECTED_NAME = "kuotunyu"
EXPECTED_EMAIL = "61350295+kuotunyu@users.noreply.github.com"
MAX_TRACKED_BYTES = 10 * 1024 * 1024
MODEL_LICENSE_MAX_AGE = timedelta(hours=24)
FINAL_FIGURES = (
    "reports/figures/real_scaling_curve.png",
    "reports/figures/synthetic_volume_curve.png",
    "reports/figures/main_comparison_table.png",
    "reports/figures/segmentation_table.png",
    "reports/figures/quality_vs_downstream.png",
    "reports/figures/sample_grid_pcb1.png",
    "reports/figures/sample_grid_capsules.png",
)
FINAL_EVIDENCE_REPORTS = (
    "reports/classifier_matrix_validation.json",
    "reports/segmentation_validation.json",
    "reports/phase2_figures_validation.json",
    "reports/sample_grids_validation.json",
    "reports/readme_validation.json",
    "reports/license_chain_validation.json",
    "reports/model_license_verification.json",
    "reports/hf_package_validation.json",
    "reports/hf_upload_plan.json",
    "reports/demo_checkpoint_selection.json",
    "reports/demo_validation.json",
    "reports/phase2_visual_review.json",
)
REQUIRED_PATHS = (
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "THIRD_PARTY_NOTICES.md",
    "SECURITY.md",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "uv.lock",
    "docs",
    "notebooks",
    "splits/split_manifest.json",
    "splits/MANIFEST.sha256",
    "splits/defect_types.json",
    "splits/test_blocklist.json",
    "results/classification.csv",
    "results/segmentation.csv",
    "reports/segmentation_results.md",
    "reports/segmentation_validation.json",
    "reports/phase2_figures_validation.json",
    "reports/readme_validation.json",
    "reports/model_license_verification.json",
    "reports/hf_package_validation.json",
    "reports/hf_upload_plan.json",
    "reports/demo_checkpoint_selection.json",
    "reports/demo_validation.json",
    "reports/phase2_visual_review.json",
    "reports/release_acceptance.md",
    "scripts/verify_readme.py",
    "scripts/verify_license_chain.py",
    "reports/license_chain_validation.json",
    "hf_cards/dataset/README.md",
    "hf_cards/model/README.md",
    "assets/demo.gif",
    "assets/github-social-preview.png",
    ".claude/skills/defectforge/SKILL.md",
    ".claude/skills/df-guard/SKILL.md",
    *FINAL_FIGURES,
)
# Two agent skills are published on purpose (ADR-028): the orchestrator and the
# anti-leakage guard. They are carved out of the blanket ".claude/" owner-local rule
# below, and REQUIRED_PATHS above keeps them from silently disappearing again.
PUBLIC_SKILL_PREFIXES = (
    ".claude/skills/defectforge/",
    ".claude/skills/df-guard/",
)
OWNER_LOCAL_PATHS = (
    ".claude/",
    ".github/",
    "CLAUDE.md",
    "PLAN.md",
    "PRODUCT.md",
    "instructions_for_me.md",
    "docs/autonomy_policy.md",
    "docs/publish_spec.md",
    "docs/skills_roadmap.md",
    "docs/worklog.md",
    "reports/handoff/",
    "assets/github-social-preview.philosophy.md",
    "scripts/verify_phase1.py",
    "tests/test_verify_phase1.py",
    "tests/test_ci_workflow.py",
)
SECRET_PATTERNS = {
    "github_token": re.compile(r"gh[oprsu]_[A-Za-z0-9]{20,}"),
    "huggingface_token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
    "openai_key": re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    "google_api_key": re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
}
PERSONAL_WINDOWS_PATH = re.compile(r"(?i)[A-Z]:[\\/]+Users[\\/]+[^<%\\/][^\\/]*[\\/]")
COAUTHOR_TRAILER = re.compile(r"(?im)^Co-Authored-By:\s")


class PublishVerificationError(RuntimeError):
    """Raised when the M24 publication gate cannot be evaluated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PublishVerificationError(message)


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    require(completed.returncode == 0, completed.stderr.strip() or f"git {' '.join(args)} failed")
    return completed.stdout


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tracked_paths(repo: Path) -> list[Path]:
    output = _git(repo, "ls-files", "-z")
    return [repo / value for value in output.split("\0") if value]


def scan_text(path: Path, text: str) -> list[dict[str, Any]]:
    """Return location-only findings so a report never repeats a credential."""

    findings: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(line):
                findings.append({"kind": label, "path": path.as_posix(), "line": line_number})
        if PERSONAL_WINDOWS_PATH.search(line):
            findings.append(
                {"kind": "personal_windows_path", "path": path.as_posix(), "line": line_number}
            )
    return findings


def scan_tracked_tree(repo: Path, paths: Sequence[Path]) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    tracked_env: list[str] = []
    oversized: list[dict[str, Any]] = []
    for path in paths:
        relative = path.relative_to(repo).as_posix()
        if path.name == ".env" or path.name.startswith(".env."):
            tracked_env.append(relative)
        if not path.is_file():
            findings.append({"kind": "missing_tracked_file", "path": relative})
            continue
        size = path.stat().st_size
        if size > MAX_TRACKED_BYTES:
            oversized.append({"path": relative, "bytes": size})
        raw = path.read_bytes()
        if b"\0" in raw:
            continue
        text = raw.decode("utf-8", errors="replace")
        findings.extend(scan_text(Path(relative), text))
    return {
        "secret_or_path_findings": findings,
        "tracked_env": tracked_env,
        "oversized": oversized,
    }


def audit_identities(repo: Path) -> dict[str, Any]:
    rows = [
        tuple(line.split("\t"))
        for line in _git(repo, "log", "--all", "--format=%an\t%ae\t%cn\t%ce").splitlines()
        if line
    ]
    require(bool(rows), "Git history is empty")
    unique = sorted(set(rows))
    expected = (EXPECTED_NAME, EXPECTED_EMAIL, EXPECTED_NAME, EXPECTED_EMAIL)
    invalid = [row for row in unique if row != expected]
    bodies = _git(repo, "log", "--all", "--format=%B%x00")
    trailer_matches = len(COAUTHOR_TRAILER.findall(bodies))
    return {
        "commit_count": len(rows),
        "unique_author_committer_rows": [list(row) for row in unique],
        "invalid_rows": [list(row) for row in invalid],
        "coauthor_trailer_count": trailer_matches,
    }


def audit_required_paths(repo: Path) -> dict[str, Any]:
    missing = [value for value in REQUIRED_PATHS if not (repo / value).exists()]
    tracked = {
        value
        for value in _git(repo, "ls-files", "-z").split("\0")
        if value
    }

    def is_tracked(value: str) -> bool:
        path = repo / value
        if path.is_dir():
            prefix = value.rstrip("/") + "/"
            return any(item.startswith(prefix) for item in tracked)
        return value in tracked

    untracked_required = [
        value for value in REQUIRED_PATHS if (repo / value).exists() and not is_tracked(value)
    ]
    owner_local_tracked = sorted(
        item
        for item in tracked
        if not any(item.startswith(prefix) for prefix in PUBLIC_SKILL_PREFIXES)
        and any(
            item.startswith(value) if value.endswith("/") else item == value
            for value in OWNER_LOCAL_PATHS
        )
    )
    return {
        "missing": missing,
        "untracked_required": untracked_required,
        "owner_local_tracked": owner_local_tracked,
    }


def audit_plan_and_readme(repo: Path) -> dict[str, Any]:
    readme_path = repo / "README.md"
    readme = readme_path.read_text(encoding="utf-8") if readme_path.is_file() else ""
    required_sections = (
        "## 研究問題",
        "## 方法與系統架構",
        "## 實驗設計",
        "## 實驗結果",
        "## 限制與誠實揭露",
        "## 重現方式",
        "## 授權與引用",
    )
    return {
        "readme_has_required_sections": all(section in readme for section in required_sections),
        "readme_has_no_tbd_or_todo": "TBD" not in readme and "TODO" not in readme,
        "readme_status_is_final": "pending" not in readme[:500].lower(),
    }


def _load_json_object(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return None, f"invalid JSON: {error}"
    if not isinstance(payload, dict):
        return None, "top-level value is not an object"
    return payload, None


def _record_hash_binding(
    repo: Path,
    *,
    report: str,
    payload: Mapping[str, Any],
    key: str,
    target: str,
    mismatches: list[dict[str, Any]],
) -> None:
    target_path = repo / target
    expected = payload.get(key)
    observed = sha256_file(target_path) if target_path.is_file() else None
    if expected != observed:
        mismatches.append(
            {
                "report": report,
                "field": key,
                "target": target,
                "expected": expected,
                "observed": observed,
            }
        )


def audit_evidence(repo: Path) -> dict[str, Any]:
    missing: list[str] = []
    invalid: list[dict[str, str]] = []
    payloads: dict[str, dict[str, Any]] = {}
    for relative in FINAL_EVIDENCE_REPORTS:
        payload, error = _load_json_object(repo / relative)
        if error == "missing":
            missing.append(relative)
            continue
        if error is not None or payload is None:
            invalid.append({"path": relative, "reason": error or "invalid"})
            continue
        if payload.get("status") != "passed":
            invalid.append({"path": relative, "reason": "status is not passed"})
            continue
        payloads[relative] = payload

    hash_mismatches: list[dict[str, Any]] = []
    bindings = {
        "reports/segmentation_validation.json": (
            ("segmentation_csv_sha256", "results/segmentation.csv"),
            ("report_sha256", "reports/segmentation_results.md"),
        ),
        "reports/readme_validation.json": (
            ("readme_sha256", "README.md"),
            ("classification_sha256", "results/classification.csv"),
            ("segmentation_sha256", "results/segmentation.csv"),
        ),
        "reports/phase2_figures_validation.json": (
            ("classification_sha256", "results/classification.csv"),
            ("segmentation_sha256", "results/segmentation.csv"),
        ),
        "reports/demo_checkpoint_selection.json": (
            ("classification_results_sha256", "results/classification.csv"),
            ("segmentation_results_sha256", "results/segmentation.csv"),
        ),
        "reports/demo_validation.json": (
            ("classification_sha256", "results/classification.csv"),
            ("segmentation_sha256", "results/segmentation.csv"),
            ("demo_gif_sha256", "assets/demo.gif"),
        ),
    }
    for report, report_bindings in bindings.items():
        payload = payloads.get(report)
        if payload is None:
            continue
        for key, target in report_bindings:
            _record_hash_binding(
                repo,
                report=report,
                payload=payload,
                key=key,
                target=target,
                mismatches=hash_mismatches,
            )

    license_payload = payloads.get("reports/license_chain_validation.json")
    if license_payload is not None:
        for key, target in (
            ("source_sha256", "docs/license_chain.md"),
            (
                "upstream_verification_sha256",
                "reports/model_license_verification.json",
            ),
        ):
            _record_hash_binding(
                repo,
                report="reports/license_chain_validation.json",
                payload=license_payload,
                key=key,
                target=target,
                mismatches=hash_mismatches,
            )
        documents = license_payload.get("document_sha256")
        if not isinstance(documents, dict):
            invalid.append(
                {
                    "path": "reports/license_chain_validation.json",
                    "reason": "document_sha256 is missing",
                }
            )
        else:
            for target, expected in sorted(documents.items()):
                target_path = repo / target
                observed = sha256_file(target_path) if target_path.is_file() else None
                if expected != observed:
                    hash_mismatches.append(
                        {
                            "report": "reports/license_chain_validation.json",
                            "field": f"document_sha256.{target}",
                            "target": target,
                            "expected": expected,
                            "observed": observed,
                        }
                    )

    figure_payload = payloads.get("reports/phase2_figures_validation.json")
    if figure_payload is not None:
        figures = figure_payload.get("figures")
        if not isinstance(figures, dict):
            invalid.append(
                {
                    "path": "reports/phase2_figures_validation.json",
                    "reason": "figures mapping is missing",
                }
            )
        else:
            for details in figures.values():
                if not isinstance(details, dict):
                    continue
                target = details.get("path")
                expected = details.get("sha256")
                if not isinstance(target, str):
                    continue
                target_path = repo / target
                observed = sha256_file(target_path) if target_path.is_file() else None
                if expected != observed:
                    hash_mismatches.append(
                        {
                            "report": "reports/phase2_figures_validation.json",
                            "field": "figures.*.sha256",
                            "target": target,
                            "expected": expected,
                            "observed": observed,
                        }
                    )

    license_fresh = False
    model_license = payloads.get("reports/model_license_verification.json")
    if model_license is not None:
        try:
            verified_at = datetime.fromisoformat(str(model_license["verified_at"]))
            if verified_at.tzinfo is None:
                verified_at = verified_at.replace(tzinfo=UTC)
            age = datetime.now(UTC) - verified_at.astimezone(UTC)
            license_fresh = timedelta(0) <= age <= MODEL_LICENSE_MAX_AGE
        except (KeyError, TypeError, ValueError):
            invalid.append(
                {
                    "path": "reports/model_license_verification.json",
                    "reason": "verified_at is invalid",
                }
            )

    upload = payloads.get("reports/hf_upload_plan.json")
    upload_safe = bool(
        upload is not None
        and upload.get("mode") == "dry_run"
        and upload.get("changes_visibility") is False
        and upload.get("creates_or_updates_private_repositories") is False
    )
    policy_failures: list[dict[str, str]] = []

    def policy(relative: str, condition: bool, reason: str) -> None:
        if relative in payloads and not condition:
            policy_failures.append({"path": relative, "reason": reason})

    classifier = payloads.get("reports/classifier_matrix_validation.json", {})
    policy(
        "reports/classifier_matrix_validation.json",
        classifier.get("formal_runs") == 38 and classifier.get("blocklist_hits") == 0,
        "expected 38 formal runs and zero blocklist hits",
    )
    segmentation = payloads.get("reports/segmentation_validation.json", {})
    policy(
        "reports/segmentation_validation.json",
        segmentation.get("physical_runs") == 16
        and segmentation.get("logical_rows") == 18
        and segmentation.get("source") == "raw_training_report_json",
        "expected 16 physical runs, 18 logical rows, and raw-report aggregation",
    )
    phase2_figures = payloads.get("reports/phase2_figures_validation.json", {})
    policy(
        "reports/phase2_figures_validation.json",
        phase2_figures.get("visual_inspection_required") is True,
        "visual inspection requirement is missing",
    )
    sample_grids = payloads.get("reports/sample_grids_validation.json", {})
    policy(
        "reports/sample_grids_validation.json",
        sample_grids.get("visual_inspection_required") is True,
        "sample-grid visual inspection requirement is missing",
    )
    readme_validation = payloads.get("reports/readme_validation.json", {})
    policy(
        "reports/readme_validation.json",
        readme_validation.get("negative_results_preserved") is True,
        "negative-result preservation is not proven",
    )
    hf_package = payloads.get("reports/hf_package_validation.json", {})
    hf_dataset = hf_package.get("dataset")
    policy(
        "reports/hf_package_validation.json",
        isinstance(hf_dataset, dict) and hf_dataset.get("test_blocklist_hits") == 0,
        "HF dataset bundle does not prove zero blocklist hits",
    )
    selection = payloads.get("reports/demo_checkpoint_selection.json", {})
    policy(
        "reports/demo_checkpoint_selection.json",
        selection.get("selection_is_post_evaluation_demo_only") is True
        and selection.get("changes_reported_metrics") is False,
        "demo checkpoint selection could alter reported metrics",
    )
    demo_validation = payloads.get("reports/demo_validation.json", {})
    validated_outputs = demo_validation.get("validated_outputs")
    policy(
        "reports/demo_validation.json",
        demo_validation.get("share_enabled") is False
        and demo_validation.get("uses_frozen_test_images") is True
        and isinstance(validated_outputs, int)
        and not isinstance(validated_outputs, bool)
        and validated_outputs >= 2,
        "demo validation must be local, use frozen test images, and validate at least two outputs",
    )
    visual_review = payloads.get("reports/phase2_visual_review.json", {})
    reviewed_sha256 = visual_review.get("reviewed_sha256")
    policy(
        "reports/phase2_visual_review.json",
        isinstance(reviewed_sha256, dict)
        and (set(FINAL_FIGURES) | {"assets/demo.gif"}) <= set(reviewed_sha256),
        "visual review does not cover every final figure and demo GIF",
    )
    if isinstance(reviewed_sha256, dict):
        for target, expected in sorted(reviewed_sha256.items()):
            target_path = repo / target
            observed = sha256_file(target_path) if target_path.is_file() else None
            if expected != observed:
                hash_mismatches.append(
                    {
                        "report": "reports/phase2_visual_review.json",
                        "field": f"reviewed_sha256.{target}",
                        "target": target,
                        "expected": expected,
                        "observed": observed,
                    }
                )
    return {
        "missing": missing,
        "invalid": invalid,
        "policy_failures": policy_failures,
        "hash_mismatches": hash_mismatches,
        "passed_report_count": len(payloads),
        "expected_report_count": len(FINAL_EVIDENCE_REPORTS),
        "model_license_fresh_within_24h": license_fresh,
        "hf_upload_plan_is_safe_dry_run": upload_safe,
    }


def audit_final_media(repo: Path) -> dict[str, Any]:
    figure_details: dict[str, Any] = {}
    for relative in FINAL_FIGURES:
        path = repo / relative
        details: dict[str, Any] = {"exists": path.is_file()}
        if path.is_file():
            try:
                with Image.open(path) as image:
                    image.verify()
                    details.update(
                        {
                            "format": image.format,
                            "width": image.width,
                            "height": image.height,
                            "valid": image.format == "PNG"
                            and image.width >= 320
                            and image.height >= 180,
                        }
                    )
            except (OSError, ValueError) as error:
                details.update({"valid": False, "error": str(error)})
        else:
            details["valid"] = False
        figure_details[relative] = details

    gif_path = repo / "assets" / "demo.gif"
    gif: dict[str, Any] = {"exists": gif_path.is_file(), "valid": False}
    if gif_path.is_file():
        try:
            with Image.open(gif_path) as image:
                frame_count = int(getattr(image, "n_frames", 1))
                for frame in range(frame_count):
                    image.seek(frame)
                    image.load()
                gif.update(
                    {
                        "format": image.format,
                        "width": image.width,
                        "height": image.height,
                        "frames": frame_count,
                        "valid": (
                            image.format == "GIF"
                            and frame_count >= 2
                            and image.width >= 320
                            and image.height >= 180
                        ),
                    }
                )
        except (EOFError, OSError, ValueError) as error:
            gif["error"] = str(error)

    preview_path = repo / "assets" / "github-social-preview.png"
    preview: dict[str, Any] = {"exists": preview_path.is_file(), "valid": False}
    if preview_path.is_file():
        try:
            with Image.open(preview_path) as image:
                image.verify()
                preview.update(
                    {
                        "format": image.format,
                        "width": image.width,
                        "height": image.height,
                        "bytes": preview_path.stat().st_size,
                        "valid": (
                            image.format == "PNG"
                            and image.width == 1280
                            and image.height == 640
                            and preview_path.stat().st_size <= 1024 * 1024
                        ),
                    }
                )
        except (OSError, ValueError) as error:
            preview["error"] = str(error)
    return {
        "figures": figure_details,
        "all_figures_valid": all(item["valid"] for item in figure_details.values()),
        "demo_gif": gif,
        "github_social_preview": preview,
    }


def audit_closeout_metadata(repo: Path) -> dict[str, Any]:
    license_path = repo / "LICENSE"
    license_text = license_path.read_text(encoding="utf-8") if license_path.is_file() else ""
    citation_path = repo / "CITATION.cff"
    citation: dict[str, Any] = {}
    citation_error: str | None = None
    if citation_path.is_file():
        try:
            loaded = yaml.safe_load(citation_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                citation = loaded
            else:
                citation_error = "top-level value is not an object"
        except (OSError, UnicodeError, yaml.YAMLError) as error:
            citation_error = str(error)
    else:
        citation_error = "missing"

    authors = citation.get("authors")
    author_rows = authors if isinstance(authors, list) else []
    author_names = {
        str(row.get("name"))
        for row in author_rows
        if isinstance(row, dict) and row.get("name")
    }
    expected_repository = "https://github.com/kuotunyu/defectforge-visa-synthetic-data"
    citation_valid = bool(
        citation_error is None
        and str(citation.get("cff-version")) == "1.2.0"
        and citation.get("title")
        and citation.get("type") == "software"
        and citation.get("repository-code") == expected_repository
        and citation.get("license") == "MIT"
        and str(citation.get("version")) == "1.2.1"
        and "kuotunyu" in author_names
    )
    standard_mit = bool(
        license_text.startswith("MIT License\n\nCopyright (c) 2026 kuotunyu\n")
        and "Permission is hereby granted, free of charge" in license_text
        and "THE SOFTWARE IS PROVIDED \"AS IS\"" in license_text
        and "\n---\n" not in license_text
        and "NOTE:" not in license_text
    )
    notices_path = repo / "THIRD_PARTY_NOTICES.md"
    notices = notices_path.read_text(encoding="utf-8") if notices_path.is_file() else ""
    notice_markers = (
        "VisA Dataset",
        "stable-diffusion-2-inpainting",
        "stable-diffusion-xl-1.0-inpainting-0.1",
        "facebook/dinov2-base",
        "DefectForge Synthetic Images",
        "DefectForge LoRA Weights",
    )
    notices_valid = all(marker in notices for marker in notice_markers)
    return {
        "standard_mit_license": standard_mit,
        "citation_valid": citation_valid,
        "citation_error": citation_error,
        "citation_repository": citation.get("repository-code"),
        "citation_version": str(citation.get("version", "")),
        "third_party_notices_complete": notices_valid,
    }


def audit_release_acceptance(repo: Path) -> dict[str, Any]:
    path = repo / "reports" / "release_acceptance.md"
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    return {
        "exists": path.is_file(),
        "no_unchecked_items": "- [ ]" not in text,
        "has_passed_items": "## Passed" in text,
        "has_fixed_items": "## Fixed" in text,
        "has_residual_risks": "## Residual risks" in text,
    }


def build_audit(repo: Path) -> dict[str, Any]:
    repo = repo.resolve(strict=True)
    require((repo / ".git").exists(), f"Not a Git repository: {repo}")
    tree = scan_tracked_tree(repo, tracked_paths(repo))
    identities = audit_identities(repo)
    required = audit_required_paths(repo)
    content = audit_plan_and_readme(repo)
    evidence = audit_evidence(repo)
    media = audit_final_media(repo)
    closeout = audit_closeout_metadata(repo)
    acceptance = audit_release_acceptance(repo)
    checks = {
        "required_paths": not required["missing"] and not required["untracked_required"],
        "owner_local_files_untracked": not required["owner_local_tracked"],
        "no_secret_or_personal_path": not tree["secret_or_path_findings"],
        "no_tracked_env": not tree["tracked_env"],
        "no_oversized_tracked_files": not tree["oversized"],
        "single_git_identity": not identities["invalid_rows"],
        "no_coauthor_trailers": identities["coauthor_trailer_count"] == 0,
        "readme_final": (
            content["readme_has_required_sections"]
            and content["readme_has_no_tbd_or_todo"]
            and content["readme_status_is_final"]
        ),
        "final_evidence_reports_passed": (
            not evidence["missing"]
            and not evidence["invalid"]
            and not evidence["policy_failures"]
            and evidence["passed_report_count"] == evidence["expected_report_count"]
        ),
        "final_evidence_hashes_current": not evidence["hash_mismatches"],
        "model_license_verification_fresh": evidence["model_license_fresh_within_24h"],
        "hf_upload_plan_safe_dry_run": evidence["hf_upload_plan_is_safe_dry_run"],
        "final_figures_valid": media["all_figures_valid"],
        "demo_gif_valid": media["demo_gif"]["valid"],
        "github_social_preview_valid": media["github_social_preview"]["valid"],
        "closeout_metadata_valid": (
            closeout["standard_mit_license"]
            and closeout["citation_valid"]
            and closeout["third_party_notices_complete"]
        ),
        "release_acceptance_complete": all(acceptance.values()),
    }
    return {
        "schema_version": 1,
        "status": "passed" if all(checks.values()) else "incomplete",
        "publishes_or_uploads": False,
        "checks": checks,
        "required": required,
        "tree": tree,
        "git": identities,
        "content": content,
        "evidence": evidence,
        "media": media,
        "closeout": closeout,
        "release_acceptance": acceptance,
    }


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/publish_validation.json"),
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Return success for a checkpoint audit even when final M24 artefacts are pending",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    audit = build_audit(args.repo)
    atomic_write_json(args.output, audit)
    if audit["status"] != "passed":
        print(json.dumps(audit["checks"], indent=2, sort_keys=True))
        print(f"M24 remains incomplete; checkpoint report: {args.output}")
        return 0 if args.allow_incomplete else 1
    print(f"Publication audit passed: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
