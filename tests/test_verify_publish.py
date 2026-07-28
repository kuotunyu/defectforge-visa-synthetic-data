from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from scripts.verify_publish import (
    COAUTHOR_TRAILER,
    EXPECTED_EMAIL,
    EXPECTED_NAME,
    FINAL_EVIDENCE_REPORTS,
    FINAL_FIGURES,
    audit_evidence,
    audit_final_media,
    audit_identities,
    audit_plan_and_readme,
    scan_text,
    sha256_file,
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def test_scan_text_reports_locations_without_secret_values() -> None:
    fake_github = "gho_" + "a" * 24
    fake_hf = "hf_" + "b" * 24
    private_path = "C:" + "\\Users\\private\\repo"
    text = f"first={fake_github}\nsecond={fake_hf}\npath={private_path}\n"
    findings = scan_text(Path("fixture.txt"), text)
    assert [item["kind"] for item in findings] == [
        "github_token",
        "huggingface_token",
        "personal_windows_path",
    ]
    serialized = repr(findings)
    assert fake_github not in serialized
    assert fake_hf not in serialized


def test_coauthor_scan_requires_a_real_trailer_line() -> None:
    prose = "This document explains why Co-Authored-By trailers are forbidden."
    trailer = "Subject\n\nCo-Authored-By: Other <other@example.com>\n"
    assert COAUTHOR_TRAILER.search(prose) is None
    assert COAUTHOR_TRAILER.search(trailer) is not None


def test_scan_text_allows_portable_userprofile_placeholder() -> None:
    findings = scan_text(
        Path("docs/environment.md"),
        r"Cache: %USERPROFILE%\.cache\huggingface",
    )
    assert findings == []


def test_identity_audit_detects_other_contributor_and_coauthor(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.name", EXPECTED_NAME)
    _git(tmp_path, "config", "user.email", EXPECTED_EMAIL)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("first\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.txt")
    _git(tmp_path, "commit", "-m", "expected identity")

    clean = audit_identities(tmp_path)
    assert clean["invalid_rows"] == []
    assert clean["coauthor_trailer_count"] == 0

    _git(tmp_path, "config", "user.name", "Other Person")
    _git(tmp_path, "config", "user.email", "other@example.com")
    tracked.write_text("second\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.txt")
    _git(
        tmp_path,
        "commit",
        "-m",
        "invalid identity",
        "-m",
        "Co-Authored-By: Another Person <another@example.com>",
    )

    rejected = audit_identities(tmp_path)
    assert rejected["invalid_rows"] == [
        ["Other Person", "other@example.com", "Other Person", "other@example.com"]
    ]
    assert rejected["coauthor_trailer_count"] == 1


def test_plan_and_readme_audit_requires_all_prepublication_milestones(tmp_path: Path) -> None:
    plan = "\n".join(
        f"- [{'x' if number != 19 else ' '}] **M{number}** milestone" for number in range(25)
    )
    (tmp_path / "PLAN.md").write_text(plan, encoding="utf-8")
    (tmp_path / "README.md").write_text(
        """# 專案
## 研究問題
## 方法與系統架構
## 實驗設計
## 實驗結果
## 限制與誠實揭露
## 重現方式
## 授權與引用
""",
        encoding="utf-8",
    )

    incomplete = audit_plan_and_readme(tmp_path)

    assert incomplete["missing_prepublication_milestones"] == [19]
    assert incomplete["prepublication_complete"] is False

    (tmp_path / "PLAN.md").write_text(
        plan.replace("- [ ] **M19**", "- [x] **M19**"), encoding="utf-8"
    )
    complete = audit_plan_and_readme(tmp_path)
    assert complete["prepublication_complete"] is True
    assert complete["readme_has_no_tbd_or_todo"] is True

    with_extension = (tmp_path / "PLAN.md").read_text(encoding="utf-8")
    with_extension += "\n- [x] **M25** post-publication extension\n"
    (tmp_path / "PLAN.md").write_text(with_extension, encoding="utf-8")
    extended = audit_plan_and_readme(tmp_path)
    assert extended["prepublication_complete"] is True
    assert extended["checked_milestones"][-1] == 25


def test_final_media_audit_rejects_single_frame_gif(tmp_path: Path) -> None:
    for relative in FINAL_FIGURES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (640, 360), "white").save(path)
    gif_path = tmp_path / "assets" / "demo.gif"
    gif_path.parent.mkdir(parents=True)
    Image.new("RGB", (640, 360), "white").save(gif_path, format="GIF")

    single_frame = audit_final_media(tmp_path)

    assert single_frame["all_figures_valid"] is True
    assert single_frame["demo_gif"]["valid"] is False

    frames = [Image.new("RGB", (640, 360), color) for color in ("white", "black")]
    frames[0].save(
        gif_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
    )
    animated = audit_final_media(tmp_path)
    assert animated["demo_gif"]["valid"] is True
    assert animated["demo_gif"]["frames"] == 2


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_evidence_audit_binds_reports_to_current_artifacts(tmp_path: Path) -> None:
    artifact_text = {
        "README.md": "readme\n",
        "results/classification.csv": "classification\n",
        "results/segmentation.csv": "segmentation\n",
        "reports/segmentation_results.md": "segmentation report\n",
        "docs/license_chain.md": "license\n",
        "hf_cards/dataset/README.md": "dataset card\n",
        "hf_cards/model/README.md": "model card\n",
        "assets/demo.gif": "gif fixture\n",
    }
    for relative, text in artifact_text.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    for relative in FINAL_FIGURES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode("utf-8"))

    model_license = {
        "status": "passed",
        "verified_at": datetime.now(UTC).isoformat(),
    }
    _write_json(tmp_path / "reports/model_license_verification.json", model_license)
    reports: dict[str, dict[str, object]] = {
        relative: {"status": "passed"} for relative in FINAL_EVIDENCE_REPORTS
    }
    reports["reports/classifier_matrix_validation.json"].update(
        {"formal_runs": 38, "blocklist_hits": 0}
    )
    reports["reports/segmentation_validation.json"].update(
        {
            "physical_runs": 16,
            "logical_rows": 18,
            "source": "raw_training_report_json",
            "segmentation_csv_sha256": sha256_file(tmp_path / "results/segmentation.csv"),
            "report_sha256": sha256_file(tmp_path / "reports/segmentation_results.md"),
        }
    )
    reports["reports/phase2_figures_validation.json"].update(
        {
            "visual_inspection_required": True,
            "classification_sha256": sha256_file(tmp_path / "results/classification.csv"),
            "segmentation_sha256": sha256_file(tmp_path / "results/segmentation.csv"),
            "figures": {},
        }
    )
    reports["reports/sample_grids_validation.json"]["visual_inspection_required"] = True
    reports["reports/readme_validation.json"].update(
        {
            "negative_results_preserved": True,
            "readme_sha256": sha256_file(tmp_path / "README.md"),
            "classification_sha256": sha256_file(tmp_path / "results/classification.csv"),
            "segmentation_sha256": sha256_file(tmp_path / "results/segmentation.csv"),
        }
    )
    reports["reports/license_chain_validation.json"].update(
        {
            "source_sha256": sha256_file(tmp_path / "docs/license_chain.md"),
            "upstream_verification_sha256": sha256_file(
                tmp_path / "reports/model_license_verification.json"
            ),
            "document_sha256": {
                relative: sha256_file(tmp_path / relative)
                for relative in (
                    "README.md",
                    "hf_cards/dataset/README.md",
                    "hf_cards/model/README.md",
                )
            },
        }
    )
    reports["reports/hf_package_validation.json"]["dataset"] = {"test_blocklist_hits": 0}
    reports["reports/hf_upload_plan.json"].update(
        {
            "mode": "dry_run",
            "changes_visibility": False,
            "creates_or_updates_private_repositories": False,
        }
    )
    reports["reports/demo_checkpoint_selection.json"].update(
        {
            "selection_is_post_evaluation_demo_only": True,
            "changes_reported_metrics": False,
            "classification_results_sha256": sha256_file(tmp_path / "results/classification.csv"),
            "segmentation_results_sha256": sha256_file(tmp_path / "results/segmentation.csv"),
        }
    )
    reports["reports/demo_validation.json"].update(
        {
            "share_enabled": False,
            "uses_frozen_test_images": True,
            "validated_outputs": 2,
            "classification_sha256": sha256_file(tmp_path / "results/classification.csv"),
            "segmentation_sha256": sha256_file(tmp_path / "results/segmentation.csv"),
            "demo_gif_sha256": sha256_file(tmp_path / "assets/demo.gif"),
        }
    )
    reports["reports/phase2_visual_review.json"]["reviewed_sha256"] = {
        relative: sha256_file(tmp_path / relative)
        for relative in (*FINAL_FIGURES, "assets/demo.gif")
    }
    for relative, payload in reports.items():
        if relative == "reports/model_license_verification.json":
            continue
        _write_json(tmp_path / relative, payload)

    passed = audit_evidence(tmp_path)

    assert passed["missing"] == []
    assert passed["invalid"] == []
    assert passed["policy_failures"] == []
    assert passed["hash_mismatches"] == []
    assert passed["model_license_fresh_within_24h"] is True
    assert passed["hf_upload_plan_is_safe_dry_run"] is True

    (tmp_path / "results/classification.csv").write_text("changed\n", encoding="utf-8")
    stale = audit_evidence(tmp_path)
    assert {item["target"] for item in stale["hash_mismatches"]} == {"results/classification.csv"}
