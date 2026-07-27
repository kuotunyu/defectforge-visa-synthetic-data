"""Verify the complete Phase 1 handoff and contributor history.

This verifier is intentionally CPU-only. It checks the frozen milestone evidence,
reruns the lightweight independent report validators, and audits the entire local
Git history so the future GitHub contributor list can only contain ``kuotunyu``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_GIT_NAME = "kuotunyu"
EXPECTED_GIT_EMAIL = "61350295+kuotunyu@users.noreply.github.com"
MILESTONES = tuple(range(16))
REQUIRED_INSTRUCTION_ROWS = (
    "| 1. 上 Colab 方式 |",
    "| 2. Runtime 選型 |",
    "| 3. 需要的 Colab Secrets |",
    "| 4. 實測時數與 compute units |",
    "| 5. 跑完要下載哪些檔案 / 放回哪個路徑 |",
)
REQUIRED_PASSED_JSON = (
    "reports/stageA_copypaste_validation.json",
    "reports/stageA_procedural_validation.json",
    "reports/stageA_procedural_norealstats_validation.json",
    "reports/placements_validation.json",
    "reports/lora_sd2_validation.json",
    "reports/lora_sdxl_import_validation.json",
    "reports/lora_sdxl_local_validation.json",
    "reports/stageB_sd2_original_validation.json",
    "reports/stageB_sd2_searched_validation.json",
    "reports/generation_quality_validation.json",
)
REQUIRED_FILES = (
    "uv.lock",
    "splits/source_checksums.json",
    "splits/split_manifest.json",
    "splits/MANIFEST.sha256",
    "splits/test_blocklist.json",
    "splits/fewshot_selection.json",
    "splits/FEWSHOT_SELECTION.sha256",
    "splits/defect_types.json",
    "splits/DEFECT_TYPES.sha256",
    "reports/split_preparation.json",
    "reports/filter_validation.json",
    "results/generation_quality.csv",
    "notebooks/01_train_inpaint_lora_sdxl.ipynb",
)
INDEPENDENT_VALIDATORS = (
    "scripts/validate_colab_notebook.py",
    "scripts/verify_filter_report.py",
    "scripts/verify_generation_quality.py",
)


class Phase1AcceptanceError(RuntimeError):
    """Raised when the Phase 1 evidence violates the frozen contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Phase1AcceptanceError(message)


def checked_milestones(plan_text: str) -> set[int]:
    """Return milestone numbers whose canonical PLAN row is checked."""
    matches = re.finditer(
        r"^- \[(?P<state>[ x])\] \*\*M(?P<number>\d+)\*\*",
        plan_text,
        flags=re.MULTILINE,
    )
    return {
        int(match.group("number"))
        for match in matches
        if match.group("state") == "x"
    }


def validate_instruction_handoff(instructions_text: str) -> dict[str, Any]:
    """Validate the completed M11 handoff and the explicit M18 ownership boundary."""
    notebook_1_start = instructions_text.find("## Notebook 1")
    notebook_2_start = instructions_text.find("## Notebook 2")
    require(notebook_1_start >= 0, "Notebook 1 handoff section is missing")
    require(notebook_2_start > notebook_1_start, "Notebook 2 ownership section is missing")
    notebook_1 = instructions_text[notebook_1_start:notebook_2_start]
    missing_rows = [row for row in REQUIRED_INSTRUCTION_ROWS if row not in notebook_1]
    require(not missing_rows, f"Notebook 1 handoff rows are missing: {missing_rows}")
    require("TBD" not in notebook_1, "Notebook 1 handoff still contains TBD")
    require("已完成" in notebook_1, "Notebook 1 is not marked complete")

    notebook_2 = instructions_text[notebook_2_start:]
    require("M18" in notebook_2 and "尚未建立" in notebook_2, "M18 ownership is unclear")
    require(
        "M15 最終驗收" not in notebook_2,
        "Notebook 2 still creates an M15/M18 dependency cycle",
    )
    return {
        "m11_handoff_rows": len(REQUIRED_INSTRUCTION_ROWS),
        "m11_status": "complete",
        "m18_status": "owned_by_m18",
    }


def parse_identity_rows(output: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in output.splitlines():
        if not line:
            continue
        parts = line.split("\t")
        require(len(parts) == 6, f"Unexpected git identity row: {line!r}")
        commit, author_name, author_email, committer_name, committer_email, subject = parts
        rows.append(
            {
                "commit": commit,
                "author_name": author_name,
                "author_email": author_email,
                "committer_name": committer_name,
                "committer_email": committer_email,
                "subject": subject,
            }
        )
    require(bool(rows), "Git history is empty")
    return rows


def audit_git_history(project_root: Path) -> dict[str, Any]:
    identity_output = subprocess.run(
        [
            "git",
            "log",
            "--format=%H%x09%an%x09%ae%x09%cn%x09%ce%x09%s",
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    rows = parse_identity_rows(identity_output)
    for row in rows:
        require(row["author_name"] == EXPECTED_GIT_NAME, f"Wrong author: {row['commit']}")
        require(row["author_email"] == EXPECTED_GIT_EMAIL, f"Wrong author email: {row['commit']}")
        require(row["committer_name"] == EXPECTED_GIT_NAME, f"Wrong committer: {row['commit']}")
        require(
            row["committer_email"] == EXPECTED_GIT_EMAIL,
            f"Wrong committer email: {row['commit']}",
        )

    messages = subprocess.run(
        ["git", "log", "--format=%B%x00"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    require(
        re.search(r"(?im)^co-authored-by\s*:", messages) is None,
        "Git history contains a Co-Authored-By trailer",
    )

    milestone_commits: dict[str, list[str]] = {}
    for milestone in MILESTONES:
        pattern = re.compile(rf"(?<!\w)M{milestone}(?!\d)", flags=re.IGNORECASE)
        commits = [row["commit"] for row in rows if pattern.search(row["subject"])]
        require(commits, f"M{milestone} has no dedicated commit")
        milestone_commits[f"M{milestone}"] = commits

    local_name = subprocess.run(
        ["git", "config", "--local", "user.name"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    local_email = subprocess.run(
        ["git", "config", "--local", "user.email"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    require(local_name == EXPECTED_GIT_NAME, "Repo-local git user.name changed")
    require(local_email == EXPECTED_GIT_EMAIL, "Repo-local git user.email changed")
    remotes = subprocess.run(
        ["git", "remote"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.split()
    return {
        "commit_count": len(rows),
        "contributor_names": sorted({row["author_name"] for row in rows}),
        "contributor_emails": sorted({row["author_email"] for row in rows}),
        "coauthored_by_trailers": 0,
        "milestone_commits": milestone_commits,
        "remote_count": len(remotes),
        "repo_local_identity": f"{local_name} <{local_email}>",
    }


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase1AcceptanceError(f"Invalid JSON: {path}") from exc
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def validate_evidence(project_root: Path) -> dict[str, Any]:
    for relative in REQUIRED_FILES:
        path = project_root / relative
        require(path.is_file() and path.stat().st_size > 0, f"Missing Phase 1 evidence: {relative}")

    statuses: dict[str, str] = {}
    for relative in REQUIRED_PASSED_JSON:
        payload = load_json_object(project_root / relative)
        require(payload.get("status") == "passed", f"Evidence did not pass: {relative}")
        statuses[relative] = "passed"

    split_preparation = load_json_object(project_root / "reports/split_preparation.json")
    assertions = split_preparation.get("assertions")
    require(isinstance(assertions, dict), "Split preparation assertions are missing")
    require(
        assertions and set(assertions.values()) == {"passed"},
        "Split preparation has a failed assertion",
    )

    filter_validation = load_json_object(project_root / "reports/filter_validation.json")
    counts = filter_validation.get("counts")
    require(isinstance(counts, dict), "Filter counts are missing")
    require(
        counts.get("accepted", 0) + counts.get("rejected", 0) == counts.get("total"),
        "Filter acceptance counts are inconsistent",
    )
    return {
        "passed_json": statuses,
        "required_file_count": len(REQUIRED_FILES),
        "split_assertions": assertions,
    }


def run_independent_validators(project_root: Path) -> dict[str, str]:
    results: dict[str, str] = {}
    for relative in INDEPENDENT_VALIDATORS:
        process = subprocess.run(
            [sys.executable, relative],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        require(
            process.returncode == 0,
            f"Independent validator failed: {relative}\n{process.stdout}\n{process.stderr}",
        )
        results[relative] = "passed"
    return results


def render_markdown(payload: dict[str, Any]) -> str:
    git = payload["git"]
    evidence = payload["evidence"]
    validators = payload["independent_validators"]
    milestone_rows = "\n".join(
        f"| {milestone} | {len(commits)} |"
        for milestone, commits in git["milestone_commits"].items()
    )
    validator_rows = "\n".join(
        f"| `{name}` | {status} |" for name, status in validators.items()
    )
    return f"""# Phase 1 Acceptance Report

**Status:** `{payload["status"]}`

**Verified at:** `{payload["verified_at"]}`

**Policy decision:** [ADR-019](../docs/decisions.md#adr-019)

## Acceptance summary

| Gate | Result |
|---|---|
| PLAN M0-M15 checked | {len(payload["plan"]["checked_milestones"])} / 16 |
| Frozen validation JSON | {len(evidence["passed_json"])} passed |
| Independent validators | {len(validators)} passed |
| Git commits audited | {git["commit_count"]} |
| Contributor identities | `{", ".join(git["contributor_names"])}` |
| Co-Authored-By trailers | {git["coauthored_by_trailers"]} |
| Repo-local identity | `{git["repo_local_identity"]}` |
| Configured remotes | {git["remote_count"]} |

## Milestone commit coverage

| Milestone | Matching commits |
|---|---:|
{milestone_rows}

## Independent validators

| Validator | Result |
|---|---|
{validator_rows}

## Colab handoff boundary

- M11 SDXL has all five operational handoff fields and is complete.
- M18 owns creation, smoke testing, and the five concrete SegFormer handoff fields.
- M19 is the user-operated Colab run. No SegFormer action is requested during M15.
- Actual M11 elapsed time and peak VRAM are recorded. Compute units were not captured
  before the run and are explicitly reported as unavailable rather than inferred.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/phase1_acceptance.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/phase1_acceptance.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    plan_text = (project_root / "PLAN.md").read_text(encoding="utf-8")
    checked = checked_milestones(plan_text)
    missing = sorted(set(MILESTONES) - checked)
    require(not missing, f"Unchecked Phase 1 milestones: {missing}")

    instructions = (project_root / "instructions_for_me.md").read_text(encoding="utf-8")
    payload: dict[str, Any] = {
        "status": "passed",
        "verified_at": datetime.now(UTC).isoformat(),
        "plan": {"checked_milestones": sorted(checked & set(MILESTONES))},
        "instructions": validate_instruction_handoff(instructions),
        "evidence": validate_evidence(project_root),
        "independent_validators": run_independent_validators(project_root),
        "git": audit_git_history(project_root),
    }

    output = args.output if args.output.is_absolute() else project_root / args.output
    report = args.report if args.report.is_absolute() else project_root / args.report
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
