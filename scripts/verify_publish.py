"""Run the fail-closed M24 GitHub publication audit without publishing anything."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

EXPECTED_NAME = "kuotunyu"
EXPECTED_EMAIL = "61350295+kuotunyu@users.noreply.github.com"
MAX_TRACKED_BYTES = 10 * 1024 * 1024
REQUIRED_PATHS = (
    "README.md",
    "PLAN.md",
    "CLAUDE.md",
    "LICENSE",
    "uv.lock",
    "docs",
    "notebooks",
    "splits/split_manifest.json",
    "splits/MANIFEST.sha256",
    "splits/defect_types.json",
    "splits/test_blocklist.json",
    "results/classification.csv",
    "results/segmentation.csv",
    "scripts/verify_readme.py",
    "assets/demo.gif",
    ".claude/skills",
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


def tracked_paths(repo: Path) -> list[Path]:
    output = _git(repo, "ls-files", "-z")
    return [repo / value for value in output.split("\0") if value]


def scan_text(path: Path, text: str) -> list[dict[str, Any]]:
    """Return location-only findings so a report never repeats a credential."""

    findings: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(line):
                findings.append(
                    {"kind": label, "path": path.as_posix(), "line": line_number}
                )
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
    skills = sorted(
        path.parent.name
        for path in (repo / ".claude" / "skills").glob("*/SKILL.md")
        if path.is_file()
    )
    return {
        "missing": missing,
        "skill_count": len(skills),
        "skills": skills,
        "skill_count_valid": len(skills) == 13,
    }


def build_audit(repo: Path) -> dict[str, Any]:
    repo = repo.resolve(strict=True)
    require((repo / ".git").exists(), f"Not a Git repository: {repo}")
    tree = scan_tracked_tree(repo, tracked_paths(repo))
    identities = audit_identities(repo)
    required = audit_required_paths(repo)
    checks = {
        "required_paths": not required["missing"],
        "thirteen_skills": required["skill_count_valid"],
        "no_secret_or_personal_path": not tree["secret_or_path_findings"],
        "no_tracked_env": not tree["tracked_env"],
        "no_oversized_tracked_files": not tree["oversized"],
        "single_git_identity": not identities["invalid_rows"],
        "no_coauthor_trailers": identities["coauthor_trailer_count"] == 0,
    }
    return {
        "schema_version": 1,
        "status": "passed" if all(checks.values()) else "incomplete",
        "publishes_or_uploads": False,
        "checks": checks,
        "required": required,
        "tree": tree,
        "git": identities,
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
    print(f"M24 publication audit passed: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
