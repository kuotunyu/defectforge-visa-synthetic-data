"""Load and resolve DefectForge paths from the single YAML source of truth."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_VARIABLE = re.compile(r"\$\{([^}]+)\}")


class PathConfigError(ValueError):
    """Raised when the path configuration is missing or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class Paths:
    """Resolved project and data paths."""

    project_root: Path
    config_file: Path
    data_root: Path
    raw: Path
    visa_tar: Path
    visa_raw: Path
    visa_fewshot: Path
    visa_highshot: Path
    synthetic: Path
    runs: Path
    cache: Path
    splits: Path
    reports: Path
    figures: Path
    configs: Path
    notebooks: Path
    colab_results: Path
    dotenv: Path
    hf_home: Path | None
    objects: tuple[str, ...]
    seed: int


def _find_project_root(config_file: Path) -> Path:
    for candidate in (config_file.parent, *config_file.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise PathConfigError(f"Could not find pyproject.toml above {config_file}")


def _expand(value: str, variables: dict[str, str]) -> str:
    previous = value
    for _ in range(10):
        expanded = _VARIABLE.sub(
            lambda match: variables.get(match.group(1), match.group(0)),
            previous,
        )
        if expanded == previous:
            unresolved = _VARIABLE.findall(expanded)
            if unresolved:
                raise PathConfigError(
                    f"Unresolved path variables {unresolved!r} in {value!r}"
                )
            return expanded
        previous = expanded
    raise PathConfigError(f"Path expansion exceeded 10 passes for {value!r}")


def _as_path(value: str, *, project_root: Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve(strict=False)


def load_paths(cfg: str | Path = "configs/paths.yaml") -> Paths:
    """Load ``configs/paths.yaml`` and return fully expanded absolute paths."""

    config_file = Path(cfg).resolve(strict=True)
    project_root = _find_project_root(config_file)
    with config_file.open("r", encoding="utf-8") as handle:
        raw_config: dict[str, Any] = yaml.safe_load(handle)

    if not isinstance(raw_config, dict) or not isinstance(raw_config.get("paths"), dict):
        raise PathConfigError("paths.yaml must contain a top-level 'paths' mapping")

    data_root_value = raw_config.get("data_root")
    if not isinstance(data_root_value, str):
        raise PathConfigError("paths.yaml must contain a string 'data_root'")

    variables = {"data_root": data_root_value}
    resolved: dict[str, Path] = {}
    for name, value in raw_config["paths"].items():
        if not isinstance(value, str):
            raise PathConfigError(f"paths.{name} must be a string")
        resolved[name] = _as_path(_expand(value, variables), project_root=project_root)

    required = {
        "raw",
        "visa_tar",
        "visa_raw",
        "visa_fewshot",
        "visa_highshot",
        "synthetic",
        "runs",
        "cache",
        "splits",
        "reports",
        "figures",
        "configs",
        "notebooks",
        "colab_results",
    }
    missing = sorted(required - resolved.keys())
    if missing:
        raise PathConfigError(f"paths.yaml is missing required paths: {missing}")

    dotenv_value = raw_config.get("dotenv")
    if not isinstance(dotenv_value, str):
        raise PathConfigError("paths.yaml must contain a string 'dotenv'")

    hf_home_value = raw_config.get("hf_home")
    hf_home = (
        None
        if hf_home_value is None
        else _as_path(_expand(str(hf_home_value), variables), project_root=project_root)
    )

    objects = raw_config.get("objects")
    if not isinstance(objects, list) or not all(isinstance(item, str) for item in objects):
        raise PathConfigError("paths.yaml 'objects' must be a list of strings")

    return Paths(
        project_root=project_root,
        config_file=config_file,
        data_root=_as_path(data_root_value, project_root=project_root),
        dotenv=_as_path(_expand(dotenv_value, variables), project_root=project_root),
        hf_home=hf_home,
        objects=tuple(objects),
        seed=int(raw_config.get("seed", 42)),
        **resolved,
    )
