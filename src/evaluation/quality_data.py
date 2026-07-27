"""M14 source audit and deterministic crop materialization."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from skimage.measure import label

from src.common.integrity import sha256_file
from src.common.paths import Paths
from src.filtering.dataset import FilterSample
from src.filtering.metrics import context_bbox

SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
CROP_PIPELINE_VERSION = "0.1.0"
LOGGER = logging.getLogger("quality_data")


class QualityDataError(RuntimeError):
    """M14 input provenance or materialized crops failed validation."""


@dataclass(frozen=True, slots=True)
class CropEntry:
    """One crop and its grouping metadata."""

    key: str
    kind: str
    object_name: str
    defect_type: str
    input_name: str | None
    sample_id: str
    relative_path: str
    sha256: str
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class SourceAudit:
    """Hash-complete proof that no M14 source belongs to the test blocklist."""

    paths_checked: int
    hashes_checked: int
    blocklist_hits: int
    sha256: str


def _type_id(type_record: Mapping[str, Any]) -> str:
    trigger = str(type_record["trigger_token"])
    match = re.fullmatch(r"<[^>]+-(type\d+)>", trigger)
    if match is None:
        raise QualityDataError(f"Cannot derive defect type from trigger: {trigger}")
    return match.group(1)


def load_defect_types(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or not isinstance(value.get("objects"), dict):
        raise QualityDataError(f"Invalid defect type map: {path}")
    return value


def component_mask(paths: Paths, component: Mapping[str, Any]) -> np.ndarray:
    """Reconstruct one frozen connected component in full-image coordinates."""

    mask_path = paths.visa_raw / str(component["mask_path"])
    with Image.open(mask_path) as handle:
        full_mask = np.asarray(handle.convert("L")) > 0
    labels = label(full_mask, connectivity=2)
    component_id = int(component["component_id"])
    selected = labels == component_id + 1
    if int(selected.sum()) != int(component["area_px"]):
        raise QualityDataError(
            f"Frozen component changed: {component['mask_path']} #{component_id}"
        )
    return selected


def _collect_source_paths(
    paths: Paths,
    samples: Sequence[FilterSample],
    defect_types: Mapping[str, Any],
) -> dict[Path, str | None]:
    sources: dict[Path, str | None] = {}
    for sample in samples:
        sources[sample.image_path] = None
        sources[sample.mask_path] = None
        source = sample.record["source"]
        background = (paths.visa_raw / str(source["background_image"])).resolve(strict=True)
        expected_background = str(source["background_sha256"])
        previous = sources.get(background)
        if previous is not None and previous != expected_background:
            raise QualityDataError(f"Conflicting expected hash for {background}")
        sources[background] = expected_background
        for field in ("defect_source_image", "defect_source_mask"):
            relative = source[field]
            if relative is not None:
                sources[(paths.visa_raw / str(relative)).resolve(strict=True)] = None
    for object_record in defect_types["objects"].values():
        for type_record in object_record["types"]:
            for component in type_record["components"]:
                for field in ("image_path", "mask_path"):
                    sources[
                        (paths.visa_raw / str(component[field])).resolve(strict=True)
                    ] = None
    return sources


def audit_sources_against_blocklist(
    paths: Paths,
    samples: Sequence[FilterSample],
    defect_types: Mapping[str, Any],
    blocklist_path: Path,
) -> SourceAudit:
    """Hash every unique provenance source and fail closed on any test hit."""

    with blocklist_path.open("r", encoding="utf-8") as handle:
        blocklist_payload = json.load(handle)
    blocked = {str(value) for value in blocklist_payload["sha256"]}
    sources = _collect_source_paths(paths, samples, defect_types)
    rows: list[tuple[str, str]] = []
    hits: list[tuple[str, str]] = []
    ordered_sources = sorted(sources.items(), key=lambda item: str(item[0]))
    for index, (path, expected) in enumerate(ordered_sources):
        observed = sha256_file(path)
        if expected is not None and observed != expected:
            raise QualityDataError(
                f"Frozen source hash changed: {path}, expected {expected}, observed {observed}"
            )
        rows.append((str(path), observed))
        if observed in blocked:
            hits.append((str(path), observed))
        if (index + 1) % 250 == 0:
            LOGGER.info("M14 source audit: %d/%d", index + 1, len(ordered_sources))
    if hits:
        raise QualityDataError(f"M14 sources hit test blocklist: {hits[:3]}")
    digest = hashlib.sha256(
        json.dumps(rows, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return SourceAudit(
        paths_checked=len(rows),
        hashes_checked=len({digest for _, digest in rows}),
        blocklist_hits=0,
        sha256=digest,
    )


def crop_cache_key(
    *,
    metadata_sha256: str,
    defect_types_sha256: str,
    source_audit_sha256: str,
    ratio: float,
) -> str:
    payload = {
        "pipeline_version": CROP_PIPELINE_VERSION,
        "algorithm": "square-mask-context-v1",
        "metadata_sha256": metadata_sha256,
        "defect_types_sha256": defect_types_sha256,
        "source_audit_sha256": source_audit_sha256,
        "ratio": ratio,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:24]


def _save_crop(
    image_path: Path,
    mask: np.ndarray,
    output_path: Path,
    *,
    ratio: float,
) -> tuple[str, int, int]:
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
        if (image.height, image.width) != mask.shape:
            raise QualityDataError(f"Image/mask shape mismatch: {image_path}")
        x0, y0, x1, y1 = context_bbox(mask, ratio=ratio)
        crop = image.crop((x0, y0, x1, y1))
        crop.save(output_path, format="PNG", optimize=True)
        width, height = crop.size
        crop.close()
    return sha256_file(output_path), width, height


def _safe(value: str) -> str:
    return SAFE_NAME.sub("_", value).strip("_")


def _materialize_fresh(
    root: Path,
    paths: Paths,
    samples: Sequence[FilterSample],
    defect_types: Mapping[str, Any],
    *,
    ratio: float,
    seed: int,
    source_audit_sha256: str,
) -> list[CropEntry]:
    entries: list[CropEntry] = []
    real_entries: list[CropEntry] = []
    for object_name, object_record in sorted(defect_types["objects"].items()):
        for type_record in object_record["types"]:
            defect_type = _type_id(type_record)
            for component in type_record["components"]:
                sample_id = (
                    f"{Path(component['image_path']).stem}"
                    f"__component{int(component['component_id']):02d}"
                )
                key = f"real__{object_name}__{defect_type}__{sample_id}"
                relative_path = f"real/{_safe(key)}.png"
                output_path = root / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                crop_sha, width, height = _save_crop(
                    paths.visa_raw / str(component["image_path"]),
                    component_mask(paths, component),
                    output_path,
                    ratio=ratio,
                )
                entry = CropEntry(
                    key=key,
                    kind="real",
                    object_name=str(object_name),
                    defect_type=defect_type,
                    input_name=None,
                    sample_id=sample_id,
                    relative_path=relative_path,
                    sha256=crop_sha,
                    width=width,
                    height=height,
                )
                entries.append(entry)
                real_entries.append(entry)

    for sample in samples:
        object_name = str(sample.record["object"])
        defect_type = str(sample.record["defect_type"])
        key = f"generated__{sample.input_name}__{sample.record['sample_id']}"
        relative_path = f"generated/{_safe(key)}.png"
        output_path = root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(sample.mask_path) as mask_handle:
            generated_mask = np.asarray(mask_handle.convert("L")) > 0
        crop_sha, width, height = _save_crop(
            sample.image_path,
            generated_mask,
            output_path,
            ratio=ratio,
        )
        entries.append(
            CropEntry(
                key=key,
                kind="generated",
                object_name=object_name,
                defect_type=defect_type,
                input_name=sample.input_name,
                sample_id=str(sample.record["sample_id"]),
                relative_path=relative_path,
                sha256=crop_sha,
                width=width,
                height=height,
            )
        )

    for index, real_entry in enumerate(real_entries):
        key = f"noise__{real_entry.object_name}__{real_entry.defect_type}__{index:03d}"
        relative_path = f"noise/{_safe(key)}.png"
        output_path = root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        identity = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, identity])))
        noise = rng.integers(
            0,
            256,
            size=(real_entry.height, real_entry.width, 3),
            dtype=np.uint8,
        )
        Image.fromarray(noise, mode="RGB").save(output_path, format="PNG", optimize=True)
        entries.append(
            CropEntry(
                key=key,
                kind="noise",
                object_name=real_entry.object_name,
                defect_type=real_entry.defect_type,
                input_name=None,
                sample_id=real_entry.sample_id,
                relative_path=relative_path,
                sha256=sha256_file(output_path),
                width=real_entry.width,
                height=real_entry.height,
            )
        )

    manifest = {
        "schema_version": 1,
        "pipeline_version": CROP_PIPELINE_VERSION,
        "source_audit_sha256": source_audit_sha256,
        "entries": [
            {
                "key": entry.key,
                "kind": entry.kind,
                "object_name": entry.object_name,
                "defect_type": entry.defect_type,
                "input_name": entry.input_name,
                "sample_id": entry.sample_id,
                "relative_path": entry.relative_path,
                "sha256": entry.sha256,
                "width": entry.width,
                "height": entry.height,
            }
            for entry in entries
        ],
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return entries


def _read_crop_manifest(root: Path, *, source_audit_sha256: str) -> list[CropEntry]:
    manifest_path = root / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest["source_audit_sha256"] != source_audit_sha256:
        raise QualityDataError("Crop cache source audit changed")
    entries = [CropEntry(**record) for record in manifest["entries"]]
    for entry in entries:
        path = root / entry.relative_path
        if not path.is_file() or sha256_file(path) != entry.sha256:
            raise QualityDataError(f"Crop cache payload changed: {path}")
    return entries


def materialize_quality_crops(
    cache_parent: Path,
    cache_key: str,
    paths: Paths,
    samples: Sequence[FilterSample],
    defect_types: Mapping[str, Any],
    *,
    ratio: float,
    seed: int,
    source_audit_sha256: str,
) -> tuple[Path, list[CropEntry]]:
    """Create or verify an immutable crop cache directory."""

    root = cache_parent / f"m14_quality_crops_{cache_key}"
    if root.is_dir():
        return root, _read_crop_manifest(
            root,
            source_audit_sha256=source_audit_sha256,
        )
    temporary = root.with_name(f".{root.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise QualityDataError(f"Temporary crop cache already exists: {temporary}")
    temporary.mkdir(parents=True)
    entries = _materialize_fresh(
        temporary,
        paths,
        samples,
        defect_types,
        ratio=ratio,
        seed=seed,
        source_audit_sha256=source_audit_sha256,
    )
    os.replace(temporary, root)
    return root, entries
