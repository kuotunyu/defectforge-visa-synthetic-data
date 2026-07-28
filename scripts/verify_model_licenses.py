"""Recheck pinned upstream model visibility, revision, and license metadata on HF."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.integrity import write_text_lf

MODEL_RULES = {
    "sd2-community/stable-diffusion-2-inpainting": {
        "revision": "5f74973cbb64c8568780732c17f43eb269d63a0d",
        "license": "openrail++",
        "role": "SD2 inpainting base",
    },
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1": {
        "revision": "115134f363124c53c7d878647567d04daf26e41e",
        "license": "openrail++",
        "role": "SDXL inpainting base",
    },
    "facebook/dinov2-base": {
        "revision": "f9e44c814b77203eaa57a6bdbbd535f21ede1415",
        "license": "apache-2.0",
        "role": "quality and filtering encoder",
    },
    "nvidia/segformer-b0-finetuned-ade-512-512": {
        "revision": "489d5cd81a0b59fab9b7ea758d3548ebe99677da",
        "license": "other",
        "role": "segmentation initialization",
    },
}


class ModelLicenseVerificationError(RuntimeError):
    """Raised when upstream HF metadata no longer matches the publication lock."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ModelLicenseVerificationError(message)


def validate_model_metadata(
    model_id: str,
    *,
    observed: Mapping[str, Any],
    expected: Mapping[str, str],
) -> dict[str, Any]:
    require(not observed.get("private"), f"Upstream model became private: {model_id}")
    require(not observed.get("gated"), f"Upstream model became gated: {model_id}")
    require(not observed.get("disabled"), f"Upstream model was disabled: {model_id}")
    require(
        observed.get("revision") == expected["revision"],
        f"Upstream HEAD changed from the pinned revision: {model_id}",
    )
    require(
        observed.get("license") == expected["license"],
        f"Upstream license metadata changed: {model_id}",
    )
    return {
        "role": expected["role"],
        "revision": observed["revision"],
        "license": observed["license"],
        "private": False,
        "gated": False,
        "disabled": False,
        "url": f"https://huggingface.co/{model_id}",
    }


def fetch_metadata(api: HfApi, model_id: str) -> dict[str, Any]:
    info = api.model_info(model_id, files_metadata=False)
    card = info.card_data.to_dict() if info.card_data is not None else {}
    return {
        "revision": info.sha,
        "license": card.get("license"),
        "private": info.private,
        "gated": info.gated,
        "disabled": info.disabled,
    }


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_text_lf(
        temporary,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/model_license_verification.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    api = HfApi()
    verified = {
        model_id: validate_model_metadata(
            model_id,
            observed=fetch_metadata(api, model_id),
            expected=expected,
        )
        for model_id, expected in MODEL_RULES.items()
    }
    payload = {
        "status": "passed",
        "schema_version": 1,
        "verified_at": datetime.now(UTC).isoformat(),
        "source": "Hugging Face Hub model_info API",
        "models": verified,
    }
    atomic_write_json(args.output, payload)
    print(f"Verified upstream model metadata: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
