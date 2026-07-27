import json
import os
from pathlib import Path

from PIL import Image

from src.common.paths import Paths
from src.filtering.dataset import (
    filtered_record,
    publish_views,
    read_filter_inputs,
)


def paths(tmp_path: Path) -> Paths:
    root = tmp_path / "project"
    synthetic = tmp_path / "data" / "synthetic"
    root.mkdir()
    synthetic.mkdir(parents=True)
    common = {
        "project_root": root,
        "config_file": root / "configs" / "paths.yaml",
        "data_root": tmp_path / "data",
        "raw": tmp_path / "data" / "raw",
        "visa_tar": tmp_path / "data" / "raw" / "visa.tar",
        "visa_raw": tmp_path / "data" / "raw" / "visa",
        "visa_fewshot": tmp_path / "data" / "few",
        "visa_highshot": tmp_path / "data" / "high",
        "synthetic": synthetic,
        "runs": tmp_path / "data" / "runs",
        "cache": tmp_path / "data" / "cache",
        "splits": root / "splits",
        "reports": root / "reports",
        "figures": root / "reports" / "figures",
        "configs": root / "configs",
        "notebooks": root / "notebooks",
        "colab_results": root / "results" / "colab",
        "dotenv": tmp_path / ".env",
        "hf_home": None,
        "objects": ("pcb1", "capsules"),
        "seed": 42,
    }
    return Paths(**common)


def metadata(sample_id: str = "sample") -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "object": "pcb1",
        "defect_type": "type0",
        "trigger_token": "<pcb1-type0>",
        "generator": "stageA_copypaste",
        "bucket": None,
        "image_path": "images/sample.png",
        "mask_path": "masks/sample.png",
        "source": {
            "background_image": "pcb1/Data/Images/Normal/001.JPG",
            "background_sha256": "0" * 64,
            "defect_source_image": None,
            "defect_source_mask": None,
            "defect_source_component_id": None,
        },
        "placement": {
            "roi_bbox": [0, 0, 8, 8],
            "mask_bbox": [2, 2, 2, 2],
            "affine": {
                "dx": 2,
                "dy": 2,
                "rotation_deg": 0.0,
                "scale": 1.0,
                "flip": False,
            },
            "mask_area_px": 4,
            "mask_area_ratio": 0.0625,
        },
        "generation": {
            "seed": 42,
            "base_model": None,
            "lora_path": None,
            "prompt": None,
            "negative_prompt": None,
            "guidance_scale": None,
            "num_inference_steps": None,
            "strength": None,
            "crop_ratio": None,
            "crop_bbox": None,
            "model_resolution": None,
            "blend": "alpha",
        },
        "filter": None,
        "pipeline_version": "0.1.0",
        "created_at": "2026-07-27T00:00:00+00:00",
    }


def write_input(paths_value: Paths) -> None:
    root = paths_value.synthetic / "source"
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    Image.new("RGB", (8, 8)).save(root / "images" / "sample.png")
    Image.new("L", (8, 8), 255).save(root / "masks" / "sample.png")
    (root / "metadata.jsonl").write_text(
        json.dumps(metadata()) + "\n",
        encoding="utf-8",
    )


def test_read_filter_inputs_and_publish_hardlinked_views(tmp_path: Path) -> None:
    paths_value = paths(tmp_path)
    write_input(paths_value)
    samples = read_filter_inputs(paths_value, ["source"])
    assert len(samples) == 1
    record = filtered_record(
        samples[0],
        scores={"roi_containment": 1.0},
        reject_reasons=[],
        thresholds={"minimum_containment": 1.0},
        pipeline_version="0.1.0",
    )

    filtered_root, unfiltered_root = publish_views(
        paths_value,
        samples,
        [record],
        filtered_name="filtered",
        unfiltered_name="unfiltered",
        link_mode="hardlink",
    )

    filtered_image = filtered_root / record["image_path"]
    unfiltered_image = unfiltered_root / record["image_path"]
    assert os.path.samefile(samples[0].image_path, filtered_image)
    assert os.path.samefile(samples[0].image_path, unfiltered_image)
    assert len((filtered_root / "metadata.jsonl").read_text().splitlines()) == 1
    assert len((unfiltered_root / "metadata.jsonl").read_text().splitlines()) == 1


def test_rejected_record_is_only_in_unfiltered_view(tmp_path: Path) -> None:
    paths_value = paths(tmp_path)
    write_input(paths_value)
    sample = read_filter_inputs(paths_value, ["source"])[0]
    rejected = filtered_record(
        sample,
        scores={"roi_containment": 0.9},
        reject_reasons=["ROI_OVERFLOW"],
        thresholds={"minimum_containment": 1.0},
        pipeline_version="0.1.0",
    )
    filtered_root, unfiltered_root = publish_views(
        paths_value,
        [sample],
        [rejected],
        filtered_name="filtered",
        unfiltered_name="unfiltered",
        link_mode="hardlink",
    )
    assert not (filtered_root / rejected["image_path"]).exists()
    assert (filtered_root / "metadata.jsonl").read_text(encoding="utf-8") == ""
    assert (unfiltered_root / rejected["image_path"]).is_file()
