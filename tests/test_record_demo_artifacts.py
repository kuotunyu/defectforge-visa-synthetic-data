from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from scripts.record_demo_artifacts import (
    FRAME_SIZE,
    build_demo_frame,
    record_object,
    write_animated_gif,
)


class FakeRuntime:
    object_name = "pcb1"

    def predict(
        self,
        image: Image.Image | np.ndarray | None,
    ) -> tuple[dict[str, float], np.ndarray, np.ndarray, str]:
        assert image is not None
        rgb = np.asarray(image)
        height, width = rgb.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[height // 4 : height // 2, width // 4 : width // 2] = 255
        heatmap = np.full((height, width, 3), (16, 103, 128), dtype=np.uint8)
        return (
            {"Defect": 0.75, "Normal": 0.25},
            mask,
            heatmap,
            "**12.3 ms** end-to-end  ·  mask coverage **6.25%**",
        )


def test_build_demo_frame_has_fixed_release_size() -> None:
    image = np.full((80, 120, 3), 90, dtype=np.uint8)
    mask = np.zeros((80, 120), dtype=np.uint8)
    heatmap = np.full((80, 120, 3), 120, dtype=np.uint8)

    frame = build_demo_frame(
        image,
        mask=mask,
        heatmap=heatmap,
        object_name="pcb1",
        label="bad",
        probabilities={"Defect": 0.75, "Normal": 0.25},
        latency="12.3 ms",
    )

    assert frame.mode == "RGB"
    assert frame.size == FRAME_SIZE


def test_record_object_and_gif_preserve_provenance(tmp_path: Path) -> None:
    image_path = tmp_path / "test.png"
    Image.new("RGB", (120, 80), "#505050").save(image_path)
    samples = [
        {
            "object": "pcb1",
            "label": label,
            "relative_path": f"pcb1/{label}.png",
            "image_path": image_path,
            "image_sha256": f"image-{label}",
            "manifest_sha256": "manifest",
        }
        for label in ("good", "bad")
    ]

    frames, validations = record_object(FakeRuntime(), samples)
    gif_path = tmp_path / "demo.gif"
    write_animated_gif(gif_path, frames, duration_ms=10)

    assert len(frames) == 2
    assert [item["label"] for item in validations] == ["good", "bad"]
    assert all(item["mask_coverage"] > 0 for item in validations)
    with Image.open(gif_path) as gif:
        assert gif.n_frames == 2
        assert gif.size == FRAME_SIZE
