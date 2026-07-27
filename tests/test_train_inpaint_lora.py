import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from src.training.train_inpaint_lora import (
    TrainingSample,
    checkpoint_candidates,
    encode_text_conditioning,
    heldout_placement,
    output_directory,
    sample_tensors,
    square_context_bbox,
    step_sample_indices,
)


class DummyTextEncoder(torch.nn.Module):
    def __init__(self, width: int, *, projected: bool = False) -> None:
        super().__init__()
        self.width = width
        self.projected = projected

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        output_hidden_states: bool,
    ):
        del attention_mask
        batch, sequence = input_ids.shape
        hidden = torch.ones((batch, sequence, self.width))
        if not output_hidden_states:
            return (hidden,)
        return SimpleNamespace(
            hidden_states=(hidden * 0, hidden * 2, hidden * 3),
            text_embeds=(
                torch.full((batch, self.width), 4.0) if self.projected else None
            ),
        )


def test_step_schedule_is_reproducible_and_covers_each_epoch() -> None:
    first = [step_sample_indices(7, 1, step, seed=42, object_name="pcb1")[0] for step in range(14)]
    second = [step_sample_indices(7, 1, step, seed=42, object_name="pcb1")[0] for step in range(14)]

    assert first == second
    assert sorted(first[:7]) == list(range(7))
    assert sorted(first[7:]) == list(range(7))
    assert first[:7] != first[7:]


def test_component_crop_resize_and_flip_are_aligned(tmp_path: Path) -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[:, :10] = (255, 0, 0)
    image[:, 10:] = (0, 255, 0)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[6:14, 3:8] = 255
    Image.fromarray(image).save(tmp_path / "image.png")
    Image.fromarray(mask).save(tmp_path / "mask.png")
    sample = TrainingSample(
        object_name="pcb1",
        cluster_id=0,
        trigger_token="<pcb1-type0>",
        image_path="image.png",
        mask_path="mask.png",
        component_id=0,
        area_px=40,
        crop_bbox=(0, 0, 20, 20),
    )
    paths = SimpleNamespace(visa_raw=tmp_path)

    image_tensor, mask_tensor, masked_tensor = sample_tensors(
        paths,
        sample,
        resolution=32,
        flip=True,
    )

    assert image_tensor.shape == (3, 32, 32)
    assert mask_tensor.shape == (1, 32, 32)
    assert masked_tensor.shape == (3, 32, 32)
    assert torch.all(masked_tensor[mask_tensor.expand_as(masked_tensor) > 0] == 0)
    assert mask_tensor[:, :, 16:].any()
    assert not mask_tensor[:, :, :8].any()


def test_square_context_bbox_is_square_and_in_bounds() -> None:
    mask = np.zeros((70, 100), dtype=bool)
    mask[2:12, 90:99] = True

    x0, y0, x1, y1 = square_context_bbox(mask, crop_ratio=3.0)

    assert x1 - x0 == y1 - y0
    assert 0 <= x0 < x1 <= 100
    assert 0 <= y0 < y1 <= 70
    assert mask[y0:y1, x0:x1].sum() == mask.sum()


def test_smoke_output_is_separate_and_checkpoint_scan_is_fail_closed(
    tmp_path: Path,
) -> None:
    paths = SimpleNamespace(runs=tmp_path)
    config = {"output": {"name": "lora_sd2"}}
    formal = output_directory(
        paths,
        config,
        object_name="pcb1",
        seed=42,
        smoke=False,
        override=None,
    )
    smoke = output_directory(
        paths,
        config,
        object_name="pcb1",
        seed=42,
        smoke=True,
        override=None,
    )
    incomplete = smoke / "checkpoint-000001"
    incomplete.mkdir(parents=True)

    assert formal != smoke
    assert formal.parts[-3:] == ("lora_sd2", "pcb1", "seed_42")
    assert smoke.parts[-3:] == ("lora_sd2_smoke", "pcb1", "seed_42")
    assert checkpoint_candidates(smoke) == []


def test_heldout_placement_selects_requested_trigger_token(tmp_path: Path) -> None:
    placements = tmp_path / "synthetic" / "placements" / "pcb1"
    raw = tmp_path / "raw"
    placements.mkdir(parents=True)
    raw.mkdir()
    Image.new("RGB", (8, 8), "blue").save(raw / "background.png")
    for index in range(2):
        Image.new("L", (8, 8), 255 if index else 0).save(placements / f"mask{index}.png")
    records = [
        {
            "background_image": "background.png",
            "mask_path": "mask0.png",
            "trigger_token": "<pcb1-type0>",
        },
        {
            "background_image": "background.png",
            "mask_path": "mask1.png",
            "trigger_token": "<pcb1-type1>",
        },
    ]
    (placements / "placements.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    paths = SimpleNamespace(synthetic=tmp_path / "synthetic", visa_raw=raw)

    _, mask, record = heldout_placement(paths, "pcb1", "<pcb1-type1>")

    assert record["trigger_token"] == "<pcb1-type1>"
    assert np.asarray(mask).all()


def test_sdxl_conditioning_concatenates_dual_encoders_and_time_ids() -> None:
    batch = {
        "input_ids": torch.ones((2, 5), dtype=torch.long),
        "attention_mask": torch.ones((2, 5), dtype=torch.long),
        "input_ids_2": torch.ones((2, 5), dtype=torch.long),
        "attention_mask_2": torch.ones((2, 5), dtype=torch.long),
    }
    hidden, added = encode_text_conditioning(
        DummyTextEncoder(3),
        batch,
        family="sdxl",
        resolution=1024,
        device=torch.device("cpu"),
        text_encoder_2=DummyTextEncoder(4, projected=True),
    )
    assert hidden.shape == (2, 5, 7)
    assert torch.all(hidden[..., :3] == 2)
    assert torch.all(hidden[..., 3:] == 2)
    assert added is not None
    assert added["text_embeds"].shape == (2, 4)
    assert added["time_ids"].tolist() == [
        [1024, 1024, 0, 0, 1024, 1024],
        [1024, 1024, 0, 0, 1024, 1024],
    ]
