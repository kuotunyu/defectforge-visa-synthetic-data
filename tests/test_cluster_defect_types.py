import numpy as np

from scripts.cluster_defect_types import (
    ComponentSample,
    choose_clustering,
    square_context_bbox,
)


def _sample(index: int) -> ComponentSample:
    from PIL import Image

    return ComponentSample(
        object_name="pcb1",
        image_path=f"pcb1/{index:02d}.jpg",
        mask_path=f"pcb1/{index:02d}.png",
        component_id=0,
        bbox=(1, 1, 2, 2),
        crop_bbox=(0, 0, 4, 4),
        area_px=4,
        morphology=[0.0] * 8,
        crop=Image.new("RGB", (4, 4)),
        crop_mask=np.ones((4, 4), dtype=bool),
    )


def test_square_context_bbox_stays_inside_image() -> None:
    assert square_context_bbox(
        (95, 75, 5, 5),
        image_width=100,
        image_height=80,
        minimum_side=20,
    ) == (80, 60, 100, 80)


def test_choose_clustering_respects_minimum_cluster_size() -> None:
    features = np.asarray(
        [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [5.0, 5.0], [5.1, 5.0], [5.0, 5.1]]
    )
    samples = [_sample(index) for index in range(6)]

    labels, candidates, fallback = choose_clustering(
        features,
        samples,
        k_min=1,
        k_max=3,
        min_cluster_size=3,
    )

    assert not fallback
    assert sorted(np.bincount(labels)) == [3, 3]
    assert any(candidate["k"] == 2 and candidate["eligible"] for candidate in candidates)
