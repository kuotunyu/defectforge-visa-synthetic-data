from scripts.sample_fewshot import deterministic_sample, stable_json_bytes


def test_deterministic_sample_is_byte_stable() -> None:
    records = [{"image_path": f"image-{index:02d}.png"} for index in range(20)]

    first = deterministic_sample(records, count=10, seed=42)
    second = deterministic_sample(list(reversed(records)), count=10, seed=42)

    assert stable_json_bytes({"selected": first}) == stable_json_bytes({"selected": second})
    assert len({record["image_path"] for record in first}) == 10
