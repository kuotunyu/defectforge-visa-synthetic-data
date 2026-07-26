from pathlib import Path

from src.common.paths import load_paths


def test_load_paths_resolves_data_and_project_paths() -> None:
    paths = load_paths()

    assert paths.data_root == Path("D:/sdg-data/01-defectforge")
    assert paths.visa_tar == paths.data_root / "raw" / "VisA_20220922.tar"
    assert paths.splits == paths.project_root / "splits"
    assert paths.objects == ("pcb1", "capsules")
    assert paths.seed == 42
