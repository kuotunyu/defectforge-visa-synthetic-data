from scripts.freeze_manifest import build_groups


def test_phash_groups_use_transitive_closure() -> None:
    # 0 differs from 1 by one bit; 1 differs from 3 by one bit; 0 and 3 differ by two.
    groups, nearest, edge_count = build_groups([0, 1, 3], threshold=1)

    assert groups == [0, 0, 0]
    assert nearest == [1, 1, 1]
    assert edge_count == 2
