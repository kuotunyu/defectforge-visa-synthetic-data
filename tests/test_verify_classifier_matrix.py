import pytest

from scripts.verify_classifier_matrix import (
    aggregate_replicates,
    expected_plan,
)


def test_expected_plan_has_exact_formal_coverage() -> None:
    plan = expected_plan()

    assert len(plan) == 38
    assert len({run_name for _, _, _, run_name in plan}) == 38
    assert {seed for group, _, seed, _ in plan if group == "real_only"} == {
        42,
        43,
        44,
    }
    assert not {"real_60", "syn_500", "base_sd2"} & {group for group, _, _, _ in plan}


def test_aggregate_replicates_reports_mean_and_sample_std() -> None:
    runs = []
    for group in ("real_only", "filtered_syn"):
        for object_name in ("pcb1", "capsules"):
            for seed, value in zip((42, 43, 44), (0.7, 0.8, 0.9), strict=True):
                runs.append(
                    {
                        "run_name": f"m16_{group}_{object_name}_seed_{seed}",
                        "object": object_name,
                        "seed": seed,
                        "macro_f1": value,
                        "auroc": value + 0.05,
                    }
                )

    result = aggregate_replicates(runs)

    assert len(result) == 4
    assert result["real_only/pcb1"]["macro_f1_mean"] == pytest.approx(0.8)
    assert round(result["filtered_syn/capsules"]["macro_f1_std"], 6) == 0.1
