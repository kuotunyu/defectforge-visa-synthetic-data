from scripts.verify_classifier_tuning import choose_candidate


def test_choose_candidate_uses_macro_f1_before_auroc() -> None:
    rows = [
        {
            "learning_rate": 1e-5,
            "mean_macro_f1": 0.8,
            "mean_auroc": 0.9,
        },
        {
            "learning_rate": 3e-5,
            "mean_macro_f1": 0.79,
            "mean_auroc": 0.99,
        },
    ]
    assert choose_candidate(rows)["learning_rate"] == 1e-5


def test_choose_candidate_prefers_lower_lr_after_metric_tie() -> None:
    rows = [
        {
            "learning_rate": 1e-5,
            "mean_macro_f1": 0.8,
            "mean_auroc": 0.9,
        },
        {
            "learning_rate": 3e-5,
            "mean_macro_f1": 0.8,
            "mean_auroc": 0.9,
        },
    ]
    assert choose_candidate(rows)["learning_rate"] == 1e-5
