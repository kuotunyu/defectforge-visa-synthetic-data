from pathlib import Path

from src.evaluation.quality_reporting import (
    embedded_summary,
    render_markdown,
    summary_sha256,
    write_csv,
)


def summary() -> dict[str, object]:
    return {
        "schema_version": 1,
        "pipeline_version": "0.1.0",
        "sanity_passed": True,
        "source_audit": {
            "paths_checked": 2,
            "hashes_checked": 2,
            "blocklist_hits": 0,
            "sha256": "a" * 64,
        },
        "model_revision": "revision",
        "clean_fid_version": "0.1.35",
        "metric_policy": {
            "formal_kid": "unbiased_degree3_polynomial_mmd",
            "sanity_kid": "biased_degree3_polynomial_mmd",
            "fid": "clean_fid_features_exact_low_rank",
        },
        "caches": {
            "generated_dino": {"path": "generated.npz", "sha256": "b" * 64},
            "reference_dino": {"path": "reference.npz", "sha256": "c" * 64},
            "clean_features": {"path": "clean.npz", "sha256": "d" * 64},
        },
        "sanity": [
            {
                "object": "pcb1",
                "defect_type": "type0",
                "n": 3,
                "self_nn_min": 1.0,
                "self_mnn": 1.0,
                "self_kid": 0.0,
                "self_fid": 0.0,
                "noise_nn_mean": 0.1,
                "tau_low": 0.7,
                "noise_kid": 1.0,
                "noise_fid": 100.0,
                "passed": True,
            }
        ],
        "rows": [
            {
                "view": "filtered",
                "input_name": "source",
                "object": "pcb1",
                "defect_type": "type0",
                "real_scope": "type:type0",
                "n_real": 3,
                "n_generated": 4,
                "status": "ok",
                "nn_mean": 0.8,
                "nn_median": 0.81,
                "nn_p05": 0.7,
                "nn_p95": 0.9,
                "mnn_score": 2 / 3,
                "kid": 0.1,
                "fid": 2.0,
            }
        ],
    }


def test_quality_summary_roundtrips_through_markdown() -> None:
    value = summary()
    markdown = render_markdown(value)
    assert embedded_summary(markdown) == value
    assert len(summary_sha256(value)) == 64


def test_csv_preserves_locked_columns_and_nulls(tmp_path: Path) -> None:
    row = dict(summary()["rows"][0])
    row["status"] = "empty"
    for field in (
        "nn_mean",
        "nn_median",
        "nn_p05",
        "nn_p95",
        "mnn_score",
        "kid",
        "fid",
    ):
        row[field] = None
    path = tmp_path / "quality.csv"
    write_csv(path, [row])
    text = path.read_text(encoding="utf-8")
    assert text.startswith("view,input_name,object,defect_type")
    assert "None" not in text
