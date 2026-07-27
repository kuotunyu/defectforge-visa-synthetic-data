"""Deterministic M13 rule ordering and rejection decisions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from enum import StrEnum

from src.filtering.metrics import phash_distance


class RejectReason(StrEnum):
    """Locked rejection enum from ``docs/filtering_spec.md``."""

    ROI_OVERFLOW = "ROI_OVERFLOW"
    AREA_OUT_OF_RANGE = "AREA_OUT_OF_RANGE"
    ASPECT_OUT_OF_RANGE = "ASPECT_OUT_OF_RANGE"
    PHASH_DUPLICATE = "PHASH_DUPLICATE"
    NN_TOO_LOW = "NN_TOO_LOW"
    NN_TOO_HIGH_COPY = "NN_TOO_HIGH_COPY"
    EMBEDDING_OUTLIER = "EMBEDDING_OUTLIER"
    SEAM_POOR = "SEAM_POOR"


RULE_ORDER = ("roi", "area", "aspect", "phash", "dinov2", "seam")


def minimum_phash_distance(value: str, accepted: Sequence[str]) -> int | None:
    """Return the nearest previously accepted pHash, or ``None`` for the first."""

    if not accepted:
        return None
    return min(phash_distance(value, other) for other in accepted)


def rejection_reasons(
    scores: Mapping[str, float | int | None],
    *,
    config: Mapping[str, object],
    disabled: set[str] | None = None,
) -> list[str]:
    """Apply locked rules in funnel order and return enum values."""

    disabled = disabled or set()
    unknown = disabled - set(RULE_ORDER)
    if unknown:
        raise ValueError(f"Unknown disabled rules: {sorted(unknown)}")
    rules = config["rules"]
    if not isinstance(rules, Mapping):
        raise TypeError("Filter config is missing rules")
    reasons: list[str] = []

    if "roi" not in disabled:
        roi = _section(rules, "roi")
        if float(scores["roi_containment"]) < float(roi["minimum_containment"]):
            reasons.append(RejectReason.ROI_OVERFLOW)

    if "area" not in disabled:
        area = _section(rules, "area")
        area_zscore = float(scores["area_zscore"])
        outside_quantiles = not (
            float(scores["area_p05"])
            <= float(scores["area_ratio"])
            <= float(scores["area_p95"])
        )
        if abs(area_zscore) > float(area["maximum_abs_zscore"]) or (
            bool(area["enforce_real_quantiles"]) and outside_quantiles
        ):
            reasons.append(RejectReason.AREA_OUT_OF_RANGE)

    if "aspect" not in disabled:
        aspect = _section(rules, "aspect")
        if abs(float(scores["aspect_zscore"])) > float(aspect["maximum_abs_zscore"]):
            reasons.append(RejectReason.ASPECT_OUT_OF_RANGE)

    if "phash" not in disabled:
        phash = _section(rules, "phash")
        distance = scores["phash_min_dist"]
        if distance is not None and int(distance) < int(phash["minimum_hamming_distance"]):
            reasons.append(RejectReason.PHASH_DUPLICATE)

    if "dinov2" not in disabled:
        dinov2 = _section(rules, "dinov2")
        nn_score = float(scores["nn_score"])
        if nn_score < float(scores["tau_low"]):
            reasons.append(RejectReason.NN_TOO_LOW)
        if nn_score >= float(dinov2["tau_copy"]):
            reasons.append(RejectReason.NN_TOO_HIGH_COPY)
        if float(scores["outlier_score"]) > float(scores["tau_outlier"]):
            reasons.append(RejectReason.EMBEDDING_OUTLIER)

    if "seam" not in disabled:
        seam = _section(rules, "seam")
        if float(scores["seam_score"]) < float(seam["minimum_score"]):
            reasons.append(RejectReason.SEAM_POOR)

    if any(not math.isfinite(float(value)) for value in scores.values() if value is not None):
        raise ValueError("Filter scores contain a non-finite value")
    return [str(reason) for reason in reasons]


def _section(rules: Mapping[str, object], name: str) -> Mapping[str, object]:
    section = rules.get(name)
    if not isinstance(section, Mapping):
        raise TypeError(f"Filter config is missing rules.{name}")
    return section
