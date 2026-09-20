"""Evaluate labeled outcomes without fitting or claiming calibrated confidence.

Stdlib only. Select thresholds using validation groups, then report test groups.
Unknown test outcomes remain unknown and cannot establish the error budget.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def probability(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("scores and risks must be numbers")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("scores and risks must be finite in [0, 1]")
    return value


def validate(dataset):
    if dataset.get("schema_version") != 1 or not isinstance(dataset.get("rows"), list):
        raise ValueError("schema_version 1 and rows list required")
    if not isinstance(dataset.get("probability_target"), bool):
        raise ValueError("declare probability_target true or false")
    ids, groups, methods, splits = set(), {}, None, set()
    for row in dataset["rows"]:
        for key in ("id", "group", "score_source", "model_version"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError("nonempty id/group/score_source/model_version required")
        if row["id"] in ids:
            raise ValueError("duplicate task ID")
        ids.add(row["id"])
        split = row.get("split")
        if split not in ("validation", "test"):
            raise ValueError("split must be validation or test")
        splits.add(split)
        if groups.setdefault(row["group"], split) != split:
            raise ValueError("related groups cannot cross validation/test splits")
        if "correct" not in row or (row["correct"] is not None and type(row["correct"]) is not bool):
            raise ValueError("correct must be boolean or null")
        if split == "validation" and row["correct"] is None:
            raise ValueError("validation outcomes must be labeled before threshold selection")
        scores = row.get("scores")
        if not isinstance(scores, dict) or not scores or any(not isinstance(k, str) or not k for k in scores):
            raise ValueError("nonempty named scores required")
        if "risk_rule" in scores:
            raise ValueError("risk_rule is reserved for the independent risk baseline")
        if not isinstance(row.get("slice", "unspecified"), str) or not row.get("slice", "unspecified"):
            raise ValueError("slice must be a nonempty string")
        if methods is None:
            methods = set(scores)
        if set(scores) != methods:
            raise ValueError("all rows must carry the same score methods")
        for value in scores.values():
            probability(value)
        probability(row.get("tool_risk"))
    if splits != {"validation", "test"}:
        raise ValueError("nonempty validation and test splits required")
    return sorted(methods)


def wilson(errors, count):
    """Two-sided 95% Wilson interval; no trials is unknown, not zero risk."""
    if not count:
        return None
    z = 1.959963984540054
    p = errors / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def metrics(rows, accepts):
    accepted = [r for r in rows if accepts(r)]
    labeled = [r for r in accepted if r["correct"] is not None]
    errors = sum(not r["correct"] for r in labeled)
    unknown = len(accepted) - len(labeled)
    return {
        "eligible": len(rows), "accepted": len(accepted),
        "coverage": len(accepted) / len(rows) if rows else None,
        "review_rate": 1 - len(accepted) / len(rows) if rows else None,
        "accepted_unknown": unknown, "accepted_labeled": len(labeled),
        "accepted_errors": errors,
        "accepted_error_rate_labeled": errors / len(labeled) if labeled else None,
        "accepted_error_interval_labeled_95": wilson(errors, len(labeled)),
        "observed_errors_per_eligible": errors / len(rows) if rows else None,
        "error_budget_assessable": bool(labeled) and unknown == 0,
    }


def evaluate(dataset, error_budget):
    methods = validate(dataset)
    probability(error_budget)
    validation = [r for r in dataset["rows"] if r["split"] == "validation"]
    test = [r for r in dataset["rows"] if r["split"] == "test"]
    result = {"schema_version": 1, "error_budget": error_budget,
              "selection": "validation only; nonzero coverage and Wilson upper bound within budget",
              "validation_count": len(validation), "test_count": len(test),
              "baselines": {"always_proceed": metrics(test, lambda r: True),
                            "always_abstain": metrics(test, lambda r: False)}, "methods": {}}
    # Risk is an independent rule baseline, never interpreted as correctness probability.
    for name in methods + ["risk_rule"]:
        if name == "risk_rule":
            value = lambda r: r["tool_risk"]
            accepts = lambda threshold: lambda r: value(r) <= threshold
        else:
            value = lambda r: r["scores"][name]
            accepts = lambda threshold: lambda r: value(r) >= threshold
        curve = []
        for threshold in sorted({0.0, 1.0} | {value(r) for r in validation}):
            curve.append({"threshold": threshold, **metrics(validation, accepts(threshold))})
        candidates = [r for r in curve if r["accepted"] and r["accepted_error_interval_labeled_95"][1] <= error_budget]
        # Prefer the more conservative boundary when validation coverage ties.
        chosen = max(candidates, key=lambda r: (r["coverage"], -r["threshold"] if name == "risk_rule" else r["threshold"])) if candidates else None
        entry = {"validation_curve": curve, "selected_threshold": chosen["threshold"] if chosen else None,
                 "status": "selected" if chosen else "no_nonzero_coverage_threshold_meets_validation_budget",
                 "heldout": metrics(test, accepts(chosen["threshold"])) if chosen else None}
        if chosen:
            heldout = entry["heldout"]
            entry["heldout_budget_supported"] = bool(heldout["error_budget_assessable"] and
                heldout["accepted_error_interval_labeled_95"][1] <= error_budget)
            entry["slices"] = {slice_name: metrics([r for r in test if r.get("slice", "unspecified") == slice_name],
                                                    accepts(chosen["threshold"]))
                               for slice_name in sorted({r.get("slice", "unspecified") for r in test})}
        if dataset["probability_target"] and name != "risk_rule":
            labeled = [r for r in test if r["correct"] is not None]
            entry["brier_score_labeled"] = (sum((value(r) - int(r["correct"])) ** 2 for r in labeled) / len(labeled)
                                              if labeled else None)
            entry["reliability_bins"] = []
            for index in range(10):
                bucket = [r for r in labeled if min(9, int(value(r) * 10)) == index]
                if bucket:
                    entry["reliability_bins"].append({"lower": index / 10, "upper": (index + 1) / 10,
                        "count": len(bucket), "mean_score": sum(value(r) for r in bucket) / len(bucket),
                        "observed_accuracy": sum(r["correct"] for r in bucket) / len(bucket)})
        result["methods"][name] = entry
    result["limitations"] = ["No calibrator fitted; scores and labels are supplied by the evaluator.",
                              "Intervals do not prove rare-event safety or account for residual dependence.",
                              "Unknown outcomes are not successes; repeated test-set tuning invalidates holdout use."]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--error-budget", type=float, required=True)
    args = parser.parse_args()
    raw = args.dataset.read_bytes()
    report = evaluate(json.loads(raw), args.error_budget)
    report["dataset_sha256"] = hashlib.sha256(raw).hexdigest()
    print(json.dumps(report, indent=2, allow_nan=False))
