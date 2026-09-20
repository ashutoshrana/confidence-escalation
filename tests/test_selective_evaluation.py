"""Holdout leakage and missing labels must not produce favorable evaluation claims."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("evaluate_dataset", Path(__file__).parents[1] / "benchmarks/evaluate_dataset.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def dataset():
    return {"schema_version": 1, "probability_target": False, "rows": [
        {"id": f"{split}-{i}", "group": f"{split}-{i}", "split": split,
         "score_source": "synthetic test only", "model_version": "none",
         "scores": {"composite": .9, "single_signal": .8}, "tool_risk": .1, "correct": True}
        for split in ("validation", "test") for i in range(40)]}


def test_threshold_selection_does_not_read_test_labels():
    data = dataset()
    first = evaluation.evaluate(data, .1)
    assert first["methods"]["composite"]["selected_threshold"] == .9
    assert first["methods"]["risk_rule"]["selected_threshold"] == .1
    for row in data["rows"]:
        if row["split"] == "test":
            row["correct"] = False
    second = evaluation.evaluate(data, .1)
    for name in first["methods"]:
        assert first["methods"][name]["selected_threshold"] == second["methods"][name]["selected_threshold"]
        assert first["methods"][name]["heldout_budget_supported"]
        assert not second["methods"][name]["heldout_budget_supported"]
    assert second["methods"]["composite"]["heldout"]["accepted_error_rate_labeled"] == 1


def test_unknown_and_zero_coverage_are_not_safety():
    data = dataset()
    data["rows"][-1]["correct"] = None
    result = evaluation.evaluate(data, .1)
    assert result["methods"]["composite"]["heldout"]["accepted_unknown"] == 1
    assert not result["methods"]["composite"]["heldout_budget_supported"]
    assert result["baselines"]["always_abstain"]["accepted_error_interval_labeled_95"] is None
    assert evaluation.evaluate(data, .001)["methods"]["composite"]["selected_threshold"] is None


def test_group_leakage_and_invalid_inputs_rejected():
    data = dataset()
    data["rows"][-1]["group"] = data["rows"][0]["group"]
    with pytest.raises(ValueError, match="groups"):
        evaluation.evaluate(data, .1)
    data = dataset()
    data["rows"][0]["scores"]["composite"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        evaluation.evaluate(data, .1)
    data = dataset()
    data["rows"][0]["correct"] = None
    with pytest.raises(ValueError, match="labeled"):
        evaluation.evaluate(data, .1)


def test_probability_metrics_require_declared_target():
    data = dataset()
    assert "brier_score_labeled" not in evaluation.evaluate(data, .1)["methods"]["composite"]
    data["probability_target"] = True
    result = evaluation.evaluate(data, .1)
    assert result["methods"]["composite"]["brier_score_labeled"] == pytest.approx(.01)
    assert "brier_score_labeled" not in result["methods"]["risk_rule"]


def test_repeated_conversation_is_not_independent_evidence():
    data = dataset()
    for row in data["rows"]:
        row["group"] = row["split"] + "-single-conversation"
    with pytest.raises(ValueError, match="one outcome per independent group"):
        evaluation.evaluate(data, .05)
