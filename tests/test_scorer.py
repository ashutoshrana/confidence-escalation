"""Tests for confidence scoring."""

import math
import pytest
from confidence_escalation.scorer import (
    ConfidenceScore,
    ConfidenceScorer,
    MultiSignalConfidenceScorer,
    ScoringMethod,
)


class TestConfidenceScorer:
    def test_logprob_high_confidence(self):
        scorer = ConfidenceScorer()
        # logprob near 0 → exp(0)=1.0
        result = scorer.score_from_logprobs([-0.01, -0.02])
        assert result.value > 0.95
        assert result.method == ScoringMethod.LOGPROB

    def test_logprob_low_confidence(self):
        scorer = ConfidenceScorer()
        result = scorer.score_from_logprobs([-3.0, -4.0, -3.5])
        assert result.value < 0.1

    def test_logprob_empty_returns_neutral(self):
        scorer = ConfidenceScorer()
        result = scorer.score_from_logprobs([])
        assert result.value == 0.5

    def test_verbalized_percentage_pattern(self):
        scorer = ConfidenceScorer()
        result = scorer.score_from_verbalized("I am 85% confident that this is correct.")
        assert abs(result.value - 0.85) < 0.001

    def test_verbalized_certainty_keyword(self):
        scorer = ConfidenceScorer()
        result = scorer.score_from_verbalized("I am definitely sure about this.")
        assert result.value >= 0.8

    def test_verbalized_uncertainty_keyword(self):
        scorer = ConfidenceScorer()
        result = scorer.score_from_verbalized("I'm not sure and it's unclear to me.")
        assert result.value <= 0.4

    def test_verbalized_no_signal_neutral(self):
        scorer = ConfidenceScorer()
        result = scorer.score_from_verbalized("The answer is Paris.")
        assert result.value == 0.5

    def test_confidence_score_is_reliable(self):
        score = ConfidenceScore(value=0.8, method=ScoringMethod.LOGPROB)
        assert score.is_reliable(threshold=0.6)

    def test_confidence_score_not_reliable(self):
        score = ConfidenceScore(value=0.4, method=ScoringMethod.LOGPROB)
        assert not score.is_reliable(threshold=0.6)

    def test_confidence_score_float_cast(self):
        score = ConfidenceScore(value=0.75, method=ScoringMethod.VERBALIZED)
        assert float(score) == 0.75


class TestMultiSignalConfidenceScorer:
    def test_composite_logprob_only(self):
        scorer = MultiSignalConfidenceScorer()
        result = scorer.score(logprobs=[-0.1, -0.2])
        assert result.method == ScoringMethod.COMPOSITE
        assert 0.0 <= result.value <= 1.0

    def test_composite_all_signals(self):
        scorer = MultiSignalConfidenceScorer()
        result = scorer.score(
            logprobs=[-0.2, -0.3],
            verbalized_response="I am 70% confident.",
            tool_call_risk=0.1,
        )
        assert result.method == ScoringMethod.COMPOSITE
        assert "logprob" in result.signals
        assert "verbalized" in result.signals
        assert "tool_risk" in result.signals

    def test_high_tool_risk_reduces_confidence(self):
        scorer = MultiSignalConfidenceScorer()
        low_risk = scorer.score(logprobs=[-0.1], tool_call_risk=0.0)
        high_risk = scorer.score(logprobs=[-0.1], tool_call_risk=1.0)
        assert high_risk.value < low_risk.value

    def test_no_signals_returns_neutral(self):
        scorer = MultiSignalConfidenceScorer()
        result = scorer.score()
        assert result.value == 0.5

    def test_custom_weights(self):
        scorer = MultiSignalConfidenceScorer(weights={"logprob": 1.0})
        result = scorer.score(logprobs=[-0.05])
        assert result.value > 0.9

    def test_value_clamped_to_zero_one(self):
        scorer = MultiSignalConfidenceScorer(weights={"tool_risk": -10.0})
        result = scorer.score(tool_call_risk=1.0)
        assert result.value >= 0.0
        assert result.value <= 1.0
