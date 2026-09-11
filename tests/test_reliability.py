import math
import pytest
from confidence_escalation import ConfidenceScore, ScoringMethod, MultiSignalConfidenceScorer, ConfidenceScorer, ThresholdPolicy, ConfidenceEscalationMiddleware

@pytest.mark.parametrize('value', [math.nan, math.inf, -0.1, 1.1])
def test_invalid_confidence_and_threshold(value):
    with pytest.raises(ValueError):
        ConfidenceScore(value, ScoringMethod.COMPOSITE)
    with pytest.raises(ValueError):
        ThresholdPolicy(threshold=value)
    with pytest.raises(ValueError):
        MultiSignalConfidenceScorer().score(tool_call_risk=value)

@pytest.mark.parametrize('value', [math.nan, math.inf, 0.1])
def test_invalid_logprob(value):
    with pytest.raises(ValueError):
        ConfidenceScorer().score_from_logprobs([value])

def test_guard_prevents_side_effects_and_missing_evidence():
    effects = []
    gate = ConfidenceEscalationMiddleware(policy=ThresholdPolicy(threshold=0.6))
    for score in [ConfidenceScore(0.2, ScoringMethod.COMPOSITE), gate.score(), gate.score(response_text='hello')]:
        with pytest.raises(PermissionError):
            gate.call_guarded(effects.append, score, 'unsafe')
    assert effects == []
    gate.call_guarded(effects.append, ConfidenceScore(0.8, ScoringMethod.COMPOSITE), 'safe')
    assert effects == ['safe']

def test_empty_logprobs_are_missing_not_neutral_evidence():
    score = MultiSignalConfidenceScorer().score(logprobs=[])
    assert score.metadata['has_evidence'] is False
    assert score.signals == {}
