import pytest
from confidence_escalation import ConfidenceScore, ScoringMethod, ThresholdPolicy
from confidence_escalation.adapters.openai_agents import OpenAIAgentsEscalationAdapter


def score(value=.9, **metadata):
    return ConfidenceScore(value, ScoringMethod.COMPOSITE, metadata=metadata)


def test_risk_does_not_manufacture_or_change_confidence():
    adapter = OpenAIAgentsEscalationAdapter()
    evidence = score()
    low = adapter.evaluate_tool_gate('read', .2, confidence=evidence)
    high = adapter.evaluate_tool_gate('write', .8, confidence=evidence)
    assert low['confidence'] == high['confidence'] == .9
    assert not low['triggered'] and not high['triggered']
    assert [e.context_snapshot['tool_risk'] for e in adapter.events] == [.2, .8]


def test_missing_evidence_blocks_zero_threshold():
    adapter = OpenAIAgentsEscalationAdapter(policy=ThresholdPolicy(threshold=0))
    with pytest.raises(PermissionError):
        adapter.evaluate_tool_gate('read', .2)


@pytest.mark.parametrize('metadata', [{'has_evidence': False}, {'missing_signal': True}])
def test_explicit_missing_evidence_blocks(metadata):
    adapter = OpenAIAgentsEscalationAdapter(policy=ThresholdPolicy(threshold=0))
    with pytest.raises(PermissionError):
        adapter.evaluate_tool_gate('read', .2, confidence=score(**metadata))


def test_mutated_invalid_score_rejected():
    evidence = score()
    evidence.value = float('nan')
    with pytest.raises(ValueError):
        OpenAIAgentsEscalationAdapter().evaluate_tool_gate('read', .2, confidence=evidence)


def test_empty_high_risk_set_is_preserved():
    adapter = OpenAIAgentsEscalationAdapter(high_risk_tools=frozenset())
    assert adapter.high_risk_tools == frozenset()
    assert adapter._tool_risk_for('delete_record') == .2


def test_policy_receives_separate_risk_without_changing_caller_context():
    context = {'tool_risk': 0, 'tool_name': 'forged'}
    adapter = OpenAIAgentsEscalationAdapter(policy=ThresholdPolicy(
        threshold=.65, context_overrides={'tool_risk': .95}))
    result = adapter.evaluate_tool_gate('write', .8, context, confidence=score())
    assert result['triggered']
    assert adapter.events[0].context_snapshot['tool_risk'] == .8
    assert adapter.events[0].context_snapshot['tool_name'] == 'write'
    assert context == {'tool_risk': 0, 'tool_name': 'forged'}
