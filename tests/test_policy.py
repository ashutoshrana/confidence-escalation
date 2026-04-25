"""Tests for escalation policies."""

import pytest
from confidence_escalation.policy import (
    CompositePolicy,
    EscalationAction,
    PolicyResult,
    ThresholdPolicy,
)
from confidence_escalation.scorer import ConfidenceScore, ScoringMethod


def make_score(value: float) -> ConfidenceScore:
    return ConfidenceScore(value=value, method=ScoringMethod.COMPOSITE)


class TestThresholdPolicy:
    def test_triggered_below_threshold(self):
        policy = ThresholdPolicy(threshold=0.65, action=EscalationAction.HUMAN_IN_LOOP)
        result = policy.evaluate(make_score(0.4))
        assert result.triggered
        assert result.action == EscalationAction.HUMAN_IN_LOOP

    def test_not_triggered_above_threshold(self):
        policy = ThresholdPolicy(threshold=0.65)
        result = policy.evaluate(make_score(0.9))
        assert not result.triggered
        assert result.action == EscalationAction.NONE

    def test_critical_threshold_takes_priority(self):
        policy = ThresholdPolicy(
            threshold=0.65,
            action=EscalationAction.HUMAN_IN_LOOP,
            critical_threshold=0.3,
            critical_action=EscalationAction.ABORT,
        )
        result = policy.evaluate(make_score(0.2))
        assert result.triggered
        assert result.action == EscalationAction.ABORT

    def test_above_critical_but_below_normal_uses_normal_action(self):
        policy = ThresholdPolicy(
            threshold=0.65,
            action=EscalationAction.HUMAN_IN_LOOP,
            critical_threshold=0.3,
            critical_action=EscalationAction.ABORT,
        )
        result = policy.evaluate(make_score(0.45))
        assert result.triggered
        assert result.action == EscalationAction.HUMAN_IN_LOOP

    def test_context_override_tightens_threshold(self):
        policy = ThresholdPolicy(
            threshold=0.65,
            context_overrides={"high_stakes": 0.85},
        )
        low_stakes = policy.evaluate(make_score(0.7), context={"high_stakes": False})
        high_stakes = policy.evaluate(make_score(0.7), context={"high_stakes": True})
        assert not low_stakes.triggered
        assert high_stakes.triggered

    def test_callback_invoked_on_trigger(self):
        called = []
        policy = ThresholdPolicy(
            threshold=0.65,
            on_escalation=lambda r: called.append(r),
        )
        policy.evaluate(make_score(0.4))
        assert len(called) == 1

    def test_callback_not_invoked_when_not_triggered(self):
        called = []
        policy = ThresholdPolicy(
            threshold=0.65,
            on_escalation=lambda r: called.append(r),
        )
        policy.evaluate(make_score(0.9))
        assert len(called) == 0

    def test_policy_result_should_escalate(self):
        result = PolicyResult(triggered=True, action=EscalationAction.HUMAN_IN_LOOP)
        assert result.should_escalate

    def test_policy_result_none_action_no_escalate(self):
        result = PolicyResult(triggered=True, action=EscalationAction.NONE)
        assert not result.should_escalate


class TestCompositePolicy:
    def test_first_triggered_policy_wins(self):
        abort_policy = ThresholdPolicy(threshold=0.3, action=EscalationAction.ABORT)
        hil_policy = ThresholdPolicy(threshold=0.65, action=EscalationAction.HUMAN_IN_LOOP)
        composite = CompositePolicy(policies=[abort_policy, hil_policy])

        result = composite.evaluate(make_score(0.2))
        assert result.action == EscalationAction.ABORT

    def test_second_policy_used_when_first_not_triggered(self):
        abort_policy = ThresholdPolicy(threshold=0.3, action=EscalationAction.ABORT)
        hil_policy = ThresholdPolicy(threshold=0.65, action=EscalationAction.HUMAN_IN_LOOP)
        composite = CompositePolicy(policies=[abort_policy, hil_policy])

        result = composite.evaluate(make_score(0.4))
        assert result.action == EscalationAction.HUMAN_IN_LOOP

    def test_no_policy_triggered(self):
        composite = CompositePolicy(policies=[
            ThresholdPolicy(threshold=0.3, action=EscalationAction.ABORT),
            ThresholdPolicy(threshold=0.5, action=EscalationAction.HUMAN_IN_LOOP),
        ])
        result = composite.evaluate(make_score(0.9))
        assert not result.triggered
