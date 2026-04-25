"""Tests for ConfidenceEscalationMiddleware."""

import pytest
from confidence_escalation.middleware import ConfidenceEscalationMiddleware, EscalationEvent
from confidence_escalation.policy import ThresholdPolicy, EscalationAction
from confidence_escalation.handlers import ComplianceLoggingHandler
from confidence_escalation.scorer import MultiSignalConfidenceScorer


class TestConfidenceEscalationMiddleware:
    def test_call_returns_response_and_escalation(self):
        middleware = ConfidenceEscalationMiddleware()
        result = middleware.call(
            agent_step=lambda: "I am 90% confident the answer is correct.",
        )
        assert "response" in result
        assert "confidence" in result
        assert "escalation" in result

    def test_escalation_not_triggered_high_confidence(self):
        policy = ThresholdPolicy(threshold=0.65)
        middleware = ConfidenceEscalationMiddleware(policy=policy)
        result = middleware.call(
            agent_step=lambda: "I am 95% confident.",
        )
        assert not result["escalation"]["triggered"]

    def test_escalation_triggered_low_confidence(self):
        logged = []
        policy = ThresholdPolicy(threshold=0.65, action=EscalationAction.HUMAN_IN_LOOP)
        handlers = [ComplianceLoggingHandler(log_sink=lambda m: logged.append(m))]
        middleware = ConfidenceEscalationMiddleware(policy=policy, handlers=handlers)
        result = middleware.call(
            agent_step=lambda: "I am not sure and it's unclear.",
        )
        assert result["escalation"]["triggered"]

    def test_events_accumulate(self):
        middleware = ConfidenceEscalationMiddleware()
        middleware.call(agent_step=lambda: "Answer 1.")
        middleware.call(agent_step=lambda: "Answer 2.")
        assert len(middleware.events) == 2

    def test_event_sink_called(self):
        events = []
        middleware = ConfidenceEscalationMiddleware(event_sink=lambda e: events.append(e))
        middleware.call(agent_step=lambda: "Some response.")
        assert len(events) == 1
        assert isinstance(events[0], EscalationEvent)

    def test_score_with_logprobs(self):
        middleware = ConfidenceEscalationMiddleware()
        score = middleware.score(logprobs=[-0.1, -0.2])
        assert 0.0 <= score.value <= 1.0

    def test_event_to_dict_serializable(self):
        middleware = ConfidenceEscalationMiddleware()
        middleware.call(agent_step=lambda: "Test response.")
        event_dict = middleware.events[0].to_dict()
        assert "timestamp" in event_dict
        assert "triggered" in event_dict
        assert "confidence_score" in event_dict
        assert "action" in event_dict

    def test_no_handlers_still_runs(self):
        middleware = ConfidenceEscalationMiddleware(handlers=[])
        result = middleware.call(agent_step=lambda: "I am not sure about this.")
        assert "response" in result
