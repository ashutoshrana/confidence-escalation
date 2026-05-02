"""Tests for OpenAI Agents SDK confidence-escalation adapter."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from confidence_escalation.adapters.openai_agents import (
    OpenAIAgentsEscalationAdapter,
    OpenAIAgentsHooks,
)
from confidence_escalation.policy import EscalationAction, ThresholdPolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(name: str = "search_documents") -> MagicMock:
    tool = MagicMock()
    tool.name = name
    return tool


def _make_ctx(tool_call_id: str = "tc_001", tool_arguments: dict | None = None) -> MagicMock:
    ctx = MagicMock()
    ctx.tool_call_id = tool_call_id
    ctx.tool_arguments = tool_arguments or {}
    ctx.raw_responses = []
    return ctx


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Adapter initialisation
# ---------------------------------------------------------------------------


class TestOpenAIAgentsAdapterInit:
    def test_default_threshold(self):
        adapter = OpenAIAgentsEscalationAdapter()
        assert adapter.threshold == 0.65

    def test_custom_threshold(self):
        adapter = OpenAIAgentsEscalationAdapter(threshold=0.80)
        assert adapter.threshold == 0.80

    def test_custom_high_risk_tools(self):
        adapter = OpenAIAgentsEscalationAdapter(
            high_risk_tools=frozenset({"my_risky_tool"})
        )
        assert "my_risky_tool" in adapter.high_risk_tools

    def test_as_hooks_returns_hooks_instance(self):
        adapter = OpenAIAgentsEscalationAdapter()
        hooks = adapter.as_hooks()
        assert isinstance(hooks, OpenAIAgentsHooks)


class TestToolRiskScoring:
    def test_default_high_risk_tool(self):
        adapter = OpenAIAgentsEscalationAdapter()
        assert adapter._tool_risk_for("delete_record") == 0.8

    def test_default_low_risk_tool(self):
        adapter = OpenAIAgentsEscalationAdapter()
        assert adapter._tool_risk_for("search_documents") == 0.2

    def test_custom_high_risk_tool(self):
        adapter = OpenAIAgentsEscalationAdapter(
            high_risk_tools=frozenset({"custom_tool"})
        )
        assert adapter._tool_risk_for("custom_tool") == 0.8
        assert adapter._tool_risk_for("delete_record") == 0.2


# ---------------------------------------------------------------------------
# Tool gate evaluation
# ---------------------------------------------------------------------------


class TestEvaluateToolGate:
    def test_not_triggered_when_threshold_zero(self):
        """With threshold=0.0 (always pass), gate should never trigger."""
        from confidence_escalation.policy import ThresholdPolicy, EscalationAction
        adapter = OpenAIAgentsEscalationAdapter(
            policy=ThresholdPolicy(
                threshold=0.0,
                action=EscalationAction.HUMAN_IN_LOOP,
                critical_threshold=None,
            )
        )
        result = adapter.evaluate_tool_gate(
            tool_name="search_documents",
            tool_risk=0.2,
        )
        assert result["triggered"] is False
        assert "confidence" in result

    def test_high_threshold_triggers_escalation(self):
        """With threshold=0.99, any real signal always triggers."""
        adapter = OpenAIAgentsEscalationAdapter(threshold=0.99, critical_threshold=0.0)
        result = adapter.evaluate_tool_gate(
            tool_name="delete_record",
            tool_risk=0.8,
        )
        assert result["triggered"] is True

    def test_context_attached_to_gate_result(self):
        adapter = OpenAIAgentsEscalationAdapter(threshold=0.99, critical_threshold=0.0)
        context = {"session_id": "sess_abc", "regulation": "EU AI Act Art. 14"}
        result = adapter.evaluate_tool_gate(
            tool_name="delete_record",
            tool_risk=0.8,
            context=context,
        )
        assert result["triggered"] is True
        assert result["tool_name"] == "delete_record"

    def test_events_log_after_gate_triggered(self):
        adapter = OpenAIAgentsEscalationAdapter(threshold=0.99, critical_threshold=0.0)
        adapter.evaluate_tool_gate("delete_record", tool_risk=0.8)
        assert len(adapter.events) == 1
        event = adapter.events[0]
        assert event.triggered is True

    def test_events_log_after_gate_not_triggered(self):
        from confidence_escalation.policy import ThresholdPolicy, EscalationAction
        adapter = OpenAIAgentsEscalationAdapter(
            policy=ThresholdPolicy(threshold=0.0, action=EscalationAction.NONE, critical_threshold=None)
        )
        adapter.evaluate_tool_gate("search_docs", tool_risk=0.2)
        # Even non-triggered gate is recorded
        assert len(adapter.events) == 1
        assert adapter.events[0].triggered is False


# ---------------------------------------------------------------------------
# Score response
# ---------------------------------------------------------------------------


class TestScoreResponse:
    def test_high_confidence_response_not_triggered(self):
        from confidence_escalation.policy import ThresholdPolicy, EscalationAction
        adapter = OpenAIAgentsEscalationAdapter(
            policy=ThresholdPolicy(threshold=0.5, action=EscalationAction.HUMAN_IN_LOOP, critical_threshold=None)
        )
        result = adapter.score_response(
            response_text="I am 95% confident the answer is correct.",
        )
        assert result["triggered"] is False
        assert result["confidence"] > 0.5

    def test_uncertain_response_triggers(self):
        adapter = OpenAIAgentsEscalationAdapter(threshold=0.95, critical_threshold=0.0)
        result = adapter.score_response(
            response_text="I am uncertain and not sure about this.",
        )
        assert result["triggered"] is True

    def test_logprobs_used_when_provided(self):
        from confidence_escalation.policy import ThresholdPolicy, EscalationAction
        adapter = OpenAIAgentsEscalationAdapter(
            policy=ThresholdPolicy(threshold=0.0, action=EscalationAction.NONE, critical_threshold=None)
        )
        result = adapter.score_response(
            response_text="",
            logprobs=[-0.05, -0.10, -0.08],
        )
        assert "confidence" in result
        assert result["confidence"] > 0.7  # high logprobs → high confidence


# ---------------------------------------------------------------------------
# Hooks interface (async)
# ---------------------------------------------------------------------------


class TestOpenAIAgentsHooks:
    def test_on_tool_start_zero_threshold_no_raise(self):
        from confidence_escalation.policy import ThresholdPolicy, EscalationAction
        adapter = OpenAIAgentsEscalationAdapter(
            policy=ThresholdPolicy(threshold=0.0, action=EscalationAction.NONE, critical_threshold=None)
        )
        hooks = adapter.as_hooks()
        ctx = _make_ctx()
        tool = _make_tool("search_documents")
        _run(hooks.on_tool_start(ctx, tool))

    def test_on_llm_end_no_error_without_raw_responses(self):
        adapter = OpenAIAgentsEscalationAdapter()
        hooks = adapter.as_hooks()
        ctx = MagicMock()
        ctx.raw_responses = []
        agent = MagicMock()
        agent.name = "test-agent"
        _run(hooks.on_llm_end(ctx, agent))

    def test_on_llm_end_extracts_text_from_raw_response(self):
        from confidence_escalation.policy import ThresholdPolicy, EscalationAction
        adapter = OpenAIAgentsEscalationAdapter(
            policy=ThresholdPolicy(threshold=0.0, action=EscalationAction.NONE, critical_threshold=None)
        )
        hooks = adapter.as_hooks()

        choice = MagicMock()
        choice.message.content = "I am 90% confident this is correct."
        choice.logprobs = None
        raw_response = MagicMock()
        raw_response.choices = [choice]

        ctx = MagicMock()
        ctx.raw_responses = [raw_response]
        agent = MagicMock()
        agent.name = "test-agent"

        _run(hooks.on_llm_end(ctx, agent))
        assert len(adapter.events) == 1

    def test_on_agent_start_end_no_error(self):
        adapter = OpenAIAgentsEscalationAdapter()
        hooks = adapter.as_hooks()
        ctx = MagicMock()
        agent = MagicMock()
        _run(hooks.on_agent_start(ctx, agent))
        _run(hooks.on_agent_end(ctx, agent))

    def test_on_handoff_no_error(self):
        adapter = OpenAIAgentsEscalationAdapter()
        hooks = adapter.as_hooks()
        ctx = MagicMock()
        _run(hooks.on_handoff(ctx, MagicMock(), MagicMock()))

    def test_tool_name_fallback_string_tool(self):
        adapter = OpenAIAgentsEscalationAdapter(threshold=0.99, critical_threshold=0.0)
        hooks = adapter.as_hooks()
        ctx = _make_ctx()
        # Should not raise AttributeError — fallback to str(tool)
        _run(hooks.on_tool_start(ctx, "bare_string_tool"))

    def test_high_threshold_escalation_captured(self):
        captured = []

        def on_esc(result):
            captured.append(result)

        adapter = OpenAIAgentsEscalationAdapter(
            threshold=0.99,
            policy=ThresholdPolicy(
                threshold=0.99,
                action=EscalationAction.HUMAN_IN_LOOP,
                critical_threshold=None,
                on_escalation=on_esc,
            ),
        )
        hooks = adapter.as_hooks()
        ctx = _make_ctx()
        tool = _make_tool("delete_record")
        _run(hooks.on_tool_start(ctx, tool))
        assert len(captured) == 1
        assert captured[0].triggered is True

    def test_eu_ai_act_art14_context_in_gate(self):
        """EU AI Act Art. 14 regulation citation must appear in escalation context."""
        captured_contexts = []

        def on_esc(result):
            captured_contexts.append(result.metadata)

        adapter = OpenAIAgentsEscalationAdapter(
            threshold=0.99,
            policy=ThresholdPolicy(
                threshold=0.99,
                action=EscalationAction.HUMAN_IN_LOOP,
                critical_threshold=None,
                on_escalation=on_esc,
            ),
        )
        hooks = adapter.as_hooks()
        ctx = _make_ctx()
        _run(hooks.on_tool_start(ctx, _make_tool("send_email")))
        assert len(captured_contexts) == 1
