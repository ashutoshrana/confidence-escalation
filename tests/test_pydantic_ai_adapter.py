"""Tests for Pydantic AI confidence-escalation adapter."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from confidence_escalation.adapters.pydantic_ai import (
    PydanticAIEscalationAdapter,
    PydanticAIHooks,
    SkipToolExecution,
)
from confidence_escalation.policy import EscalationAction, ThresholdPolicy
from confidence_escalation.scorer import ConfidenceScore, ScoringMethod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_call(tool_name: str = "search_docs", tool_call_id: str = "tc_001") -> MagicMock:
    call = MagicMock()
    call.tool_name = tool_name
    call.tool_call_id = tool_call_id
    call.args = {}
    return call


def _make_tool_def(name: str = "search_docs") -> MagicMock:
    td = MagicMock()
    td.name = name
    td.description = "Search documents"
    return td


def _make_ctx(messages: list | None = None) -> MagicMock:
    ctx = MagicMock()
    ctx.messages = messages or []
    ctx.usage = MagicMock()
    ctx.deps = None
    return ctx


def _run(coro):
    return asyncio.run(coro)


def _always_pass_adapter(**kwargs) -> PydanticAIEscalationAdapter:
    """Return adapter that never triggers (threshold=0, no critical threshold)."""
    kwargs.setdefault("evidence_provider", lambda *_: ConfidenceScore(0.95, ScoringMethod.COMPOSITE))
    return PydanticAIEscalationAdapter(
        policy=ThresholdPolicy(
            threshold=0.0,
            action=EscalationAction.NONE,
            critical_threshold=None,
        ),
        **kwargs,
    )


def _always_trigger_adapter(**kwargs) -> PydanticAIEscalationAdapter:
    """Return adapter that always triggers (threshold=0.99, no critical)."""
    kwargs.setdefault("evidence_provider", lambda *_: ConfidenceScore(0.5, ScoringMethod.COMPOSITE))
    return PydanticAIEscalationAdapter(
        threshold=0.99,
        critical_threshold=0.0,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# SkipToolExecution
# ---------------------------------------------------------------------------


class TestSkipToolExecution:
    def test_default_message_mentions_threshold(self):
        exc = SkipToolExecution()
        assert "threshold" in exc.result.lower() or "confidence" in exc.result.lower()

    def test_custom_message(self):
        exc = SkipToolExecution("Custom block reason")
        assert exc.result == "Custom block reason"

    def test_is_exception(self):
        with pytest.raises(SkipToolExecution):
            raise SkipToolExecution()

    def test_str_representation(self):
        exc = SkipToolExecution("blocked")
        assert str(exc) == "blocked"


# ---------------------------------------------------------------------------
# Adapter initialisation
# ---------------------------------------------------------------------------


class TestPydanticAIAdapterInit:
    def test_default_threshold(self):
        adapter = PydanticAIEscalationAdapter()
        assert adapter.threshold == 0.65

    def test_custom_threshold(self):
        adapter = PydanticAIEscalationAdapter(threshold=0.80)
        assert adapter.threshold == 0.80

    def test_as_hooks_returns_hooks(self):
        adapter = PydanticAIEscalationAdapter()
        assert isinstance(adapter.as_hooks(), PydanticAIHooks)

    def test_high_risk_tools_defaults_include_delete(self):
        adapter = PydanticAIEscalationAdapter()
        assert "delete_record" in adapter.high_risk_tools

    def test_custom_high_risk_tools_override(self):
        adapter = PydanticAIEscalationAdapter(high_risk_tools=frozenset({"my_tool"}))
        assert "my_tool" in adapter.high_risk_tools
        assert "delete_record" not in adapter.high_risk_tools


# ---------------------------------------------------------------------------
# Gate method
# ---------------------------------------------------------------------------


class TestGateMethod:
    def test_always_trigger_adapter_fires(self):
        adapter = _always_trigger_adapter()
        result = adapter.gate("search_docs", tool_risk=0.2, confidence=ConfidenceScore(0.5, ScoringMethod.COMPOSITE))
        assert result["triggered"] is True
        assert result["action"] == "human_in_loop"

    def test_always_pass_adapter_does_not_trigger(self):
        adapter = _always_pass_adapter()
        result = adapter.gate("search_docs", tool_risk=0.2, confidence=ConfidenceScore(0.5, ScoringMethod.COMPOSITE))
        assert result["triggered"] is False
        assert "confidence" in result

    def test_explicit_high_confidence_passes_with_moderate_threshold(self):
        adapter = PydanticAIEscalationAdapter(
            policy=ThresholdPolicy(threshold=0.5, action=EscalationAction.HUMAN_IN_LOOP, critical_threshold=None)
        )
        result = adapter.gate(
            "search_docs",
            tool_risk=0.1,
            confidence=ConfidenceScore(0.95, ScoringMethod.COMPOSITE),
        )
        assert result["triggered"] is False

    def test_explicit_low_confidence_high_threshold_triggers(self):
        adapter = PydanticAIEscalationAdapter(
            policy=ThresholdPolicy(threshold=0.7, action=EscalationAction.HUMAN_IN_LOOP, critical_threshold=None)
        )
        result = adapter.gate(
            "delete_record",
            tool_risk=0.5,
            confidence=ConfidenceScore(0.1, ScoringMethod.COMPOSITE),
        )
        assert result["triggered"] is True

    def test_events_logged_when_triggered(self):
        adapter = _always_trigger_adapter()
        adapter.gate("delete_record", tool_risk=0.8, confidence=ConfidenceScore(0.5, ScoringMethod.COMPOSITE))
        assert len(adapter.events) == 1
        assert adapter.events[0].triggered is True

    def test_tool_name_in_result(self):
        adapter = _always_trigger_adapter()
        result = adapter.gate("my_tool", confidence=ConfidenceScore(0.5, ScoringMethod.COMPOSITE))
        assert result["tool_name"] == "my_tool"


# ---------------------------------------------------------------------------
# Hooks: before_tool_execute
# ---------------------------------------------------------------------------


class TestPydanticAIHooks:
    def test_always_pass_returns_args_unchanged(self):
        adapter = _always_pass_adapter()
        hooks = adapter.as_hooks()
        ctx = _make_ctx()
        call = _make_call("search_docs")
        tool_def = _make_tool_def("search_docs")
        args = {"query": "hello"}

        result = _run(hooks.before_tool_execute(ctx, call=call, tool_def=tool_def, args=args))
        assert result == args

    def test_always_trigger_raises_skip_execution(self):
        adapter = _always_trigger_adapter()
        hooks = adapter.as_hooks()
        ctx = _make_ctx()
        call = _make_call("delete_record")
        tool_def = _make_tool_def("delete_record")

        with pytest.raises(SkipToolExecution) as exc_info:
            _run(hooks.before_tool_execute(ctx, call=call, tool_def=tool_def, args={}))

        assert "delete_record" in exc_info.value.result

    def test_skip_message_contains_eu_ai_act_art14(self):
        adapter = _always_trigger_adapter()
        hooks = adapter.as_hooks()

        with pytest.raises(SkipToolExecution) as exc_info:
            _run(hooks.before_tool_execute(
                _make_ctx(),
                call=_make_call("submit_form"),
                tool_def=_make_tool_def("submit_form"),
                args={},
            ))

        assert "EU AI Act" in exc_info.value.result

    def test_skip_contains_confidence_value(self):
        adapter = _always_trigger_adapter()
        hooks = adapter.as_hooks()

        with pytest.raises(SkipToolExecution) as exc_info:
            _run(hooks.before_tool_execute(
                _make_ctx(), call=_make_call("delete_record"),
                tool_def=_make_tool_def("delete_record"), args={},
            ))

        # Should mention confidence or threshold in message
        msg = exc_info.value.result.lower()
        assert "confidence" in msg or "threshold" in msg

    def test_before_run_no_error(self):
        adapter = _always_pass_adapter()
        hooks = adapter.as_hooks()
        _run(hooks.before_run(_make_ctx()))

    def test_after_tool_execute_no_error(self):
        adapter = _always_pass_adapter()
        hooks = adapter.as_hooks()
        _run(hooks.after_tool_execute(
            _make_ctx(),
            call=_make_call(), tool_def=_make_tool_def(), args={}, result="ok",
        ))

    def test_tool_name_from_tool_def_when_call_missing_attribute(self):
        adapter = _always_trigger_adapter()
        hooks = adapter.as_hooks()
        call = MagicMock(spec=[])  # no .tool_name attribute
        tool_def = _make_tool_def("submit_form")

        with pytest.raises(SkipToolExecution) as exc_info:
            _run(hooks.before_tool_execute(_make_ctx(), call=call, tool_def=tool_def, args={}))

        assert "submit_form" in exc_info.value.result

    def test_high_confidence_history_is_not_evidence(self):
        """Untrusted history cannot supply pre-action confidence evidence."""
        adapter = PydanticAIEscalationAdapter(
            policy=ThresholdPolicy(threshold=0.5, action=EscalationAction.HUMAN_IN_LOOP, critical_threshold=None)
        )
        hooks = adapter.as_hooks()

        msg = MagicMock()
        part = MagicMock()
        part.content = "I am 98% confident this is correct."
        msg.parts = [part]
        ctx = _make_ctx(messages=[msg])

        with pytest.raises(SkipToolExecution, match="missing or invalid confidence evidence"):
            _run(hooks.before_tool_execute(
                ctx, call=_make_call("search_docs"),
                tool_def=_make_tool_def("search_docs"), args={"query": "test"},
            ))

    def test_escalation_callback_fires(self):
        """EU AI Act Art. 14: escalation event must be captured."""
        captured = []

        def on_esc(result):
            captured.append(result)

        adapter = _always_trigger_adapter(
            policy=ThresholdPolicy(
                threshold=0.99,
                action=EscalationAction.HUMAN_IN_LOOP,
                critical_threshold=None,
                on_escalation=on_esc,
            )
        )
        hooks = adapter.as_hooks()

        with pytest.raises(SkipToolExecution):
            _run(hooks.before_tool_execute(
                _make_ctx(), call=_make_call("delete_record"),
                tool_def=_make_tool_def("delete_record"), args={},
            ))

        assert len(captured) == 1
        assert captured[0].triggered is True


@pytest.mark.parametrize("score", [None, 0.99, ConfidenceScore(0.99, ScoringMethod.COMPOSITE, metadata={"missing_signal": True})])
def test_direct_gate_rejects_missing_or_invalid_evidence_at_zero(score):
    adapter = _always_pass_adapter()
    with pytest.raises((PermissionError, TypeError)):
        adapter.gate("write_record", confidence=score)


def test_empty_high_risk_set_is_preserved():
    assert PydanticAIEscalationAdapter(high_risk_tools=frozenset())._tool_risk_for("delete_record") == 0.2


def test_handler_error_is_not_misreported_as_invalid_evidence():
    failure = ValueError("notifier failed")

    class Broken:
        def supports(self, action): return True
        def handle(self, result, context): raise failure

    adapter = _always_trigger_adapter(handlers=[Broken()])
    with pytest.raises(ValueError) as exc:
        _run(adapter.as_hooks().before_tool_execute(
            _make_ctx(), call=_make_call(), tool_def=_make_tool_def(), args={},
        ))
    assert exc.value is failure
