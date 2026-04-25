"""Tests for escalation handlers."""

import pytest
from confidence_escalation.handlers import (
    ComplianceLoggingHandler,
    HumanInLoopHandler,
    ModelUpgradeHandler,
    ToolRestrictionHandler,
)
from confidence_escalation.policy import EscalationAction, PolicyResult


def make_result(action=EscalationAction.HUMAN_IN_LOOP, confidence=0.4, threshold=0.65):
    return PolicyResult(
        triggered=True,
        action=action,
        confidence_score=confidence,
        threshold_used=threshold,
        reason=f"Confidence {confidence} below {threshold}",
        metadata={"signals": {"logprob": confidence}},
    )


class TestHumanInLoopHandler:
    def test_callback_invoked(self):
        called = []
        handler = HumanInLoopHandler(callback=lambda ctx, r: called.append((ctx, r)))
        handler.handle(make_result(), context={"session_id": "s1"})
        assert len(called) == 1
        assert called[0][0]["session_id"] == "s1"

    def test_raises_when_configured(self):
        handler = HumanInLoopHandler(raise_on_trigger=True)
        with pytest.raises(HumanInLoopHandler.HumanReviewRequired):
            handler.handle(make_result())

    def test_does_not_raise_by_default(self):
        handler = HumanInLoopHandler()
        result = handler.handle(make_result())
        assert result["handler"] == "HumanInLoopHandler"

    def test_supports_correct_action(self):
        handler = HumanInLoopHandler()
        assert handler.supports(EscalationAction.HUMAN_IN_LOOP)
        assert not handler.supports(EscalationAction.MODEL_UPGRADE)


class TestModelUpgradeHandler:
    def test_upgrades_known_model(self):
        handler = ModelUpgradeHandler()
        result = handler.handle(make_result(action=EscalationAction.MODEL_UPGRADE), context={"model": "claude-sonnet-4-6"})
        assert result["upgraded_model"] == "claude-opus-4-7"
        assert result["current_model"] == "claude-sonnet-4-6"

    def test_falls_back_to_default_for_unknown_model(self):
        handler = ModelUpgradeHandler(default_upgraded_model="claude-opus-4-7")
        result = handler.handle(make_result(), context={"model": "unknown-model"})
        assert result["upgraded_model"] == "claude-opus-4-7"

    def test_on_upgrade_callback(self):
        upgrades = []
        handler = ModelUpgradeHandler(on_upgrade=lambda old, new: upgrades.append((old, new)))
        handler.handle(make_result(), context={"model": "gpt-4o-mini"})
        assert upgrades == [("gpt-4o-mini", "gpt-4o")]


class TestToolRestrictionHandler:
    def test_high_risk_tools_restricted(self):
        handler = ToolRestrictionHandler(high_risk_tools=["delete_record", "send_email"])
        result = handler.handle(
            make_result(action=EscalationAction.TOOL_RESTRICTION),
            context={"available_tools": ["delete_record", "get_customer", "send_email"]},
        )
        assert "delete_record" in result["restricted_tools"]
        assert "send_email" in result["restricted_tools"]
        assert "get_customer" in result["allowed_tools"]

    def test_read_only_tools_always_allowed(self):
        handler = ToolRestrictionHandler(high_risk_tools=["write_file"], allow_read_only=True)
        result = handler.handle(
            make_result(),
            context={"available_tools": ["read_file", "write_file", "list_files"]},
        )
        assert "read_file" in result["allowed_tools"]
        assert "list_files" in result["allowed_tools"]
        assert "write_file" in result["restricted_tools"]

    def test_on_restriction_callback(self):
        events = []
        handler = ToolRestrictionHandler(
            high_risk_tools=["delete"],
            on_restriction=lambda r, a: events.append((r, a)),
        )
        handler.handle(make_result(), context={"available_tools": ["delete", "get"]})
        assert len(events) == 1


class TestComplianceLoggingHandler:
    def test_entry_logged_and_stored(self):
        logged = []
        handler = ComplianceLoggingHandler(log_sink=lambda msg: logged.append(msg))
        result = handler.handle(make_result(), context={"session_id": "sess1"})
        assert result["logged"]
        assert len(logged) == 1
        assert len(handler.entries) == 1

    def test_entry_contains_session_id(self):
        handler = ComplianceLoggingHandler(include_context_keys=["session_id"])
        handler.handle(make_result(), context={"session_id": "s123", "other": "x"})
        entry = handler.entries[0]
        assert entry.context_snapshot["session_id"] == "s123"
        assert "other" not in entry.context_snapshot

    def test_unstructured_logging(self):
        logged = []
        handler = ComplianceLoggingHandler(log_sink=lambda msg: logged.append(msg), structured=False)
        handler.handle(make_result())
        assert "COMPLIANCE" in logged[0]

    def test_supports_all_actions(self):
        handler = ComplianceLoggingHandler()
        for action in EscalationAction:
            assert handler.supports(action)
