"""
Escalation handlers — side-effecting actions triggered by policy evaluation.

Each handler receives a PolicyResult and executes the appropriate response:
routing to human review, upgrading the model, restricting tool access,
or writing a compliance audit log entry.
"""

from __future__ import annotations

import datetime
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from confidence_escalation.policy import EscalationAction, PolicyResult

__all__ = [
    "EscalationHandler",
    "HumanInLoopHandler",
    "ModelUpgradeHandler",
    "ToolRestrictionHandler",
    "ComplianceLoggingHandler",
]

logger = logging.getLogger(__name__)


class EscalationHandler(ABC):
    """Base class for all escalation handlers."""

    @abstractmethod
    def handle(self, result: PolicyResult, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the escalation action. Returns a dict with execution metadata."""

    def supports(self, action: EscalationAction) -> bool:
        """Whether this handler supports the given action."""
        return True


class HumanInLoopHandler(EscalationHandler):
    """
    Pause agent execution and route the turn to a human reviewer.

    Calls ``callback`` with the current context dict and PolicyResult.
    Optionally raises ``HumanReviewRequired`` to interrupt the framework loop.

    Example::

        def my_review_callback(context, result):
            queue.publish({"session": context["session_id"], "score": result.confidence_score})

        handler = HumanInLoopHandler(callback=my_review_callback, raise_on_trigger=True)
    """

    class HumanReviewRequired(Exception):
        def __init__(self, result: PolicyResult):
            self.result = result
            super().__init__(f"Human review required: confidence={result.confidence_score:.3f}")

    def __init__(
        self,
        callback: Optional[Callable[[Dict[str, Any], PolicyResult], None]] = None,
        raise_on_trigger: bool = False,
        queue_name: str = "human_review_queue",
    ):
        self.callback = callback
        self.raise_on_trigger = raise_on_trigger
        self.queue_name = queue_name

    def supports(self, action: EscalationAction) -> bool:
        return action == EscalationAction.HUMAN_IN_LOOP

    def handle(self, result: PolicyResult, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ctx = context or {}
        logger.warning(
            "Human-in-loop triggered: confidence=%.3f reason=%s",
            result.confidence_score or 0.0,
            result.reason,
        )
        if self.callback:
            self.callback(ctx, result)

        outcome = {
            "handler": "HumanInLoopHandler",
            "queue": self.queue_name,
            "confidence": result.confidence_score,
            "reason": result.reason,
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }

        if self.raise_on_trigger:
            raise self.HumanReviewRequired(result)

        return outcome


class ModelUpgradeHandler(EscalationHandler):
    """
    Suggest or apply a model upgrade when confidence is insufficient.

    Returns the upgraded model name in the response dict so the calling
    framework can re-invoke with the stronger model.

    Example::

        handler = ModelUpgradeHandler(
            upgrade_map={
                "claude-haiku-4-5": "claude-sonnet-4-6",
                "claude-sonnet-4-6": "claude-opus-4-7",
            }
        )
    """

    DEFAULT_UPGRADE_MAP = {
        "claude-haiku-4-5": "claude-sonnet-4-6",
        "claude-sonnet-4-6": "claude-opus-4-7",
        "gemini-flash": "gemini-pro",
        "gpt-4o-mini": "gpt-4o",
        "gpt-4o": "o1",
    }

    def __init__(
        self,
        upgrade_map: Optional[Dict[str, str]] = None,
        default_upgraded_model: str = "claude-opus-4-7",
        on_upgrade: Optional[Callable[[str, str], None]] = None,
    ):
        self.upgrade_map = upgrade_map or self.DEFAULT_UPGRADE_MAP
        self.default_upgraded_model = default_upgraded_model
        self.on_upgrade = on_upgrade

    def supports(self, action: EscalationAction) -> bool:
        return action == EscalationAction.MODEL_UPGRADE

    def handle(self, result: PolicyResult, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ctx = context or {}
        current_model = ctx.get("model", "unknown")
        upgraded_model = self.upgrade_map.get(current_model, self.default_upgraded_model)

        logger.info("Model upgrade: %s → %s (confidence=%.3f)", current_model, upgraded_model, result.confidence_score or 0.0)

        if self.on_upgrade:
            self.on_upgrade(current_model, upgraded_model)

        return {
            "handler": "ModelUpgradeHandler",
            "current_model": current_model,
            "upgraded_model": upgraded_model,
            "confidence": result.confidence_score,
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }


class ToolRestrictionHandler(EscalationHandler):
    """
    Restrict available tools when confidence is below threshold.

    Removes high-risk tools from the active tool list and returns the
    filtered set. Integrate with framework tool registries as needed.

    Example::

        handler = ToolRestrictionHandler(
            high_risk_tools=["delete_record", "send_email", "execute_sql"],
            allow_read_only=True,
        )
    """

    def __init__(
        self,
        high_risk_tools: Optional[List[str]] = None,
        allow_read_only: bool = True,
        read_only_prefixes: Optional[List[str]] = None,
        on_restriction: Optional[Callable[[List[str], List[str]], None]] = None,
    ):
        self.high_risk_tools = set(high_risk_tools or [])
        self.allow_read_only = allow_read_only
        self.read_only_prefixes = read_only_prefixes or ["get_", "list_", "search_", "read_", "fetch_"]
        self.on_restriction = on_restriction

    def supports(self, action: EscalationAction) -> bool:
        return action == EscalationAction.TOOL_RESTRICTION

    def _is_read_only(self, tool_name: str) -> bool:
        return any(tool_name.startswith(pfx) for pfx in self.read_only_prefixes)

    def handle(self, result: PolicyResult, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ctx = context or {}
        available_tools: List[str] = ctx.get("available_tools", [])

        restricted: List[str] = []
        allowed: List[str] = []
        for tool in available_tools:
            if tool in self.high_risk_tools:
                restricted.append(tool)
            elif self.allow_read_only and self._is_read_only(tool):
                allowed.append(tool)
            elif tool not in self.high_risk_tools:
                allowed.append(tool)
            else:
                restricted.append(tool)

        logger.info("Tool restriction applied: %d restricted, %d allowed", len(restricted), len(allowed))

        if self.on_restriction:
            self.on_restriction(restricted, allowed)

        return {
            "handler": "ToolRestrictionHandler",
            "restricted_tools": restricted,
            "allowed_tools": allowed,
            "confidence": result.confidence_score,
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }


@dataclass
class ComplianceLogEntry:
    timestamp: str
    session_id: str
    confidence_score: float
    threshold: float
    action: str
    reason: str
    signals: Dict[str, float] = field(default_factory=dict)
    context_snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "confidence_score": self.confidence_score,
            "threshold": self.threshold,
            "action": self.action,
            "reason": self.reason,
            "signals": self.signals,
            "context_snapshot": self.context_snapshot,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class ComplianceLoggingHandler(EscalationHandler):
    """
    Write a structured compliance audit log entry for every escalation event.

    Satisfies EU AI Act Article 12 (logging/monitoring) and OWASP ASI-09
    (Human-Agent Trust Exploitation) audit requirements.

    Example::

        handler = ComplianceLoggingHandler(
            log_sink=my_audit_logger.info,
            include_context_keys=["session_id", "user_id", "intent"],
        )
    """

    def __init__(
        self,
        log_sink: Optional[Callable[[str], None]] = None,
        include_context_keys: Optional[List[str]] = None,
        structured: bool = True,
    ):
        self.log_sink = log_sink or logger.info
        self.include_context_keys = include_context_keys or ["session_id", "user_id", "intent", "turn_count"]
        self.structured = structured
        self._entries: List[ComplianceLogEntry] = []

    def supports(self, action: EscalationAction) -> bool:
        return True  # Log all actions

    def handle(self, result: PolicyResult, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ctx = context or {}
        context_snapshot = {k: ctx[k] for k in self.include_context_keys if k in ctx}

        entry = ComplianceLogEntry(
            timestamp=datetime.datetime.utcnow().isoformat(),
            session_id=str(ctx.get("session_id", "unknown")),
            confidence_score=result.confidence_score or 0.0,
            threshold=result.threshold_used or 0.0,
            action=result.action.value,
            reason=result.reason,
            signals=result.metadata.get("signals", {}),
            context_snapshot=context_snapshot,
        )
        self._entries.append(entry)

        if self.structured:
            self.log_sink(entry.to_json())
        else:
            self.log_sink(
                f"[COMPLIANCE] escalation action={entry.action} "
                f"confidence={entry.confidence_score:.3f} reason={entry.reason}"
            )

        return {
            "handler": "ComplianceLoggingHandler",
            "logged": True,
            "entry": entry.to_dict(),
        }

    @property
    def entries(self) -> List[ComplianceLogEntry]:
        return list(self._entries)
