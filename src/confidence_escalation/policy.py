"""
Threshold-based escalation policies for confidence-gated LLM agent workflows.

Policies evaluate a ConfidenceScore and decide whether/how to escalate.
Framework-agnostic; composed with handlers to produce full escalation pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from confidence_escalation.scorer import ConfidenceScore

__all__ = [
    "EscalationAction",
    "PolicyResult",
    "EscalationPolicy",
    "ThresholdPolicy",
]


class EscalationAction(str, Enum):
    NONE = "none"
    HUMAN_IN_LOOP = "human_in_loop"
    MODEL_UPGRADE = "model_upgrade"
    TOOL_RESTRICTION = "tool_restriction"
    COMPLIANCE_LOG = "compliance_log"
    ABORT = "abort"


@dataclass
class PolicyResult:
    triggered: bool
    action: EscalationAction = EscalationAction.NONE
    confidence_score: Optional[float] = None
    threshold_used: Optional[float] = None
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def should_escalate(self) -> bool:
        return self.triggered and self.action != EscalationAction.NONE


class EscalationPolicy:
    """Abstract base for escalation policies."""

    def evaluate(
        self,
        score: ConfidenceScore,
        context: Optional[Dict[str, Any]] = None,
    ) -> PolicyResult:
        raise NotImplementedError


class ThresholdPolicy(EscalationPolicy):
    """
    Escalate when confidence falls below a threshold.

    Supports per-context overrides and multi-level action dispatch.

    Example::

        policy = ThresholdPolicy(
            threshold=0.65,
            action=EscalationAction.HUMAN_IN_LOOP,
            critical_threshold=0.3,
            critical_action=EscalationAction.ABORT,
        )
        result = policy.evaluate(score, context={"tool_calls_pending": 3})
    """

    def __init__(
        self,
        threshold: float = 0.65,
        action: EscalationAction = EscalationAction.HUMAN_IN_LOOP,
        critical_threshold: Optional[float] = None,
        critical_action: EscalationAction = EscalationAction.ABORT,
        context_overrides: Optional[Dict[str, float]] = None,
        on_escalation: Optional[Callable[[PolicyResult], None]] = None,
    ):
        self.threshold = threshold
        self.action = action
        self.critical_threshold = critical_threshold
        self.critical_action = critical_action
        self.context_overrides = context_overrides or {}
        self.on_escalation = on_escalation

    def _effective_threshold(self, context: Optional[Dict[str, Any]]) -> float:
        if not context:
            return self.threshold
        for key, override in self.context_overrides.items():
            if context.get(key):
                return override
        return self.threshold

    def evaluate(
        self,
        score: ConfidenceScore,
        context: Optional[Dict[str, Any]] = None,
    ) -> PolicyResult:
        effective = self._effective_threshold(context)

        # Check critical threshold first (lower threshold = worst confidence)
        if self.critical_threshold is not None and score.value <= self.critical_threshold:
            result = PolicyResult(
                triggered=True,
                action=self.critical_action,
                confidence_score=score.value,
                threshold_used=self.critical_threshold,
                reason=f"Confidence {score.value:.3f} below critical threshold {self.critical_threshold}",
                metadata={"signals": score.signals, "method": score.method.value},
            )
            if self.on_escalation:
                self.on_escalation(result)
            return result

        if score.value < effective:
            result = PolicyResult(
                triggered=True,
                action=self.action,
                confidence_score=score.value,
                threshold_used=effective,
                reason=f"Confidence {score.value:.3f} below threshold {effective}",
                metadata={"signals": score.signals, "method": score.method.value},
            )
            if self.on_escalation:
                self.on_escalation(result)
            return result

        return PolicyResult(
            triggered=False,
            action=EscalationAction.NONE,
            confidence_score=score.value,
            threshold_used=effective,
            reason="Confidence above threshold",
        )


class CompositePolicy(EscalationPolicy):
    """
    Evaluate multiple policies in priority order; return the first triggered result.

    Example::

        policy = CompositePolicy(policies=[
            ThresholdPolicy(threshold=0.3, action=EscalationAction.ABORT),
            ThresholdPolicy(threshold=0.6, action=EscalationAction.HUMAN_IN_LOOP),
            ThresholdPolicy(threshold=0.75, action=EscalationAction.COMPLIANCE_LOG),
        ])
    """

    def __init__(self, policies: List[EscalationPolicy]):
        self.policies = policies

    def evaluate(
        self,
        score: ConfidenceScore,
        context: Optional[Dict[str, Any]] = None,
    ) -> PolicyResult:
        for policy in self.policies:
            result = policy.evaluate(score, context)
            if result.triggered:
                return result
        return PolicyResult(triggered=False, action=EscalationAction.NONE)
