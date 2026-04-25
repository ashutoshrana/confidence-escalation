"""
Framework-agnostic confidence escalation middleware.

ConfidenceEscalationMiddleware wraps any callable agent step and intercepts
execution to score confidence, evaluate policy, and dispatch handlers — all
without requiring framework-specific integration.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from confidence_escalation.handlers import EscalationHandler
from confidence_escalation.policy import EscalationAction, EscalationPolicy, PolicyResult
from confidence_escalation.scorer import ConfidenceScore, MultiSignalConfidenceScorer

__all__ = [
    "EscalationEvent",
    "ConfidenceEscalationMiddleware",
]


@dataclass
class EscalationEvent:
    """Immutable record of a single escalation decision."""

    timestamp: str
    triggered: bool
    action: EscalationAction
    confidence_score: float
    reason: str
    handler_results: List[Dict[str, Any]] = field(default_factory=list)
    context_snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "triggered": self.triggered,
            "action": self.action.value,
            "confidence_score": self.confidence_score,
            "reason": self.reason,
            "handler_results": self.handler_results,
        }


class ConfidenceEscalationMiddleware:
    """
    Wraps an agent step function with confidence scoring and escalation.

    On each invocation:
    1. Calls the wrapped step to get a response
    2. Scores confidence from logprobs / verbalized output / tool risk
    3. Evaluates the policy
    4. Dispatches matching handlers
    5. Returns the response with an escalation event attached

    Example::

        scorer = MultiSignalConfidenceScorer()
        policy = ThresholdPolicy(threshold=0.65, action=EscalationAction.HUMAN_IN_LOOP)
        handlers = [HumanInLoopHandler(callback=notify_human), ComplianceLoggingHandler()]

        middleware = ConfidenceEscalationMiddleware(
            scorer=scorer,
            policy=policy,
            handlers=handlers,
        )

        # Wrap your agent step
        result = middleware.call(
            agent_step=my_llm_call,
            messages=messages,
            context={"session_id": "abc123", "model": "claude-sonnet-4-6"},
        )
    """

    def __init__(
        self,
        scorer: Optional[MultiSignalConfidenceScorer] = None,
        policy: Optional[EscalationPolicy] = None,
        handlers: Optional[List[EscalationHandler]] = None,
        event_sink: Optional[Callable[[EscalationEvent], None]] = None,
    ):
        self.scorer = scorer or MultiSignalConfidenceScorer()
        self.policy = policy
        self.handlers = handlers or []
        self.event_sink = event_sink
        self._events: List[EscalationEvent] = []

    def score(
        self,
        response_text: Optional[str] = None,
        logprobs: Optional[List[float]] = None,
        tool_call_risk: Optional[float] = None,
        additional_signals: Optional[Dict[str, float]] = None,
    ) -> ConfidenceScore:
        return self.scorer.score(
            logprobs=logprobs,
            verbalized_response=response_text,
            tool_call_risk=tool_call_risk,
            additional_signals=additional_signals,
        )

    def evaluate(
        self,
        confidence: ConfidenceScore,
        context: Optional[Dict[str, Any]] = None,
    ) -> PolicyResult:
        if self.policy is None:
            from confidence_escalation.policy import ThresholdPolicy
            return ThresholdPolicy().evaluate(confidence, context)
        return self.policy.evaluate(confidence, context)

    def dispatch(
        self,
        result: PolicyResult,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        handler_results = []
        for handler in self.handlers:
            if handler.supports(result.action):
                hr = handler.handle(result, context)
                handler_results.append(hr)
        return handler_results

    def call(
        self,
        agent_step: Callable[..., Any],
        *args: Any,
        context: Optional[Dict[str, Any]] = None,
        logprobs: Optional[List[float]] = None,
        tool_call_risk: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Invoke ``agent_step(*args, **kwargs)``, score its output, and escalate if needed.

        The wrapped function's return value is included in the result dict
        under key ``"response"``. The escalation event is under ``"escalation"``.
        """
        response = agent_step(*args, **kwargs)
        response_text = response if isinstance(response, str) else str(response)

        confidence = self.score(
            response_text=response_text,
            logprobs=logprobs,
            tool_call_risk=tool_call_risk,
        )
        policy_result = self.evaluate(confidence, context)
        handler_results = self.dispatch(policy_result, context) if policy_result.triggered else []

        event = EscalationEvent(
            timestamp=datetime.datetime.utcnow().isoformat(),
            triggered=policy_result.triggered,
            action=policy_result.action,
            confidence_score=confidence.value,
            reason=policy_result.reason,
            handler_results=handler_results,
            context_snapshot=context or {},
        )
        self._events.append(event)

        if self.event_sink:
            self.event_sink(event)

        return {
            "response": response,
            "confidence": confidence.value,
            "escalation": event.to_dict(),
        }

    @property
    def events(self) -> List[EscalationEvent]:
        return list(self._events)
