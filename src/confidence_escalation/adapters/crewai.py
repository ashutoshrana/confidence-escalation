"""
CrewAI adapter for confidence-escalation.

Wraps a CrewAI Agent's step_callback to intercept task output,
score confidence, and escalate when needed.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from confidence_escalation.middleware import ConfidenceEscalationMiddleware
from confidence_escalation.policy import EscalationPolicy, ThresholdPolicy, EscalationAction
from confidence_escalation.handlers import EscalationHandler, HumanInLoopHandler, ComplianceLoggingHandler
from confidence_escalation.scorer import MultiSignalConfidenceScorer

__all__ = ["CrewAIEscalationAdapter"]


class CrewAIEscalationAdapter:
    """
    CrewAI step_callback-compatible adapter.

    Assign to an agent's ``step_callback`` to score confidence on each
    agent action and escalate on low confidence.

    Example::

        adapter = CrewAIEscalationAdapter(threshold=0.65)

        agent = Agent(
            role="Research Agent",
            goal="...",
            step_callback=adapter.step_callback,
        )

    Or wrap task output explicitly::

        result = adapter.evaluate_task_output(output_text, context={"task": "research"})
        if result["triggered"]:
            raise HumanReviewRequired("Low confidence on task output")
    """

    def __init__(
        self,
        threshold: float = 0.65,
        policy: Optional[EscalationPolicy] = None,
        handlers: Optional[List[EscalationHandler]] = None,
        scorer: Optional[MultiSignalConfidenceScorer] = None,
    ):
        self._middleware = ConfidenceEscalationMiddleware(
            scorer=scorer,
            policy=policy or ThresholdPolicy(threshold=threshold, action=EscalationAction.HUMAN_IN_LOOP),
            handlers=handlers or [HumanInLoopHandler(), ComplianceLoggingHandler()],
        )

    def step_callback(self, agent_output: Any) -> None:
        """Pass as ``step_callback`` to a CrewAI Agent."""
        text = str(agent_output) if not isinstance(agent_output, str) else agent_output
        confidence = self._middleware.score(response_text=text)
        result = self._middleware.evaluate(confidence)
        if result.triggered:
            self._middleware.dispatch(result)

    def evaluate_task_output(
        self,
        output: str,
        logprobs: Optional[List[float]] = None,
        tool_call_risk: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Explicitly evaluate a task output string."""
        confidence = self._middleware.score(
            response_text=output,
            logprobs=logprobs,
            tool_call_risk=tool_call_risk,
        )
        result = self._middleware.evaluate(confidence, context)
        handler_results = self._middleware.dispatch(result, context) if result.triggered else []
        return {
            "triggered": result.triggered,
            "action": result.action.value,
            "confidence": confidence.value,
            "handler_results": handler_results,
        }

    @property
    def events(self):
        return self._middleware.events
