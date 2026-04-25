"""
Google ADK (Agent Development Kit) adapter for confidence-escalation.

Integrates with ADK's BaseAgent / LlmAgent event loop to intercept
LlmResponse events and apply confidence-gated escalation before tool
execution or response delivery.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from confidence_escalation.middleware import ConfidenceEscalationMiddleware
from confidence_escalation.policy import EscalationPolicy, ThresholdPolicy, EscalationAction
from confidence_escalation.handlers import EscalationHandler, HumanInLoopHandler, ComplianceLoggingHandler
from confidence_escalation.scorer import MultiSignalConfidenceScorer

__all__ = ["ADKEscalationAdapter"]


class ADKEscalationAdapter:
    """
    Google ADK BaseAgent-compatible adapter.

    Inject into an ADK agent's ``_run_async_impl`` to intercept LLM responses:

        class MyGovernedAgent(BaseAgent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._escalation = ADKEscalationAdapter(threshold=0.65)

            async def _run_async_impl(self, ctx):
                async for event in self._llm_agent._run_async_impl(ctx):
                    if event.is_final_response():
                        result = self._escalation.evaluate_event(event, ctx)
                        if result["triggered"]:
                            yield self._escalation.build_escalation_event(result)
                            return
                    yield event

    For multi-agent orchestrators, use ``evaluate_sub_agent_output`` to gate
    sub-agent responses before the orchestrator acts on them.
    """

    def __init__(
        self,
        threshold: float = 0.65,
        policy: Optional[EscalationPolicy] = None,
        handlers: Optional[List[EscalationHandler]] = None,
        scorer: Optional[MultiSignalConfidenceScorer] = None,
        escalation_message: str = "I need to transfer you to a human specialist for this request.",
    ):
        self._middleware = ConfidenceEscalationMiddleware(
            scorer=scorer,
            policy=policy or ThresholdPolicy(threshold=threshold, action=EscalationAction.HUMAN_IN_LOOP),
            handlers=handlers or [HumanInLoopHandler(), ComplianceLoggingHandler()],
        )
        self.escalation_message = escalation_message

    def evaluate_event(
        self,
        event: Any,
        ctx: Optional[Any] = None,
        logprobs: Optional[List[float]] = None,
        tool_call_risk: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate an ADK Event object.

        Extracts text from ``event.content.parts[*].text`` if available,
        or falls back to ``str(event)``.
        """
        text = _extract_adk_event_text(event)
        context = _extract_adk_context(ctx)

        confidence = self._middleware.score(
            response_text=text,
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
            "escalation_message": self.escalation_message if result.triggered else None,
        }

    def evaluate_sub_agent_output(
        self,
        agent_name: str,
        output: str,
        logprobs: Optional[List[float]] = None,
        tool_call_risk: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Gate a sub-agent's output in a multi-agent orchestration flow."""
        ctx = dict(context or {})
        ctx["agent_name"] = agent_name

        confidence = self._middleware.score(
            response_text=output,
            logprobs=logprobs,
            tool_call_risk=tool_call_risk,
        )
        result = self._middleware.evaluate(confidence, ctx)
        handler_results = self._middleware.dispatch(result, ctx) if result.triggered else []

        return {
            "triggered": result.triggered,
            "action": result.action.value,
            "confidence": confidence.value,
            "agent_name": agent_name,
            "handler_results": handler_results,
        }

    def build_escalation_event(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Build a synthetic ADK-compatible escalation event dict."""
        return {
            "type": "escalation",
            "content": {"parts": [{"text": self.escalation_message}]},
            "is_final_response": True,
            "escalation_metadata": result,
        }

    @property
    def events(self):
        return self._middleware.events


def _extract_adk_event_text(event: Any) -> str:
    try:
        parts = event.content.parts
        texts = [p.text for p in parts if hasattr(p, "text") and p.text]
        return " ".join(texts)
    except AttributeError:
        return str(event)


def _extract_adk_context(ctx: Any) -> Dict[str, Any]:
    if ctx is None:
        return {}
    if isinstance(ctx, dict):
        return ctx
    result: Dict[str, Any] = {}
    for attr in ("session_id", "user_id", "invocation_id"):
        val = getattr(ctx, attr, None)
        if val is not None:
            result[attr] = val
    return result
