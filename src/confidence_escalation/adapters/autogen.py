"""
Microsoft AutoGen adapter for confidence-escalation.

Wraps AutoGen ConversableAgent reply functions and GroupChat speaker selection
to intercept responses and apply confidence-gated escalation.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

from confidence_escalation.middleware import ConfidenceEscalationMiddleware
from confidence_escalation.policy import EscalationPolicy, ThresholdPolicy, EscalationAction
from confidence_escalation.handlers import EscalationHandler, HumanInLoopHandler
from confidence_escalation.scorer import MultiSignalConfidenceScorer

__all__ = ["AutoGenEscalationAdapter"]


class AutoGenEscalationAdapter:
    """
    AutoGen ConversableAgent-compatible adapter.

    Use ``wrap_reply_func`` to intercept an agent's ``generate_reply`` method,
    or call ``evaluate_message`` directly from a custom reply function.

    Example::

        adapter = AutoGenEscalationAdapter(threshold=0.7)

        # Wrap an existing reply function
        original_reply = agent.generate_reply
        agent.generate_reply = adapter.wrap_reply_func(original_reply)

    For GroupChat-based setups, evaluate speaker messages::

        def custom_speaker_selection(last_speaker, groupchat):
            msg = groupchat.messages[-1]["content"]
            result = adapter.evaluate_message(msg)
            if result["triggered"]:
                return human_proxy_agent
            return next_speaker
    """

    def __init__(
        self,
        threshold: float = 0.65,
        policy: Optional[EscalationPolicy] = None,
        handlers: Optional[List[EscalationHandler]] = None,
        scorer: Optional[MultiSignalConfidenceScorer] = None,
        human_proxy_name: str = "Human",
    ):
        self._middleware = ConfidenceEscalationMiddleware(
            scorer=scorer,
            policy=policy or ThresholdPolicy(threshold=threshold, action=EscalationAction.HUMAN_IN_LOOP),
            handlers=handlers or [HumanInLoopHandler()],
        )
        self.human_proxy_name = human_proxy_name

    def wrap_reply_func(
        self,
        original_func: Callable[..., Any],
    ) -> Callable[..., Any]:
        """Return a wrapped version of an AutoGen generate_reply function."""
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            response = original_func(*args, **kwargs)
            if isinstance(response, str):
                self.evaluate_message(response)
            elif isinstance(response, (list, tuple)) and response:
                self.evaluate_message(str(response[-1]))
            return response
        return wrapped

    def evaluate_message(
        self,
        message: str,
        logprobs: Optional[List[float]] = None,
        tool_call_risk: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        confidence = self._middleware.score(
            response_text=message,
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
            "route_to_human": result.triggered and result.action == EscalationAction.HUMAN_IN_LOOP,
        }

    @property
    def events(self):
        return self._middleware.events
