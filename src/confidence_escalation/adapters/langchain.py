"""
LangChain / LangGraph adapter for confidence-escalation.

Drop-in callback handler that scores confidence on every LLM end event
and evaluates the configured policy. Raise HumanReviewRequired to pause
a LangGraph graph at any node.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from confidence_escalation.middleware import ConfidenceEscalationMiddleware
from confidence_escalation.policy import EscalationPolicy, ThresholdPolicy, EscalationAction
from confidence_escalation.handlers import EscalationHandler, HumanInLoopHandler
from confidence_escalation.scorer import MultiSignalConfidenceScorer

__all__ = ["LangChainEscalationAdapter"]


class LangChainEscalationAdapter:
    """
    LangChain CallbackHandler-compatible adapter.

    Use as a callback in any LangChain chain or LangGraph node:

        adapter = LangChainEscalationAdapter(
            threshold=0.65,
            handlers=[HumanInLoopHandler(raise_on_trigger=True)],
        )
        chain = LLMChain(llm=llm, callbacks=[adapter.as_callback()])

    For LangGraph, inject at the node level:

        def my_node(state):
            response = llm.invoke(state["messages"])
            adapter.on_llm_end(response.text, logprobs=response.logprobs)
            return {"response": response.text}
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
            handlers=handlers or [HumanInLoopHandler()],
        )

    def on_llm_end(
        self,
        response_text: str,
        logprobs: Optional[List[float]] = None,
        tool_call_risk: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Call from LangChain on_llm_end callback or LangGraph node."""
        confidence = self._middleware.score(
            response_text=response_text,
            logprobs=logprobs,
            tool_call_risk=tool_call_risk,
        )
        result = self._middleware.evaluate(confidence, context)
        if result.triggered:
            return {"triggered": True, "results": self._middleware.dispatch(result, context)}
        return {"triggered": False, "confidence": confidence.value}

    def as_callback(self) -> "LangChainCallbackShim":
        """Return a minimal LangChain BaseCallbackHandler shim."""
        return LangChainCallbackShim(adapter=self)

    @property
    def events(self):
        return self._middleware.events


class LangChainCallbackShim:
    """Minimal shim implementing LangChain's BaseCallbackHandler interface."""

    def __init__(self, adapter: LangChainEscalationAdapter):
        self._adapter = adapter

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        text = ""
        if hasattr(response, "generations"):
            gen = response.generations
            if gen and gen[0]:
                text = getattr(gen[0][0], "text", str(gen[0][0]))
        elif isinstance(response, str):
            text = response
        self._adapter.on_llm_end(response_text=text)

    def on_chain_start(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_chain_end(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_tool_start(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_tool_end(self, *args: Any, **kwargs: Any) -> None:
        pass
