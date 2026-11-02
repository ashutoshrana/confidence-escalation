"""
Async confidence escalation middleware for streaming LLM agents.

AsyncConfidenceEscalationMiddleware wraps async agent steps — LangGraph nodes,
CrewAI Flows callbacks (v0.28+), AutoGen async handlers — with non-blocking
confidence scoring and concurrent handler dispatch.

Accepts async iterators of text tokens. Score structured framework events explicitly.
"""

from __future__ import annotations

import asyncio
import datetime
import inspect
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from confidence_escalation.handlers import EscalationHandler
from confidence_escalation.middleware import EscalationEvent
from confidence_escalation.policy import EscalationPolicy, PolicyResult
from confidence_escalation.scorer import ConfidenceScore, MultiSignalConfidenceScorer

__all__ = ["AsyncConfidenceEscalationMiddleware"]


class AsyncConfidenceEscalationMiddleware:
    """
    Async variant of ConfidenceEscalationMiddleware for streaming agent pipelines.

    Features:
    - Awaits async agent steps; runs sync steps in executor (never blocks event loop)
    - Streaming token accumulation via async generator wrapper
    - Concurrent handler dispatch via asyncio.gather
    - Post-stream decisions for text-token iterators

    Example::

        middleware = AsyncConfidenceEscalationMiddleware(
            scorer=MultiSignalConfidenceScorer(),
            policy=ThresholdPolicy(threshold=0.65, action=EscalationAction.HUMAN_IN_LOOP),
            handlers=[ComplianceLoggingHandler(), HumanInLoopHandler(notify=alert)],
        )

        # Wrap an async agent step
        result = await middleware.call(my_async_llm_call, messages=msgs, context=ctx)

        # Stream with inline confidence tracking
        async for chunk in middleware.stream(my_streaming_llm_call, messages=msgs, context=ctx):
            print(chunk)
    """

    def __init__(
        self,
        scorer: Optional[MultiSignalConfidenceScorer] = None,
        policy: Optional[EscalationPolicy] = None,
        handlers: Optional[List[EscalationHandler]] = None,
        event_sink: Optional[Callable[[EscalationEvent], None]] = None,
    ) -> None:
        self.scorer = scorer or MultiSignalConfidenceScorer()
        self.policy = policy
        self.handlers = handlers or []
        self.event_sink = event_sink
        self._events: List[EscalationEvent] = []

    async def call_guarded(
        self, action: Callable[..., Any], confidence: ConfidenceScore,
        *args: Any, context: Optional[Dict[str, Any]] = None, **kwargs: Any,
    ) -> Any:
        """Check fresh confidence evidence before invoking and awaiting an action.

        Confidence gating does not replace caller/resource authorization.
        Cancellation cannot undo an action already started, including worker threads.
        """
        confidence.require_evidence()
        decision = self._evaluate(confidence, context)
        if decision.triggered:
            _, error = await self._dispatch(decision, context)
            if error is not None:
                raise error
            raise PermissionError(f"Action blocked: {decision.reason}")
        return await self._invoke(action, *args, **kwargs)

    @staticmethod
    async def _invoke(action: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if inspect.iscoroutinefunction(action) or inspect.iscoroutinefunction(getattr(action, "__call__", None)):
            result = action(*args, **kwargs)
        else:
            result = await asyncio.to_thread(action, *args, **kwargs)
        return await result if inspect.isawaitable(result) else result

    async def call(
        self,
        agent_step: Callable[..., Any],
        *args: Any,
        context: Optional[Dict[str, Any]] = None,
        logprobs: Optional[List[float]] = None,
        tool_call_risk: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Invoke agent_step, score its output, dispatch handlers if escalation triggered.

        Coroutine functions are awaited directly; sync callables run in the default
        executor so the event loop is never blocked.
        """
        response = await self._invoke(agent_step, *args, **kwargs)

        response_text = response if isinstance(response, str) else str(response)
        confidence = self.scorer.score(
            logprobs=logprobs,
            verbalized_response=response_text,
            tool_call_risk=tool_call_risk,
        )
        policy_result = self._evaluate(confidence, context)
        event = await self._dispatch_and_record(confidence, policy_result, context)
        return {"response": response, "confidence": confidence.value, "escalation": event.to_dict()}

    def _record_event(
        self, confidence: ConfidenceScore, policy_result: PolicyResult,
        handler_results: List[Dict[str, Any]], context: Optional[Dict[str, Any]],
    ) -> EscalationEvent:
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
            outcome = self.event_sink(event)
            if inspect.isawaitable(outcome):
                if inspect.iscoroutine(outcome):
                    outcome.close()
                raise TypeError("event_sink must complete synchronously")
        return event

    async def stream(
        self,
        agent_step: Callable[..., Any],
        *args: Any,
        context: Optional[Dict[str, Any]] = None,
        tool_call_risk: Optional[float] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Yield tokens from an async generator agent step with post-stream confidence scoring.

        Tokens are yielded immediately as they arrive. Once the stream is exhausted,
        the full response is scored and an escalation event is emitted.
        This is post-response scoring: tokens have already been delivered. Use
        call_guarded for pre-action gating. Pass text tokens, not framework event dicts.
        """
        accumulated: List[str] = []
        async for chunk in agent_step(*args, **kwargs):
            token = chunk if isinstance(chunk, str) else str(chunk)
            accumulated.append(token)
            yield token

        full_response = "".join(accumulated)
        confidence = self.scorer.score(
            verbalized_response=full_response,
            tool_call_risk=tool_call_risk,
        )
        policy_result = self._evaluate(confidence, context)
        await self._dispatch_and_record(confidence, policy_result, context)

    def _evaluate(
        self,
        confidence: ConfidenceScore,
        context: Optional[Dict[str, Any]],
    ) -> PolicyResult:
        if self.policy is None:
            from confidence_escalation.policy import ThresholdPolicy
            return ThresholdPolicy().evaluate(confidence, context)
        return self.policy.evaluate(confidence, context)

    async def _dispatch_and_record(
        self, confidence: ConfidenceScore, decision: PolicyResult,
        context: Optional[Dict[str, Any]],
    ) -> EscalationEvent:
        outcomes, error = await self._dispatch(decision, context) if decision.triggered else ([], None)
        try:
            event = self._record_event(confidence, decision, outcomes, context)
        except BaseException as sink_error:
            if error is not None:
                raise error from sink_error
            raise
        if error is not None:
            raise error
        return event

    async def _dispatch(
        self,
        result: PolicyResult,
        context: Optional[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], Optional[BaseException]]:
        """Finish every matching handler and retain failures alongside successes."""
        matching = [h for h in self.handlers if h.supports(result.action)]
        results = await asyncio.gather(
            *(self._invoke(handler.handle, result, context) for handler in matching),
            return_exceptions=True,
        )
        outcomes: List[Dict[str, Any]] = []
        error: Optional[BaseException] = None
        for handler, outcome in zip(matching, results):
            if isinstance(outcome, BaseException):
                outcomes.append({"handler": type(handler).__name__, "status": "failed", "error_type": type(outcome).__name__})
                if error is None:
                    error = outcome
            elif isinstance(outcome, dict):
                outcomes.append(outcome)
        return outcomes, error

    @property
    def events(self) -> List[EscalationEvent]:
        return list(self._events)
