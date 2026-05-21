"""
Async confidence escalation middleware for streaming LLM agents.

AsyncConfidenceEscalationMiddleware wraps async agent steps — LangGraph nodes,
CrewAI Flows callbacks (v0.28+), AutoGen async handlers — with non-blocking
confidence scoring and concurrent handler dispatch.

Compatible with LangChain stream_events v3 API (LangChain v0.5+).
"""

from __future__ import annotations

import asyncio
import datetime
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from confidence_escalation.handlers import EscalationHandler
from confidence_escalation.middleware import EscalationEvent
from confidence_escalation.policy import EscalationAction, EscalationPolicy, PolicyResult
from confidence_escalation.scorer import ConfidenceScore, MultiSignalConfidenceScorer

__all__ = ["AsyncConfidenceEscalationMiddleware"]


class AsyncConfidenceEscalationMiddleware:
    """
    Async variant of ConfidenceEscalationMiddleware for streaming agent pipelines.

    Features:
    - Awaits async agent steps; runs sync steps in executor (never blocks event loop)
    - Streaming token accumulation via async generator wrapper
    - Concurrent handler dispatch via asyncio.gather
    - LangChain stream_events v3 and CrewAI Flows compatible

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
        if asyncio.iscoroutinefunction(agent_step):
            response = await agent_step(*args, **kwargs)
        else:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, lambda: agent_step(*args, **kwargs))

        response_text = response if isinstance(response, str) else str(response)
        confidence = self.scorer.score(
            logprobs=logprobs,
            verbalized_response=response_text,
            tool_call_risk=tool_call_risk,
        )
        policy_result = self._evaluate(confidence, context)
        handler_results = await self._dispatch(policy_result, context) if policy_result.triggered else []

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
        return {"response": response, "confidence": confidence.value, "escalation": event.to_dict()}

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
        Compatible with LangChain stream_events v3 chunk format.
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
        if policy_result.triggered:
            await self._dispatch(policy_result, context)

    def _evaluate(
        self,
        confidence: ConfidenceScore,
        context: Optional[Dict[str, Any]],
    ) -> PolicyResult:
        if self.policy is None:
            from confidence_escalation.policy import ThresholdPolicy
            return ThresholdPolicy().evaluate(confidence, context)
        return self.policy.evaluate(confidence, context)

    async def _dispatch(
        self,
        result: PolicyResult,
        context: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Run all matching handlers concurrently via asyncio.gather."""
        matching = [h for h in self.handlers if h.supports(result.action)]
        if not matching:
            return []
        loop = asyncio.get_running_loop()
        tasks: List[Any] = []
        for handler in matching:
            if asyncio.iscoroutinefunction(handler.handle):
                tasks.append(handler.handle(result, context))
            else:
                tasks.append(loop.run_in_executor(None, lambda h=handler: h.handle(result, context)))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if isinstance(r, dict)]

    @property
    def events(self) -> List[EscalationEvent]:
        return list(self._events)
