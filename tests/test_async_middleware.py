"""Tests for AsyncConfidenceEscalationMiddleware."""
from __future__ import annotations
import asyncio
import pytest
from confidence_escalation.async_middleware import AsyncConfidenceEscalationMiddleware
from confidence_escalation.policy import ThresholdPolicy, EscalationAction


async def _high_confidence_step(*args, **kwargs) -> str:
    return "I am 95% confident the answer is correct."


async def _low_confidence_step(*args, **kwargs) -> str:
    return "I am uncertain about this."


async def _streaming_step(*args, **kwargs):
    for token in ["I ", "am ", "70% ", "confident."]:
        yield token


def _sync_step(*args, **kwargs) -> str:
    return "I am 80% confident."


@pytest.mark.asyncio
async def test_call_high_confidence_no_escalation():
    middleware = AsyncConfidenceEscalationMiddleware(
        policy=ThresholdPolicy(threshold=0.5, action=EscalationAction.HUMAN_IN_LOOP)
    )
    result = await middleware.call(_high_confidence_step)
    assert result["escalation"]["triggered"] is False
    assert result["confidence"] > 0.5


@pytest.mark.asyncio
async def test_call_low_confidence_triggers_escalation():
    middleware = AsyncConfidenceEscalationMiddleware(
        policy=ThresholdPolicy(threshold=0.8, action=EscalationAction.HUMAN_IN_LOOP)
    )
    result = await middleware.call(_low_confidence_step)
    assert result["escalation"]["triggered"] is True


@pytest.mark.asyncio
async def test_call_sync_step_runs_in_executor():
    middleware = AsyncConfidenceEscalationMiddleware()
    result = await middleware.call(_sync_step)
    assert "response" in result
    assert result["response"] == "I am 80% confident."


@pytest.mark.asyncio
async def test_streaming_accumulates_tokens():
    middleware = AsyncConfidenceEscalationMiddleware()
    tokens = []
    async for token in middleware.stream(_streaming_step):
        tokens.append(token)
    assert "".join(tokens) == "I am 70% confident."


@pytest.mark.asyncio
async def test_events_are_recorded():
    middleware = AsyncConfidenceEscalationMiddleware()
    await middleware.call(_high_confidence_step)
    await middleware.call(_low_confidence_step)
    assert len(middleware.events) == 2


@pytest.mark.asyncio
async def test_event_sink_called():
    received = []
    middleware = AsyncConfidenceEscalationMiddleware(event_sink=received.append)
    await middleware.call(_high_confidence_step)
    assert len(received) == 1
    assert received[0].confidence_score > 0


@pytest.mark.asyncio
async def test_concurrent_calls_independent():
    middleware = AsyncConfidenceEscalationMiddleware()
    results = await asyncio.gather(
        middleware.call(_high_confidence_step, context={"id": "a"}),
        middleware.call(_low_confidence_step, context={"id": "b"}),
    )
    assert len(results) == 2
    assert results[0]["confidence"] != results[1]["confidence"]


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold,triggered", [(0.5, False), (0.9, True)])
async def test_stream_records_and_delivers_final_decision(threshold, triggered):
    received = []
    middleware = AsyncConfidenceEscalationMiddleware(
        policy=ThresholdPolicy(threshold=threshold), event_sink=received.append,
    )
    assert "".join([token async for token in middleware.stream(_streaming_step)]) == "I am 70% confident."
    assert len(middleware.events) == 1
    assert received == middleware.events
    assert received[0].triggered is triggered


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_handler_failure_propagates_after_all_handlers_finish(streaming):
    completed = []
    received = []
    failure = RuntimeError("notification failed")

    class Broken:
        def supports(self, action):
            return True

        async def handle(self, result, context):
            raise failure

    class Successful(Broken):
        async def handle(self, result, context):
            await asyncio.sleep(0)
            completed.append(True)
            return {"delivered": True}

    middleware = AsyncConfidenceEscalationMiddleware(
        policy=ThresholdPolicy(threshold=0.99), handlers=[Broken(), Successful()], event_sink=received.append,
    )
    with pytest.raises(RuntimeError, match="notification failed") as exc:
        if streaming:
            _ = [token async for token in middleware.stream(_streaming_step)]
        else:
            await middleware.call(_low_confidence_step)
    assert exc.value is failure
    assert completed == [True]
    assert len(middleware.events) == 1
    assert received == middleware.events
    assert middleware.events[0].handler_results == [
        {"handler": "Broken", "status": "failed", "error_type": "RuntimeError"},
        {"delivered": True},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("handler_fails", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
async def test_sink_error_preserves_record_and_does_not_mask_handler_error(handler_fails, streaming):
    handler_failure = RuntimeError("handler failed")
    sink_failure = ValueError("sink failed")

    class Handler:
        def supports(self, action): return True
        async def handle(self, result, context):
            if handler_fails:
                raise handler_failure
            return {"delivered": True}

    def sink(event):
        raise sink_failure

    middleware = AsyncConfidenceEscalationMiddleware(
        policy=ThresholdPolicy(threshold=0.99), handlers=[Handler()], event_sink=sink,
    )
    with pytest.raises((RuntimeError, ValueError)) as exc:
        if streaming:
            _ = [token async for token in middleware.stream(_streaming_step)]
        else:
            await middleware.call(_low_confidence_step)
    assert len(middleware.events) == 1
    if handler_fails:
        assert exc.value is handler_failure
        assert exc.value.__cause__ is sink_failure
    else:
        assert exc.value is sink_failure


@pytest.mark.asyncio
async def test_awaitable_sink_is_rejected_after_local_event_retention():
    received = []

    async def sink(event):
        received.append(event)

    middleware = AsyncConfidenceEscalationMiddleware(event_sink=sink)
    with pytest.raises(TypeError, match="synchronously"):
        await middleware.call(_low_confidence_step)
    assert len(middleware.events) == 1
    assert received == []
