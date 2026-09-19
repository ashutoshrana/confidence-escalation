import asyncio
import pytest
from confidence_escalation import ConfidenceEscalationMiddleware, ConfidenceScore, ScoringMethod, MultiSignalConfidenceScorer
from confidence_escalation.async_middleware import AsyncConfidenceEscalationMiddleware

@pytest.mark.asyncio
async def test_async_gate_blocks_low_and_missing_evidence_before_side_effect():
    effects = []
    async def action():
        effects.append('executed')
        return 'ok'
    gate = AsyncConfidenceEscalationMiddleware()
    for score in [ConfidenceScore(.1, ScoringMethod.COMPOSITE), MultiSignalConfidenceScorer().score()]:
        with pytest.raises(PermissionError):
            await gate.call_guarded(action, score)
    assert effects == []
    assert await gate.call_guarded(action, ConfidenceScore(.9, ScoringMethod.COMPOSITE)) == 'ok'
    assert effects == ['executed']

@pytest.mark.asyncio
async def test_async_callable_objects_are_awaited():
    class Action:
        async def __call__(self):
            return 'I am 90% confident'
    gate = AsyncConfidenceEscalationMiddleware()
    result = await gate.call(Action())
    assert result['response'] == 'I am 90% confident'
    assert await gate.call_guarded(Action(), ConfidenceScore(.9, ScoringMethod.COMPOSITE)) == 'I am 90% confident'

def test_sync_gate_rejects_deferred_coroutine_execution():
    async def action():
        return 1
    with pytest.raises(TypeError, match='async'):
        ConfidenceEscalationMiddleware().call_guarded(action, ConfidenceScore(.9, ScoringMethod.COMPOSITE))

def test_sync_gate_closes_coroutine_returned_by_wrapper():
    effects = []
    async def action():
        effects.append('executed')
    with pytest.raises(TypeError, match='async'):
        ConfidenceEscalationMiddleware().call_guarded(lambda: action(), ConfidenceScore(.9, ScoringMethod.COMPOSITE))
    assert effects == []
