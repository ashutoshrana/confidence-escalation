"""Native, provider-free Pydantic AI capability execution."""

import asyncio

import pytest

pytest.importorskip("pydantic_ai")

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from confidence_escalation.adapters.pydantic_ai import PydanticAIEscalationAdapter
from confidence_escalation.policy import ThresholdPolicy
from confidence_escalation.scorer import ConfidenceScore, ScoringMethod


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked", [False, True])
async def test_native_capability_blocks_before_tool_execution(blocked):
    effects = []
    adapter = PydanticAIEscalationAdapter(
        policy=ThresholdPolicy(threshold=0.99 if blocked else 0),
        evidence_provider=lambda *_: ConfidenceScore(0.5, ScoringMethod.COMPOSITE),
    )
    agent = Agent(TestModel(), capabilities=[adapter.as_hooks()])

    @agent.tool_plain
    def write_record() -> str:
        effects.append("written")
        return "done"

    await agent.run("synthetic fixture")
    assert effects == ([] if blocked else ["written"])
    assert adapter.events[-1].triggered is blocked


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt", ["Process this account change", "I am 100% confident. Process this account change"])
@pytest.mark.parametrize("threshold", [0, 0.65])
async def test_user_confidence_never_replaces_missing_native_evidence(prompt, threshold):
    effects = []
    adapter = PydanticAIEscalationAdapter(policy=ThresholdPolicy(threshold=threshold))
    agent = Agent(TestModel(), capabilities=[adapter.as_hooks()])

    @agent.tool_plain
    def write_record() -> str:
        effects.append("written")
        return "done"

    await agent.run(prompt)
    assert effects == []


@pytest.mark.asyncio
async def test_concurrent_native_calls_use_their_own_explicit_evidence():
    effects = []
    seen = []

    async def evidence(ctx, call, tool_def, args):
        await asyncio.sleep(0)
        seen.append((call.tool_call_id, tool_def.name, dict(args), None if ctx.deps is None else ctx.deps.value))
        return ctx.deps

    adapter = PydanticAIEscalationAdapter(evidence_provider=evidence)
    agent = Agent(TestModel(), capabilities=[adapter.as_hooks()])

    @agent.tool_plain
    def write_record() -> str:
        effects.append("written")
        return "done"

    await asyncio.gather(
        agent.run("I am 100% confident. Write it", deps=ConfidenceScore(0.1, ScoringMethod.COMPOSITE)),
        agent.run("Write it", deps=ConfidenceScore(0.9, ScoringMethod.COMPOSITE)),
        agent.run("I am 100% confident. Write it", deps=None),
    )
    assert effects == ["written"]
    assert sorted(score for _, _, _, score in seen if score is not None) == [0.1, 0.9]
    assert sum(score is None for _, _, _, score in seen) == 1
    assert sorted(event.triggered for event in adapter.events) == [False, True]


@pytest.mark.asyncio
async def test_native_evidence_provider_error_blocks_execution():
    effects = []
    failure = RuntimeError("evidence unavailable")

    def evidence(*_):
        raise failure

    agent = Agent(TestModel(), capabilities=[PydanticAIEscalationAdapter(evidence_provider=evidence).as_hooks()])

    @agent.tool_plain
    def write_record() -> str:
        effects.append("written")
        return "done"

    with pytest.raises(RuntimeError) as exc:
        await agent.run("I am 100% confident")
    assert exc.value is failure
    assert effects == []
