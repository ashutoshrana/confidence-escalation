"""Native, provider-free Pydantic AI capability execution."""

import pytest

pytest.importorskip("pydantic_ai")

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from confidence_escalation.adapters.pydantic_ai import PydanticAIEscalationAdapter
from confidence_escalation.policy import ThresholdPolicy


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked", [False, True])
async def test_native_capability_blocks_before_tool_execution(blocked):
    effects = []
    adapter = PydanticAIEscalationAdapter(policy=ThresholdPolicy(threshold=0.99 if blocked else 0))
    agent = Agent(TestModel(), capabilities=[adapter.as_hooks()])

    @agent.tool_plain
    def write_record() -> str:
        effects.append("written")
        return "done"

    await agent.run("synthetic fixture")
    assert effects == ([] if blocked else ["written"])
    assert adapter.events[-1].triggered is blocked
