"""Exercise the installed SDK Runner; fake model, tracing disabled, no provider calls."""
import pytest

pytest.importorskip('agents')
from agents import Agent, Model, ModelResponse, Runner, RunConfig, Usage, function_tool
from agents.exceptions import ToolInputGuardrailTripwireTriggered
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText
from confidence_escalation.adapters.openai_agents import OpenAIAgentsEscalationAdapter
from confidence_escalation import ThresholdPolicy


class ScriptedModel(Model):
    def __init__(self):
        self.calls = 0

    async def get_response(self, *args, **kwargs):
        self.calls += 1
        output = ([ResponseFunctionToolCall(id='fc_1', call_id='call_1', name='write_record', arguments='{}', type='function_call')]
                  if self.calls == 1 else [ResponseOutputMessage(id='msg_1', type='message', role='assistant', status='completed',
                    content=[ResponseOutputText(type='output_text', text='I am 90% confident', annotations=[])])])
        return ModelResponse(output=output, usage=Usage(), response_id='fake')

    async def stream_response(self, *args, **kwargs):
        raise NotImplementedError
        yield


@pytest.mark.asyncio
@pytest.mark.parametrize('allow', [False, True])
async def test_native_sdk_prevents_effect_and_hooks_complete(allow):
    adapter = OpenAIAgentsEscalationAdapter(policy=ThresholdPolicy(threshold=0 if allow else .8))
    effects = []

    @function_tool(tool_input_guardrails=[adapter.as_tool_guardrail()])
    async def write_record() -> str:
        effects.append('written')
        return 'done'

    agent = Agent(name='synthetic', model=ScriptedModel(), tools=[write_record])
    invoke = Runner.run(agent, 'synthetic fixture', hooks=adapter.as_hooks(), run_config=RunConfig(tracing_disabled=True))
    if allow:
        result = await invoke
        assert result.final_output == 'I am 90% confident'
        assert effects == ['written']
    else:
        with pytest.raises(ToolInputGuardrailTripwireTriggered):
            await invoke
        assert effects == []
