"""Exercise the installed SDK Runner; fake model, tracing disabled, no provider calls."""
import pytest

pytest.importorskip('agents')
from agents import Agent, Model, ModelResponse, Runner, RunConfig, Usage, function_tool
from agents.exceptions import ToolInputGuardrailTripwireTriggered, UserError
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText
from confidence_escalation.adapters.openai_agents import OpenAIAgentsEscalationAdapter
from confidence_escalation import ThresholdPolicy, ConfidenceScore, ScoringMethod


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
@pytest.mark.parametrize('mode', ['allow', 'low', 'missing', 'missing_flag', 'invalid', 'async_allow', 'no_provider', 'provider_error'])
async def test_native_sdk_prevents_effect_and_hooks_complete(mode):
    # Missing/invalid evidence must block even when the threshold would allow zero.
    adapter = OpenAIAgentsEscalationAdapter(policy=ThresholdPolicy(threshold=.8 if mode == 'low' else 0))
    effects = []
    evidence_requests = []

    def evidence(data):
        evidence_requests.append(data.context.tool_name)
        if mode == 'provider_error':
            raise RuntimeError('evidence unavailable')
        if mode == 'missing':
            return None
        value = ConfidenceScore(.2 if mode == 'low' else .9, ScoringMethod.COMPOSITE,
                                metadata={'has_evidence': mode != 'missing_flag'})
        if mode == 'invalid':
            value.value = float('nan')
        return value

    async def async_evidence(data):
        return evidence(data)

    provider = None if mode == 'no_provider' else (async_evidence if mode == 'async_allow' else evidence)

    @function_tool(tool_input_guardrails=[adapter.as_tool_guardrail(provider)])
    async def write_record() -> str:
        effects.append('written')
        return 'done'

    agent = Agent(name='synthetic', model=ScriptedModel(), tools=[write_record])
    invoke = Runner.run(agent, 'synthetic fixture', hooks=adapter.as_hooks(), run_config=RunConfig(tracing_disabled=True))
    if mode in ('allow', 'async_allow'):
        result = await invoke
        assert result.final_output == 'I am 90% confident'
        assert effects == ['written']
    elif mode == 'provider_error':
        with pytest.raises(UserError, match='evidence unavailable'):
            await invoke
        assert effects == []
    else:
        with pytest.raises(ToolInputGuardrailTripwireTriggered):
            await invoke
        assert effects == []
    assert evidence_requests == ([] if mode == 'no_provider' else ['write_record'])
