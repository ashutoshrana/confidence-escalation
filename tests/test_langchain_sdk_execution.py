"""Provider-free native LangChain callback contract."""

import pytest

pytest.importorskip("langchain_core")

from langchain_core.callbacks.manager import CallbackManager
from langchain_core.outputs import Generation, LLMResult

from confidence_escalation.adapters.langchain import LangChainEscalationAdapter
from confidence_escalation.policy import ThresholdPolicy


@pytest.mark.parametrize("score,blocked", [(20, True), (90, False)])
def test_native_callback_manager_receives_post_response_decision(score, blocked):
    decisions = []
    adapter = LangChainEscalationAdapter(policy=ThresholdPolicy(threshold=0.65, on_escalation=decisions.append))
    manager = CallbackManager([adapter.as_callback()])
    runs = manager.on_llm_start({"name": "synthetic"}, ["synthetic prompt"])
    runs[0].on_llm_end(LLMResult(generations=[[Generation(text=f"I am {score}% confident")]]))
    assert bool(decisions) is blocked
    if blocked:
        assert decisions[0].confidence_score == 0.2


def test_native_callback_errors_propagate_after_generation():
    failure = RuntimeError("notification failed")

    class Broken:
        def supports(self, action): return True
        def handle(self, result, context): raise failure

    adapter = LangChainEscalationAdapter(handlers=[Broken()])
    manager = CallbackManager([adapter.as_callback()])
    run = manager.on_llm_start({"name": "synthetic"}, ["synthetic prompt"])[0]
    with pytest.raises(RuntimeError) as exc:
        run.on_llm_end(LLMResult(generations=[[Generation(text="I am 20% confident")]]))
    assert exc.value is failure
