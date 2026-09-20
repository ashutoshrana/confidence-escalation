# confidence-escalation

**Framework-agnostic confidence-gated escalation middleware for LLM agents.**

[![PyPI version](https://badge.fury.io/py/confidence-escalation.svg)](https://badge.fury.io/py/confidence-escalation)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Coverage](https://codecov.io/gh/ashutoshrana/confidence-escalation/branch/main/graph/badge.svg)](https://codecov.io/gh/ashutoshrana/confidence-escalation)
[![CI](https://github.com/ashutoshrana/confidence-escalation/actions/workflows/ci.yml/badge.svg)](https://github.com/ashutoshrana/confidence-escalation/actions/workflows/ci.yml)

Multi-signal confidence scoring (logprob + verbalized + ASR + tool risk) with threshold-based escalation policies and pluggable handlers. Works with **LangChain**, **LangGraph**, **CrewAI**, **AutoGen**, **Google ADK**, and any Python agent framework.

Provides confidence-based controls relevant to human-agent trust risks; execution is blocked only where your application explicitly wires in a pre-action gate.

---

## What it does and when to use it

This Python library helps agent application developers decide whether to continue, request review, restrict tools, or try another model when available confidence signals are weak. Use it when you can supply model or tool-risk signals and need a consistent escalation policy. The current published version is [0.3.0](https://pypi.org/project/confidence-escalation/0.3.0/).

For example, a support agent proposes an account change. Your application first checks the user's permission, then evaluates confidence and invokes the change through the pre-action gate. If evidence is missing or the policy escalates, the action is not invoked. A review handler can record or route the request; your application must recheck permission and policy before resuming it.

### Choose the right execution boundary

| Need | Use | Limit |
|---|---|---|
| Review an answer or trigger a retry after generation | `ConfidenceEscalationMiddleware.call` | Scores after the wrapped operation; cannot undo its side effects |
| Check confidence before invoking an action | `ConfidenceEscalationMiddleware.call_guarded` | Requires evidence available before execution; does not grant resource access |
| Gate an async action | `AsyncConfidenceEscalationMiddleware.call_guarded` | Await the gate; cancellation cannot undo effects already started |
| Gate OpenAI Agents SDK function tools | `as_tool_guardrail()` on `OpenAIAgentsEscalationAdapter` | Configure each tool; lifecycle hooks alone do not block execution |

Start with [basic scoring](#basic-scoring) and the [policy example](#threshold-policy--human-in-loop). For execution details, see the [sync middleware](src/confidence_escalation/middleware.py), [async middleware](src/confidence_escalation/async_middleware.py), and [SDK guardrail example](#september-2026-boundary-review-030). The [project guide](https://github.com/ashutoshrana/ashutoshrana/blob/main/PROJECT_GUIDE.md) compares the related libraries.

### What the score does not prove

A score is a weighted heuristic, not a verified probability that an answer is correct. Verbalized confidence can be wrong; missing signals must not be treated as reassuring evidence. Choose thresholds using labeled outcomes from your own use case. The included [six-case synthetic evaluation](benchmarks/results.json) illustrates metrics, not production calibration. Confidence gating also does not replace identity checks, resource authorization, durable approval storage, or legal review.

---

## Features

- **Multi-signal scoring** — combine logprobs, verbalized confidence, and tool-call risk into a single composite score
- **Threshold policies** — single-threshold, dual-threshold (normal + critical), composite multi-policy chains
- **Pluggable handlers** — human-in-loop, model upgrade, tool restriction, compliance logging
- **Framework adapters** — LangChain callbacks, CrewAI step_callback, AutoGen reply function wrapper, Google ADK event interceptor
- **EU AI Act Article 12 audit logging** — structured JSON compliance log on every escalation
- **Zero required dependencies** — core library runs with no dependencies; framework integrations are optional extras

---

## Quick Start

### Installation

```bash
pip install confidence-escalation
# With LangChain:
pip install "confidence-escalation[langchain]"
# With all frameworks:
pip install "confidence-escalation[all]"
```

### Basic Scoring

```python
from confidence_escalation import MultiSignalConfidenceScorer

scorer = MultiSignalConfidenceScorer(
    weights={"logprob": 0.5, "verbalized": 0.3, "tool_risk": -0.2}
)

score = scorer.score(
    logprobs=[-0.1, -0.3, -0.2],
    verbalized_response="I am 70% confident about this answer.",
    tool_call_risk=0.15,
)

print(f"Confidence: {score.value:.3f}")   # e.g. 0.712
print(f"Reliable: {score.is_reliable()}")  # True (above 0.6 default)
```

### Threshold Policy + Human-in-Loop

```python
from confidence_escalation import (
    ThresholdPolicy,
    EscalationAction,
    HumanInLoopHandler,
    ComplianceLoggingHandler,
    ConfidenceEscalationMiddleware,
)

def notify_human(ctx, result):
    print(f"Routing to human review: session={ctx['session_id']}, confidence={result.confidence_score:.3f}")

policy = ThresholdPolicy(
    threshold=0.65,
    action=EscalationAction.HUMAN_IN_LOOP,
    critical_threshold=0.3,
    critical_action=EscalationAction.ABORT,
)

middleware = ConfidenceEscalationMiddleware(
    policy=policy,
    handlers=[
        HumanInLoopHandler(callback=notify_human),
        ComplianceLoggingHandler(),
    ],
)

result = middleware.call(
    agent_step=lambda: my_llm.invoke(messages),
    context={"session_id": "abc123", "model": "claude-sonnet-4-6"},
    logprobs=[-0.4, -0.5],
)

if result["escalation"]["triggered"]:
    print("Escalated — stopping agent execution.")
```

### Model Upgrade Handler

```python
from confidence_escalation import ModelUpgradeHandler, ThresholdPolicy, EscalationAction

handler = ModelUpgradeHandler(
    upgrade_map={
        "claude-haiku-4-5": "claude-sonnet-4-6",
        "claude-sonnet-4-6": "claude-opus-4-7",
    }
)

policy = ThresholdPolicy(threshold=0.7, action=EscalationAction.MODEL_UPGRADE)
result = policy.evaluate(score, context={"model": "claude-haiku-4-5"})

if result.triggered:
    upgrade_info = handler.handle(result, context={"model": "claude-haiku-4-5"})
    print(f"Retry with: {upgrade_info['upgraded_model']}")
```

### Tool Restriction

```python
from confidence_escalation import ToolRestrictionHandler, ThresholdPolicy, EscalationAction

handler = ToolRestrictionHandler(
    high_risk_tools=["delete_record", "send_email", "execute_sql"],
    allow_read_only=True,
)

policy = ThresholdPolicy(threshold=0.65, action=EscalationAction.TOOL_RESTRICTION)
result = policy.evaluate(score, context={"available_tools": ["get_customer", "delete_record"]})

if result.triggered:
    restriction = handler.handle(result, context={"available_tools": agent_tools})
    safe_tools = restriction["allowed_tools"]
    # Re-invoke agent with only safe_tools
```

### LangChain Integration

```python
from confidence_escalation.adapters.langchain import LangChainEscalationAdapter
from confidence_escalation.handlers import HumanInLoopHandler

adapter = LangChainEscalationAdapter(
    threshold=0.65,
    handlers=[HumanInLoopHandler(raise_on_trigger=True)],
)

# Attach as LangChain callback
chain = LLMChain(llm=llm, callbacks=[adapter.as_callback()])

# Or call directly from a LangGraph node
def research_node(state):
    response = llm.invoke(state["messages"])
    try:
        adapter.on_llm_end(response.content, logprobs=response.response_metadata.get("logprobs"))
    except HumanInLoopHandler.HumanReviewRequired:
        return {"status": "escalated"}
    return {"response": response.content}
```

### CrewAI Integration

```python
from crewai import Agent
from confidence_escalation.adapters.crewai import CrewAIEscalationAdapter

adapter = CrewAIEscalationAdapter(threshold=0.65)

agent = Agent(
    role="Research Specialist",
    goal="Analyze market trends",
    backstory="...",
    step_callback=adapter.step_callback,
)
```

### Google ADK Integration

```python
from google.adk.agents import BaseAgent
from confidence_escalation.adapters.google_adk import ADKEscalationAdapter

class GovernedAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._escalation = ADKEscalationAdapter(threshold=0.65)

    async def _run_async_impl(self, ctx):
        async for event in self._llm_agent._run_async_impl(ctx):
            if event.is_final_response():
                result = self._escalation.evaluate_event(event, ctx)
                if result["triggered"]:
                    yield self._escalation.build_escalation_event(result)
                    return
            yield event
```

---

## Composite Policy Chains

```python
from confidence_escalation import ThresholdPolicy, EscalationAction
from confidence_escalation.policy import CompositePolicy

policy = CompositePolicy(policies=[
    ThresholdPolicy(threshold=0.25, action=EscalationAction.ABORT),
    ThresholdPolicy(threshold=0.55, action=EscalationAction.HUMAN_IN_LOOP),
    ThresholdPolicy(threshold=0.75, action=EscalationAction.COMPLIANCE_LOG),
])

result = policy.evaluate(score, context={"session_id": "abc"})
# First matching threshold wins
```

---

## OWASP Agentic AI Coverage

| OWASP ASI ID | Risk | Coverage |
|-------------|------|----------|
| ASI-09 | Human-Agent Trust Exploitation | Confidence gating before high-stakes actions |
| ASI-02 | Tool Misuse | Tool restriction handler removes high-risk tools at low confidence |
| ASI-03 | Identity/Privilege Abuse | ComplianceLoggingHandler creates immutable audit trail |

---

## Related Packages

- [voice-ai-governance](https://github.com/ashutoshrana/voice-ai-governance) — HIPAA/FERPA/EU AI Act compliance for voice AI pipelines
- [regulated-ai-governance](https://github.com/ashutoshrana/regulated-ai-governance) — Runtime tool authorization and capability scoping
- [enterprise-rag-patterns](https://github.com/ashutoshrana/enterprise-rag-patterns) — FERPA/HIPAA/GDPR-compliant RAG patterns

---

## License

MIT License. See [LICENSE](LICENSE).

## Reliability updates (0.2.0)

Explicit call_guarded pre-action gating; finite/range input validation and missing-signal metadata; reproducible synthetic evaluation. Introduced in 0.2.0.

`ConfidenceEscalationMiddleware.call` scores after execution. Use `middleware.call_guarded(action, confidence, *args, context=context, **kwargs)` before side effects. Escalation or explicitly missing evidence raises `PermissionError` without invoking the action. Re-evaluate after human review; callbacks do not grant approval. This gate does not replace identity/resource authorization.

Scores/thresholds must be finite in [0,1], log probabilities finite and non-positive. Missing signals are marked; weights are heuristics, not calibrated probabilities. Run `python benchmarks/evaluate.py` to reproduce [results](benchmarks/results.json). Six synthetic hand-labeled cases illustrate coverage, error/abstention, Brier score, and aggregate confidence bias. They are not representative or fitted calibration; collect independently labeled domain outcomes before choosing deployment thresholds.

## September 2026 boundary review (0.3.0)

Add async pre-action gating, await async callable objects, reject deferred execution through the synchronous gate, and exercise native SDK tool input guardrails with the real Runner.

Use `await AsyncConfidenceEscalationMiddleware().call_guarded(action, confidence)` for async side effects. The async class lives in `confidence_escalation.async_middleware`. Low or missing confidence blocks before invocation; sync callables run in a worker thread, async callables and returned awaitables are awaited. Cancellation cannot undo effects already started. The synchronous gate rejects async actions and awaitable results; use the async API instead. Confidence checks do not replace identity/resource authorization.

For OpenAI Agents SDK integration, install `pip install ".[dev,openai-agents]"` from this checkout on Python 3.10+. SDK 0.22.3 is tested through its actual Runner with a scripted local model and tracing disabled:

```python
from agents import function_tool
from confidence_escalation.adapters.openai_agents import OpenAIAgentsEscalationAdapter
adapter = OpenAIAgentsEscalationAdapter()

# Populate from application-controlled pre-action evaluation, keyed by call ID.
# An empty mapping blocks every call; never populate it with a constant approval.
evidence_by_call_id = {}

def evidence_for_call(data):
    return evidence_by_call_id.get(data.context.tool_call_id)

@function_tool(tool_input_guardrails=[adapter.as_tool_guardrail(evidence_for_call)])
async def protected_action() -> str:
    return "performed"
```

Attach `adapter.as_hooks()` to Runner for lifecycle observations; hooks alone are not the execution boundary. Tool input guardrails raise the SDK tripwire before a triggered function tool executes. **Migration to explicit evidence:** direct `evaluate_tool_gate()` calls must supply
`confidence=ConfidenceScore(...)`; native guardrails use a synchronous or asynchronous
`evidence_provider(data)` returning a `ConfidenceScore` for that invocation. Missing
or invalid evidence blocks even at threshold zero. Provider failures also stop the
call; the SDK may wrap the exception. No score from an earlier response is reused.
Tool risk is separate `tool_risk` policy context and never converted into correctness
confidence. The default threshold policy compares the supplied confidence; use an
explicit policy if risk should alter the decision. `high_risk_tools=None` selects
defaults, while an empty set now stays empty. Lifecycle tool hooks do not evaluate
confidence policies without evidence. Post-response scoring is unchanged.

The application owns evidence provenance, freshness, binding to tool arguments,
and resource authorization. Supplying a number does not prove calibration or
correctness. The example mapping is an integration placeholder, not durable approval
storage. This does not cover hosted tools or calls outside the configured SDK tool path.

[Official tool guardrail guidance](https://openai.github.io/openai-agents-python/guardrails/) distinguishes input guardrails before execution from output checks afterward. [Lifecycle signatures](https://openai.github.io/openai-agents-python/ref/lifecycle/) include agent, tool, response, and output arguments exercised by the regression test. Run `python -m pytest tests/test_openai_sdk_execution.py`; the dedicated CI installs the SDK so this check cannot silently skip there.
