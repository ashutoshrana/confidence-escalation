# confidence-escalation

**Framework-agnostic confidence-gated escalation middleware for LLM agents.**

[![PyPI version](https://badge.fury.io/py/confidence-escalation.svg)](https://badge.fury.io/py/confidence-escalation)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Coverage](https://codecov.io/gh/ashutoshrana/confidence-escalation/branch/main/graph/badge.svg)](https://codecov.io/gh/ashutoshrana/confidence-escalation)
[![CI](https://github.com/ashutoshrana/confidence-escalation/actions/workflows/ci.yml/badge.svg)](https://github.com/ashutoshrana/confidence-escalation/actions/workflows/ci.yml)

Multi-signal confidence scoring (logprob + verbalized + ASR + tool risk) with threshold-based escalation policies and pluggable handlers. Works with **LangChain**, **LangGraph**, **CrewAI**, **AutoGen**, **Google ADK**, and any Python agent framework.

Addresses **OWASP Agentic AI Top 10 ASI-09**: Human-Agent Trust Exploitation — prevents agents from taking high-stakes actions when confidence is insufficient.

---

## The Problem

LLM agents fail silently. When an agent is uncertain, it still returns a response — often confidently-worded — with no mechanism to:
- Detect that confidence is low before executing a high-risk tool call
- Route uncertain responses to a human reviewer
- Escalate to a stronger model when needed
- Produce a compliance audit trail of every escalation event

`confidence-escalation` solves all four.

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
