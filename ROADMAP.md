# Public Roadmap

## Current state (v0.1.0)

Core confidence-gated escalation is implemented and tested across 4 major
agent frameworks: LangChain, CrewAI, AutoGen, and Google ADK.

---

## Near-term milestones

### 1. Additional framework adapters

- **Pydantic AI adapter** — wraps as a Pydantic AI agent hook
- **OpenAI Agents SDK adapter** — wraps as a tool/handoff guard
- **LangGraph adapter** — integrates with graph node execution boundaries
- **Smolagents adapter** — HuggingFace smolagents integration

### 2. EU AI Act Article 14 end-to-end example

A complete example demonstrating how confidence-gated escalation satisfies
EU AI Act Article 14 human oversight requirements for a high-risk AI system
(Annex III — education or employment sector).

### 3. Streaming confidence evaluation

Support for confidence evaluation on streaming LLM responses, where logprobs
are available token-by-token rather than as a complete response.

### 4. Confidence calibration utilities

Tools for calibrating threshold values against historical data from a specific
LLM and use case, to reduce both false positives (unnecessary escalation) and
false negatives (missed escalation).

### 5. OWASP Agentic AI Top 10 coverage

Explicit mapping of confidence-gated escalation to each relevant OWASP
Agentic AI Top 10 (2026) vulnerability and mitigation.
