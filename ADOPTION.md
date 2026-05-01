# Adoption

## PyPI Downloads

Verified via [pypistats.org](https://pypistats.org/packages/confidence-escalation).

| Week of | Downloads |
|---------|-----------|
| 2026-04-20 | ~0 (new package) |

Downloads are organic — no self-installs, no promotional campaigns.
Weekly downloads tracked from PyPI release date.

## How It Is Used

`confidence-escalation` implements confidence-gated escalation patterns for
LLM agents. Developers use it to:

1. **Route uncertain LLM responses to human review** before they reach
   consequential decision points (EU AI Act Article 14 compliance)
2. **Combine multiple confidence signals** — logprobs, verbalized confidence,
   tool call risk scores — into a unified escalation decision
3. **Enforce compliance audit logging** on every confidence evaluation
   (EU AI Act Article 12 structured audit records)

## Why Confidence-Gated Escalation

Standard agent frameworks handle failures at the tool or exception level.
Confidence-gated escalation catches *uncertainty before failure* — when an
agent's answer is statistically unreliable but syntactically valid.

This is architecturally required for:
- **OWASP Agentic AI ASI-09** (Human-Agent Trust Exploitation): agents that
  proceed confidently on uncertain outputs create exploitable trust vectors
- **EU AI Act Article 14 §1(d)**: high-risk AI systems must allow human
  override at every consequential decision boundary

## Related Packages

- [regulated-ai-governance](https://pypi.org/project/regulated-ai-governance/) — Policy enforcement for AI agents (CrewAI, AutoGen, LangChain, Google ADK)
- [enterprise-rag-patterns](https://pypi.org/project/enterprise-rag-patterns/) — FERPA/HIPAA/GDPR-compliant RAG retrieval patterns
- [voice-ai-governance](https://pypi.org/project/voice-ai-governance/) — Compliance enforcement for voice AI pipelines
