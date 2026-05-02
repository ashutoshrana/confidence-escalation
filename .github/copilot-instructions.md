# Copilot Instructions — confidence-escalation

## Project Purpose
`confidence-escalation` is a Python middleware library implementing **confidence-gated escalation** for LLM agents. It routes to human reviewers when an agent's confidence falls below a configurable threshold — directly implementing EU AI Act Article 14 human oversight obligations and OWASP Agentic AI ASI-09.

## Core Concepts
- **Confidence-Gated Dispatch (CGD)** — the central pattern: measure multi-signal confidence, compare to threshold, escalate if below
- **Multi-signal scoring** — combines logprobs, verbalized confidence, tool risk, session entropy
- **ThresholdPolicy** — configurable per use-case (GDPR, HIPAA, FERPA, TCPA, EU AI Act)
- **HumanInLoopHandler** — queues escalated tasks; implements Art. 14 §1(d) override capability
- **ComplianceLoggingHandler** — 8-field JSON audit log per Art. 12 §4.4

## Package Structure
```
src/confidence_escalation/
  policy.py          — ThresholdPolicy, PolicyConfig
  handler.py         — HumanInLoopHandler, ComplianceLoggingHandler
  adapters/          — LangChain, LlamaIndex, Google ADK, AutoGen framework adapters
  signals/           — logprob scorer, verbalized scorer, tool risk scorer
examples/
  basic_confidence_gate.py   — minimal working example
tests/
  test_policy.py, test_handler.py, test_adapters.py
```

## Code Conventions
- All core types are `@dataclass(frozen=True)`
- Every escalation decision produces a `ComplianceLog` entry with: timestamp, session_id, confidence_score, threshold, action, reason, signals, context_snapshot
- Adapters must not import from each other — only from `confidence_escalation.policy` and `confidence_escalation.handler`
- Tests use `pytest` with no external network calls

## Regulatory Citations
- EU AI Act Art. 14 — human oversight; Art. 12 — audit logging
- OWASP Agentic AI ASI-09 — confidence-gated dispatch pattern
- HIPAA §164.312 — access controls for PHI in AI pipelines
- TCPA 47 U.S.C. § 227 — consent verification before AI-initiated contact

## What NOT to Include
- No customer names, institution names, or product names (ELLA, Falcon, Polaris, ASTRUM, SEI, Capella, Strayer)
- No production-specific configuration or environment variables
- No vendor-specific cloud infrastructure (GCP project IDs, AWS account IDs)
- Patterns must be framework-agnostic — adapters live in `adapters/` not in core

## PR Standards
- PR title: conventional commits format — `feat: add AutoGen confidence adapter` / `fix: logprob scorer edge case`
- Every new adapter needs: implementation file + test file + README entry
- Tests must run without network access (`pytest tests/ --no-header`)
