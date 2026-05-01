# Changelog

All notable changes to this project are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] — 2026-04-25

### Added — Initial release

**Core middleware**
- `ConfidenceEscalationMiddleware` — main entry point; wraps any LLM call with
  confidence evaluation and policy-based escalation
- Multi-signal confidence scoring: logprob aggregation, verbalized confidence
  extraction, tool-call risk assessment
- `ThresholdPolicy` — single-threshold escalation (route to human below threshold)
- `DualThresholdPolicy` — two-level policy (soft escalation vs. hard stop)
- `CompositePolicyChain` — compose multiple policies in sequence

**Handlers**
- `HumanInLoopHandler` — enqueues context for human review; implements EU AI Act
  Article 14 §1(d) override capability
- `ModelUpgradeHandler` — retries with a higher-capability model on low confidence
- `ComplianceLoggingHandler` — structured JSON audit log (8-field schema per
  EU AI Act Article 12: timestamp, session_id, confidence_score, threshold,
  action, reason, signals, context_snapshot)

**Framework adapters**
- LangChain adapter — wraps as a `BaseCallbackHandler`
- CrewAI adapter — wraps as a crew-level callback
- AutoGen adapter — wraps as a message hook
- Google ADK adapter — wraps as a `BeforeModelInvocationCallback`

**Tests**
- 50 tests across core middleware, policies, handlers, and all 4 adapters
- All tests use duck-typed stubs; no optional framework dependencies required

---

## [Unreleased]

### Planned
- Pydantic AI adapter
- OpenAI Agents SDK adapter
- EU AI Act Article 14 compliance example (end-to-end)
- Streaming confidence evaluation support
