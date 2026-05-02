---
description: How to add a new framework adapter or confidence signal to confidence-escalation
---

# Skill: Add a New Framework Adapter or Confidence Signal

Use this when extending confidence-escalation with a new LLM framework integration or a new confidence scoring signal.

## Adding a Framework Adapter

### Files to create
1. `src/confidence_escalation/adapters/{framework}_adapter.py` — the adapter
2. `tests/test_{framework}_adapter.py` — adapter tests

### Adapter structure

```python
from __future__ import annotations
from dataclasses import dataclass
from confidence_escalation.policy import ThresholdPolicy, PolicyDecision
from confidence_escalation.handler import HumanInLoopHandler, ComplianceLoggingHandler

@dataclass(frozen=True)
class {Framework}ConfidenceAdapter:
    """Confidence-gated dispatch adapter for {Framework}."""
    
    policy: ThresholdPolicy
    handler: HumanInLoopHandler
    logger: ComplianceLoggingHandler

    def evaluate(self, response: dict, context: dict) -> PolicyDecision:
        """Evaluate confidence and route to human if below threshold."""
        score = self._compute_score(response)
        decision = self.policy.evaluate(score)
        
        self.logger.log(
            session_id=context.get("session_id"),
            confidence_score=score,
            threshold=self.policy.threshold,
            action=decision.action,
            reason=decision.reason,
            signals=decision.signals,
            context_snapshot=context,
        )
        
        if decision.requires_escalation:
            self.handler.queue(context, reason=decision.reason)
        
        return decision

    def _compute_score(self, response: dict) -> float:
        # Combine: logprob score + verbalized confidence + tool risk
        raise NotImplementedError
```

### Compliance requirements
Every adapter MUST:
- Log every decision (EU AI Act Art. 12)
- Queue to `HumanInLoopHandler` when `decision.requires_escalation` is True (EU AI Act Art. 14)
- Include `regulation_citation` in the log entry

### Test requirements
Minimum 5 tests per adapter:
1. High confidence → PROCEED decision
2. Low confidence → ESCALATE decision + handler queued
3. Borderline confidence at exact threshold
4. Missing/malformed response → safe default (escalate)
5. Audit log entry created with correct 8 fields

## Adding a Confidence Signal

### Files to create
1. `src/confidence_escalation/signals/{signal_name}.py`
2. `tests/test_{signal_name}_signal.py`

### Signal structure

```python
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class {SignalName}Scorer:
    """Compute {signal_name} confidence signal from LLM response."""
    
    weight: float = 1.0  # weighting in composite score

    def score(self, response: dict) -> float:
        """Return float in [0.0, 1.0] — higher = more confident."""
        raise NotImplementedError
```

### Signal types
- `logprob` — token log-probability from LLM API
- `verbalized` — model's explicit "I am X% confident" parsing
- `tool_risk` — risk level of the tool being invoked (high = lower effective confidence)
- `session_entropy` — conversation complexity and uncertainty accumulation

## README update (required after every new adapter)

Add to the "Supported Frameworks" table:
```
| {Framework} | `{Framework}ConfidenceAdapter` | `pip install confidence-escalation[{framework}]` |
```

Update test count in README header:
`50 tests` → `55 tests` (or actual new count)

## CHANGELOG entry

```markdown
## [vX.Y.Z] — YYYY-MM-DD

### Added — {Framework} Adapter (`{framework}_adapter.py`)

- `{Framework}ConfidenceAdapter` — confidence-gated dispatch for {Framework}
- EU AI Act Art. 12 audit logging + Art. 14 human escalation
- N new tests. Total: **NN passed**.
```
