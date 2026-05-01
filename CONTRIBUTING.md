# Contributing to confidence-escalation

Thank you for your interest in contributing. This library provides
framework-agnostic confidence-gated escalation middleware for LLM agents,
addressing OWASP Agentic AI ASI-09 (Human-Agent Trust Exploitation) and
EU AI Act Article 14 (Human Oversight).

---

## Table of contents

1. [Development setup](#1-development-setup)
2. [Repository structure](#2-repository-structure)
3. [How to add a new framework adapter](#3-how-to-add-a-new-framework-adapter)
4. [How to add a new escalation handler](#4-how-to-add-a-new-escalation-handler)
5. [PR checklist](#5-pr-checklist)

---

## 1. Development setup

```bash
git clone https://github.com/ashutoshrana/confidence-escalation.git
cd confidence-escalation

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e ".[dev]"

pytest tests/ -v
```

The `[dev]` extra installs `pytest`, `pytest-cov`, `ruff`, and `mypy`.
No LLM framework dependencies are required for the core test suite —
all framework-specific code uses lazy imports with duck-typed stubs in tests.

---

## 2. Repository structure

```
src/confidence_escalation/
├── middleware.py          # ConfidenceEscalationMiddleware — main entry point
├── policies.py            # ThresholdPolicy, DualThresholdPolicy, CompositePolicyChain
├── handlers.py            # HumanInLoopHandler, ModelUpgradeHandler, ComplianceLoggingHandler
├── signals.py             # LogprobSignal, VerbalizedSignal, ToolRiskSignal
├── adapters/              # Framework-specific adapter wrappers
│   ├── langchain.py
│   ├── crewai.py
│   ├── autogen.py
│   └── google_adk.py
└── audit.py               # Structured audit log schema (EU AI Act Article 12)
tests/
examples/
```

---

## 3. How to add a new framework adapter

### Step 1 — Open an issue first

Before writing code, open an issue with the label `new-adapter`. Describe
the framework, its callback/hook mechanism, and the minimum version.

### Step 2 — Create the adapter file

Create `src/confidence_escalation/adapters/<framework>.py`:

```python
"""
<framework>.py — <Framework> adapter for confidence-gated escalation.

Lazy import: <framework-package> is imported inside methods only.

Regulatory context:
  EU AI Act Article 14 §1(d): Human override capability at every agent
  decision boundary.
"""

from __future__ import annotations
from typing import Any
from ..middleware import ConfidenceEscalationMiddleware


class <Framework>ConfidenceAdapter:
    """
    Wraps ConfidenceEscalationMiddleware as a <Framework> callback/hook.

    Lazy import: <framework-package> imported inside __init__ or run().
    """

    def __init__(self, middleware: ConfidenceEscalationMiddleware) -> None:
        self.middleware = middleware
```

### Step 3 — Write tests using duck-typed stubs

Tests must not import the optional framework:

```python
class _StubAgent:
    def __init__(self): self.calls = []
    def run(self, input): return {"output": "stub", "logprobs": [0.9]}
```

### Step 4 — Update ECOSYSTEM.md and exports

Add to `src/confidence_escalation/adapters/__init__.py` and document
in ECOSYSTEM.md.

---

## 4. How to add a new escalation handler

Create a class implementing the `EscalationHandler` protocol:

```python
from confidence_escalation.handlers import EscalationHandler, EscalationContext

class MyHandler(EscalationHandler):
    def handle(self, context: EscalationContext) -> None:
        # Route to human review, upgrade model, restrict tools, etc.
        ...
```

---

## 5. PR checklist

- [ ] `pytest tests/ -v` passes
- [ ] `ruff check src/ tests/` clean
- [ ] `mypy src/` clean
- [ ] Framework imports are lazy (inside methods, not at module level)
- [ ] Tests use duck-typed stubs (no optional framework imports in tests)
- [ ] New handler/adapter exported from relevant `__init__.py`
- [ ] CHANGELOG.md updated under `## [Unreleased]`
- [ ] Regulatory citations present in docstrings for compliance-relevant methods
- [ ] ECOSYSTEM.md updated if adding a new adapter

## Out of scope

- Hard framework dependencies (all framework imports must be lazy/optional)
- Contributions that suppress or bypass escalation for non-test purposes
- Vendor-specific sales material or proprietary implementation details
