## Summary
<!-- What does this PR add or fix? -->

## Motivation
<!-- What compliance gap or use case does this address? -->

## Changes
- [ ] New framework adapter
- [ ] New confidence signal
- [ ] Bug fix
- [ ] Documentation
- [ ] Tests

## Regulatory context
<!-- Which regulations does this implement? EU AI Act Art. 14? OWASP ASI-09? -->

## Tests
<!-- Describe the tests added or updated -->

## Checklist
- [ ] Tests pass (`pytest tests/ --no-header`)
- [ ] Lint passes (`ruff check src/ tests/`)
- [ ] Every adapter decision logs a `ComplianceLog` entry (EU AI Act Art. 12)
- [ ] Low-confidence path routes to `HumanInLoopHandler` (EU AI Act Art. 14)
- [ ] No customer data, institution-specific logic, or vendor-specific artifacts included
- [ ] Patterns are framework-agnostic (adapters live in `adapters/`, not in core)
