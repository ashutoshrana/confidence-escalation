
## Common Failure Patterns

| Symptom | Root cause | Fix |
|---|---|---|
| Low confidence action already executed | Response scoring happens after side effects | Add call_guarded and blocked-invocation/missing-evidence regressions |
| Package and runtime versions diverge or reuse an existing release | Release metadata was not validated against runtime, tag, and artifacts | Synchronize versions, verify wheel/sdist metadata and isolated imports, publish only the validated artifact |
| Async actions bypass intended gating and SDK hooks fail at runtime | Missing async gate and mock-only obsolete lifecycle contract | Share evidence validation, await callable results, and test native tool guardrails through installed SDK Runner |

| SDK gates collapse every tool-risk-only score to zero or permit missing evidence at threshold zero | Negative risk weight was treated as correctness confidence and the gate skipped evidence validation | Require per-call ConfidenceScore evidence, validate it before policy evaluation, pass risk separately, and exercise native SDK blocked-invocation cases |
| Evaluation appears safe when all tasks abstain or test outcomes are unknown | Zero accepted cases and missing labels were treated as favorable evidence | Report unknown intervals and labels explicitly; require nonzero labeled coverage and select thresholds only on disjoint validation groups |
| Completed token streams have no final event and failed escalation notifications appear successful | Stream omitted event recording, and gather results silently discarded exceptions | Share final event recording and propagate original handler failures after all dispatched handlers finish |
| A Pydantic AI hooks object passes direct-call tests but fails in the native Agent | The object did not inherit AbstractCapability and its skip exception was not the SDK exception | Use native base types when installed, preserve standalone behavior, and verify blocked tools with the native provider-free TestModel |
| Release workflow uses an advisory-affected or mutable publishing action | A dependency update in one repository left equivalent release workflows unchanged | Pin every publishing workflow to the verified patched action commit and validate each workflow before release |
| Python 3.9 CI fails before tests start | ubuntu-latest moved to Ubuntu 26.04 without Python 3.9 builds | Pin the supported Python test matrix to Ubuntu 24.04 and verify the official Python version manifest |
