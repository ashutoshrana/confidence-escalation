
## Common Failure Patterns

| Symptom | Root cause | Fix |
|---|---|---|
| Low confidence action already executed | Response scoring happens after side effects | Add call_guarded and blocked-invocation/missing-evidence regressions |
| Package and runtime versions diverge or reuse an existing release | Release metadata was not validated against runtime, tag, and artifacts | Synchronize versions, verify wheel/sdist metadata and isolated imports, publish only the validated artifact |
| Async actions bypass intended gating and SDK hooks fail at runtime | Missing async gate and mock-only obsolete lifecycle contract | Share evidence validation, await callable results, and test native tool guardrails through installed SDK Runner |

| SDK gates collapse every tool-risk-only score to zero or permit missing evidence at threshold zero | Negative risk weight was treated as correctness confidence and the gate skipped evidence validation | Require per-call ConfidenceScore evidence, validate it before policy evaluation, pass risk separately, and exercise native SDK blocked-invocation cases |
