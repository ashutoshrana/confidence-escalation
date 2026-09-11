
## Common Failure Patterns

| Symptom | Root cause | Fix |
|---|---|---|
| Low confidence action already executed | Response scoring happens after side effects | Add call_guarded and blocked-invocation/missing-evidence regressions |
| Package and runtime versions diverge or reuse an existing release | Release metadata was not validated against runtime, tag, and artifacts | Synchronize versions, verify wheel/sdist metadata and isolated imports, publish only the validated artifact |
