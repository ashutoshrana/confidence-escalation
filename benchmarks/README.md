# Evaluating escalation on held-out outcomes

`evaluate.py` and `synthetic.json` remain a six-case illustration. They are not a
calibration dataset. `evaluate_dataset.py` accepts your own scored, labeled tasks
and compares supplied scoring methods with always-proceed, always-abstain and a
separate tool-risk rule. It uses the Python standard library only.

```sh
python benchmarks/evaluate_dataset.py outcomes.json --error-budget 0.05 > evaluation.json
```

Choose the error budget with the application owner before examining test results.
The output includes the input file hash for reproducibility. It contains aggregate
metrics and method/slice names, not task text or IDs; still review these labels
before sharing the report.

## Dataset contract

```json
{
  "schema_version": 1,
  "probability_target": false,
  "rows": [
    {
      "id": "task-001",
      "group": "conversation-001",
      "split": "validation",
      "score_source": "versioned evaluator configuration",
      "model_version": "exact model and scoring revision",
      "scores": {"composite": 0.8, "single_signal": 0.7},
      "tool_risk": 0.2,
      "correct": true,
      "slice": "workflow-A"
    }
  ]
}
```

The row above illustrates the schema, not a usable evaluation dataset. Supply
nonempty validation and test splits with disjoint conversation/task-family groups,
one independently labeled outcome per group, unique task IDs, and the same named
score methods on every row. Repeated groups are rejected: this tool's row-level
intervals cannot treat turns from one conversation as independent trials. Aggregate
the outcome at the independent unit using a predeclared rubric, or use a separate
cluster-aware evaluation method. `risk_rule` is
reserved. All scores and risks must be finite numbers in [0,1]. Validation labels
must be boolean; test labels can be null, which remains unknown. Record labeling
rubric, disagreement, cohort selection and model versions alongside the dataset.
Use independent outcome labels, not the model's self-reported certainty as truth.

Keep any model/scorer fitting and calibration on a separate training/calibration
cohort, before this tool. Do not place related cases from that cohort in these
splits. The tool cannot detect semantic duplicates or verify label independence.

## Selection and reporting

For each method, select the highest-coverage nonempty validation threshold whose
95% Wilson upper error bound meets the chosen budget; ties prefer the more
conservative threshold. Thresholds are never selected using test labels. If no
threshold qualifies, report that result rather than presenting zero automation
as success. Inspect validation risk–coverage points in `validation_curve`.

Evaluate the selected boundary once on the held-out test set. Report coverage,
review rate, accepted errors, accepted unknowns, labeled accepted-error rate and
its interval, plus slice results. `heldout_budget_supported` is false if accepted
outcomes are missing, no tasks are accepted, or the error upper bound exceeds the
budget. Observed errors per eligible task is not a rate of harmful outcomes;
harm severity requires separate labels and analysis.

Set `probability_target` true only if every supplied score claims to estimate the
probability of the defined correctness label. This adds Brier scores and reliability
bins; a heuristic's numeric range alone does not make that interpretation valid.
The risk rule is never evaluated as a correctness probability.

Intervals assume sufficiently independent trials and do not address unmodeled
dependence, label bias, distribution shift or rare catastrophic events. Comparing
many methods and repeatedly inspecting test results can invalidate the holdout.
An initial 500 independently labeled tasks is a planning cohort, not a safety
certificate or sufficient evidence for very rare errors. No such cohort or
improvement claim is included here. If a composite fails to beat the simpler
baseline at the agreed error budget, retain the simpler rule.

## Runnable mechanics check

```sh
python -m pytest tests/test_selective_evaluation.py -q
```

These tests use explicitly synthetic cases to verify selection isolation, group
leakage rejection, unknown-label handling and probability-metric opt-in. They do
not establish predictive quality or represent real users.
