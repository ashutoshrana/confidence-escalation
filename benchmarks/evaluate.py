"""Deterministic synthetic scoring illustration; no model calls or calibration claim."""
import json
from pathlib import Path
from confidence_escalation import ConfidenceEscalationMiddleware, ConfidenceScore, ScoringMethod, ThresholdPolicy


def evaluate(rows, threshold):
    gate = ConfidenceEscalationMiddleware(policy=ThresholdPolicy(threshold=threshold))
    accepted = []
    for row in rows:
        score = ConfidenceScore(row['confidence'], ScoringMethod.COMPOSITE)
        try:
            gate.call_guarded(accepted.append, score, row)
        except PermissionError:
            pass
    return {
        'threshold': threshold,
        'coverage': len(accepted) / len(rows),
        'unsafe_acceptance_count': sum(not r['correct'] for r in accepted),
        'accepted_error_rate': sum(not r['correct'] for r in accepted) / len(accepted) if accepted else None,
        'unnecessary_escalation_count': sum(r['correct'] and r not in accepted for r in rows),
        'brier_score': sum((r['confidence'] - int(r['correct'])) ** 2 for r in rows) / len(rows),
        'mean_confidence_minus_accuracy': sum(r['confidence'] - int(r['correct']) for r in rows) / len(rows),
    }

if __name__ == '__main__':
    rows = json.loads(Path(__file__).with_name('synthetic.json').read_text())
    result = [evaluate(rows, t) for t in (0.3, 0.6, 0.9)]
    assert result[1]['unsafe_acceptance_count'] == 1
    assert result[1]['unnecessary_escalation_count'] == 1
    assert result[2]['coverage'] < result[1]['coverage']
    print(json.dumps({'dataset':'synthetic illustration, n=6; not representative or fitted calibration', 'results':result}, indent=2))
