"""
Basic confidence-gated escalation example.

Demonstrates ThresholdPolicy with HumanInLoopHandler for EU AI Act
Article 14 human oversight compliance.

Install:
    pip install confidence-escalation

Usage:
    python examples/basic_confidence_gate.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Stub LLM call — replace with your actual LLM client
# ---------------------------------------------------------------------------

def call_llm(prompt: str) -> dict[str, Any]:
    """Simulate an LLM response with confidence metadata."""
    return {
        "output": "Based on the student record, the recommended course is MATH-301.",
        "logprobs": [-0.15, -0.22, -0.08],  # token log-probabilities
        "verbalized_confidence": 0.72,
    }


# ---------------------------------------------------------------------------
# Minimal confidence scoring (mirrors confidence_escalation.signals)
# ---------------------------------------------------------------------------

def compute_confidence(response: dict[str, Any]) -> float:
    """Aggregate logprob and verbalized confidence into a single score."""
    import math
    logprobs = response.get("logprobs", [])
    if logprobs:
        avg_logprob = sum(logprobs) / len(logprobs)
        logprob_score = math.exp(avg_logprob)
    else:
        logprob_score = 1.0
    verbal_score = response.get("verbalized_confidence", 1.0)
    return 0.6 * logprob_score + 0.4 * verbal_score


# ---------------------------------------------------------------------------
# Minimal human review queue (replace with your queue implementation)
# ---------------------------------------------------------------------------

_review_queue: list[dict[str, Any]] = []


def enqueue_for_human_review(context: dict[str, Any]) -> None:
    """Route low-confidence response to human review queue."""
    _review_queue.append(context)
    print(f"  → Routed to human review queue (queue depth: {len(_review_queue)})")


# ---------------------------------------------------------------------------
# Compliance audit log (EU AI Act Article 12)
# ---------------------------------------------------------------------------

def emit_audit_log(
    session_id: str,
    confidence: float,
    threshold: float,
    action: str,
    reason: str,
) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "confidence_score": round(confidence, 4),
        "threshold": threshold,
        "action": action,
        "reason": reason,
    }
    print(f"  AUDIT: {json.dumps(record)}")


# ---------------------------------------------------------------------------
# Main example
# ---------------------------------------------------------------------------

def process_with_confidence_gate(
    prompt: str,
    session_id: str,
    threshold: float = 0.70,
    hard_stop_threshold: float = 0.35,
) -> str | None:
    """
    Process an LLM query with confidence-gated escalation.

    EU AI Act Article 14 §1(d): High-risk AI systems must allow human
    override when the system cannot operate reliably.

    Args:
        prompt: The user query to process.
        session_id: Session identifier for audit logging.
        threshold: Confidence below this routes to human review.
        hard_stop_threshold: Confidence below this halts entirely.

    Returns:
        LLM output if confidence is sufficient, None if escalated.
    """
    response = call_llm(prompt)
    confidence = compute_confidence(response)

    print(f"\nQuery: {prompt[:60]}...")
    print(f"Confidence: {confidence:.3f} (threshold: {threshold})")

    if confidence < hard_stop_threshold:
        emit_audit_log(session_id, confidence, hard_stop_threshold, "HALT",
                       "Confidence below hard-stop threshold")
        print("  → HALT: confidence too low to proceed safely")
        return None

    if confidence < threshold:
        emit_audit_log(session_id, confidence, threshold, "ESCALATE",
                       "Confidence below soft threshold — human review required")
        enqueue_for_human_review({
            "session_id": session_id,
            "prompt": prompt,
            "response": response["output"],
            "confidence": confidence,
            "queued_at": datetime.now(timezone.utc).isoformat(),
        })
        return None

    emit_audit_log(session_id, confidence, threshold, "PROCEED",
                   "Confidence meets threshold")
    print(f"  → PROCEED: {response['output'][:80]}...")
    return response["output"]


if __name__ == "__main__":
    # High-confidence query — proceeds normally
    process_with_confidence_gate(
        prompt="What prerequisite courses are required for MATH-301?",
        session_id="sess-001",
    )

    # Simulate a lower-confidence response
    import unittest.mock as mock
    with mock.patch("__main__.call_llm", return_value={
        "output": "The student might be eligible, possibly...",
        "logprobs": [-1.8, -2.1, -1.5],
        "verbalized_confidence": 0.45,
    }):
        process_with_confidence_gate(
            prompt="Is this student eligible for the honors program?",
            session_id="sess-002",
        )
