"""
EU AI Act Article 14 — End-to-End Compliance Example
=====================================================

This example demonstrates how confidence-escalation implements EU AI Act
Article 14 (human oversight) obligations for AI agents invoking tools.

EU AI Act Art. 14 §1(d): High-risk AI systems must "allow the natural persons
to whom human oversight is assigned to intervene in the operation of the
high-risk AI system."

OWASP Agentic AI ASI-09: Confidence-gated dispatch prevents autonomous
execution when the agent's certainty falls below a configured threshold.

What this example shows:
  1. PydanticAIEscalationAdapter blocking a low-confidence tool call
  2. OpenAIAgentsEscalationAdapter scoring a post-LLM response
  3. Compliance audit log entries (EU AI Act Art. 12)
  4. EscalationEvent inspection for downstream reporting
  5. Escalation callback (on_escalation) capturing human-review events

Install:
    pip install confidence-escalation

Usage:
    python examples/eu_ai_act_art14_compliance.py
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock

from confidence_escalation.adapters.pydantic_ai import (
    PydanticAIEscalationAdapter,
    SkipToolExecution,
)
from confidence_escalation.adapters.openai_agents import OpenAIAgentsEscalationAdapter
from confidence_escalation.policy import EscalationAction, ThresholdPolicy


# ---------------------------------------------------------------------------
# Section 1: PydanticAI — blocking a low-confidence tool call
# ---------------------------------------------------------------------------

SECTION_DIVIDER = "=" * 60


def section(title: str) -> None:
    print(f"\n{SECTION_DIVIDER}")
    print(f"  {title}")
    print(SECTION_DIVIDER)


def demo_pydantic_ai_block() -> None:
    """EU AI Act Art. 14: SkipToolExecution blocks low-confidence tool call."""
    section("1. Pydantic AI — LOW CONFIDENCE → tool blocked")

    adapter = PydanticAIEscalationAdapter(
        threshold=0.70,
        high_risk_tools=frozenset({"update_student_record", "send_financial_aid_email"}),
    )
    hooks = adapter.as_hooks()

    # Simulate RunContext with no prior high-confidence message
    ctx = MagicMock()
    ctx.messages = []  # no verbalized confidence in history

    # Simulate a ToolCallPart for a high-risk operation
    call = MagicMock()
    call.tool_name = "update_student_record"
    call.tool_call_id = "tc_hipaa_001"

    tool_def = MagicMock()
    tool_def.name = "update_student_record"
    tool_def.description = "Update a student's academic record (FERPA-protected)"

    args = {"student_id": "S12345", "grade": "A", "course_id": "CS-401"}

    print("\n[Tool call] update_student_record — high-risk, no prior confidence signal")

    try:
        asyncio.run(hooks.before_tool_execute(ctx, call=call, tool_def=tool_def, args=args))
        print("  ⚠️  Gate passed (unexpected in this scenario)")
    except SkipToolExecution as exc:
        print(f"  ✅ SkipToolExecution raised — tool BLOCKED")
        print(f"  📋 Result returned to agent: {exc.result}")

    print(f"\n  Escalation events recorded: {len(adapter.events)}")
    for ev in adapter.events:
        print(f"    triggered={ev.triggered}  action={ev.action.value}")
        print(f"    confidence={ev.confidence_score:.3f}  reason={ev.reason}")


def demo_pydantic_ai_pass() -> None:
    """EU AI Act Art. 14: High-confidence response allows tool to execute."""
    section("2. Pydantic AI — HIGH CONFIDENCE → tool allowed")

    adapter = PydanticAIEscalationAdapter(
        threshold=0.65,
        high_risk_tools=frozenset({"delete_record"}),
    )
    hooks = adapter.as_hooks()

    # Simulate RunContext with a prior high-confidence assistant message
    msg = MagicMock()
    part = MagicMock()
    part.content = (
        "I am 92% confident the student's transcript is correct and the "
        "requested update is valid based on professor confirmation."
    )
    msg.parts = [part]
    ctx = MagicMock()
    ctx.messages = [msg]

    call = MagicMock()
    call.tool_name = "search_student_records"
    call.tool_call_id = "tc_read_001"
    tool_def = MagicMock()
    tool_def.name = "search_student_records"

    args = {"student_id": "S12345", "fields": ["gpa", "courses"]}

    print("\n[Tool call] search_student_records — low risk, 92% confidence in history")

    try:
        result_args = asyncio.run(
            hooks.before_tool_execute(ctx, call=call, tool_def=tool_def, args=args)
        )
        print(f"  ✅ Gate passed — args returned unchanged: {list(result_args.keys())}")
    except SkipToolExecution as exc:
        print(f"  ❌ Unexpectedly blocked: {exc.result}")


# ---------------------------------------------------------------------------
# Section 2: OpenAI Agents — scoring a post-LLM response
# ---------------------------------------------------------------------------

def demo_openai_agents_scoring() -> None:
    """EU AI Act Art. 12: audit log entry for every LLM response scored."""
    section("3. OpenAI Agents — scoring post-LLM response (Art. 12 audit)")

    adapter = OpenAIAgentsEscalationAdapter(
        threshold=0.65,
        high_risk_tools=frozenset({"send_financial_aid_email", "enroll_student"}),
    )

    print("\n[Response 1] High confidence response from LLM")
    r1 = adapter.score_response(
        response_text="I am 88% confident this financial aid calculation is correct.",
        context={"agent": "financial-aid-agent", "regulation": "FERPA §99.31"},
    )
    print(f"  confidence={r1['confidence']:.3f}  triggered={r1['triggered']}")

    print("\n[Response 2] Uncertain response from LLM — triggers escalation")
    r2 = adapter.score_response(
        response_text=(
            "I am uncertain about this financial aid calculation "
            "and not sure which eligibility rule applies here."
        ),
        context={"agent": "financial-aid-agent", "regulation": "FERPA §99.31"},
    )
    print(f"  confidence={r2['confidence']:.3f}  triggered={r2['triggered']}")

    print(f"\n  Total events in audit log: {len(adapter.events)}")
    for i, ev in enumerate(adapter.events, 1):
        print(f"  [{i}] triggered={ev.triggered}  action={ev.action.value}  "
              f"confidence={ev.confidence_score:.3f}")


# ---------------------------------------------------------------------------
# Section 3: Escalation callback (EU AI Act Art. 14 notification chain)
# ---------------------------------------------------------------------------

def demo_escalation_callback() -> None:
    """EU AI Act Art. 14: escalation callback fires on every blocked tool call."""
    section("4. Escalation callback — human review notification chain")

    escalation_log: list[Any] = []

    def on_escalation(result: Any) -> None:
        escalation_log.append({
            "timestamp": "2026-05-01T00:00:00Z",
            "action": result.action.value,
            "confidence": result.confidence_score,
            "reason": result.reason,
            "regulation": "EU AI Act Art. 14 §1(d)",
        })
        print(f"  🔔 ESCALATION CALLBACK fired → action={result.action.value}  "
              f"confidence={result.confidence_score:.3f}")

    adapter = PydanticAIEscalationAdapter(
        policy=ThresholdPolicy(
            threshold=0.80,
            action=EscalationAction.HUMAN_IN_LOOP,
            critical_threshold=None,
            on_escalation=on_escalation,
        ),
        high_risk_tools=frozenset({"process_payment", "update_credentials"}),
    )
    hooks = adapter.as_hooks()

    ctx = MagicMock()
    ctx.messages = []
    call = MagicMock()
    call.tool_name = "process_payment"
    call.tool_call_id = "tc_pay_001"
    tool_def = MagicMock()
    tool_def.name = "process_payment"

    print("\n[Tool call] process_payment (high risk, no confidence signal)")
    try:
        asyncio.run(hooks.before_tool_execute(ctx, call=call, tool_def=tool_def, args={}))
    except SkipToolExecution:
        pass

    print(f"\n  Escalation log entries: {len(escalation_log)}")
    print("  " + json.dumps(escalation_log[0], indent=4).replace("\n", "\n  "))


# ---------------------------------------------------------------------------
# Section 4: OpenAI Agents on_tool_start hook
# ---------------------------------------------------------------------------

async def demo_openai_agents_hooks() -> None:
    """EU AI Act Art. 14: OpenAI Agents on_tool_start gate."""
    section("5. OpenAI Agents Hooks — on_tool_start pre-tool gate")

    captured: list[Any] = []

    def on_esc(result: Any) -> None:
        captured.append(result)

    adapter = OpenAIAgentsEscalationAdapter(
        policy=ThresholdPolicy(
            threshold=0.90,
            action=EscalationAction.HUMAN_IN_LOOP,
            critical_threshold=None,
            on_escalation=on_esc,
        ),
        high_risk_tools=frozenset({"send_financial_aid_email"}),
    )
    hooks = adapter.as_hooks()

    # Simulate ToolContext
    ctx = MagicMock()
    ctx.tool_call_id = "tc_email_001"
    ctx.tool_arguments = {"recipient": "student@university.edu"}
    ctx.raw_responses = []

    tool = MagicMock()
    tool.name = "send_financial_aid_email"

    print("\n[on_tool_start] send_financial_aid_email — high risk, threshold=0.90")
    await hooks.on_tool_start(ctx, tool)

    print(f"\n  Gate result: triggered={bool(captured)}")
    if captured:
        print(f"  Escalation captured: action={captured[0].action.value}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n🏛️  EU AI Act Article 14 — Confidence-Gated Tool Execution")
    print("   Demonstrates human oversight obligations for AI agents\n")

    demo_pydantic_ai_block()
    demo_pydantic_ai_pass()
    demo_openai_agents_scoring()
    demo_escalation_callback()
    asyncio.run(demo_openai_agents_hooks())

    print(f"\n{SECTION_DIVIDER}")
    print("  ✅ All EU AI Act Art. 14 compliance scenarios completed")
    print(f"{SECTION_DIVIDER}\n")


if __name__ == "__main__":
    main()
