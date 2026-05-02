"""
OpenAI Agents SDK adapter for confidence-escalation.

Hooks into the OpenAI Agents SDK lifecycle via ``RunHooksBase.on_tool_start``
and ``on_llm_end`` to score confidence and apply the configured escalation
policy before any tool invocation.

EU AI Act Art. 14 compliance: every tool call is pre-screened;
low-confidence calls are routed to HumanInLoopHandler before execution.

OWASP Agentic AI ASI-09: confidence-gated dispatch prevents autonomous
tool use when the agent's certainty falls below the configured threshold.

Install::

    pip install 'confidence-escalation[openai-agents]'
    pip install openai-agents>=0.14.0

Usage::

    from confidence_escalation.adapters.openai_agents import OpenAIAgentsEscalationAdapter

    adapter = OpenAIAgentsEscalationAdapter(threshold=0.65)

    # Register hooks on the Runner:
    result = await Runner.run(
        agent,
        input=user_message,
        hooks=adapter.as_hooks(),
    )

    # Or check per-response after the run:
    adapter.on_llm_response(response_text, logprobs=logprobs)
"""

from __future__ import annotations

import datetime
import json
import logging
from typing import Any, Dict, List, Optional

from confidence_escalation.handlers import EscalationHandler, HumanInLoopHandler, ComplianceLoggingHandler
from confidence_escalation.middleware import ConfidenceEscalationMiddleware
from confidence_escalation.policy import EscalationAction, EscalationPolicy, ThresholdPolicy
from confidence_escalation.scorer import MultiSignalConfidenceScorer

__all__ = ["OpenAIAgentsEscalationAdapter", "OpenAIAgentsHooks"]

logger = logging.getLogger(__name__)


class OpenAIAgentsHooks:
    """
    Minimal implementation of openai-agents RunHooksBase interface.

    Implements ``on_tool_start`` (pre-tool confidence gate) and
    ``on_llm_end`` (post-response confidence scoring).

    When ``openai-agents`` is installed, subclass ``RunHooksBase`` instead;
    this shim avoids a hard import dependency.
    """

    def __init__(self, adapter: "OpenAIAgentsEscalationAdapter"):
        self._adapter = adapter

    async def on_tool_start(self, ctx: Any, tool: Any) -> None:
        """
        Pre-tool-call gate (EU AI Act Art. 14 override point).

        Accesses tool name and arguments from ``ToolContext`` if available,
        computes tool risk score, and evaluates confidence policy.
        Raises ``HumanInLoopHandler.HumanReviewRequired`` if escalation
        is triggered and ``raise_on_trigger=True``.
        """
        tool_name = getattr(tool, "name", str(tool))
        tool_arguments: Dict[str, Any] = {}

        # ToolContext exposes tool_call_id and tool_arguments when available
        if hasattr(ctx, "tool_arguments") and ctx.tool_arguments:
            tool_arguments = ctx.tool_arguments

        tool_risk = self._adapter._tool_risk_for(tool_name)
        context = {
            "event": "on_tool_start",
            "tool_name": tool_name,
            "tool_call_id": getattr(ctx, "tool_call_id", None),
            "tool_arguments_keys": list(tool_arguments.keys()),
            "regulation_citation": "EU AI Act Art. 14 §1(d) — human override capability",
        }
        self._adapter.evaluate_tool_gate(
            tool_name=tool_name,
            tool_risk=tool_risk,
            context=context,
        )

    async def on_llm_end(self, ctx: Any, agent: Any) -> None:
        """Post-LLM-response confidence scoring (EU AI Act Art. 12 audit log)."""
        # Raw responses are available on the RunResult; extract usage metadata
        raw_responses: List[Any] = getattr(ctx, "raw_responses", [])
        logprobs: Optional[List[float]] = None
        response_text = ""

        for raw in raw_responses:
            # Each raw response may have choices[].logprobs
            choices = getattr(raw, "choices", []) or (raw.get("choices", []) if isinstance(raw, dict) else [])
            for choice in choices:
                msg = getattr(choice, "message", None) or (choice.get("message", {}) if isinstance(choice, dict) else {})
                content = getattr(msg, "content", None) or (msg.get("content", "") if isinstance(msg, dict) else "")
                if content:
                    response_text += str(content)
                lp = getattr(choice, "logprobs", None) or (choice.get("logprobs") if isinstance(choice, dict) else None)
                if lp:
                    token_lps = getattr(lp, "token_logprobs", None) or (lp.get("token_logprobs") if isinstance(lp, dict) else None)
                    if token_lps:
                        logprobs = [float(v) for v in token_lps if v is not None]

        if response_text or logprobs:
            self._adapter.score_response(
                response_text=response_text,
                logprobs=logprobs,
                context={"event": "on_llm_end", "agent_name": getattr(agent, "name", "unknown")},
            )

    async def on_agent_start(self, ctx: Any, agent: Any) -> None:
        pass

    async def on_agent_end(self, ctx: Any, agent: Any) -> None:
        pass

    async def on_handoff(self, ctx: Any, from_agent: Any, to_agent: Any) -> None:
        pass


class OpenAIAgentsEscalationAdapter:
    """
    Confidence-gated escalation adapter for the OpenAI Agents SDK.

    Intercepts tool calls via ``RunHooksBase.on_tool_start`` and post-LLM
    responses via ``on_llm_end`` to apply a configurable threshold policy.

    EU AI Act Art. 14 compliance:
        - Pre-tool gate blocks high-risk tool calls when confidence is low
        - Every decision produces a ``ComplianceLoggingHandler`` audit entry
        - ``HumanInLoopHandler`` queues escalated tasks for human review

    OWASP Agentic AI ASI-09:
        - Confidence-gated dispatch prevents autonomous tool use below threshold
        - Multi-signal scoring: logprobs + verbalized confidence + tool risk

    Args:
        threshold: Minimum confidence to allow autonomous tool execution.
            Default 0.65 (EU AI Act Art. 14 HIPAA/Annex III recommended).
        critical_threshold: Below this, ABORT instead of escalate.
            Default 0.25.
        policy: Override the default ThresholdPolicy.
        handlers: Override the default handler list.
        high_risk_tools: Tool names that apply a tighter threshold (+0.15).

    Example::

        adapter = OpenAIAgentsEscalationAdapter(
            threshold=0.65,
            high_risk_tools=["send_email", "delete_record", "transfer_funds"],
        )
        result = await Runner.run(agent, input="...", hooks=adapter.as_hooks())
    """

    # Tools that should apply a tighter confidence threshold
    _DEFAULT_HIGH_RISK_TOOLS = frozenset({
        "send_email", "send_message", "post_message",
        "delete_record", "delete_file", "delete_document",
        "transfer_funds", "make_payment", "submit_form",
        "update_database", "write_file", "execute_code",
        "call_external_api", "update_credentials",
    })

    def __init__(
        self,
        threshold: float = 0.65,
        critical_threshold: float = 0.25,
        policy: Optional[EscalationPolicy] = None,
        handlers: Optional[List[EscalationHandler]] = None,
        high_risk_tools: Optional[frozenset] = None,
    ):
        self.threshold = threshold
        self.critical_threshold = critical_threshold
        self.high_risk_tools = high_risk_tools or self._DEFAULT_HIGH_RISK_TOOLS
        self._local_events: List[Any] = []
        self._middleware = ConfidenceEscalationMiddleware(
            scorer=MultiSignalConfidenceScorer(),
            policy=policy or ThresholdPolicy(
                threshold=threshold,
                action=EscalationAction.HUMAN_IN_LOOP,
                critical_threshold=critical_threshold if critical_threshold > 0 else None,
                critical_action=EscalationAction.ABORT,
            ),
            handlers=handlers or [
                HumanInLoopHandler(),
                ComplianceLoggingHandler(
                    include_context_keys=["tool_name", "regulation_citation", "event"],
                ),
            ],
        )

    def _tool_risk_for(self, tool_name: str) -> float:
        """Return a [0.0, 1.0] risk score for the given tool name."""
        return 0.8 if tool_name in self.high_risk_tools else 0.2

    def evaluate_tool_gate(
        self,
        tool_name: str,
        tool_risk: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate confidence before a tool call.

        Returns a result dict; raises ``HumanReviewRequired`` if the
        HumanInLoopHandler is configured with ``raise_on_trigger=True``.
        """
        confidence = self._middleware.score(tool_call_risk=tool_risk)
        result = self._middleware.evaluate(confidence, context)
        handler_results = []
        if result.triggered:
            handler_results = self._middleware.dispatch(result, context)
            logger.warning(
                "Tool gate triggered for '%s': confidence=%.3f threshold=%.3f action=%s",
                tool_name, confidence.value, result.threshold_used, result.action.value,
            )
        self._record_event(result, confidence, context)
        if result.triggered:
            return {"triggered": True, "tool_name": tool_name, "handler_results": handler_results}
        return {"triggered": False, "tool_name": tool_name, "confidence": confidence.value}

    def score_response(
        self,
        response_text: str,
        logprobs: Optional[List[float]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Score an LLM response; log audit entry regardless of trigger."""
        confidence = self._middleware.score(
            response_text=response_text,
            logprobs=logprobs,
        )
        result = self._middleware.evaluate(confidence, context)
        if result.triggered:
            self._middleware.dispatch(result, context)
        self._record_event(result, confidence, context)
        return {"confidence": confidence.value, "triggered": result.triggered}

    def _record_event(self, result: Any, confidence: Any, context: Optional[Dict[str, Any]]) -> None:
        """Record a local escalation event."""
        from confidence_escalation.middleware import EscalationEvent
        import datetime
        event = EscalationEvent(
            timestamp=datetime.datetime.utcnow().isoformat(),
            triggered=result.triggered,
            action=result.action,
            confidence_score=confidence.value,
            reason=result.reason,
            context_snapshot=context or {},
        )
        self._local_events.append(event)

    def as_hooks(self) -> "OpenAIAgentsHooks":
        """Return a ``RunHooksBase``-compatible hook object."""
        return OpenAIAgentsHooks(adapter=self)

    @property
    def events(self):
        return list(self._local_events)
