"""OpenAI Agents SDK adapter with explicit pre-action evidence.

Attach ``as_tool_guardrail(evidence_provider)`` to each protected function tool.
The provider receives SDK ToolInputGuardrailData and returns a ConfidenceScore
(or an awaitable score) for that invocation. Missing evidence blocks execution.
Lifecycle hooks observe responses; they are not an authorization boundary.
Install the optional dependency with ``confidence-escalation[openai-agents]``.
"""

from __future__ import annotations

import datetime
import json
import logging
import inspect
from typing import Any, Callable, Dict, List, Optional

from confidence_escalation.handlers import EscalationHandler, HumanInLoopHandler, ComplianceLoggingHandler
from confidence_escalation.middleware import ConfidenceEscalationMiddleware
from confidence_escalation.policy import EscalationAction, EscalationPolicy, ThresholdPolicy
from confidence_escalation.scorer import ConfidenceScore, MultiSignalConfidenceScorer, validate_probability

__all__ = ["OpenAIAgentsEscalationAdapter", "OpenAIAgentsHooks"]

logger = logging.getLogger(__name__)


try:
    from agents.lifecycle import RunHooksBase as _RunHooksBase
except ImportError:
    class _RunHooksBase:  # Optional SDK; native integration requires installation.
        pass


class OpenAIAgentsHooks(_RunHooksBase):
    """
    Minimal implementation of openai-agents RunHooksBase interface.

    Implements ``on_tool_start`` (tool-risk observation) and
    ``on_llm_end`` (post-response confidence scoring).

    Subclasses the installed SDK base when available. Use as_tool_guardrail
    for blocking; hooks record lifecycle observations only.
    """

    def __init__(self, adapter: "OpenAIAgentsEscalationAdapter"):
        self._adapter = adapter

    async def on_tool_start(self, ctx: Any, agent: Any, tool: Any = None) -> None:
        """Observe tool impact without generating a confidence or approval verdict."""
        if tool is None:  # Retain legacy direct two-argument calls.
            tool = agent
        tool_name = getattr(tool, "name", str(tool))
        tool_risk = self._adapter._tool_risk_for(tool_name)
        logger.debug("Observed tool start: %s (risk=%.2f)", tool_name, tool_risk)

    async def on_llm_end(self, ctx: Any, agent: Any, response: Any = None) -> None:
        """Post-LLM-response confidence scoring (EU AI Act Art. 12 audit log)."""
        if response is not None:
            text = "".join(
                getattr(content, "text", "")
                for item in getattr(response, "output", [])
                for content in getattr(item, "content", [])
                if getattr(content, "type", None) == "output_text"
            )
            if text:
                self._adapter.score_response(text, context={"event": "on_llm_end"})
            return
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

    async def on_llm_start(self, context: Any, agent: Any, system_prompt: Any, input_items: Any) -> None:
        pass

    async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: Any) -> None:
        pass

    async def on_agent_start(self, ctx: Any, agent: Any) -> None:
        pass

    async def on_agent_end(self, ctx: Any, agent: Any, output: Any = None) -> None:
        pass

    async def on_handoff(self, ctx: Any, from_agent: Any, to_agent: Any) -> None:
        pass


class OpenAIAgentsEscalationAdapter:
    """Evaluate explicit evidence at native function-tool input boundaries.

    Thresholds are application choices, not regulatory recommendations. Tool risk
    is passed separately in policy context and is not a correctness probability.
    high_risk_tools=None uses defaults; an empty set classifies no tools as high
    risk. Configure a policy that consumes tool_risk if action impact should alter
    the decision. Resource authorization remains the application's responsibility.
    """

    # Default tool impact labels supplied to policy context
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
        self.high_risk_tools = self._DEFAULT_HIGH_RISK_TOOLS if high_risk_tools is None else high_risk_tools
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
        *,
        confidence: Optional[ConfidenceScore] = None,
    ) -> Dict[str, Any]:
        """Evaluate explicit pre-action evidence; risk remains separate policy context.

        Missing evidence raises PermissionError even for a zero threshold. No score
        is inferred from tool risk, tool names, or a previous response. The caller
        owns evidence provenance, freshness, and resource authorization.
        """
        validate_probability(tool_risk, "tool_risk")
        if confidence is None:
            raise PermissionError("Action blocked: pre-action confidence evidence is required")
        if not isinstance(confidence, ConfidenceScore):
            raise TypeError("confidence must be a ConfidenceScore")
        confidence.require_evidence()
        context = dict(context or {})
        context.update(tool_name=tool_name, tool_risk=tool_risk)
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

    def as_tool_guardrail(self, evidence_provider: Optional[Callable[[Any], Any]] = None) -> Any:
        """Build an SDK function-tool input guardrail using per-invocation evidence.

        evidence_provider receives the SDK ToolInputGuardrailData and returns a
        ConfidenceScore (or an awaitable score). None/missing/invalid evidence blocks
        execution. Provider errors propagate; no previous score is reused. Hooks
        remain observations and do not substitute for this execution boundary.
        """
        from agents.tool_guardrails import tool_input_guardrail, ToolGuardrailFunctionOutput

        @tool_input_guardrail
        async def confidence_gate(data: Any) -> Any:
            name = data.context.tool_name
            confidence = evidence_provider(data) if evidence_provider is not None else None
            if inspect.isawaitable(confidence):
                confidence = await confidence
            try:
                result = self.evaluate_tool_gate(
                    name, self._tool_risk_for(name), {"event": "tool_input_guardrail"},
                    confidence=confidence,
                )
            except (PermissionError, ValueError, TypeError):
                return ToolGuardrailFunctionOutput.raise_exception({"reason": "missing or invalid confidence evidence"})
            if result["triggered"]:
                return ToolGuardrailFunctionOutput.raise_exception({"reason": "confidence escalation"})
            return ToolGuardrailFunctionOutput.allow()

        return confidence_gate

    def as_hooks(self) -> "OpenAIAgentsHooks":
        """Return a ``RunHooksBase``-compatible hook object."""
        return OpenAIAgentsHooks(adapter=self)

    @property
    def events(self):
        return list(self._local_events)
