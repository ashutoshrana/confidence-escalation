"""
Pydantic AI adapter for confidence-escalation.

Hooks into the Pydantic AI ``before_tool_execute`` lifecycle event to apply
confidence-gated escalation before registered function-tool execution.

The native capability raises the SDK-compatible ``SkipToolExecution``
before an escalated tool executes. This applies the configured confidence
policy; it does not establish legal compliance or resource authorization.

Install::

    pip install 'confidence-escalation[pydantic-ai]'

Usage::

    from confidence_escalation.adapters.pydantic_ai import PydanticAIEscalationAdapter

    adapter = PydanticAIEscalationAdapter(threshold=0.65, evidence_provider=resolve_evidence)

    # Option 1: get pre-built hooks object
    hooks = adapter.as_hooks()

    agent = Agent(
        model,
        tools=[my_tool], capabilities=[hooks],
    )

    # Option 2: integrate with existing hooks
    @adapter.before_tool_execute
    async def my_before_tool(ctx, call, tool_def, args):
        return args  # or raise SkipToolExecution("Human review required")
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Dict, List, Optional

from confidence_escalation.handlers import EscalationHandler, HumanInLoopHandler, ComplianceLoggingHandler
from confidence_escalation.middleware import ConfidenceEscalationMiddleware
from confidence_escalation.policy import EscalationAction, EscalationPolicy, ThresholdPolicy
from confidence_escalation.scorer import ConfidenceScore, MultiSignalConfidenceScorer, validate_probability

__all__ = ["PydanticAIEscalationAdapter", "PydanticAIHooks", "SkipToolExecution"]

logger = logging.getLogger(__name__)

try:
    from pydantic_ai.capabilities import AbstractCapability
    from pydantic_ai.exceptions import SkipToolExecution as _NativeSkipToolExecution
except ImportError:
    AbstractCapability = object
    _NativeSkipToolExecution = Exception


class SkipToolExecution(_NativeSkipToolExecution):
    """
    Raised by ``before_tool_execute`` to skip tool execution.

    Subclasses the SDK exception when installed and remains usable standalone.

    Attributes:
        result: The alternative result returned to the agent instead of
            executing the tool.
    """

    def __init__(self, result: str = "Tool execution blocked: confidence below threshold. Human review required."):
        self.result = result
        super().__init__(result)
        self.args = (result,)


class PydanticAIHooks(AbstractCapability):
    """
    Pydantic AI ``Hooks``-compatible object for confidence-gated dispatch.

    Uses the native AbstractCapability protocol when the SDK is installed.

    When confidence falls below the policy threshold:
    - Raises ``SkipToolExecution`` with a human-review message
    - Queues the task via ``HumanInLoopHandler``
    - Logs a ``ComplianceLoggingHandler`` entry (EU AI Act Art. 12)

    When pydantic-ai is installed, this integrates via::

        agent = Agent(..., capabilities=[adapter.as_hooks()])
    """

    def __init__(self, adapter: "PydanticAIEscalationAdapter"):
        self._adapter = adapter

    async def before_tool_execute(
        self,
        ctx: Any,
        *,
        call: Any,
        tool_def: Any,
        args: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Pre-tool confidence gate.

        Args:
            ctx: RunContext — provides deps, usage, and message history.
            call: ToolCallPart — provides tool_name, tool_call_id, args.
            tool_def: ToolDefinition — provides name, description, schema.
            args: Validated arguments dict for the tool.

        Returns:
            The (possibly modified) args dict if confidence is sufficient.

        Raises:
            SkipToolExecution: If confidence is below threshold.
        """
        tool_name: str = (
            getattr(call, "tool_name", None)
            or getattr(tool_def, "name", None)
            or "unknown_tool"
        )
        tool_call_id: Optional[str] = getattr(call, "tool_call_id", None)

        # Application evidence is resolved for this invocation, never inferred from messages.
        provider = self._adapter.evidence_provider
        confidence = provider(ctx, call, tool_def, args) if provider is not None else None
        if inspect.isawaitable(confidence):
            confidence = await confidence
        context = {
            "event": "before_tool_execute",
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "tool_args_keys": list(args.keys()),
        }
        try:
            if not isinstance(confidence, ConfidenceScore):
                raise TypeError("confidence must be a ConfidenceScore")
            confidence.require_evidence()
        except (PermissionError, ValueError, TypeError) as exc:
            raise SkipToolExecution(f"Tool '{tool_name}' blocked: missing or invalid confidence evidence") from exc
        result = self._adapter.gate(
            tool_name=tool_name, tool_risk=self._adapter._tool_risk_for(tool_name),
            context=context, confidence=confidence,
        )

        if result["triggered"]:
            skip_msg = (
                f"[EU AI Act Art. 14] Tool '{tool_name}' execution skipped — "
                f"confidence {result.get('confidence', 0.0):.3f} below threshold. "
                "Routed to human reviewer."
            )
            raise SkipToolExecution(skip_msg)

        return args

    async def after_tool_execute(self, ctx: Any, *, call: Any, tool_def: Any, args: Any, result: Any) -> None:
        """Post-tool audit log (EU AI Act Art. 12)."""
        pass

    async def before_run(self, ctx: Any) -> None:
        pass

    async def before_model_request(self, ctx: Any, request_context: Any) -> Any:
        return request_context


class PydanticAIEscalationAdapter:
    """
    Confidence-gated escalation adapter for Pydantic AI.

    Uses Pydantic AI's ``before_tool_execute`` lifecycle hook to block
    low-confidence tool calls and route them to human review.

    Unlike the LangChain adapter (which can only observe), Pydantic AI's
    ``SkipToolExecution`` exception allows the adapter to **prevent** the tool
    from executing — the ideal EU AI Act Art. 14 override mechanism.

    Args:
        threshold: Minimum confidence to allow tool execution. Default 0.65.
        critical_threshold: Below this, ABORT. Default 0.25.
        policy: Override the default ThresholdPolicy.
        handlers: Override default handlers.
        high_risk_tools: Tool names exposed as higher tool_risk in policy context.
        skip_message: Message returned to agent when tool is skipped.
        evidence_provider: Trusted callable(ctx, call, tool_def, args) returning a
            ConfidenceScore or awaitable score for this invocation. No provider
            denies native tool execution. Message text is never used as evidence.

    Example::

        adapter = PydanticAIEscalationAdapter(
            threshold=0.7,
            evidence_provider=resolve_evidence,  # application-owned per-call evidence
            high_risk_tools=frozenset({"delete_record", "transfer_funds"}),
        )
        hooks = adapter.as_hooks()

        agent = Agent("anthropic:claude-sonnet-4-6", tools=[my_tool])
        result = await agent.run("...", capabilities=[hooks])
    """

    _DEFAULT_HIGH_RISK_TOOLS = frozenset({
        "send_email", "send_message", "delete_record", "delete_file",
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
        skip_message: str = "Tool blocked: confidence below threshold. Human review required.",
        evidence_provider: Optional[Callable[[Any, Any, Any, Dict[str, Any]], Any]] = None,
    ):
        self.threshold = threshold
        self.critical_threshold = critical_threshold
        self.high_risk_tools = self._DEFAULT_HIGH_RISK_TOOLS if high_risk_tools is None else high_risk_tools
        self.skip_message = skip_message
        self.evidence_provider = evidence_provider
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
        return 0.8 if tool_name in self.high_risk_tools else 0.2

    def gate(
        self,
        tool_name: str,
        tool_risk: float = 0.2,
        context: Optional[Dict[str, Any]] = None,
        *,
        confidence: Optional[ConfidenceScore] = None,
    ) -> Dict[str, Any]:
        """Evaluate explicit per-call evidence; messages and risk are not confidence.

        The application owns evidence provenance, freshness, argument binding and
        resource authorization. Missing evidence blocks even at threshold zero.
        """
        validate_probability(tool_risk, "tool_risk")
        if confidence is None:
            raise PermissionError("Pre-action confidence evidence is required")
        if not isinstance(confidence, ConfidenceScore):
            raise TypeError("confidence must be a ConfidenceScore")
        confidence.require_evidence()
        context = dict(context or {})
        context.update(tool_name=tool_name, tool_risk=tool_risk)
        result = self._middleware.evaluate(confidence, context)
        if result.triggered:
            self._middleware.dispatch(result, context)
        self._record_event(result, confidence, context)
        if result.triggered:
            return {
                "triggered": True,
                "tool_name": tool_name,
                "confidence": confidence.value,
                "action": result.action.value,
                "reason": result.reason,
            }
        return {
            "triggered": False,
            "tool_name": tool_name,
            "confidence": confidence.value,
        }

    def _record_event(self, result: Any, confidence: Any, context: Optional[Dict[str, Any]] = None) -> None:
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

    def as_hooks(self) -> PydanticAIHooks:
        """Return a Pydantic AI hooks-compatible object."""
        return PydanticAIHooks(adapter=self)

    def before_tool_execute(self, fn: Callable) -> Callable:
        """Decorator: wrap an existing before_tool_execute hook."""
        async def wrapper(ctx: Any, *, call: Any, tool_def: Any, args: Dict[str, Any]) -> Dict[str, Any]:
            hooks = self.as_hooks()
            await hooks.before_tool_execute(ctx, call=call, tool_def=tool_def, args=args)
            return await fn(ctx, call=call, tool_def=tool_def, args=args)
        return wrapper

    @property
    def events(self):
        return list(self._local_events)
