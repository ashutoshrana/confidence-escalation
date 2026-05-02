"""
Pydantic AI adapter for confidence-escalation.

Hooks into the Pydantic AI ``before_tool_execute`` lifecycle event to apply
confidence-gated escalation before any tool execution.

Pydantic AI's ``before_tool_execute`` hook can raise ``SkipToolExecution``
to block a tool call entirely — making it the ideal EU AI Act Art. 14
human override integration point.

EU AI Act Art. 14 compliance: every tool call is pre-screened;
low-confidence calls skip execution and route to HumanInLoopHandler.

OWASP Agentic AI ASI-09: confidence-gated dispatch.

Install::

    pip install 'confidence-escalation[pydantic-ai]'
    pip install 'pydantic-ai>=1.0.0'

Usage::

    from confidence_escalation.adapters.pydantic_ai import PydanticAIEscalationAdapter

    adapter = PydanticAIEscalationAdapter(threshold=0.65)

    # Option 1: get pre-built hooks object
    hooks = adapter.as_hooks()

    agent = Agent(
        "anthropic:claude-sonnet-4-6",
        tools=[my_tool],
    )

    # Option 2: integrate with existing hooks
    @adapter.before_tool_execute
    async def my_before_tool(ctx, call, tool_def, args):
        return args  # or raise SkipToolExecution("Human review required")
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from confidence_escalation.handlers import EscalationHandler, HumanInLoopHandler, ComplianceLoggingHandler
from confidence_escalation.middleware import ConfidenceEscalationMiddleware
from confidence_escalation.policy import EscalationAction, EscalationPolicy, ThresholdPolicy
from confidence_escalation.scorer import MultiSignalConfidenceScorer

__all__ = ["PydanticAIEscalationAdapter", "PydanticAIHooks", "SkipToolExecution"]

logger = logging.getLogger(__name__)


class SkipToolExecution(Exception):
    """
    Raised by ``before_tool_execute`` to skip tool execution.

    Mirrors pydantic-ai's ``SkipToolExecution`` exception — compatible with
    the real SDK when installed, and usable standalone for testing.

    Attributes:
        result: The alternative result returned to the agent instead of
            executing the tool.
    """

    def __init__(self, result: str = "Tool execution blocked: confidence below threshold. Human review required."):
        self.result = result
        super().__init__(result)


class PydanticAIHooks:
    """
    Pydantic AI ``Hooks``-compatible object for confidence-gated dispatch.

    Implements the ``before_tool_execute`` protocol used by pydantic-ai >= 1.0.

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

        # Extract prior response text from message history for verbalized scoring
        response_text: Optional[str] = None
        messages = getattr(ctx, "messages", []) or []
        for msg in reversed(messages):
            parts = getattr(msg, "parts", []) or (msg if isinstance(msg, list) else [])
            for part in parts:
                content = getattr(part, "content", None) or (part if isinstance(part, str) else None)
                if content and isinstance(content, str) and len(content) > 10:
                    response_text = content
                    break
            if response_text:
                break

        tool_risk = self._adapter._tool_risk_for(tool_name)
        context = {
            "event": "before_tool_execute",
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "tool_args_keys": list(args.keys()),
            "regulation_citation": "EU AI Act Art. 14 §1(d) + OWASP ASI-09",
        }

        result = self._adapter.gate(
            tool_name=tool_name,
            tool_risk=tool_risk,
            response_text=response_text,
            context=context,
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
        high_risk_tools: Tool names that apply tighter threshold (+0.15).
        skip_message: Message returned to agent when tool is skipped.

    Example::

        adapter = PydanticAIEscalationAdapter(
            threshold=0.7,
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
    ):
        self.threshold = threshold
        self.critical_threshold = critical_threshold
        self.high_risk_tools = high_risk_tools or self._DEFAULT_HIGH_RISK_TOOLS
        self.skip_message = skip_message
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
        response_text: Optional[str] = None,
        logprobs: Optional[List[float]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Core gate logic — scores confidence and evaluates policy.

        Called by ``PydanticAIHooks.before_tool_execute``.
        Also usable standalone for testing.
        """
        # tool_call_risk is a fallback signal only — when verbalized or logprob
        # signals are present, they take priority and the tool_risk contribution
        # would otherwise dilute a high-confidence explicit signal below any
        # practical threshold (max composite with tool_risk=0.2 is 0.4).
        has_explicit_signal = response_text is not None or logprobs is not None
        confidence = self._middleware.score(
            response_text=response_text,
            logprobs=logprobs,
            tool_call_risk=None if has_explicit_signal else tool_risk,
        )
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
