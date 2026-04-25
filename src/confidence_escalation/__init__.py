"""
confidence-escalation: Framework-agnostic confidence-gated escalation for LLM agents.

Computes multi-signal confidence scores (logprob, semantic consistency,
tool-call risk) and applies configurable threshold policies to trigger
escalation actions: human-in-loop callback, model upgrade, tool restriction,
or compliance logging.

Integrates with LangChain, LangGraph, CrewAI, AutoGen, Google ADK,
Semantic Kernel, and any framework supporting tool-call interception.

OWASP Agentic AI Top 10 mitigation: ASI-09 (Human-Agent Trust Exploitation).
"""

from confidence_escalation.scorer import (
    ConfidenceScorer,
    ConfidenceScore,
    ScoringMethod,
    MultiSignalConfidenceScorer,
)
from confidence_escalation.policy import (
    EscalationPolicy,
    ThresholdPolicy,
    PolicyResult,
    EscalationAction,
)
from confidence_escalation.handlers import (
    EscalationHandler,
    HumanInLoopHandler,
    ModelUpgradeHandler,
    ToolRestrictionHandler,
    ComplianceLoggingHandler,
)
from confidence_escalation.middleware import (
    ConfidenceEscalationMiddleware,
    EscalationEvent,
)
from confidence_escalation.adapters.langchain import LangChainEscalationAdapter
from confidence_escalation.adapters.crewai import CrewAIEscalationAdapter
from confidence_escalation.adapters.autogen import AutoGenEscalationAdapter
from confidence_escalation.adapters.google_adk import ADKEscalationAdapter

__version__ = "0.1.0"
__all__ = [
    # Scorer
    "ConfidenceScorer",
    "ConfidenceScore",
    "ScoringMethod",
    "MultiSignalConfidenceScorer",
    # Policy
    "EscalationPolicy",
    "ThresholdPolicy",
    "PolicyResult",
    "EscalationAction",
    # Handlers
    "EscalationHandler",
    "HumanInLoopHandler",
    "ModelUpgradeHandler",
    "ToolRestrictionHandler",
    "ComplianceLoggingHandler",
    # Middleware
    "ConfidenceEscalationMiddleware",
    "EscalationEvent",
    # Adapters
    "LangChainEscalationAdapter",
    "CrewAIEscalationAdapter",
    "AutoGenEscalationAdapter",
    "ADKEscalationAdapter",
]
