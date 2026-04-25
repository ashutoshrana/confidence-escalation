"""
Multi-signal confidence scoring for LLM agents.

Computes composite confidence scores from logprob, semantic consistency,
tool-call risk, and verbalized confidence signals. Framework-agnostic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

__all__ = [
    "ScoringMethod",
    "ConfidenceScore",
    "ConfidenceScorer",
    "MultiSignalConfidenceScorer",
]


class ScoringMethod(str, Enum):
    LOGPROB = "logprob"
    VERBALIZED = "verbalized"
    SEMANTIC_CONSISTENCY = "semantic_consistency"
    TOOL_CALL_RISK = "tool_call_risk"
    COMPOSITE = "composite"


@dataclass
class ConfidenceScore:
    value: float  # [0.0, 1.0]
    method: ScoringMethod
    signals: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_reliable(self, threshold: float = 0.6) -> bool:
        return self.value >= threshold

    def __float__(self) -> float:
        return self.value


class ConfidenceScorer:
    """Single-method confidence scorer."""

    def __init__(self, method: ScoringMethod = ScoringMethod.VERBALIZED):
        self.method = method

    def score_from_logprobs(self, logprobs: List[float]) -> ConfidenceScore:
        if not logprobs:
            return ConfidenceScore(value=0.5, method=ScoringMethod.LOGPROB)
        avg_logprob = sum(logprobs) / len(logprobs)
        # Convert log probability to [0, 1] range
        confidence = math.exp(avg_logprob)
        return ConfidenceScore(
            value=min(1.0, max(0.0, confidence)),
            method=ScoringMethod.LOGPROB,
            signals={"avg_logprob": avg_logprob, "num_tokens": len(logprobs)},
        )

    def score_from_verbalized(self, verbalized: str) -> ConfidenceScore:
        """
        Parse verbalized confidence from LLM response.
        e.g. "I am 85% confident that..." → 0.85
        """
        import re
        patterns = [
            r"(\d{1,3})%\s*confident",
            r"confidence[:\s]+(\d{1,3})%",
            r"(\d{1,3})%\s*sure",
            r"certainty[:\s]+(\d{1,3})%",
        ]
        for pattern in patterns:
            match = re.search(pattern, verbalized, re.IGNORECASE)
            if match:
                pct = float(match.group(1))
                return ConfidenceScore(
                    value=min(1.0, pct / 100.0),
                    method=ScoringMethod.VERBALIZED,
                    signals={"raw_percentage": pct},
                )
        # Keyword-based fallback — check uncertainty first (more specific phrases win)
        text_lower = verbalized.lower()
        if any(w in text_lower for w in ["uncertain", "not sure", "unclear", "possibly", "not certain"]):
            return ConfidenceScore(value=0.35, method=ScoringMethod.VERBALIZED)
        if any(w in text_lower for w in ["certain", "definitely", "sure"]):
            return ConfidenceScore(value=0.85, method=ScoringMethod.VERBALIZED)
        return ConfidenceScore(value=0.5, method=ScoringMethod.VERBALIZED)


class MultiSignalConfidenceScorer:
    """
    Composite confidence scorer combining multiple signal sources.

    Weights are configurable per deployment context.
    Default weights are calibrated for enterprise voice AI.

    Example:
        scorer = MultiSignalConfidenceScorer(
            weights={"logprob": 0.5, "verbalized": 0.3, "tool_risk": 0.2}
        )
        score = scorer.score(
            logprobs=[-0.1, -0.3, -0.2],
            verbalized_response="I am 70% confident about this.",
            tool_call_risk=0.2,
        )
    """

    DEFAULT_WEIGHTS = {
        "logprob": 0.50,
        "verbalized": 0.25,
        "tool_risk": -0.25,  # High tool risk reduces confidence
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS
        self._scorer = ConfidenceScorer()

    def score(
        self,
        logprobs: Optional[List[float]] = None,
        verbalized_response: Optional[str] = None,
        tool_call_risk: Optional[float] = None,
        additional_signals: Optional[Dict[str, float]] = None,
    ) -> ConfidenceScore:
        signals = {}
        composite = 0.0
        weight_sum = 0.0

        if logprobs is not None:
            lp_score = self._scorer.score_from_logprobs(logprobs)
            w = self.weights.get("logprob", 0.5)
            composite += lp_score.value * w
            weight_sum += abs(w)
            signals["logprob"] = lp_score.value

        if verbalized_response is not None:
            v_score = self._scorer.score_from_verbalized(verbalized_response)
            w = self.weights.get("verbalized", 0.25)
            composite += v_score.value * w
            weight_sum += abs(w)
            signals["verbalized"] = v_score.value

        if tool_call_risk is not None:
            w = self.weights.get("tool_risk", -0.25)
            # High tool risk subtracts from confidence
            composite += tool_call_risk * w
            weight_sum += abs(w)
            signals["tool_risk"] = tool_call_risk

        if additional_signals:
            for name, value in additional_signals.items():
                w = self.weights.get(name, 0.1)
                composite += value * w
                weight_sum += abs(w)
                signals[name] = value

        if weight_sum > 0:
            normalized = composite / weight_sum
        else:
            normalized = 0.5

        return ConfidenceScore(
            value=max(0.0, min(1.0, normalized)),
            method=ScoringMethod.COMPOSITE,
            signals=signals,
        )
