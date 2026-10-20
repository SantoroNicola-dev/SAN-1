"""
3-Factor Confidence-Gated Router for S1stem Hybrid Vision Pipeline.
===================================================================
Evaluates System-1 outputs using a rigorous multi-factor arbitration policy:
1. Top-1 Confidence Threshold (tau): Ensures sufficient peak probability.
2. Margin Separation Threshold (delta): Ensures clear separation between top-1 and top-2 candidates.
3. Shannon Entropy Threshold (H_max): Guarantees low probability dispersion/uncertainty.
"""

from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass, field
from .s1_engine import S1DecisionResult


class RouteTarget(str, Enum):
    FAST_PATH = "System-1 (Fast Path)"
    DELIBERATIVE_PATH = "System-2 (Deliberation Engine)"


@dataclass
class RouteDecision:
    target: RouteTarget
    is_fast_path: bool
    confidence: float
    margin: float
    entropy: float
    flags: List[str]
    escalation_reason: Optional[str]
    thresholds: Dict[str, float]
    telemetry: Dict[str, Any] = field(default_factory=dict)


class ConfidenceGatedRouter:
    """
    Arbitrates whether an image prediction is certified directly by System-1
    or escalated to System-2 Deliberation Engine.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.75,
        margin_threshold: float = 0.15,
        entropy_threshold: float = 1.50,
    ):
        self.confidence_threshold = float(confidence_threshold)
        self.margin_threshold = float(margin_threshold)
        self.entropy_threshold = float(entropy_threshold)

    def evaluate(self, s1_result: S1DecisionResult) -> RouteDecision:
        """
        Executes the 3-Factor evaluation on the S1stem prediction result.
        Returns a detailed RouteDecision object.
        """
        flags: List[str] = []

        if s1_result.confidence < self.confidence_threshold:
            flags.append(
                f"LOW_CONFIDENCE: {s1_result.confidence:.3f} < {self.confidence_threshold:.3f}"
            )

        if s1_result.margin < self.margin_threshold:
            flags.append(
                f"NARROW_MARGIN: {s1_result.margin:.3f} < {self.margin_threshold:.3f}"
            )

        if s1_result.entropy > self.entropy_threshold:
            flags.append(
                f"HIGH_ENTROPY: {s1_result.entropy:.3f} > {self.entropy_threshold:.3f}"
            )

        is_fast_path = (len(flags) == 0)
        target = RouteTarget.FAST_PATH if is_fast_path else RouteTarget.DELIBERATIVE_PATH
        escalation_reason = "; ".join(flags) if flags else None

        thresholds = {
            "confidence_threshold": self.confidence_threshold,
            "margin_threshold": self.margin_threshold,
            "entropy_threshold": self.entropy_threshold,
        }

        telemetry = {
            "latency_ms": s1_result.latency_ms,
            "winner_id": s1_result.winner_id,
            "winner_label": s1_result.winner_label,
            "num_candidates": len(s1_result.probabilities),
            "top2_ids": [c["id"] for c in s1_result.top_candidates[:2]],
            "flags_count": len(flags),
        }

        return RouteDecision(
            target=target,
            is_fast_path=is_fast_path,
            confidence=s1_result.confidence,
            margin=s1_result.margin,
            entropy=s1_result.entropy,
            flags=flags,
            escalation_reason=escalation_reason,
            thresholds=thresholds,
            telemetry=telemetry,
        )
