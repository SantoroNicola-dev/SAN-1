"""
Hybrid Vision Pipeline: S1stem Fast Path + System-2 Deliberation Path Orchestrator.
===================================================================================
Orchestrates high-speed deterministic image classification (Fast Path)
and escalates borderline/ambiguous cases to System-2 Deliberation Engine.
"""

from typing import Union, List, Dict, Any, Optional
from pathlib import Path
import time
from PIL import Image
import numpy as np

from .preprocessing import ImagePreprocessor
from .s1_engine import S1stemVisionEngine, S1DecisionResult
from .router import ConfidenceGatedRouter, RouteDecision, RouteTarget
from .system2 import BaseDeliberator, DualStrategySystem2, DeliberationOutcome


class HybridVisionPipeline:
    """
    Unified end-to-end coordinator for Hybrid System-1 / System-2 Image Classification.
    """

    def __init__(
        self,
        s1_engine: Optional[S1stemVisionEngine] = None,
        router: Optional[ConfidenceGatedRouter] = None,
        system2: Optional[BaseDeliberator] = None,
        preprocessor: Optional[ImagePreprocessor] = None,
        confidence_threshold: float = 0.75,
        margin_threshold: float = 0.15,
        entropy_threshold: float = 1.50,
        model_path: Optional[str] = None,
        text_model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
    ):
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.s1_engine = s1_engine or S1stemVisionEngine(
            model_path=model_path,
            text_model_path=text_model_path,
            tokenizer_path=tokenizer_path,
        )
        self.router = router or ConfidenceGatedRouter(
            confidence_threshold=confidence_threshold,
            margin_threshold=margin_threshold,
            entropy_threshold=entropy_threshold,
        )
        self.system2 = system2 or DualStrategySystem2(mode="adaptive")

        # Telemetry aggregation
        self.stats = {
            "total_requests": 0,
            "fast_path_count": 0,
            "deliberation_count": 0,
            "total_fast_path_ms": 0.0,
            "total_deliberation_ms": 0.0,
        }

    def register_classes(self, classes: List[Union[str, Dict[str, Any]]]) -> int:
        """Registers candidate classes in S1stem RAM vector cache."""
        return self.s1_engine.register_classes(classes)

    def process(
        self,
        image_input: Union[str, Path, bytes, Image.Image, np.ndarray],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end classification pipeline:
        1. Preprocess input image to normalized tensor.
        2. Fast S1stem inference (<15 ms).
        3. 3-Factor Gated routing evaluation.
        4. If confident -> Certify immediately (Fast Path).
        5. If uncertain -> Escalate to System-2 Deliberation Engine.
        """
        t_start = time.perf_counter()
        metadata = metadata or {}
        self.stats["total_requests"] += 1

        # 1. Preprocess Image
        tensor = self.preprocessor.preprocess(image_input)

        # 2. S1stem Fast Path Inference
        s1_result: S1DecisionResult = self.s1_engine.predict(tensor)

        # 3. 3-Factor Confidence Gating
        route_decision: RouteDecision = self.router.evaluate(s1_result)

        if route_decision.is_fast_path:
            # === FAST PATH EXIT (Zero LLM Tokens, <15 ms) ===
            self.stats["fast_path_count"] += 1
            self.stats["total_fast_path_ms"] += s1_result.latency_ms
            total_duration_ms = (time.perf_counter() - t_start) * 1000.0

            return {
                "route": RouteTarget.FAST_PATH.value,
                "is_fast_path": True,
                "class_id": s1_result.winner_id,
                "label": s1_result.winner_label,
                "confidence": s1_result.confidence,
                "margin": s1_result.margin,
                "entropy": s1_result.entropy,
                "top_candidates": s1_result.top_candidates,
                "latent_vector": s1_result.latent_vector,
                "flags": [],
                "deliberation": None,
                "telemetry": {
                    "s1_latency_ms": s1_result.latency_ms,
                    "pipeline_total_ms": round(total_duration_ms, 3),
                    "token_cost": 0.0,
                    "offload_rate_pct": self.get_offload_rate(),
                },
            }

        # === SYSTEM-2 DELIBERATIVE PATH EXIT ===
        self.stats["deliberation_count"] += 1
        outcome: DeliberationOutcome = self.system2.deliberate(
            s1_result=s1_result,
            route_decision=route_decision,
            image_input=image_input,
            metadata=metadata,
        )
        self.stats["total_deliberation_ms"] += outcome.deliberation_latency_ms
        total_duration_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "route": RouteTarget.DELIBERATIVE_PATH.value,
            "is_fast_path": False,
            "class_id": outcome.final_class_id,
            "label": outcome.final_class_label,
            "confidence": outcome.deliberation_confidence,
            "margin": s1_result.margin,
            "entropy": s1_result.entropy,
            "top_candidates": s1_result.top_candidates,
            "latent_vector": s1_result.latent_vector,
            "flags": route_decision.flags,
            "deliberation": {
                "strategy": outcome.strategy_used,
                "reasoning": outcome.reasoning_summary,
                "context_used": outcome.context_used,
                "deliberation_latency_ms": outcome.deliberation_latency_ms,
            },
            "telemetry": {
                "s1_latency_ms": s1_result.latency_ms,
                "deliberation_latency_ms": outcome.deliberation_latency_ms,
                "pipeline_total_ms": round(total_duration_ms, 3),
                "escalation_reasons": route_decision.flags,
                "offload_rate_pct": self.get_offload_rate(),
            },
        }

    def get_offload_rate(self) -> float:
        """Returns the percentage of requests resolved directly by S1stem Fast Path."""
        total = self.stats["total_requests"]
        if total == 0:
            return 100.0
        return round((self.stats["fast_path_count"] / total) * 100.0, 2)
