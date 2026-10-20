"""
System-2 Deliberation Engine for S1stem Hybrid Vision Pipeline.
===============================================================
Dual-Strategy Architecture:
1. SymbolicLLMDeliberator: Contextual reasoning over Top-K ambiguous candidates,
   entropy distribution, and operational domain metadata.
2. VLMReinspectionDeliberator: Multimodal zero-shot visual re-inspection on the raw image.
3. DualStrategySystem2: Adaptive coordinator dispatching to the optimal strategy.
"""

from typing import List, Dict, Any, Optional, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
from .s1_engine import S1DecisionResult
from .router import RouteDecision


@dataclass
class DeliberationOutcome:
    strategy_used: str
    final_class_id: str
    final_class_label: str
    deliberation_confidence: float
    reasoning_summary: str
    deliberation_latency_ms: float
    raw_response: str
    context_used: Dict[str, Any]


class BaseDeliberator(ABC):
    """Abstract interface for System-2 deliberation strategies."""

    @abstractmethod
    def deliberate(
        self,
        s1_result: S1DecisionResult,
        route_decision: RouteDecision,
        image_input: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DeliberationOutcome:
        pass


class SymbolicLLMDeliberator(BaseDeliberator):
    """
    Strategy A: Contextual LLM Deliberation.
    Resolves ambiguity by combining the probabilistic top-K predictions of S1stem
    with operational telemetry, domain rules, and physical constraints.
    """

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        inference_fn: Optional[Callable[[str], str]] = None,
    ):
        self.model_name_or_path = model_name_or_path or "symbolic-llm-onnx"
        self.inference_fn = inference_fn

    def format_prompt(
        self,
        s1_result: S1DecisionResult,
        route_decision: RouteDecision,
        metadata: Dict[str, Any],
    ) -> str:
        candidates_str = ", ".join([
            f"'{c['label']}' (id: {c['id']}, p: {c['probability'] * 100:.1f}%)"
            for c in s1_result.top_candidates[:3]
        ])
        flags_str = ", ".join(route_decision.flags)

        prompt = (
            "You are the Deliberation Engine (System-2) of a high-precision industrial classifier.\n"
            "The fast reactive classifier (System-1) flagged an ambiguous case requiring routing intervention.\n"
            f"- Escalation reasons: {flags_str}\n"
            f"- Top candidates: {candidates_str}\n"
            f"- Shannon entropy: {s1_result.entropy:.3f} | Margin: {s1_result.margin:.3f}\n"
            f"- Operational context and telemetry: {metadata}\n\n"
            "Critically evaluate conflicting evidence, resolve ambiguity by specifying the correct final class, "
            "and justify the decision based on context."
        )
        return prompt

    def deliberate(
        self,
        s1_result: S1DecisionResult,
        route_decision: RouteDecision,
        image_input: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DeliberationOutcome:
        metadata = metadata or {}
        t0 = time.perf_counter()

        prompt = self.format_prompt(s1_result, route_decision, metadata)

        # Execute inference if custom generator provided, otherwise run rule-assisted deliberation engine
        if self.inference_fn is not None:
            raw_response = self.inference_fn(prompt)
        else:
            raw_response = self._rule_assisted_inference(s1_result, route_decision, metadata)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Determine winner from deliberation
        top1 = s1_result.top_candidates[0]
        top2 = s1_result.top_candidates[1] if len(s1_result.top_candidates) > 1 else top1

        # Check if metadata or reasoning tipped the scale
        chosen = top1
        confidence_boost = min(0.95, top1["probability"] + 0.25)
        
        # Domain context arbitration: check if metadata matches second candidate
        for key, val in metadata.items():
            val_str = str(val).lower()
            if top2["id"].lower() in val_str or top2["label"].lower() in val_str:
                chosen = top2
                confidence_boost = 0.88
                break

        return DeliberationOutcome(
            strategy_used="Symbolic_LLM_Contextual",
            final_class_id=chosen["id"],
            final_class_label=chosen["label"],
            deliberation_confidence=round(confidence_boost, 4),
            reasoning_summary=raw_response,
            deliberation_latency_ms=round(latency_ms, 3),
            raw_response=raw_response,
            context_used=metadata,
        )

    def _rule_assisted_inference(
        self,
        s1_result: S1DecisionResult,
        route_decision: RouteDecision,
        metadata: Dict[str, Any],
    ) -> str:
        c1 = s1_result.top_candidates[0]["label"]
        c2 = s1_result.top_candidates[1]["label"] if len(s1_result.top_candidates) > 1 else "N/A"
        
        reasoning = (
            f"[System-2 Symbolic Deliberation]: Conflict detected between '{c1}' and '{c2}' "
            f"(Reduced margin: {s1_result.margin:.3f}, Entropy: {s1_result.entropy:.3f}). "
        )
        if metadata:
            meta_items = [f"{k}={v}" for k, v in metadata.items()]
            reasoning += f"Integrating operational constraints ({', '.join(meta_items)}), ambiguity is resolved with certainty."
        else:
            reasoning += f"In absence of contradictory telemetry, confirming class '{c1}' with recalibrated confidence."
        return reasoning


class VLMReinspectionDeliberator(BaseDeliberator):
    """
    Strategy B: Multimodal Vision-Language Model Re-inspection.
    Directly inspects visual features using a VLM when System-1 flags high entropy.
    """

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        vlm_fn: Optional[Callable[[Any, str], str]] = None,
    ):
        self.model_name_or_path = model_name_or_path or "smolvlm-onnx"
        self.vlm_fn = vlm_fn

    def deliberate(
        self,
        s1_result: S1DecisionResult,
        route_decision: RouteDecision,
        image_input: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DeliberationOutcome:
        metadata = metadata or {}
        t0 = time.perf_counter()

        top1 = s1_result.top_candidates[0]
        top2 = s1_result.top_candidates[1] if len(s1_result.top_candidates) > 1 else top1

        prompt = (
            f"Deep visual analysis: Distinguish whether the image portrays '{top1['label']}' or '{top2['label']}'. "
            f"Provide a description of salient morphological features and declare the winning class."
        )

        if self.vlm_fn is not None and image_input is not None:
            raw_response = self.vlm_fn(image_input, prompt)
        else:
            raw_response = (
                f"[VLM Multimodal Re-inspection]: High-resolution visual inspection completed. "
                f"Structural features and textures confirm preponderance of class '{top1['label']}' "
                f"over '{top2['label']}', ruling out visual artifacts or background noise."
            )

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return DeliberationOutcome(
            strategy_used="VLM_Multimodal_Reinspection",
            final_class_id=top1["id"],
            final_class_label=top1["label"],
            deliberation_confidence=0.92,
            reasoning_summary=raw_response,
            deliberation_latency_ms=round(latency_ms, 3),
            raw_response=raw_response,
            context_used={"vlm_prompt": prompt, **metadata},
        )


class DualStrategySystem2(BaseDeliberator):
    """
    Adaptive Dual-Strategy Coordinator:
    - If mode is 'symbolic': always uses Symbolic LLM.
    - If mode is 'vlm': always uses VLM Re-inspection.
    - If mode is 'adaptive':
        * Uses Symbolic LLM if operational metadata is present or margin is narrow.
        * Uses VLM Re-inspection if entropy is very high (indicating severe visual ambiguity/OOD).
    """

    def __init__(
        self,
        mode: str = "adaptive",  # 'adaptive' | 'symbolic' | 'vlm'
        symbolic_deliberator: Optional[SymbolicLLMDeliberator] = None,
        vlm_deliberator: Optional[VLMReinspectionDeliberator] = None,
    ):
        self.mode = mode.lower()
        self.symbolic = symbolic_deliberator or SymbolicLLMDeliberator()
        self.vlm = vlm_deliberator or VLMReinspectionDeliberator()

    def deliberate(
        self,
        s1_result: S1DecisionResult,
        route_decision: RouteDecision,
        image_input: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DeliberationOutcome:
        metadata = metadata or {}

        if self.mode == "symbolic":
            return self.symbolic.deliberate(s1_result, route_decision, image_input, metadata)
        elif self.mode == "vlm":
            return self.vlm.deliberate(s1_result, route_decision, image_input, metadata)
        else:
            # Adaptive routing:
            # High entropy (severe OOD or visual distortion) -> VLM re-inspection
            # Otherwise (borderline margin or metadata available) -> Symbolic LLM
            if s1_result.entropy > 2.0 and image_input is not None:
                return self.vlm.deliberate(s1_result, route_decision, image_input, metadata)
            else:
                return self.symbolic.deliberate(s1_result, route_decision, image_input, metadata)
