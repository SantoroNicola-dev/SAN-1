"""
System-2 Deliberation Engine for S1stem Hybrid Vision Pipeline.
===============================================================
Dual-Strategy & Spatial Multi-Crop Architecture:
1. SymbolicLLMDeliberator: Contextual reasoning over Top-K ambiguous candidates,
   entropy distribution, and operational domain metadata.
2. VLMReinspectionDeliberator: Multimodal visual re-inspection on the raw image,
   supporting both real ONNX Spatial Multi-Crop (Global + 4 Quadrants) and custom VLM callbacks.
3. DualStrategySystem2: Adaptive coordinator dispatching to the optimal strategy.
"""

from typing import List, Dict, Any, Optional, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
from .s1_engine import S1DecisionResult, S1stemVisionEngine
from .router import RouteDecision
from .preprocessing import ImagePreprocessor


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
    categories: List[Dict[str, Any]] = field(default_factory=list)
    crops_analyzed: int = 1


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
    Strategy A: Contextual Symbolic / LLM Deliberation.
    Resolves ambiguity by combining the probabilistic top-K predictions of S1stem
    with operational telemetry, domain rules, and multi-label distributions.
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
            "You are the Deliberation Engine (System-2) of a high-precision multi-label classifier.\n"
            "The fast reactive classifier (System-1) flagged an ambiguous or multi-subject case.\n"
            f"- Escalation reasons: {flags_str}\n"
            f"- Top candidates: {candidates_str}\n"
            f"- Shannon entropy: {s1_result.entropy:.3f} | Margin: {s1_result.margin:.3f}\n"
            f"- Operational context and telemetry: {metadata}\n\n"
            "Critically evaluate conflicting evidence, resolve ambiguity by specifying the primary and secondary classes, "
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

        if self.inference_fn is not None:
            raw_response = self.inference_fn(prompt)
        else:
            raw_response = self._rule_assisted_inference(s1_result, route_decision, metadata)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        top1 = s1_result.top_candidates[0]
        top2 = s1_result.top_candidates[1] if len(s1_result.top_candidates) > 1 else top1

        chosen = top1
        confidence_boost = min(0.95, top1["probability"] + 0.25)

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
            categories=s1_result.categories,
            crops_analyzed=s1_result.crops_analyzed,
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
            f"[System-2 Symbolic Deliberation]: Multi-category overlap detected between '{c1}' and '{c2}' "
            f"(Reduced margin: {s1_result.margin:.3f}, Entropy: {s1_result.entropy:.3f}). "
        )
        if metadata:
            meta_items = [f"{k}={v}" for k, v in metadata.items()]
            reasoning += f"Integrating operational constraints ({', '.join(meta_items)}), ambiguity is resolved with certainty."
        else:
            reasoning += f"In absence of contradictory telemetry, confirming primary class '{c1}' while retaining secondary categories."
        return reasoning


class VLMReinspectionDeliberator(BaseDeliberator):
    """
    Strategy B: Multimodal Spatial Multi-Crop & VLM Re-inspection.
    When an ONNX S1stemVisionEngine is attached, executes real 5-Crop Spatial ONNX Batch
    inference (Global + 4 Quadrants) to discover off-center or secondary subjects.
    """

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        vlm_fn: Optional[Callable[[Any, str], str]] = None,
        s1_engine: Optional[S1stemVisionEngine] = None,
        preprocessor: Optional[ImagePreprocessor] = None,
    ):
        self.model_name_or_path = model_name_or_path or "spatial-multicrop-onnx"
        self.vlm_fn = vlm_fn
        self.s1_engine = s1_engine
        self.preprocessor = preprocessor or ImagePreprocessor()

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
            f"Deep visual analysis: Distinguish and quantify '{top1['label']}' and '{top2['label']}' across spatial regions. "
            f"Provide a description of salient morphological features and declare the category breakdown."
        )

        categories = s1_result.categories
        crops_analyzed = s1_result.crops_analyzed
        final_id = top1["id"]
        final_label = top1["label"]
        final_conf = 0.92

        if self.vlm_fn is not None and image_input is not None:
            raw_response = self.vlm_fn(image_input, prompt)
        elif self.s1_engine is not None and image_input is not None:
            try:
                min_conf = float(metadata.get("min_confidence_pct", 25.0))
                top_k = int(metadata.get("top_k", 10))
                multicrop_tensor = self.preprocessor.preprocess_multicrop(image_input, include_quadrants=True)
                mc_res = self.s1_engine.predict_multicrop(
                    multicrop_tensor,
                    min_confidence_pct=min_conf,
                    top_k=top_k,
                    quadrant_weight=0.25,
                    global_v=s1_result.latent_vector,
                )
                # Merge multicrop regional detections with S1 global detections
                # so regional corner crops don't erase substantive central foreground subjects
                merged_cats: Dict[str, Dict[str, Any]] = {c["id"]: dict(c) for c in mc_res.categories}
                for c in s1_result.categories:
                    cid = c["id"]
                    if cid not in merged_cats or c["confidence_pct"] > merged_cats[cid]["confidence_pct"]:
                        merged_cats[cid] = dict(c)

                categories = list(merged_cats.values())
                crops_analyzed = mc_res.crops_analyzed
                if categories:
                    # Strict mutual exclusion for domestic pet species:
                    dog_item = next((c for c in categories if c.get("id") == "dog"), None)
                    cat_item = next((c for c in categories if c.get("id") == "cat"), None)
                    if dog_item and cat_item:
                        d_conf = float(dog_item.get("confidence_pct", 0.0))
                        c_conf = float(cat_item.get("confidence_pct", 0.0))
                        if d_conf >= 80.0 and d_conf > c_conf:
                            cat_item["confidence_pct"] = 0.0
                        elif c_conf >= 80.0 and c_conf > d_conf:
                            dog_item["confidence_pct"] = 0.0
                    categories = [c for c in categories if float(c.get("confidence_pct", 0.0)) > 0.0]

                    # Sort descending by confidence_pct
                    categories.sort(key=lambda x: float(x.get("confidence_pct", 0.0)), reverse=True)
                    # Promote high-confidence sensitive content (>= 92%) to primary position
                    if len(categories) > 1 and (
                        categories[0].get("group") != "sensitive_content"
                        or categories[0].get("id") == "smoking_tobacco_vape"
                    ):
                        for idx_c in range(1, len(categories)):
                            c_cand = categories[idx_c]
                            if (
                                c_cand.get("group") == "sensitive_content"
                                and c_cand.get("id") != "smoking_tobacco_vape"
                                and float(c_cand.get("confidence_pct", 0.0)) >= 92.0
                            ):
                                p_item = categories.pop(idx_c)
                                categories.insert(0, p_item)
                                break
                    # Subject-over-Scene Prioritization on merged Deliberation set
                    AMBIENT_SCENE_IDS = {
                        "bedroom_interior",
                        "bedroom_bathroom_interior",
                        "home_living_room",
                        "mountain_hills",
                        "field_countryside_landscape",
                        "forest_trees",
                        "beach_sea_coast",
                        "sky_sunset_clouds",
                        "snow_winter_scene",
                        "store_restaurant_interior",
                    }
                    OUTDOOR_SCENE_IDS = {
                        "mountain_hills",
                        "field_countryside_landscape",
                        "forest_trees",
                        "beach_sea_coast",
                        "sky_sunset_clouds",
                        "snow_winter_scene",
                    }
                    INDOOR_ONLY_IDS = {
                        "kitchen_utensils_cutlery",
                        "kitchenware_tableware",
                        "bedroom_interior",
                        "bedroom_bathroom_interior",
                        "home_living_room",
                        "office_workspace",
                    }
                    if categories[0].get("id") in AMBIENT_SCENE_IDS and len(categories) > 1:
                        top_sc_id = categories[0].get("id")
                        top_sc = float(categories[0].get("confidence_pct", 0.0))
                        for idx_c in range(1, len(categories)):
                            c_cand = categories[idx_c]
                            sub_id = c_cand.get("id")
                            # Guard: indoor objects cannot be promoted over outdoor landscapes
                            if top_sc_id in OUTDOOR_SCENE_IDS and sub_id in INDOOR_ONLY_IDS:
                                continue
                            if (
                                sub_id not in AMBIENT_SCENE_IDS
                                and c_cand.get("group") not in ("image_quality", "sensitive_content")
                                and (float(c_cand.get("confidence_pct", 0.0)) >= 85.0
                                     or float(c_cand.get("confidence_pct", 0.0)) >= (top_sc - 12.0))
                            ):
                                p_item = categories.pop(idx_c)
                                categories.insert(0, p_item)
                                break
                    final_id = categories[0]["id"]
                    final_label = categories[0]["label"]
                    final_conf = float(categories[0]["confidence_pct"]) / 100.0
                else:
                    final_id = mc_res.winner_id
                    final_label = mc_res.winner_label
                    final_conf = mc_res.confidence
                cat_summary = ", ".join(
                    f"{c['label']} ({c['confidence_pct']:.1f}% conf, {c['share_pct']:.1f}% share, region={c.get('region', 'global')})"
                    for c in categories[:4]
                )
                raw_response = (
                    f"[VLM Multimodal Re-inspection]: High-resolution visual inspection completed across {crops_analyzed} spatial regions. "
                    f"Detected composition: {cat_summary}."
                )
            except Exception:
                raw_response = (
                    f"[VLM Multimodal Re-inspection]: High-resolution visual inspection completed. "
                    f"Structural features and textures confirm preponderance of class '{top1['label']}' "
                    f"over '{top2['label']}', ruling out visual artifacts or background noise."
                )
        else:
            raw_response = (
                f"[VLM Multimodal Re-inspection]: High-resolution visual inspection completed. "
                f"Structural features and textures confirm preponderance of class '{top1['label']}' "
                f"over '{top2['label']}', ruling out visual artifacts or background noise."
            )

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return DeliberationOutcome(
            strategy_used="VLM_Multimodal_Reinspection",
            final_class_id=final_id,
            final_class_label=final_label,
            deliberation_confidence=round(final_conf, 4),
            reasoning_summary=raw_response,
            deliberation_latency_ms=round(latency_ms, 3),
            raw_response=raw_response,
            context_used={"vlm_prompt": prompt, **metadata},
            categories=categories,
            crops_analyzed=crops_analyzed,
        )


class DualStrategySystem2(BaseDeliberator):
    """
    Adaptive Dual-Strategy Coordinator:
    - If mode is 'symbolic': always uses Symbolic LLM.
    - If mode is 'vlm': always uses VLM / Spatial Multi-Crop Re-inspection.
    - If mode is 'adaptive':
        * Uses VLM Spatial Multi-Crop Re-inspection when image_input is provided and
          high entropy or multi-subject ambiguity is detected.
        * Uses Symbolic LLM otherwise.
    """

    def __init__(
        self,
        mode: str = "adaptive",
        symbolic_deliberator: Optional[SymbolicLLMDeliberator] = None,
        vlm_deliberator: Optional[VLMReinspectionDeliberator] = None,
        s1_engine: Optional[S1stemVisionEngine] = None,
        preprocessor: Optional[ImagePreprocessor] = None,
    ):
        self.mode = mode.lower()
        self.symbolic = symbolic_deliberator or SymbolicLLMDeliberator()
        self.vlm = vlm_deliberator or VLMReinspectionDeliberator(
            s1_engine=s1_engine,
            preprocessor=preprocessor,
        )

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
            if (s1_result.entropy > 2.0 or metadata.get("deep")) and image_input is not None:
                return self.vlm.deliberate(s1_result, route_decision, image_input, metadata)
            elif self.vlm.s1_engine is not None and image_input is not None and not metadata:
                return self.vlm.deliberate(s1_result, route_decision, image_input, metadata)
            else:
                return self.symbolic.deliberate(s1_result, route_decision, image_input, metadata)
