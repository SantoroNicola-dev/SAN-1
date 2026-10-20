"""
S1stem Vision: Hybrid System-1 / System-2 Image Classification Pipeline
========================================================================
Architecture based on Contrastive Language Model (CLM) Two-Tower Decision Engine:
- Fast Path (System-1): S1stem Contrastive Vision Engine in ONNX (<15 ms).
- Gating Layer: 3-Factor Confidence Gated Router (Confidence, Margin, Entropy).
- Deliberative Path (System-2): Dual-Strategy Deliberation Engine (Symbolic LLM + Multimodal VLM).
"""

from .s1_engine import S1stemVisionEngine, S1DecisionResult
from .router import ConfidenceGatedRouter, RouteDecision, RouteTarget
from .system2 import (
    BaseDeliberator,
    SymbolicLLMDeliberator,
    VLMReinspectionDeliberator,
    DualStrategySystem2,
)
from .pipeline import HybridVisionPipeline
from .preprocessing import ImagePreprocessor
from .selfie_guardrail import SelfieGuardrail, SelfieValidationResult
from .guardrail import SAN1Guardrail, SAN1Result

__all__ = [
    "SAN1Guardrail",
    "SAN1Result",
    "S1stemVisionEngine",
    "S1DecisionResult",
    "ConfidenceGatedRouter",
    "RouteDecision",
    "RouteTarget",
    "BaseDeliberator",
    "SymbolicLLMDeliberator",
    "VLMReinspectionDeliberator",
    "DualStrategySystem2",
    "HybridVisionPipeline",
    "ImagePreprocessor",
    "SelfieGuardrail",
    "SelfieValidationResult",
]
