"""
SAN-1: Universal Multimodal Multi-Label Content Classification & Vision Pipeline
================================================================================
Architecture based on Contrastive Language Model (CLM) Two-Tower Decision Engine:
- Fast Path (System-1): S1stem Multi-Prompt Contrastive Vision Engine in ONNX (<15 ms).
- Gating Layer: 3-Factor Confidence Gated Router (Confidence, Margin, Entropy).
- Deliberative Path (System-2): Dual-Strategy Deliberation Engine (Spatial Multi-Crop ONNX + Symbolic).
- Hybrid OCR & Multilingual NLP: RapidOCR + Sentence-Transformer ONNX for text categorization.
"""

from .classifier import SAN1Classifier, ClassificationResult, CategoryPrediction
from .guardrail import SAN1Guardrail, SAN1Result
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

__version__ = "1.4.0"

__all__ = [
    "__version__",
    "SAN1Classifier",
    "ClassificationResult",
    "CategoryPrediction",
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
