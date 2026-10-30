"""
SAN-1 Unified Classifier & Guardrail Engine (Production Interface)
==================================================================
High-performance, multi-modal content classifier powered by ONNX Runtime.
Coordinates:
1. S1stem Vision Multi-Label Two-Tower Classifier (60+ categories, multi-prompt ensembling)
2. Hybrid OCR Categorizer (Aho-Corasick + Syntactic Blasphemy + RapidOCR)
3. Multilingual Text Semantic NLP Categorizer (Sentence-Transformer ONNX)
"""

from typing import Optional, Union, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass, field
import os
import time
from PIL import Image
import numpy as np

from .classifier import SAN1Classifier, ClassificationResult
from .selfie_guardrail import SelfieGuardrail, SelfieValidationResult
from .ocr_guardrail import OCRTextGuardrail, OCRGuardrailResult
from .text_semantic_guardrail import TextSemanticGuardrail


@dataclass
class SAN1Result:
    is_allowed: bool = True
    status: str = "ANALYZED"
    rejection_code: Optional[str] = None
    rejection_reason: Optional[str] = None
    remediation_hint: Optional[str] = None
    category_id: Optional[str] = None
    category_label: Optional[str] = None
    confidence: float = 1.0
    confidence_pct: float = 100.0
    latency_ms: float = 0.0
    primary_category: Optional[Dict[str, Any]] = None
    categories: List[Dict[str, Any]] = field(default_factory=list)
    extracted_text: Optional[str] = None
    image_result: Optional[SelfieValidationResult] = None
    ocr_result: Optional[OCRGuardrailResult] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "primary_category": self.primary_category,
            "categories": self.categories,
            "category_id": self.category_id,
            "category_label": self.category_label,
            "confidence_pct": round(float(self.confidence_pct), 2),
            "confidence": round(float(self.confidence), 4),
            "latency_ms": round(float(self.latency_ms), 2),
            "extracted_text": self.extracted_text,
            "is_allowed": self.is_allowed,
            "status": self.status,
            "rejection_code": self.rejection_code,
            "rejection_reason": self.rejection_reason,
            "remediation_hint": self.remediation_hint,
            "details": self.details,
        }
        return d


class SAN1Guardrail(SAN1Classifier):
    """
    Unified SAN-1 Production Classifier and Compatibility Guardrail.
    Inherits directly from SAN1Classifier to provide universal multi-label
    categorization across 60+ categories with independent and share percentages.
    """

    def __init__(
        self,
        models_dir: Optional[Union[str, Path]] = None,
        categories_path: Optional[Union[str, Path]] = None,
        policy_path: Optional[Union[str, Path]] = None,
        cache_path: Optional[Union[str, Path]] = None,
        allow_benign_content: bool = True,
        enable_ocr: bool = True,
        enable_nlp: bool = True,
        default_min_confidence: float = 25.0,
        default_top_k: int = 10,
    ):
        cat_path = categories_path or policy_path
        super().__init__(
            models_dir=models_dir,
            categories_path=cat_path,
            cache_path=cache_path,
            enable_ocr=enable_ocr,
            enable_nlp=enable_nlp,
            default_min_confidence=default_min_confidence,
            default_top_k=default_top_k,
        )
        self.allow_benign_content = allow_benign_content

        # Retain self.vision for legacy callers/tests if needed
        base_dir = Path(__file__).resolve().parent.parent
        leg_policy = policy_path or (base_dir / "config" / "vision_policy.json")
        vision_model = self.models_dir / "clip" / "onnx" / "vision_model_quantized.onnx"
        text_model = self.models_dir / "clip" / "onnx" / "text_model_quantized.onnx"
        tokenizer_path = self.models_dir / "clip" / "tokenizer.json"
        leg_cache = self.models_dir / "clip" / "guardrail_embeddings_cache.npz"

        self.vision = SelfieGuardrail(
            policy_path=leg_policy,
            model_path=str(vision_model),
            text_model_path=str(text_model),
            tokenizer_path=str(tokenizer_path),
            cache_path=leg_cache,
            allow_benign_content=self.allow_benign_content,
        )

    def validate_image(
        self,
        image_input: Union[str, Path, bytes, Image.Image, np.ndarray],
        check_embedded_text: bool = True,
        deep: bool = False,
        top_k: Optional[int] = None,
        min_confidence: Optional[float] = None,
    ) -> SAN1Result:
        """
        Classifies an image across all categories with multi-label percentages.
        Returns SAN1Result containing full category distribution.
        """
        c_res = self.classify_image(
            image_input=image_input,
            top_k=top_k,
            min_confidence=min_confidence,
            deep=deep,
            check_embedded_text=check_embedded_text,
        )

        return SAN1Result(
            is_allowed=True,
            status="APPROVED",
            category_id=c_res.category_id,
            category_label=c_res.category_label,
            confidence=c_res.confidence,
            confidence_pct=c_res.confidence_pct,
            primary_category=c_res.primary_category,
            categories=c_res.categories,
            extracted_text=c_res.extracted_text,
            latency_ms=c_res.latency_ms,
            details=c_res.details,
        )

    def validate_text(
        self,
        text_input: str,
        top_k: Optional[int] = None,
        min_confidence: Optional[float] = None,
    ) -> SAN1Result:
        """
        Classifies a text string across semantic and lexical categories.
        Returns SAN1Result containing full category distribution.
        """
        c_res = self.classify_text(
            text_input=text_input,
            top_k=top_k,
            min_confidence=min_confidence,
        )

        # Check if sensitive content was detected for legacy caller compatibility
        is_bad = any(
            c.get("id") in {
                "blasphemy_religious_slur",
                "hate_speech_text",
                "threats_violence_text",
                "cyber_malware_text",
                "scam_fraud_text",
            }
            for c in c_res.categories
        )

        return SAN1Result(
            is_allowed=not is_bad,
            status="BLOCKED" if is_bad else "APPROVED",
            rejection_code=c_res.category_id if is_bad else None,
            rejection_reason="Sensitive or prohibited content detected in text." if is_bad else None,
            category_id=c_res.category_id,
            category_label=c_res.category_label,
            confidence=c_res.confidence,
            confidence_pct=c_res.confidence_pct,
            primary_category=c_res.primary_category,
            categories=c_res.categories,
            extracted_text=c_res.extracted_text,
            latency_ms=c_res.latency_ms,
            details=c_res.details,
        )

    def validate(
        self,
        image_input: Optional[Union[str, Path, bytes, Image.Image, np.ndarray]] = None,
        text_input: Optional[str] = None,
        top_k: Optional[int] = None,
        min_confidence: Optional[float] = None,
        deep: bool = False,
    ) -> SAN1Result:
        """Unified classification for multimodal requests."""
        c_res = self.classify(
            image_input=image_input,
            text_input=text_input,
            top_k=top_k,
            min_confidence=min_confidence,
            deep=deep,
        )

        return SAN1Result(
            is_allowed=True,
            status="ANALYZED",
            category_id=c_res.category_id,
            category_label=c_res.category_label,
            confidence=c_res.confidence,
            confidence_pct=c_res.confidence_pct,
            primary_category=c_res.primary_category,
            categories=c_res.categories,
            extracted_text=c_res.extracted_text,
            latency_ms=c_res.latency_ms,
            details=c_res.details,
        )
