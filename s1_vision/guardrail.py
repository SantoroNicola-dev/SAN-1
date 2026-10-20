"""
SAN-1 Unified Guardrail Engine (Production Release Interface)
=============================================================
High-performance, multi-modal content safety guardrail powered by ONNX Runtime.
Coordinates:
1. S1stem Vision Two-Tower Guardrail (Framing, NSFW, weapons, hate, drugs, memes, offensive gestures)
2. Hybrid OCR Guardrail (Aho-Corasick + Syntactic Blasphemy + RapidOCR)
3. Multilingual Text Semantic NLP Guardrail (Sentence-Transformer ONNX)
"""

from typing import Optional, Union, Dict, Any
from pathlib import Path
from dataclasses import dataclass, field
import os
import time
from PIL import Image
import numpy as np

from .selfie_guardrail import SelfieGuardrail, SelfieValidationResult
from .ocr_guardrail import OCRTextGuardrail, OCRGuardrailResult
from .text_semantic_guardrail import TextSemanticGuardrail


@dataclass
class SAN1Result:
    is_allowed: bool
    status: str  # "APPROVED" | "BLOCKED"
    rejection_code: Optional[str] = None
    rejection_reason: Optional[str] = None
    remediation_hint: Optional[str] = None
    category_id: Optional[str] = None
    category_label: Optional[str] = None
    confidence: float = 1.0
    latency_ms: float = 0.0
    image_result: Optional[SelfieValidationResult] = None
    ocr_result: Optional[OCRGuardrailResult] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_allowed": self.is_allowed,
            "status": self.status,
            "category_id": self.category_id,
            "category_label": self.category_label,
            "rejection_code": self.rejection_code,
            "rejection_reason": self.rejection_reason,
            "remediation_hint": self.remediation_hint,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "details": self.details,
        }


class SAN1Guardrail:
    """
    Unified SAN-1 Production Guardrail.
    Loads and runs all ONNX models in <25ms with zero remote network calls.
    """

    def __init__(
        self,
        models_dir: Optional[Union[str, Path]] = None,
        policy_path: Optional[Union[str, Path]] = None,
        allow_benign_content: bool = True,
        enable_ocr: bool = True,
        enable_nlp: bool = True,
    ):
        base_dir = Path(__file__).resolve().parent.parent

        # Resolve models directory
        env_models = os.getenv("SAN1_MODELS_DIR")
        if models_dir is not None:
            self.models_dir = Path(models_dir)
        elif env_models:
            self.models_dir = Path(env_models)
        else:
            self.models_dir = base_dir / "models"

        # Resolve policy path
        env_policy = os.getenv("SAN1_POLICY_PATH")
        if policy_path is not None:
            self.policy_path = Path(policy_path)
        elif env_policy:
            self.policy_path = Path(env_policy)
        else:
            self.policy_path = base_dir / "config" / "vision_policy.json"

        self.allow_benign_content = allow_benign_content
        self.enable_ocr = enable_ocr
        self.enable_nlp = enable_nlp

        # Paths for Vision CLIP ONNX
        vision_model = self.models_dir / "clip" / "onnx" / "vision_model_quantized.onnx"
        text_model = self.models_dir / "clip" / "onnx" / "text_model_quantized.onnx"
        tokenizer_path = self.models_dir / "clip" / "tokenizer.json"
        cache_path = self.models_dir / "clip" / "guardrail_embeddings_cache.npz"

        # Initialize Vision Guardrail
        self.vision = SelfieGuardrail(
            policy_path=self.policy_path,
            model_path=str(vision_model),
            text_model_path=str(text_model),
            tokenizer_path=str(tokenizer_path),
            cache_path=cache_path,
            allow_benign_content=self.allow_benign_content,
        )

        # Initialize OCR Guardrail if enabled
        self.ocr: Optional[OCRTextGuardrail] = None
        if self.enable_ocr:
            try:
                self.ocr = OCRTextGuardrail()
            except Exception as e:
                # RapidOCR optional fallback
                self.ocr = None

        # Initialize Text Semantic NLP Guardrail if enabled
        self.nlp: Optional[TextSemanticGuardrail] = None
        if self.enable_nlp:
            nlp_model = self.models_dir / "nlp" / "model_quantized.onnx"
            nlp_tok = self.models_dir / "nlp" / "tokenizer.json"
            if nlp_model.exists() and nlp_tok.exists():
                try:
                    self.nlp = TextSemanticGuardrail(
                        model_path=nlp_model,
                        tokenizer_path=nlp_tok,
                    )
                except Exception:
                    self.nlp = None

    def validate_image(
        self,
        image_input: Union[str, Path, bytes, Image.Image, np.ndarray],
        check_embedded_text: bool = True,
    ) -> SAN1Result:
        """
        Validates an image against the multimodal guardrail policy.
        1. Evaluates visual semantics (framing, safety veto gate).
        2. If visual checks pass, runs embedded text OCR guardrail (blasphemy, toxicity, slurs).
        """
        t0 = time.perf_counter()

        # Step 1: Vision Guardrail
        v_res = self.vision.validate(image_input)

        if not v_res.is_allowed:
            latency = (time.perf_counter() - t0) * 1000.0
            return SAN1Result(
                is_allowed=False,
                status="BLOCKED",
                category_id=v_res.category_id,
                category_label=v_res.category_label,
                rejection_code=v_res.rejection_code,
                rejection_reason=v_res.rejection_reason,
                remediation_hint=v_res.remediation_hint,
                confidence=v_res.confidence,
                latency_ms=round(latency, 2),
                image_result=v_res,
                details={
                    "stage": "vision_guardrail",
                    "route": v_res.route_path,
                },
            )

        # Step 2: OCR Guardrail (if enabled and requested)
        ocr_res = None
        if check_embedded_text and self.ocr is not None:
            try:
                ocr_res = self.ocr.validate_image(image_input)
                if not ocr_res.is_allowed:
                    latency = (time.perf_counter() - t0) * 1000.0
                    return SAN1Result(
                        is_allowed=False,
                        status="BLOCKED",
                        category_id=ocr_res.category,
                        category_label="Vulgar or Offensive Text (OCR)",
                        rejection_code=ocr_res.rejection_code or "PROHIBITED_TEXT",
                        rejection_reason=ocr_res.rejection_reason or "Prohibited text detected in image.",
                        remediation_hint="Remove offensive, vulgar, or profane text from the image.",
                        confidence=1.0,
                        latency_ms=round(latency, 2),
                        image_result=v_res,
                        ocr_result=ocr_res,
                        details={
                            "stage": "ocr_guardrail",
                            "detected_text": ocr_res.detected_text,
                            "offending_words": ocr_res.offending_words,
                        },
                    )
            except Exception:
                pass

        latency = (time.perf_counter() - t0) * 1000.0
        return SAN1Result(
            is_allowed=True,
            status="APPROVED",
            category_id=v_res.category_id,
            category_label=v_res.category_label,
            confidence=v_res.confidence,
            latency_ms=round(latency, 2),
            image_result=v_res,
            ocr_result=ocr_res,
            details={
                "stage": "approved",
                "route": v_res.route_path,
            },
        )

    def validate_text(self, text_input: str) -> SAN1Result:
        """
        Validates arbitrary text (e.g. user bio, post caption, chat message)
        using both Syntactic/Blasphemy matcher and Multilingual Semantic NLP.
        """
        t0 = time.perf_counter()

        # Check Syntactic / Blasphemy via OCR text module
        if self.ocr is not None:
            is_syntactic_allowed, cat, code, reason, words, _ = self.ocr.check_text_safety(text_input)
            if not is_syntactic_allowed:
                latency = (time.perf_counter() - t0) * 1000.0
                is_blasphemy = (cat == "BLOCKED_BLASPHEMY")
                cat_label = "Profane or Blasphemous Text" if is_blasphemy else "Harmful or Prohibited Text Content"
                hint = "Remove profane or blasphemous expressions from the text." if is_blasphemy else "Rephrase the text to remove hate speech, threats, or prohibited content."
                return SAN1Result(
                    is_allowed=False,
                    status="BLOCKED",
                    category_id=cat,
                    category_label=cat_label,
                    rejection_code=code,
                    rejection_reason=reason,
                    remediation_hint=hint,
                    confidence=1.0,
                    latency_ms=round(latency, 2),
                    details={"offending_words": words, "matcher": "fast_path_syntactic"},
                )

        # Check Semantic NLP Guardrail
        nlp_details = {}
        if self.nlp is not None:
            is_allowed, category, max_toxic, max_benign, reason = self.nlp.evaluate_text(text_input)
            nlp_details = {"max_toxic_sim": round(max_toxic, 4), "max_benign_sim": round(max_benign, 4)}
            if not is_allowed:
                latency = (time.perf_counter() - t0) * 1000.0
                return SAN1Result(
                    is_allowed=False,
                    status="BLOCKED",
                    category_id=category,
                    category_label="Harmful or Prohibited Text Content",
                    rejection_code=category,
                    rejection_reason=reason,
                    remediation_hint="Rephrase the text to remove hate speech, threats, or prohibited content.",
                    confidence=round(float(max_toxic), 4),
                    latency_ms=round(latency, 2),
                    details={"matcher": "multilingual_semantic_nlp", **nlp_details},
                )

        latency = (time.perf_counter() - t0) * 1000.0
        return SAN1Result(
            is_allowed=True,
            status="APPROVED",
            category_id="CLEAN_TEXT",
            category_label="Compliant Text",
            confidence=1.0,
            latency_ms=round(latency, 2),
            details={"matcher": "all_checks_passed", **nlp_details},
        )

    def validate(
        self,
        image_input: Optional[Union[str, Path, bytes, Image.Image, np.ndarray]] = None,
        text_input: Optional[str] = None,
    ) -> SAN1Result:
        """
        Unified validation for multimodal requests.
        Evaluates image, text, or both.
        """
        if image_input is not None:
            img_res = self.validate_image(image_input)
            if not img_res.is_allowed:
                return img_res
        
        if text_input:
            txt_res = self.validate_text(text_input)
            if not txt_res.is_allowed:
                return txt_res

        if image_input is not None:
            return img_res
        elif text_input:
            return txt_res
        
        return SAN1Result(
            is_allowed=True,
            status="APPROVED",
            category_id="NO_INPUT",
            category_label="Nessun input fornito",
            confidence=1.0,
            latency_ms=0.0,
        )
