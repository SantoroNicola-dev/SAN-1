"""
Comprehensive Pytest Test Suite for S1stem Hybrid Vision Pipeline.
===================================================================
Tests all modules:
- Preprocessing and tensor formatting
- S1stem Vision CLM Engine with ONNX Runtime
- 3-Factor Confidence Gated Router arbitration
- System-2 Deliberation Engine (Symbolic LLM + VLM)
- Full End-to-End Hybrid Vision Pipeline
"""

import sys
import pytest
import numpy as np
from PIL import Image
from pathlib import Path

# Add project root to sys.path so tests can run cleanly from anywhere
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from s1_vision.preprocessing import ImagePreprocessor
from s1_vision.s1_engine import S1stemVisionEngine, S1DecisionResult
from s1_vision.router import ConfidenceGatedRouter, RouteTarget
from s1_vision.system2 import (
    SymbolicLLMDeliberator,
    VLMReinspectionDeliberator,
    DualStrategySystem2,
)
from s1_vision.pipeline import HybridVisionPipeline
from scripts.export_or_download_models import generate_local_vision_onnx


@pytest.fixture(scope="session")
def onnx_model_path(tmp_path_factory):
    fn = tmp_path_factory.mktemp("models") / "test_vision.onnx"
    generate_local_vision_onnx(output_path=str(fn), embedding_dim=512)
    return str(fn)


@pytest.fixture
def sample_classes():
    return [
        {
            "id": "INTEGERRIMO",
            "label": "Componente Integro",
            "description": "Componente metallico perfetto senza difetti visivi o crepe",
        },
        {
            "id": "DIFETTO_SALDATURA",
            "label": "Difetto di Saldatura",
            "description": "Saldatura irregolare con inclusioni di scoria o porosità superficiale",
        },
        {
            "id": "CREPA_STRUTTURALE",
            "label": "Crepa Strutturale",
            "description": "Fessurazione profonda o frattura lineare lungo la superficie metallica",
        },
        {
            "id": "OSSIDAZIONE",
            "label": "Ossidazione e Ruggine",
            "description": "Macchie scure di corrosione e ruggine diffusa",
        },
    ]


@pytest.fixture
def synthetic_image():
    # 300x300 RGB synthetic image
    arr = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_preprocessing(synthetic_image):
    prep = ImagePreprocessor(target_size=(224, 224))
    tensor = prep.preprocess(synthetic_image)
    
    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == np.float32
    assert not np.isnan(tensor).any()


def test_s1_engine_registration_and_dimensions(sample_classes):
    engine = S1stemVisionEngine(embedding_dim=512, temperature=0.05)
    count = engine.register_classes(sample_classes)

    assert count == 4
    assert engine.action_matrix is not None
    assert engine.action_matrix.shape == (4, 512)
    
    # Check unit normalization of each action row
    norms = np.linalg.norm(engine.action_matrix, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_s1_engine_prediction_with_onnx(onnx_model_path, sample_classes, synthetic_image):
    prep = ImagePreprocessor(target_size=(224, 224))
    tensor = prep.preprocess(synthetic_image)

    engine = S1stemVisionEngine(model_path=onnx_model_path, embedding_dim=512, temperature=0.05)
    engine.register_classes(sample_classes)

    # Warmup
    _ = engine.predict(tensor)
    result = engine.predict(tensor)
    assert isinstance(result, S1DecisionResult)
    assert result.winner_id in [c["id"] for c in sample_classes]
    assert 0.0 <= result.confidence <= 1.0
    assert 0.0 <= result.margin <= 1.0
    assert result.entropy >= 0.0
    assert len(result.probabilities) == 4
    np.testing.assert_allclose(np.sum(result.probabilities), 1.0, atol=1e-5)
    assert result.latency_ms > 0.0
    assert result.latency_ms < 50.0  # Ultra-fast System 1 target (warm)


def test_3factor_router_fast_path():
    router = ConfidenceGatedRouter(
        confidence_threshold=0.75,
        margin_threshold=0.15,
        entropy_threshold=1.50,
    )
    # High confidence, wide margin, low entropy
    probs = np.array([0.85, 0.08, 0.04, 0.03], dtype=np.float32)
    dummy_result = S1DecisionResult(
        winner_id="INTEGERRIMO",
        winner_label="Componente Integro",
        winner_index=0,
        confidence=0.85,
        margin=0.77,
        entropy=0.75,
        probabilities=probs,
        top_candidates=[],
        latent_vector=np.zeros(512),
        latency_ms=4.2,
    )

    decision = router.evaluate(dummy_result)
    assert decision.is_fast_path is True
    assert decision.target == RouteTarget.FAST_PATH
    assert len(decision.flags) == 0
    assert decision.escalation_reason is None


def test_3factor_router_escalation_on_low_confidence():
    router = ConfidenceGatedRouter(confidence_threshold=0.75, margin_threshold=0.15)
    probs = np.array([0.60, 0.25, 0.10, 0.05], dtype=np.float32)
    dummy_result = S1DecisionResult(
        winner_id="DIFETTO_SALDATURA",
        winner_label="Difetto di Saldatura",
        winner_index=0,
        confidence=0.60,  # Below 0.75 threshold
        margin=0.35,      # Above 0.15 margin
        entropy=1.20,     # Below 1.50 entropy
        probabilities=probs,
        top_candidates=[],
        latent_vector=np.zeros(512),
        latency_ms=3.8,
    )

    decision = router.evaluate(dummy_result)
    assert decision.is_fast_path is False
    assert decision.target == RouteTarget.DELIBERATIVE_PATH
    assert any("LOW_CONFIDENCE" in f for f in decision.flags)


def test_3factor_router_escalation_on_narrow_margin():
    router = ConfidenceGatedRouter(confidence_threshold=0.50, margin_threshold=0.15)
    # Two top candidates almost tied: 52% vs 48%
    probs = np.array([0.52, 0.48], dtype=np.float32)
    dummy_result = S1DecisionResult(
        winner_id="DIFETTO_SALDATURA",
        winner_label="Difetto",
        winner_index=0,
        confidence=0.52,
        margin=0.04,  # Narrow margin (< 0.15)
        entropy=0.99,
        probabilities=probs,
        top_candidates=[],
        latent_vector=np.zeros(512),
        latency_ms=3.5,
    )

    decision = router.evaluate(dummy_result)
    assert decision.is_fast_path is False
    assert any("NARROW_MARGIN" in f for f in decision.flags)


def test_system2_deliberation():
    symbolic = SymbolicLLMDeliberator()
    vlm = VLMReinspectionDeliberator()
    dual = DualStrategySystem2(mode="adaptive", symbolic_deliberator=symbolic, vlm_deliberator=vlm)

    probs = np.array([0.51, 0.49], dtype=np.float32)
    s1_res = S1DecisionResult(
        winner_id="A",
        winner_label="Classe A",
        winner_index=0,
        confidence=0.51,
        margin=0.02,
        entropy=1.85,
        probabilities=probs,
        top_candidates=[
            {"id": "A", "label": "Classe A", "probability": 0.51},
            {"id": "B", "label": "Classe B", "probability": 0.49},
        ],
        latent_vector=np.zeros(512),
        latency_ms=5.0,
    )
    router = ConfidenceGatedRouter()
    route_dec = router.evaluate(s1_res)

    # Test Symbolic Deliberation
    outcome_sym = symbolic.deliberate(s1_res, route_dec, metadata={"line_speed": "high"})
    assert outcome_sym.final_class_id in ["A", "B"]
    assert len(outcome_sym.reasoning_summary) > 10
    assert outcome_sym.strategy_used == "Symbolic_LLM_Contextual"

    # Test VLM Deliberation
    outcome_vlm = vlm.deliberate(s1_res, route_dec)
    assert outcome_vlm.strategy_used == "VLM_Multimodal_Reinspection"
    assert "High-resolution visual inspection" in outcome_vlm.reasoning_summary


def test_end_to_end_hybrid_pipeline(onnx_model_path, sample_classes, synthetic_image):
    pipeline = HybridVisionPipeline(
        model_path=onnx_model_path,
        confidence_threshold=0.75,
        margin_threshold=0.15,
        entropy_threshold=1.50,
    )
    pipeline.register_classes(sample_classes)

    # Run processing
    res = pipeline.process(synthetic_image, metadata={"station": "Line_Alpha", "operator": "Robot_1"})

    assert "route" in res
    assert "class_id" in res
    assert "label" in res
    assert "confidence" in res
    assert "telemetry" in res
    assert res["telemetry"]["pipeline_total_ms"] > 0
    assert pipeline.stats["total_requests"] == 1


def test_selfie_guardrail_approval_and_blocking(synthetic_image):
    from s1_vision.selfie_guardrail import SelfieGuardrail

    guardrail = SelfieGuardrail(
        model_path="models/clip/onnx/vision_model_quantized.onnx",
        text_model_path="models/clip/onnx/text_model_quantized.onnx",
        tokenizer_path="models/clip/tokenizer.json",
    )

    # 1. Test Approval for Close-up Selfie
    idx_selfie = [c.id for c in guardrail.engine.action_cache].index("ALLOWED_CLOSEUP_SELFIE")
    guardrail.engine.encode_image = lambda _: guardrail.engine.action_matrix[idx_selfie]
    res_allowed = guardrail.validate(synthetic_image)

    assert res_allowed.is_allowed is True
    assert res_allowed.status == "APPROVED"
    assert res_allowed.category_id == "ALLOWED_CLOSEUP_SELFIE"
    assert res_allowed.rejection_code is None

    # 2. Test Blocking for Landscape
    idx_land = [c.id for c in guardrail.engine.action_cache].index("BLOCKED_LANDSCAPE")
    guardrail.engine.encode_image = lambda _: guardrail.engine.action_matrix[idx_land]
    res_blocked = guardrail.validate(synthetic_image)

    assert res_blocked.is_allowed is False
    assert res_blocked.status == "BLOCKED"
    assert res_blocked.category_id == "BLOCKED_LANDSCAPE"
    assert res_blocked.rejection_code == "NO_FACE_LANDSCAPE"
    assert res_blocked.rejection_reason is not None
    assert res_blocked.remediation_hint is not None

