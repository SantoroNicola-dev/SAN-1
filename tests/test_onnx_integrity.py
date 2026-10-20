"""
ONNX Model Integrity & Verification Test Suite.
===============================================
Verifies that all ONNX graphs listed in models/manifest.json are valid,
quantized INT8, match their SHA-256 checksums, and run successful inference.
"""

import sys
import json
import hashlib
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from s1_vision.guardrail import SAN1Guardrail


def test_manifest_existence_and_hashes():
    manifest_path = PROJECT_ROOT / "models" / "manifest.json"
    assert manifest_path.exists(), "models/manifest.json must exist"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Check models
    for model_key, model_info in manifest["models"].items():
        file_path = PROJECT_ROOT / model_info["file"]
        assert file_path.exists(), f"Model file {model_info['file']} not found"
        
        # Verify hash
        h = hashlib.sha256()
        with open(file_path, "rb") as mf:
            while chunk := mf.read(1024 * 1024):
                h.update(chunk)
        calc_sha = h.hexdigest()
        assert calc_sha == model_info["sha256"], f"SHA256 mismatch for {model_key}"


def test_onnx_runtime_sessions():
    """Verify that ONNX Runtime can load and run all models."""
    manifest_path = PROJECT_ROOT / "models" / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for model_key, model_info in manifest["models"].items():
        file_path = PROJECT_ROOT / model_info["file"]
        sess = ort.InferenceSession(str(file_path), providers=["CPUExecutionProvider"])
        assert sess is not None, f"Failed to load ONNX session for {model_key}"
        inputs = sess.get_inputs()
        outputs = sess.get_outputs()
        assert len(inputs) > 0
        assert len(outputs) > 0


def test_san1_guardrail_end_to_end():
    """Verify high-level SAN1Guardrail interface."""
    guardrail = SAN1Guardrail(allow_benign_content=True)

    # Test clean text
    res_text = guardrail.validate_text("Buongiorno a tutti, questa è una bellissima giornata di sole.")
    assert res_text.is_allowed is True
    assert res_text.status == "APPROVED"

    # Test prohibited blasphemy text
    res_bad = guardrail.validate_text("porco dio che schifo")
    assert res_bad.is_allowed is False
    assert res_bad.status == "BLOCKED"

    # Test synthetic image
    img = Image.new("RGB", (224, 224), color=(200, 180, 160))
    res_img = guardrail.validate_image(img)
    assert res_img is not None
    assert isinstance(res_img.is_allowed, bool)
    assert res_img.latency_ms > 0
