"""
SAN-1 FastAPI Production Microservice
=====================================
REST API server for multimodal multi-label content classification.
Exposes:
- POST /v1/classify (Multipart image file or JSON Base64 + optional text + parameters)
- POST /v1/classify-text (Pure text multi-label categorization)
- GET  /v1/categories (List all 60+ visual and textual categories)
- POST /v1/guardrail/validate (Compatibility alias)
- POST /v1/guardrail/validate-text (Compatibility alias)
- GET  /health (System and ONNX models healthcheck)
- GET  /docs (OpenAPI Swagger Documentation)
"""

import io
import os
import base64
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
from pathlib import Path
from PIL import Image

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .classifier import SAN1Classifier, ClassificationResult

# Global Classifier instance
classifier: Optional[SAN1Classifier] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier
    print("[*] SAN-1 Microservice: Loading ONNX models and pre-warming classifier...")
    classifier = SAN1Classifier()
    # Dummy pre-warm inference
    try:
        dummy = Image.new("RGB", (224, 224), color=(200, 200, 200))
        classifier.classify(image_input=dummy, text_input="prewarm text")
        print("[+] SAN-1 Microservice: Pre-warming successfully completed. ONNX models ready.")
    except Exception as e:
        print(f"[!] Warning during pre-warm: {e}")
    yield
    print("[*] SAN-1 Microservice: Shutdown completed.")


app = FastAPI(
    title="SAN-1 Multimodal Content Classification API",
    description="High-performance, offline ONNX-powered multi-label categorizer for images and text.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextClassificationRequest(BaseModel):
    text: str = Field(..., description="Text string to categorize (caption, bio, comment, chat message)")
    top_k: Optional[int] = Field(5, description="Maximum number of top categories to return")
    min_confidence: Optional[float] = Field(25.0, description="Minimum confidence percentage threshold (0-100)")


class MultimodalJSONRequest(BaseModel):
    image_base64: Optional[str] = Field(None, description="Base64-encoded image string (PNG, JPG, WebP)")
    text: Optional[str] = Field(None, description="Optional caption or associated text")
    top_k: Optional[int] = Field(10, description="Maximum number of categories to return")
    min_confidence: Optional[float] = Field(25.0, description="Minimum confidence percentage (0-100)")
    deep: Optional[bool] = Field(False, description="Enable deep 5-crop spatial quadrant inspection")
    custom_categories: Optional[List[str]] = Field(None, description="Optional runtime zero-shot categories")


@app.get("/health", tags=["Monitoring"])
async def health_check():
    """Returns operational status and integrity of SAN-1 and its ONNX runtime models."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="ONNX models not yet initialized.")

    return {
        "status": "healthy",
        "service": "SAN-1 Multimodal Classifier",
        "version": classifier.version,
        "catalog_hash": classifier.compute_catalog_hash(),
        "total_categories": len(classifier.visual_categories) + len(classifier.text_categories),
        "models": {
            "vision_clip_onnx": classifier.engine.session is not None,
            "text_clip_onnx": classifier.engine.text_session is not None,
            "ocr_rapidocr": classifier.ocr is not None,
            "nlp_multilingual_onnx": classifier.nlp is not None,
        },
    }


@app.get("/v1/categories", tags=["Taxonomy"])
async def get_categories():
    """Lists all available visual and textual categories organized by macro-group."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Classifier service not ready.")
    return classifier.list_categories()


@app.post("/v1/classify", tags=["Classification"])
async def classify_content(
    file: Optional[UploadFile] = File(None, description="Binary image file (multipart/form-data)"),
    text: Optional[str] = Form(None, description="Optional text or caption (multipart)"),
    top_k: Optional[int] = Form(None, description="Maximum categories to return"),
    min_confidence: Optional[float] = Form(None, description="Minimum confidence percentage (0-100)"),
    deep: Optional[bool] = Form(False, description="Enable deep 5-crop spatial inspection"),
    custom_categories: Optional[str] = Form(None, description="Comma-separated zero-shot custom categories"),
    json_payload: Optional[MultimodalJSONRequest] = None,
):
    """
    Primary multimodal classification endpoint:
    Returns the primary category and full multi-label percentage breakdown
    (both independent confidence_pct and composition share_pct).
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Classifier service not ready.")

    img = None
    text_content = text
    k_limit = top_k
    min_conf = min_confidence
    is_deep = deep or False
    custom_cats: Optional[List[str]] = None

    if custom_categories:
        custom_cats = [c.strip() for c in custom_categories.split(",") if c.strip()]

    # Handle multipart upload
    if file is not None:
        try:
            content = await file.read()
            img = Image.open(io.BytesIO(content)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decode uploaded image: {e}")

    # Handle JSON payload if multipart was not provided
    elif json_payload is not None:
        text_content = json_payload.text or text_content
        k_limit = json_payload.top_k if k_limit is None else k_limit
        min_conf = json_payload.min_confidence if min_conf is None else min_conf
        is_deep = json_payload.deep if json_payload.deep is not None else is_deep
        custom_cats = json_payload.custom_categories or custom_cats

        if json_payload.image_base64:
            try:
                b64_data = json_payload.image_base64
                if "," in b64_data:
                    b64_data = b64_data.split(",", 1)[1]
                content = base64.b64decode(b64_data)
                img = Image.open(io.BytesIO(content)).convert("RGB")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to decode image_base64 field: {e}")

    if img is None and not text_content:
        raise HTTPException(
            status_code=400,
            detail="Invalid request: please provide an image file or a payload with 'image_base64' or 'text'.",
        )

    res: ClassificationResult = classifier.classify(
        image_input=img,
        text_input=text_content,
        top_k=k_limit,
        min_confidence=min_conf,
        deep=is_deep,
        custom_categories=custom_cats,
    )
    return res.to_dict()


@app.post("/v1/classify-text", tags=["Classification"])
async def classify_text_only(req: TextClassificationRequest):
    """High-speed text-only multi-label classification across semantic and lexical categories."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Classifier service not ready.")

    res = classifier.classify_text(
        req.text,
        top_k=req.top_k,
        min_confidence=req.min_confidence,
    )
    return res.to_dict()


# Backward compatibility aliases
@app.post("/v1/guardrail/validate", tags=["Compatibility Aliases"])
async def validate_content_compat(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    json_payload: Optional[MultimodalJSONRequest] = None,
):
    return await classify_content(file=file, text=text, json_payload=json_payload)


@app.post("/v1/guardrail/validate-text", tags=["Compatibility Aliases"])
async def validate_text_compat(req: TextClassificationRequest):
    return await classify_text_only(req)


def main():
    import uvicorn
    host = os.getenv("SAN1_HOST", "0.0.0.0")
    port = int(os.getenv("SAN1_PORT", "8000"))
    workers = int(os.getenv("SAN1_WORKERS", "1"))
    print(f"[*] Starting SAN-1 Classifier Microservice on http://{host}:{port} (workers={workers})...")
    uvicorn.run("s1_vision.server:app", host=host, port=port, workers=workers)


if __name__ == "__main__":
    main()
