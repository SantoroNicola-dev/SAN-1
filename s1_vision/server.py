"""
SAN-1 FastAPI Production Microservice
=====================================
REST API server for multimodal content moderation.
Exposes:
- POST /v1/guardrail/validate (Multipart image file or JSON Base64 + optional text)
- POST /v1/guardrail/validate-text (Pure text moderation)
- GET  /health (System and ONNX models healthcheck)
- GET  /docs (OpenAPI Swagger Documentation)
"""

import io
import os
import base64
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from pathlib import Path
from PIL import Image

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .guardrail import SAN1Guardrail, SAN1Result

# Global Guardrail instance
guardrail: Optional[SAN1Guardrail] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global guardrail
    print("[*] SAN-1 Microservice: Loading ONNX models and pre-warming pipeline...")
    guardrail = SAN1Guardrail()
    # Dummy pre-warm inference
    try:
        dummy = Image.new("RGB", (224, 224), color=(200, 200, 200))
        guardrail.validate(image_input=dummy, text_input="prewarm text")
        print("[+] SAN-1 Microservice: Pre-warming successfully completed. ONNX models ready.")
    except Exception as e:
        print(f"[!] Warning during pre-warm: {e}")
    yield
    print("[*] SAN-1 Microservice: Shutdown completed.")


app = FastAPI(
    title="SAN-1 Multimodal Content Guardrail API",
    description="High-performance, offline ONNX-powered safety guardrail for images and text.",
    version="1.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextValidationRequest(BaseModel):
    text: str = Field(..., description="Text string to moderate (caption, bio, comment, chat message)")


class MultimodalJSONRequest(BaseModel):
    image_base64: Optional[str] = Field(None, description="Base64-encoded image string (PNG, JPG, WebP)")
    text: Optional[str] = Field(None, description="Optional caption or associated text")


@app.get("/health", tags=["Monitoring"])
async def health_check():
    """Returns operational status and integrity of SAN-1 and its ONNX runtime models."""
    if guardrail is None:
        raise HTTPException(status_code=503, detail="ONNX models not yet initialized.")
    
    return {
        "status": "healthy",
        "service": "SAN-1 Multimodal Guardrail",
        "version": "1.3.0",
        "policy_version": guardrail.vision.policy.version,
        "policy_hash": guardrail.vision.policy.compute_policy_hash(),
        "models": {
            "vision_clip_onnx": guardrail.vision.engine.session is not None,
            "text_clip_onnx": guardrail.vision.engine.text_session is not None,
            "ocr_rapidocr": guardrail.ocr is not None,
            "nlp_multilingual_onnx": guardrail.nlp is not None,
        }
    }


@app.post("/v1/guardrail/validate", tags=["Moderation"])
async def validate_content(
    file: Optional[UploadFile] = File(None, description="Binary image file (multipart/form-data)"),
    text: Optional[str] = Form(None, description="Optional text or caption (multipart)"),
    json_payload: Optional[MultimodalJSONRequest] = None,
):
    """
    Primary multimodal moderation endpoint:
    Supports:
    1. Multipart upload: image file + optional text field
    2. JSON payload: image_base64 + optional text field
    """
    if guardrail is None:
        raise HTTPException(status_code=503, detail="Guardrail service not ready.")

    img = None
    text_content = text

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
            detail="Invalid request: please provide an image file or a payload with 'image_base64' or 'text'."
        )

    res: SAN1Result = guardrail.validate(image_input=img, text_input=text_content)
    return res.to_dict()


@app.post("/v1/guardrail/validate-text", tags=["Moderation"])
async def validate_text_only(req: TextValidationRequest):
    """High-speed text-only moderation (blasphemy, hate speech, threats, malware, fraud, etc.)."""
    if guardrail is None:
        raise HTTPException(status_code=503, detail="Guardrail service not ready.")

    res = guardrail.validate_text(req.text)
    return res.to_dict()


def main():
    import uvicorn
    host = os.getenv("SAN1_HOST", "0.0.0.0")
    port = int(os.getenv("SAN1_PORT", "8000"))
    workers = int(os.getenv("SAN1_WORKERS", "1"))
    print(f"[*] Starting SAN-1 Microservice on http://{host}:{port} (workers={workers})...")
    uvicorn.run("s1_vision.server:app", host=host, port=port, workers=workers)


if __name__ == "__main__":
    main()
