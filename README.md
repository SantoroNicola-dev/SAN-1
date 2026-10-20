# SAN-1: Safety Assurance Network - 1
### Multimodal Content Guardrail & Safety Engine

[![Runtime](https://img.shields.io/badge/runtime-ONNX%20Runtime%201.31-blue.svg)](https://onnxruntime.ai/)
[![Architecture](https://img.shields.io/badge/architecture-Two--Tower%20CLM%20%2B%20Veto%20Gate%20%2B%20NLP-purple.svg)](https://github.com/SantoroNicola-dev/S1stem)
[![Fast Path Latency](https://img.shields.io/badge/vision%20fast--path-%3C%2025%20ms-brightgreen.svg)]()
[![Docker](https://img.shields.io/badge/docker-ready-2496ED.svg)](Dockerfile)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

**SAN-1** (**Safety Assurance Network - 1**) is an ultra-high performance multimodal content moderation guardrail for images and text, engineered for mission-critical production environments with **zero external cloud API dependencies** and **fully offline execution** via quantized INT8 ONNX models.

---

## 1. What is SAN-1? (System Definition & Identity)

**SAN-1** stands for **Safety Assurance Network - 1**. It is a **Compound AI System** designed as a deterministic **Multimodal Safety Guardrail and Middleware**. It acts as an ultra-fast, local security boundary sitting between user-generated inputs (images, selfies, profile photos, documents, and text) and upstream platforms, databases, or generative AI workflows.

### 1.1 Technical Classification: Compound AI System
Rather than relying on a single monolithic model or an unpredictable cloud-hosted Large Language Model (LLM), SAN-1 implements a **Compound AI System**—combining multiple specialized AI models with deterministic decision trees and exact pattern matchers:

1. **Vision Model (ViT-B/32 ONNX INT8):** Extracts 512-dimensional visual embeddings in under 15 ms.
2. **Text Policy Encoder (CLIP Text ONNX INT8):** Projects safety policies into the shared multimodal vector space.
3. **Text Spotter & OCR Engine (RapidOCR ONNX):** Extracts embedded text, captions, and watermarks from image graphics.
4. **Multilingual NLP Model (MiniLM Sentence Transformer ONNX INT8):** Encodes text into 384-dimensional dense semantic vectors across 50+ languages.
5. **Deterministic Symbolic Veto Engine:** Enforces cosine margin gates, zero-shot contrastive vetoes, and sub-millisecond Aho-Corasick lexical automata.

### 1.2 Guardrail vs. Autonomous Agent: Why Determinism Matters
A common question in multi-model architectures is whether SAN-1 is an "AI Agent". **Architecturally, SAN-1 is a Guardrail, not an Autonomous Agent — by design:**

| Dimension | Autonomous AI Agent | SAN-1 Safety Guardrail |
| :--- | :--- | :--- |
| **Execution Flow** | Cyclic, self-reflective, dynamic tool-calling | Directed Acyclic Graph (DAG), single-pass pipeline |
| **Output Behavior** | Non-deterministic, probabilistic, generative | Strictly deterministic, immutable decision gates |
| **Hallucination Risk** | Vulnerable to prompt injection & drift | Zero hallucination risk (closed geometric & lexical bounds) |
| **Latency Profile** | High & variable (500 ms – 15+ seconds) | Ultra-low & predictable (< 25 ms vision, < 1 ms text) |
| **Resource Cost** | GPU-intensive or expensive token billing | Lightweight INT8 ONNX executing locally on standard CPU |
| **Statefulness** | Stateful memory, environment mutator | Stateless, pure input-output decision function |

> [!IMPORTANT]
> In mission-critical Trust & Safety (T&S) and compliance, **indeterminism is a liability**. SAN-1 deliberately avoids autonomous agentic loops so that every safety decision is mathematically reproducible, auditable, and instant.

### 1.3 Core Protected Threat Surface
- **Prohibited Visual Content:** Nudity, explicit/NSFW material, illicit drugs, paraphernalia, weapons, violence, gore, hate and extremist symbols, and internet memes / satirical image macros.
- **Offensive Western Gestures:** Middle finger, bras d'honneur (umbrella gesture), horned fingers (*mano cornuta* insult), fig sign (*mano a fico*), reversed V-sign (UK two-finger salute / "up yours"), chin flick (*la barbe*), cut-throat death threat gesture, wanker, and rude shocker gestures.
- **Text & OCR Violations:** Italian religious blasphemy (fused and leetspeak pattern detection in < 0.1 ms), severe profanity, multilingual hate speech, incitement to violence, death threats, malware/cyberattack promotion, and financial scams or fraud across 50+ languages.

---

## 2. Two-Tier Moderation Architecture

```
                                [Input Image / Text]
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                           ▼
          [Vision Pipeline]                           [Text Pipeline (OCR / NLP)]
                    │                                           │
                    ▼                                           ▼
       ┌─────────────────────────┐                 ┌─────────────────────────┐
       │   Tier 1: Veto Gate     │                 │   Tier 1: Fast-Path     │
       │   Fast Contrastive Veto │                 │   Aho-Corasick + Regex  │
       │   NSFW, Drugs, Weapons, │                 │   Blasphemy, Slurs,     │
       │   Hate, Memes, Gestures │                 │   Leetspeak (< 0.1 ms)  │
       └─────────────────────────┘                 └─────────────────────────┘
                    │                                           │
           ┌────────┴────────┐                         ┌────────┴────────┐
           ▼                 ▼                         ▼                 ▼
       [BLOCKED]          [PASS]                   [BLOCKED]          [PASS]
                             │                                           │
                             ▼                                           ▼
       ┌─────────────────────────┐                 ┌─────────────────────────┐
       │   Tier 2: Taxonomy      │                 │   Tier 2: Semantic NLP  │
       │   Framing, Distance,    │                 │   Multilingual Sentence │
       │   Landscapes, Documents │                 │   Transformer ONNX      │
       └─────────────────────────┘                 └─────────────────────────┘
                    │                                           │
           ┌────────┴────────┐                         ┌────────┴────────┐
           ▼                 ▼                         ▼                 ▼
       [BLOCKED]        [APPROVED]                 [BLOCKED]        [APPROVED]
```

---

## 3. Included Quantized ONNX Model Suite

All model weights and tokenizers are pre-quantized to **INT8** and tracked with cryptographic SHA-256 integrity checksums in [`models/manifest.json`](file:///d:/Projects/VibeProject/SAN-1/models/manifest.json):

| Model Component | Relative Path | File Size | Format | Role & Purpose |
| :--- | :--- | :---: | :---: | :--- |
| **Vision Encoder** | `models/clip/onnx/vision_model_quantized.onnx` | 85.0 MB | ONNX INT8 | 512-dim visual embeddings (ViT-B/32) |
| **Text Encoder** | `models/clip/onnx/text_model_quantized.onnx` | 61.5 MB | ONNX INT8 | 512-dim policy & veto prompt embeddings |
| **CLIP Tokenizer** | `models/clip/tokenizer.json` | 2.2 MB | JSON | Fast Hugging Face tokenizers format |
| **Embeddings Cache** | `models/clip/guardrail_embeddings_cache.npz` | 48.8 KB | NumPy Binary | Zero-startup latency (<1 ms) policy cache |
| **Multilingual NLP** | `models/nlp/model_quantized.onnx` | 112.8 MB | ONNX INT8 | 384-dim Sentence Transformer (50+ languages) |
| **NLP Tokenizer** | `models/nlp/tokenizer.json` | 17.1 MB | JSON | Multilingual tokenization index |
| **Vision Policy** | `config/vision_policy.json` | 13.1 KB | JSON | Canonical taxonomy items & contrastive veto thresholds |

---

## 4. Comparative Benchmark Results

Comparative evaluation performed on a test set of 160 images comprising 50 policy compliant selfies and 110 non compliant images across 11 threat categories: weapons, violence, illicit drugs, explicit content, hate symbols, documents, blurry images, vulgar quotes, profane text, offensive gestures, and memes.

| Model | Accuracy | False Negatives | False Positives | Average Latency | Cost per 100k Images |
| :--- | :---: | :---: | :---: | :---: | :---: |
| SAN-1 Offline ONNX INT8 | 98.125% | 0.00% | 2.00% | 265 ms | $0.00 |
| Azure AI Content Safety | 46.25% | 77.27% | 2.00% | 484 ms | $150.00 |
| Azure Foundry GPT-5 nano | 58.75% | 60.00% | 0.00% | 2763 ms | $20.77 |
| Azure Foundry GPT-6.1 | 100.00% | 0.00% | 0.00% | 2846 ms | $173.72 |

Key findings:
- Zero safety leakage for SAN-1: blocked 110 out of 110 violations with 0.00% false negatives, matching the security bar of GPT-6.1.
- Azure AI Content Safety and GPT-5 nano missed 77.27% and 60.00% of threat images respectively in multimodal tests.
- SAN-1 executes locally on standard CPU without GPU at an average latency of 265 ms and zero dollar operational cost, compared to 2.8 seconds and over 170 dollars per 100k requests for frontier cloud models.

---

## 5. Local Installation & Setup

### 5.1 Install via `pip`
```bash
git clone https://github.com/SantoroNicola-dev/SAN-1.git
cd SAN-1

# Install package in editable mode with dependencies
pip install -e .
```

### 5.2 Verify Model Integrity
```bash
san1 verify-models
```

---

## 6. Python Library Usage

```python
from s1_vision import SAN1Guardrail
from PIL import Image

# Unified initialization (loads offline ONNX models from models/)
guardrail = SAN1Guardrail()

# 1. Image Moderation
result = guardrail.validate_image("path/to/sample.jpg")
print(f"Status: {result.status} | Category: {result.category_label}")
if not result.is_allowed:
    print(f"Code: {result.rejection_code}")
    print(f"Reason: {result.rejection_reason}")
    print(f"Hint: {result.remediation_hint}")

# 2. Text Moderation (Bios, Comments, Captions)
res_text = guardrail.validate_text("Good morning everyone, have a wonderful day!")
print(res_text.status)  # "APPROVED"

res_bad = guardrail.validate_text("I will find your home and burn it down, you are dead")
print(res_bad.status)   # "BLOCKED" (VIOLENCE_INCITEMENT)
```

---

## 7. Command-Line Interface (CLI)

```bash
# Validate a local image
san1 validate-image path/to/image.jpg

# Validate a text string
san1 validate-text "Text content to moderate..."

# Check SHA-256 integrity and file presence of all ONNX models and assets
san1 verify-models

# Start the production REST microservice
san1 serve --host 0.0.0.0 --port 8000
```

---

## 8. Docker Container Deployment

SAN-1 includes a production-ready, self-contained multi-stage Docker setup.

### 8.1 Run with Docker Compose (Recommended)
```bash
docker compose up -d
```

### 8.2 Build and Run with Docker CLI
```bash
# Build the Docker image
docker build -t san1-guardrail:latest .

# Launch the HTTP microservice daemon
docker run -d --name san1_service -p 8000:8000 --restart unless-stopped san1-guardrail:latest

# Run CLI verification inside the container
docker run --rm san1-guardrail:latest san1 verify-models
docker run --rm san1-guardrail:latest san1 validate-text "Test query"
```

---

## 9. REST API Specifications (FastAPI)

Interactive OpenAPI Swagger UI is available at:  
👉 **`http://localhost:8000/docs`**

### 9.1 `GET /health`
Returns system health, policy hash, and ONNX engine statuses:
```json
{
  "status": "healthy",
  "service": "SAN-1 Multimodal Guardrail",
  "version": "1.3.0",
  "policy_version": "1.4",
  "policy_hash": "a3b207054fca919bb9d59400f71927c2be5f12de2db3c094b8938f6807eaf1dd",
  "models": {
    "vision_clip_onnx": true,
    "text_clip_onnx": true,
    "ocr_rapidocr": true,
    "nlp_multilingual_onnx": true
  }
}
```

### 9.2 `POST /v1/guardrail/validate` (Multipart Upload)
Accepts an image file (`multipart/form-data`) with an optional `text` field:
```bash
curl -X POST "http://localhost:8000/v1/guardrail/validate" \
  -F "file=@selfie.jpg" \
  -F "text=My profile picture"
```

**JSON Response (Approved):**
```json
{
  "is_allowed": true,
  "status": "APPROVED",
  "category_id": "ALLOWED_CLOSEUP_SELFIE",
  "category_label": "Closeup Portrait Selfie",
  "rejection_code": null,
  "rejection_reason": null,
  "remediation_hint": null,
  "confidence": 0.998,
  "latency_ms": 14.2,
  "details": {
    "stage": "approved",
    "route": "System-1 (Fast Path)"
  }
}
```

**JSON Response (Blocked - Offensive Gesture):**
```json
{
  "is_allowed": false,
  "status": "BLOCKED",
  "category_id": "BLOCKED_OFFENSIVE_GESTURE",
  "category_label": "Offensive or Obscene Gesture",
  "rejection_code": "OFFENSIVE_GESTURE",
  "rejection_reason": "The image portrays a rude, vulgar, obscene, or threatening gesture (middle finger, bras d'honneur / umbrella gesture, horned fingers, fig sign, reversed V sign, throat slit, or chin flick).",
  "remediation_hint": "Please upload a polite and respectful photo without offensive, vulgar, or threatening hand gestures.",
  "confidence": 0.92,
  "latency_ms": 8.1,
  "details": {
    "stage": "vision_guardrail",
    "route": "Safety Veto Gate (OFFENSIVE_GESTURE)"
  }
}
```

### 9.3 `POST /v1/guardrail/validate-text`
Instant moderation for bios, comments, or chat messages:
```bash
curl -X POST "http://localhost:8000/v1/guardrail/validate-text" \
  -H "Content-Type: application/json" \
  -d '{"text": "Text to moderate..."}'
```

---

## 10. Automated Testing & Quality Assurance

```bash
# Execute the complete pytest test suite
pytest
```
- `tests/test_pipeline.py`: Tests preprocessing, router logic, System-1 CLM, and System-2 escalation.
- `tests/test_onnx_integrity.py`: Validates cryptographic SHA-256 hashes and runtime execution of all ONNX models.
