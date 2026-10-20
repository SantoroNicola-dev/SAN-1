# Hybrid System-1 / LLM Architecture in ONNX for Image Classification

This document details the architectural principles, theoretical foundations, and implementation guidelines for a high-performance **Hybrid Image Classification and Visual Reasoning** pipeline based on:
1. **System-1 (Fast Path):** A compact vision classifier (CNN / Vision Transformer exported to ONNX) for deterministic low-latency inference ($< 10\text{ ms}$).
2. **Confidence-Gated Router:** A probabilistic gating mechanism that evaluates Shannon entropy, separation margin, and confidence of the System-1 output.
3. **LLM / VLM (Deliberative Path):** A language or vision-language model running under ONNX Runtime for ambiguity resolution, out-of-distribution (OOD) scenarios, and contextual arbitration.

---

## 1. Architectural Overview

```
[Input Image]
       │
       ▼
┌─────────────────────────────────────────┐
│     System-1 (CNN/ViT in ONNX)          │  ◄── Latency ~5-15 ms (FP16/INT8)
│     - Backbone: MobileNet / ConvNeXt    │
└─────────────────────────────────────────┘
       │
       ├─► Shannon Entropy & Softmax Confidence Calculation
       │
       ├───► [Confidence >= Threshold τ]  ──► [Direct Output: Certified Class]
       │                                         (Zero overhead, deterministic)
       │
       └───► [Confidence < Threshold τ]   ──► [System-2: LLM Deliberation Engine]
                                                 - Top-K ambiguous class inspection
                                                 - Generative prompt arbitration / VLM
                                                 - Context-driven resolution
```

### Why use a hybrid architecture?
* **Computational Efficiency:** 80-90% of production visual queries fall into unambiguous, well-separated classes. System-1 processes these in milliseconds on standard CPU or edge accelerators.
* **Failure Resilience:** When System-1 is uncertain (low margin or elevated Shannon entropy), the system avoids forced errors by escalating to a symbolic deliberator or VLM.
* **Unified Runtime:** Both the visual classifier and language models execute under the same runtime (`onnxruntime`), sharing CPU thread pools and hardware execution providers (CUDA, DirectML, CoreML).

---

## 2. Key Components & Decision Gating

### 2.1 3-Factor Gating Metrics
To determine whether an input is resolved by System-1 or forwarded to System-2, three coordinate parameters are measured:

1. **Top-1 Confidence ($p_{\max}$):**
   $$p_{\max} = \max_i \sigma(z)_i$$
   where $\sigma(z)$ is the softmax probability vector. If $p_{\max} \ge \tau$, prediction certainty is sufficient.

2. **Separation Margin ($\Delta$):**
   $$\Delta = p_{(1)} - p_{(2)}$$
   Difference between top-1 and top-2 probabilities. If $\Delta < \delta$, two classes are competing closely (borderline ambiguity).

3. **Shannon Entropy ($H$):**
   $$H(p) = -\sum_{i=1}^{C} p_i \log_2(p_i)$$
   High entropy indicates dispersed probabilities across multiple categories (visual noise, low quality, or OOD scenario).

### 2.2 System-2 Dual-Strategy Deliberation
When the router triggers escalation:
* **Strategy A (Symbolic LLM Contextual):** The LLM receives the top-$K$ candidates with probabilities, margins, and operational metadata (telemetry, camera specs, domain constraints) to resolve the ambiguity symbolically.
* **Strategy B (VLM Multimodal Re-inspection):** In high-entropy scenarios with severe visual distortion, the image is passed to a multimodal vision-language model (e.g. Phi-3.5-Vision or SmolVLM ONNX) for zero-shot visual re-examination.

---

## 3. Reference Implementation

```python
import numpy as np
import onnxruntime as ort

class VisionSystem1:
    """Low-latency reactive vision classifier powered by ONNX Runtime."""
    def __init__(self, model_path: str, labels: list[str]):
        self.session = ort.InferenceSession(
            model_path, 
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.labels = labels

    def predict(self, preprocessed_image: np.ndarray) -> tuple[int, str, float, np.ndarray]:
        """
        Input: preprocessed numpy tensor [1, 3, 224, 224] (float32)
        Output: class_id, label, confidence, full_probabilities
        """
        logits = self.session.run(None, {self.input_name: preprocessed_image})[0]
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probabilities = (exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)).squeeze()
        
        top1_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[top1_idx])
        label = self.labels[top1_idx]
        return top1_idx, label, confidence, probabilities


class LLMDeliberationEngine:
    """System-2: Deliberation model for uncertain or ambiguous cases."""
    def __init__(self, onnx_llm_model_path: str = None):
        self.model_path = onnx_llm_model_path

    def deliberate(self, top_candidates: list[tuple[str, float]], metadata: dict) -> str:
        candidates_str = ", ".join([f"{name} ({prob*100:.1f}%)" for name, prob in top_candidates])
        prompt = (
            f"Evaluate the detected object.\n"
            f"Top candidates from vision classifier: {candidates_str}.\n"
            f"Operational context: {metadata}.\n"
            f"Determine the correct interpretation and note any anomalies:"
        )
        simulated_reasoning = (
            f"[LLM Deliberation] Given the reflection reported in context, "
            f"class '{top_candidates[0][0]}' is confirmed over '{top_candidates[1][0]}'."
        )
        return simulated_reasoning


class HybridVisionPipeline:
    """Pipeline coordinator with confidence-gated routing."""
    def __init__(
        self, 
        system1: VisionSystem1, 
        system2: LLMDeliberationEngine, 
        confidence_threshold: float = 0.75,
        entropy_threshold: float = 1.5
    ):
        self.system1 = system1
        self.system2 = system2
        self.threshold = confidence_threshold
        self.entropy_threshold = entropy_threshold

    @staticmethod
    def _shannon_entropy(probs: np.ndarray) -> float:
        p = np.clip(probs, 1e-12, 1.0)
        return -float(np.sum(p * np.log2(p)))

    def process(self, image_tensor: np.ndarray, metadata: dict = None) -> dict:
        metadata = metadata or {}
        idx, label, conf, all_probs = self.system1.predict(image_tensor)
        entropy = self._shannon_entropy(all_probs)
        
        is_confident = (conf >= self.threshold) and (entropy <= self.entropy_threshold)
        if is_confident:
            return {
                "route": "System-1 (Fast Path)",
                "class_id": idx,
                "label": label,
                "confidence": round(conf, 4),
                "entropy": round(entropy, 4),
                "reasoning": None
            }
        
        top_k_indices = np.argsort(all_probs)[::-1][:3]
        top_candidates = [(self.system1.labels[i], float(all_probs[i])) for i in top_k_indices]
        deliberation_result = self.system2.deliberate(top_candidates, metadata)
        
        return {
            "route": "System-2 (LLM Deliberative Path)",
            "top_candidates": top_candidates,
            "confidence": round(conf, 4),
            "entropy": round(entropy, 4),
            "reasoning": deliberation_result
        }
```

---

## 4. Production Guidelines & Export

1. **Vision Backbone Training (System-1):**
   * Recommended backbones: `mobilenet_v3_small`, `efficientnet_b0`, or `convnext_nano`.
   * Standard image dimensions: $224 \times 224$ px.
   * Dynamic axes: `dynamic_axes={'input': {0: 'batch_size'}}`.

2. **Classifier INT8 Quantization:**
   * Apply static quantization (`quantize_static`) with 100-200 calibration images to minimize CPU latency without accuracy degradation.

3. **Language Model Deployment:**
   * Use `onnxruntime-genai` to load quantized SLMs (e.g., SmolLM-360M, Qwen2.5-0.5B, or Phi-3.5-Mini) in pure local ONNX format.
