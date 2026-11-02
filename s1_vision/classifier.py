"""
SAN-1 Universal Multimodal Multi-Label Content Classifier
=========================================================
Categorizes images and text across 60+ built-in categories (and arbitrary zero-shot
custom categories at runtime) with:
- Independent calibrated confidence percentage (`confidence_pct`: 0-100%)
- Normalized composition share percentage (`share_pct`: sums to 100% across detected categories)
- Adaptive System-1 (Global Fast-Path) + System-2 (5-Crop Spatial Quadrant Inspection)
- Multimodal Fusion of Vision (CLIP ViT-B/32 ONNX) + Embedded Text OCR (RapidOCR) + Semantic NLP (MiniLM ONNX)
"""

from typing import Optional, Union, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import os
import json
import hashlib
import time
from PIL import Image
import numpy as np

from .preprocessing import ImagePreprocessor
from .s1_engine import S1stemVisionEngine, S1Candidate
from .router import ConfidenceGatedRouter
from .system2 import DualStrategySystem2
from .pipeline import HybridVisionPipeline
from .ocr_guardrail import OCRTextGuardrail, OCRGuardrailResult
from .text_semantic_guardrail import TextSemanticGuardrail


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CATEGORIES_PATH = REPO_ROOT / "config" / "categories.json"
DEFAULT_CATEGORIES_CACHE_PATH = REPO_ROOT / "models" / "clip" / "categories_embeddings_cache.npz"


@dataclass
class CategoryPrediction:
    id: str
    label: str
    group: str
    confidence_pct: float
    share_pct: float
    source: str = "vision"
    description: Optional[str] = None
    similarity: Optional[float] = None
    region: str = "global"

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "label": self.label,
            "group": self.group,
            "confidence_pct": round(float(self.confidence_pct), 2),
            "share_pct": round(float(self.share_pct), 2),
            "source": self.source,
        }
        if self.region:
            d["region"] = self.region
        if self.similarity is not None:
            d["similarity"] = round(float(self.similarity), 4)
        return d


@dataclass
class ClassificationResult:
    primary_category: Dict[str, Any]
    categories: List[Dict[str, Any]]
    category_id: str
    category_label: str
    confidence_pct: float
    confidence: float
    extracted_text: Optional[str] = None
    latency_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_category": self.primary_category,
            "categories": self.categories,
            "category_id": self.category_id,
            "category_label": self.category_label,
            "confidence_pct": round(float(self.confidence_pct), 2),
            "extracted_text": self.extracted_text,
            "latency_ms": round(float(self.latency_ms), 2),
            "details": self.details,
        }


class SAN1Classifier:
    """
    Universal Multi-Label Content Categorizer for Images and Text.
    Produces neutral, multi-category percentage breakdowns without binary safe/unsafe blocking.
    """

    def __init__(
        self,
        models_dir: Optional[Union[str, Path]] = None,
        categories_path: Optional[Union[str, Path]] = None,
        cache_path: Optional[Union[str, Path]] = None,
        enable_ocr: bool = True,
        enable_nlp: bool = True,
        default_min_confidence: float = 25.0,
        default_top_k: int = 10,
        confidence_threshold: float = 0.55,
        margin_threshold: float = 0.08,
        entropy_threshold: float = 2.60,
    ):
        base_dir = REPO_ROOT

        env_models = os.getenv("SAN1_MODELS_DIR")
        if models_dir is not None:
            self.models_dir = Path(models_dir)
        elif env_models:
            self.models_dir = Path(env_models)
        else:
            self.models_dir = base_dir / "models"

        env_cats = os.getenv("SAN1_CATEGORIES_PATH")
        if categories_path is not None:
            self.categories_path = Path(categories_path)
        elif env_cats:
            self.categories_path = Path(env_cats)
        else:
            self.categories_path = DEFAULT_CATEGORIES_PATH

        if cache_path is not None:
            self.cache_path = Path(cache_path)
        else:
            self.cache_path = self.models_dir / "clip" / "categories_embeddings_cache.npz"

        self.enable_ocr = enable_ocr
        self.enable_nlp = enable_nlp
        self.default_min_confidence = float(default_min_confidence)
        self.default_top_k = int(default_top_k)

        # Load category catalog JSON
        with open(self.categories_path, "r", encoding="utf-8") as f:
            self.catalog_data = json.load(f)

        self.version = self.catalog_data.get("version", "2.0")
        self.groups = self.catalog_data.get("groups", [])
        self.visual_categories = self.catalog_data.get("categories", [])
        self.text_categories = self.catalog_data.get("text_categories", [])
        self.benign_anchors = self.catalog_data.get("benign_anchors", {})

        vision_model = self.models_dir / "clip" / "onnx" / "vision_model_quantized.onnx"
        text_model = self.models_dir / "clip" / "onnx" / "text_model_quantized.onnx"
        tokenizer_path = self.models_dir / "clip" / "tokenizer.json"

        self.preprocessor = ImagePreprocessor()
        self.engine = S1stemVisionEngine(
            model_path=str(vision_model) if vision_model.exists() else None,
            text_model_path=str(text_model) if text_model.exists() else None,
            tokenizer_path=str(tokenizer_path) if tokenizer_path.exists() else None,
            temperature=0.02,
        )
        self.router = ConfidenceGatedRouter(
            confidence_threshold=confidence_threshold,
            margin_threshold=margin_threshold,
            entropy_threshold=entropy_threshold,
        )
        self.system2 = DualStrategySystem2(
            mode="adaptive",
            s1_engine=self.engine,
            preprocessor=self.preprocessor,
        )
        self.pipeline = HybridVisionPipeline(
            s1_engine=self.engine,
            router=self.router,
            system2=self.system2,
            preprocessor=self.preprocessor,
        )

        # Load or build multi-prompt embeddings cache
        self._load_or_build_cache()

        # Initialize OCR & NLP Categorizers
        self.ocr: Optional[OCRTextGuardrail] = None
        if self.enable_ocr:
            try:
                self.ocr = OCRTextGuardrail(enable_semantic=self.enable_nlp)
            except Exception:
                self.ocr = None

        self.nlp: Optional[TextSemanticGuardrail] = None
        if self.enable_nlp:
            if self.ocr is not None and self.ocr.semantic_guard is not None and self.ocr.semantic_guard.is_ready:
                self.nlp = self.ocr.semantic_guard
            else:
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

    def compute_catalog_hash(self) -> str:
        """Computes deterministic SHA-256 hash of all category definitions and prompts."""
        payload = {
            "categories": [
                {
                    "id": c["id"],
                    "prompts": c.get("prompts", [c.get("description", "")]),
                    "baseline_threshold": c.get("baseline_threshold", 0.200),
                    "contrast_anchors": c.get("contrast_anchors", []),
                    "contrast_margin": c.get("contrast_margin", 0.0),
                }
                for c in self.visual_categories
            ],
            "benign_anchors": self.benign_anchors,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _load_or_build_cache(self) -> None:
        """Loads pre-computed multi-prompt category vectors from .npz cache or builds them."""
        current_hash = self.compute_catalog_hash()
        cache_loaded = False

        if self.cache_path.exists():
            try:
                cached = np.load(self.cache_path, allow_pickle=True)
                if str(cached.get("catalog_hash", "")) == current_hash and "action_matrix" in cached:
                    self.engine.action_matrix = cached["action_matrix"].astype(np.float32)
                    self.engine.action_cache = [
                        S1Candidate(
                            id=c["id"],
                            label=c["label"],
                            description=c.get("description", ""),
                            index=idx,
                            group=c.get("group", "general"),
                            prompts=c.get("prompts", []),
                            baseline_threshold=float(c.get("baseline_threshold", 0.200)),
                            contrast_anchors=list(c.get("contrast_anchors", [])),
                            contrast_margin=float(c.get("contrast_margin", 0.0)),
                        )
                        for idx, c in enumerate(self.visual_categories)
                    ]
                    if "benign_matrix" in cached and "benign_keys" in cached:
                        self.engine.benign_matrix = cached["benign_matrix"].astype(np.float32)
                        self.engine.benign_keys = [str(k) for k in cached["benign_keys"]]
                    cache_loaded = True
            except Exception:
                cache_loaded = False

        if not cache_loaded:
            self.engine.register_classes(self.visual_categories)
            self.engine.register_benign_anchors(self.benign_anchors)
            try:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                save_kwargs: Dict[str, Any] = {
                    "catalog_hash": current_hash,
                    "action_matrix": self.engine.action_matrix,
                }
                if self.engine.benign_matrix is not None:
                    save_kwargs["benign_matrix"] = self.engine.benign_matrix
                    save_kwargs["benign_keys"] = np.array(self.engine.benign_keys)
                np.savez_compressed(self.cache_path, **save_kwargs)
            except Exception:
                pass

    def list_categories(self) -> Dict[str, Any]:
        """Returns the full catalog of visual and textual categories and macro-groups."""
        return {
            "version": self.version,
            "total_categories": len(self.visual_categories) + len(self.text_categories),
            "visual_categories_count": len(self.visual_categories),
            "text_categories_count": len(self.text_categories),
            "groups": self.groups,
            "visual_categories": [
                {
                    "id": c["id"],
                    "label": c["label"],
                    "group": c.get("group", "general"),
                    "description": c.get("description", ""),
                }
                for c in self.visual_categories
            ],
            "text_categories": self.text_categories,
        }

    @staticmethod
    def _renormalize_shares(
        categories: List[Dict[str, Any]],
        min_confidence: float,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Filters, sorts by confidence_pct descending, caps at top_k, and normalizes share_pct to 100%."""
        if not categories:
            return []

        sorted_cats = sorted(categories, key=lambda x: float(x.get("confidence_pct", 0.0)), reverse=True)
        active = [dict(c) for c in sorted_cats if float(c.get("confidence_pct", 0.0)) >= min_confidence]
        if not active:
            active = [dict(sorted_cats[0])]

        # If top category is image_quality but a substantive semantic subject exists, prioritize the object
        if active and active[0].get("group") == "image_quality" and len(active) > 1:
            for idx_cand in range(1, len(active)):
                if active[idx_cand].get("group") != "image_quality" and float(active[idx_cand].get("confidence_pct", 0.0)) >= 28.0:
                    primary_item = active.pop(idx_cand)
                    active.insert(0, primary_item)
                    break

        # Subject-over-Scene Prioritization:
        # If top category is a generic environmental background or domestic room,
        # but a salient foreground subject (animal, object, vehicle, food, person, document, architecture)
        # is clearly present (confidence >= 70.0% or within 20.0% of the scene), prioritize the subject.
        AMBIENT_SCENE_IDS = {
            "bedroom_interior",
            "bedroom_bathroom_interior",
            "home_living_room",
            "mountain_hills",
            "field_countryside_landscape",
            "forest_trees",
            "beach_sea_coast",
            "sky_sunset_clouds",
            "snow_winter_scene",
            "store_restaurant_interior",
        }
        OUTDOOR_SCENE_IDS = {
            "mountain_hills",
            "field_countryside_landscape",
            "forest_trees",
            "beach_sea_coast",
            "sky_sunset_clouds",
            "snow_winter_scene",
        }
        INDOOR_ONLY_IDS = {
            "kitchen_utensils_cutlery",
            "kitchenware_tableware",
            "bedroom_interior",
            "bedroom_bathroom_interior",
            "home_living_room",
            "office_workspace",
        }
        if active and active[0].get("id") in AMBIENT_SCENE_IDS and len(active) > 1:
            top_scene_id = active[0].get("id")
            top_scene_conf = float(active[0].get("confidence_pct", 0.0))
            for idx_cand in range(1, len(active)):
                cand_id = active[idx_cand].get("id")
                cand_conf = float(active[idx_cand].get("confidence_pct", 0.0))
                # Guard: indoor objects cannot be promoted over outdoor landscapes
                if top_scene_id in OUTDOOR_SCENE_IDS and cand_id in INDOOR_ONLY_IDS:
                    continue
                if (
                    cand_id not in AMBIENT_SCENE_IDS
                    and active[idx_cand].get("group") not in ("image_quality", "sensitive_content")
                    and (cand_conf >= 85.0 and cand_conf >= (top_scene_conf - 12.0))
                ):
                    primary_item = active.pop(idx_cand)
                    active.insert(0, primary_item)
                    break

        # Prune low-confidence background noise and hallucinated sensitive labels from secondary positions
        SENSITIVE_TEXT_IDS = {
            "profanity_vulgar_text",
            "blasphemy_religious_slur",
            "hate_speech_text",
            "threats_violence_text",
            "cyber_malware_text",
            "scam_fraud_text",
        }
        if len(active) > 1:
            filtered = [active[0]]
            primary_id = active[0].get("id")
            for c in active[1:]:
                c_conf = float(c.get("confidence_pct", 0.0))
                cid = c.get("id")
                # Sensitive categories must have at least 85% confidence to appear as secondary
                if (c.get("group") == "sensitive_content" or cid in SENSITIVE_TEXT_IDS) and c_conf < 85.0:
                    continue
                # Strict domestic mutual exclusion:
                if primary_id == "dog" and cid == "cat":
                    continue
                if primary_id == "cat" and cid == "dog":
                    continue
                if c_conf >= 50.0:
                    filtered.append(c)
            active = filtered

        # Promote confirmed high-confidence sensitive content (>= 92%) to primary position
        if (
            len(active) > 1
            and (active[0].get("group") != "sensitive_content" or active[0].get("id") == "smoking_tobacco_vape")
            and active[0].get("id") not in SENSITIVE_TEXT_IDS
        ):
            for idx_cand in range(1, len(active)):
                cand_item = active[idx_cand]
                cand_id = cand_item.get("id")
                if (
                    (cand_item.get("group") == "sensitive_content" and cand_id != "smoking_tobacco_vape")
                    or cand_id in SENSITIVE_TEXT_IDS
                ) and float(cand_item.get("confidence_pct", 0.0)) >= 92.0:
                    primary_item = active.pop(idx_cand)
                    active.insert(0, primary_item)
                    break

        if top_k and top_k > 0:
            active = active[:top_k]

        # Power-scaled weight for share distribution:
        # A dominant primary (90%+) should take ~70-95% share, while secondary elements take realistic 5-25%
        active_weights = np.array([(float(c["confidence_pct"]) / 100.0) ** 3.5 for c in active], dtype=np.float32)
        total_weight = float(np.sum(active_weights)) if len(active_weights) > 0 else 1.0
        if total_weight < 1e-9:
            total_weight = 1.0

        # Filter out negligible secondary shares (< 3.5%) when multiple categories exist
        if len(active) > 1:
            raw_shares = (active_weights / total_weight) * 100.0
            kept = [active[0]]
            kept_weights = [active_weights[0]]
            for i in range(1, len(active)):
                if raw_shares[i] >= 3.5:
                    kept.append(active[i])
                    kept_weights.append(active_weights[i])
            active = kept
            active_weights = np.array(kept_weights, dtype=np.float32)
            total_weight = float(np.sum(active_weights)) if len(active_weights) > 0 else 1.0
            if total_weight < 1e-9:
                total_weight = 1.0

        running = 0.0
        for i, c in enumerate(active):
            c["confidence_pct"] = round(float(c["confidence_pct"]), 2)
            if i == len(active) - 1:
                share = round(max(0.0, 100.0 - running), 2)
            else:
                share = round((float(active_weights[i]) / total_weight) * 100.0, 2)
                running += share
            c["share_pct"] = share
            c.pop("index", None)
            c.pop("probability", None)
        return active

    def _score_custom_categories(
        self,
        v_s: np.ndarray,
        custom_categories: List[Union[str, Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Zero-shot runtime encoding and scoring for user-supplied custom categories."""
        scored = []
        for idx, item in enumerate(custom_categories):
            if isinstance(item, str):
                if ":" in item:
                    cid, desc = item.split(":", 1)
                    cid = cid.strip().lower().replace(" ", "_")
                    label = cid.replace("_", " ").title()
                    prompts = [desc.strip()]
                else:
                    cid = item.strip().lower().replace(" ", "_")
                    label = item.strip().title()
                    prompts = [f"a photograph of {item.strip()}"]
                group = "custom"
                baseline = 0.200
            elif isinstance(item, dict):
                cid = item.get("id") or f"custom_{idx}"
                label = item.get("label") or cid.replace("_", " ").title()
                desc = item.get("description") or label
                prompts = item.get("prompts") or [f"{label}. {desc}"]
                group = item.get("group", "custom")
                baseline = float(item.get("baseline_threshold", 0.200))
            else:
                continue

            vec = self.engine.encode_prompts_ensemble(prompts)
            sim = float(np.dot(vec, v_s))
            logit = 42.0 * (sim - baseline)
            conf = 100.0 / (1.0 + float(np.exp(-np.clip(logit, -15.0, 15.0))))
            scored.append({
                "id": cid,
                "label": label,
                "group": group,
                "description": prompts[0],
                "confidence_pct": round(float(np.clip(conf, 0.0, 99.9)), 2),
                "share_pct": 0.0,
                "similarity": round(sim, 4),
                "source": "vision_zeroshot",
                "region": "global",
            })
        return scored

    def classify_image(
        self,
        image_input: Union[str, Path, bytes, Image.Image, np.ndarray],
        top_k: Optional[int] = None,
        min_confidence: Optional[float] = None,
        deep: bool = False,
        check_embedded_text: bool = True,
        custom_categories: Optional[List[Union[str, Dict[str, Any]]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ClassificationResult:
        """
        Categorizes an image into one or more categories with independent confidence_pct
        and normalized composition share_pct (summing to 100%).
        """
        t0 = time.perf_counter()
        min_conf = self.default_min_confidence if min_confidence is None else float(min_confidence)
        k_limit = self.default_top_k if top_k is None else int(top_k)

        # 1. Vision Pipeline Execution (System-1 Fast Path or System-2 Spatial Multi-Crop)
        vis_res = self.pipeline.process(
            image_input=image_input,
            metadata=metadata,
            deep=deep,
            min_confidence_pct=min_conf,
            top_k=max(k_limit, 15),
        )

        merged_map: Dict[str, Dict[str, Any]] = {}
        for cat in vis_res.get("categories", []):
            merged_map[cat["id"]] = dict(cat)

        # 2. Zero-Shot Custom Categories (if requested)
        latent_vec = vis_res.get("latent_vector")
        if custom_categories and latent_vec is not None:
            for c_cat in self._score_custom_categories(latent_vec, custom_categories):
                cid = c_cat["id"]
                if cid not in merged_map or c_cat["confidence_pct"] > merged_map[cid]["confidence_pct"]:
                    merged_map[cid] = c_cat

        # 3. Embedded Text OCR & Semantic NLP Categorization
        extracted_text: Optional[str] = None
        matched_keywords: List[str] = []
        if check_embedded_text and self.ocr is not None:
            try:
                ocr_res: OCRGuardrailResult = self.ocr.validate_image(
                    image_input,
                    min_confidence_pct=min_conf,
                    top_k=5,
                )
                if (
                    not (ocr_res.detected_text and ocr_res.detected_text.strip())
                    and float(merged_map.get("clothing_fashion_apparel", {}).get("confidence_pct", 0.0)) >= 99.0
                ):
                    from PIL import ImageOps
                    mirrored_img = ImageOps.mirror(self.preprocessor.load_image(image_input))
                    mirror_ocr = self.ocr.validate_image(
                        mirrored_img,
                        min_confidence_pct=min_conf,
                        top_k=5,
                    )
                    if mirror_ocr.offending_words:
                        ocr_res = mirror_ocr
                if ocr_res.detected_text and ocr_res.detected_text.strip():
                    extracted_text = ocr_res.detected_text.strip()
                    matched_keywords = ocr_res.offending_words
                    for t_cat in ocr_res.categories:
                        cid = t_cat["id"]
                        if cid in merged_map:
                            # Multimodal agreement boost when both Vision and OCR detect the same concept
                            boosted = min(99.9, max(merged_map[cid]["confidence_pct"], t_cat["confidence_pct"]) + 5.0)
                            merged_map[cid]["confidence_pct"] = round(boosted, 2)
                            merged_map[cid]["source"] = "multimodal_vision_ocr"
                        else:
                            merged_map[cid] = dict(t_cat)
            except Exception:
                pass

        # 4. Final Multi-Label Filtering and 100% Share Normalization
        final_categories = self._renormalize_shares(
            list(merged_map.values()),
            min_confidence=min_conf,
            top_k=k_limit,
        )

        primary = final_categories[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        details: Dict[str, Any] = {
            "route": vis_res.get("route", "System-1 (Fast Path)"),
            "crops_analyzed": vis_res.get("crops_analyzed", 1),
            "total_categories_evaluated": len(self.visual_categories) + len(self.text_categories) + (len(custom_categories) if custom_categories else 0),
            "detected_categories_count": len(final_categories),
        }
        if matched_keywords:
            details["matched_keywords"] = matched_keywords
        if vis_res.get("deliberation"):
            details["deliberation"] = vis_res["deliberation"]

        return ClassificationResult(
            primary_category=primary,
            categories=final_categories,
            category_id=primary["id"],
            category_label=primary["label"],
            confidence_pct=primary["confidence_pct"],
            confidence=round(primary["confidence_pct"] / 100.0, 4),
            extracted_text=extracted_text,
            latency_ms=round(latency_ms, 2),
            details=details,
        )

    def classify_text(
        self,
        text_input: str,
        top_k: Optional[int] = None,
        min_confidence: Optional[float] = None,
    ) -> ClassificationResult:
        """
        Categorizes a text string across lexical and multilingual semantic categories
        with independent confidence_pct and normalized share_pct.
        """
        t0 = time.perf_counter()
        min_conf = self.default_min_confidence if min_confidence is None else float(min_confidence)
        k_limit = 5 if top_k is None else int(top_k)

        categories: List[Dict[str, Any]] = []
        matched_keywords: List[str] = []

        if self.ocr is not None:
            categories, matched_keywords = self.ocr.categorize_text(
                text_input,
                is_embedded_in_image=False,
                min_confidence_pct=min_conf,
                top_k=k_limit,
            )
        elif self.nlp is not None:
            categories = self.nlp.classify_text(
                text_input,
                min_confidence_pct=min_conf,
                top_k=k_limit,
            )

        if not categories:
            categories = [
                {
                    "id": "greeting_social_caption",
                    "label": "General Text / Caption",
                    "group": "text_content",
                    "confidence_pct": 50.0,
                    "share_pct": 100.0,
                    "source": "text_default",
                }
            ]

        primary = categories[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return ClassificationResult(
            primary_category=primary,
            categories=categories,
            category_id=primary["id"],
            category_label=primary["label"],
            confidence_pct=primary["confidence_pct"],
            confidence=round(primary["confidence_pct"] / 100.0, 4),
            extracted_text=text_input,
            latency_ms=round(latency_ms, 2),
            details={
                "route": "Multilingual Text Categorizer (Lexical + MiniLM ONNX)",
                "matched_keywords": matched_keywords,
                "detected_categories_count": len(categories),
            },
        )

    def classify(
        self,
        image_input: Optional[Union[str, Path, bytes, Image.Image, np.ndarray]] = None,
        text_input: Optional[str] = None,
        top_k: Optional[int] = None,
        min_confidence: Optional[float] = None,
        deep: bool = False,
        custom_categories: Optional[List[Union[str, Dict[str, Any]]]] = None,
    ) -> ClassificationResult:
        """
        Unified multimodal categorization for image, text, or both combined.
        """
        t0 = time.perf_counter()
        min_conf = self.default_min_confidence if min_confidence is None else float(min_confidence)
        k_limit = self.default_top_k if top_k is None else int(top_k)

        if image_input is not None and not text_input:
            return self.classify_image(
                image_input,
                top_k=k_limit,
                min_confidence=min_conf,
                deep=deep,
                custom_categories=custom_categories,
            )

        if text_input and image_input is None:
            return self.classify_text(
                text_input,
                top_k=k_limit,
                min_confidence=min_conf,
            )

        if image_input is not None and text_input:
            img_res = self.classify_image(
                image_input,
                top_k=max(k_limit, 15),
                min_confidence=min_conf,
                deep=deep,
                custom_categories=custom_categories,
            )
            txt_res = self.classify_text(
                text_input,
                top_k=max(k_limit, 10),
                min_confidence=min_conf,
            )

            merged: Dict[str, Dict[str, Any]] = {c["id"]: dict(c) for c in img_res.categories}
            for tc in txt_res.categories:
                cid = tc["id"]
                if cid in merged:
                    merged[cid]["confidence_pct"] = round(
                        min(99.9, max(merged[cid]["confidence_pct"], tc["confidence_pct"]) + 5.0),
                        2,
                    )
                    merged[cid]["source"] = "multimodal_image_text"
                else:
                    merged[cid] = dict(tc)

            final_cats = self._renormalize_shares(
                list(merged.values()),
                min_confidence=min_conf,
                top_k=k_limit,
            )
            primary = final_cats[0]
            combined_text = text_input if not img_res.extracted_text else f"{text_input} | OCR: {img_res.extracted_text}"
            latency_ms = (time.perf_counter() - t0) * 1000.0

            return ClassificationResult(
                primary_category=primary,
                categories=final_cats,
                category_id=primary["id"],
                category_label=primary["label"],
                confidence_pct=primary["confidence_pct"],
                confidence=round(primary["confidence_pct"] / 100.0, 4),
                extracted_text=combined_text,
                latency_ms=round(latency_ms, 2),
                details={
                    **img_res.details,
                    "route": f"{img_res.details.get('route')} + Text Categorizer",
                    "detected_categories_count": len(final_cats),
                },
            )

        return ClassificationResult(
            primary_category={
                "id": "no_input",
                "label": "No Input Provided",
                "group": "general",
                "confidence_pct": 0.0,
                "share_pct": 100.0,
                "source": "none",
            },
            categories=[],
            category_id="no_input",
            category_label="No Input Provided",
            confidence_pct=0.0,
            confidence=0.0,
            extracted_text=None,
            latency_ms=0.0,
            details={},
        )

    def classify_batch(
        self,
        image_inputs: List[Union[str, Path, bytes, Image.Image, np.ndarray]],
        top_k: Optional[int] = None,
        min_confidence: Optional[float] = None,
    ) -> List[ClassificationResult]:
        """High-throughput batch categorization for a list of images."""
        if not image_inputs:
            return []

        t0 = time.perf_counter()
        min_conf = self.default_min_confidence if min_confidence is None else float(min_confidence)
        k_limit = self.default_top_k if top_k is None else int(top_k)

        batch_tensors = self.preprocessor.preprocess_batch(image_inputs)
        s1_results = self.engine.predict_batch(
            batch_tensors,
            min_confidence_pct=min_conf,
            top_k=k_limit,
        )
        total_ms = (time.perf_counter() - t0) * 1000.0
        per_img_ms = round(total_ms / len(image_inputs), 2)

        results: List[ClassificationResult] = []
        for s1_res in s1_results:
            cats = self._renormalize_shares(s1_res.categories, min_confidence=min_conf, top_k=k_limit)
            primary = cats[0]
            results.append(
                ClassificationResult(
                    primary_category=primary,
                    categories=cats,
                    category_id=primary["id"],
                    category_label=primary["label"],
                    confidence_pct=primary["confidence_pct"],
                    confidence=round(primary["confidence_pct"] / 100.0, 4),
                    extracted_text=None,
                    latency_ms=per_img_ms,
                    details={
                        "route": "System-1 (Batch Fast Path)",
                        "crops_analyzed": 1,
                        "detected_categories_count": len(cats),
                    },
                )
            )
        return results

    def classify_and_route(
        self,
        image_path: Union[str, Path],
        move: bool = True,
        target_root: Optional[Union[str, Path]] = None,
        top_k: Optional[int] = None,
        min_confidence: Optional[float] = None,
        deep: bool = False,
    ) -> Tuple[ClassificationResult, Path]:
        """
        Categorizes an image and routes/moves it into a subfolder named after its
        primary detected category: `<target_root>/<primary_category_id>/<filename>`.
        """
        p = Path(image_path)
        res = self.classify_image(p, top_k=top_k, min_confidence=min_confidence, deep=deep)
        if not move or not p.exists():
            return res, p

        base_dir = Path(target_root) if target_root is not None else p.parent
        dest_dir = base_dir / res.category_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / p.name

        if p.resolve() != dest_path.resolve():
            p.replace(dest_path)

        return res, dest_path

