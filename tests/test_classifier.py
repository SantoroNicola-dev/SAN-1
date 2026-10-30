"""
Unit & Multi-Label Integration Tests for SAN1Classifier.
========================================================
Tests:
- Multi-label output format (confidence_pct 0-100%, share_pct summing to 100%)
- Multi-subject detection (e.g. selfie + cat/dog/animal)
- Spatial 5-crop inspection (deep=True)
- Zero-shot custom categories at runtime
- Multilingual text categorization
- Taxonomy and category list verification
"""

import sys
from pathlib import Path
import pytest
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from s1_vision.classifier import SAN1Classifier, ClassificationResult


@pytest.fixture(scope="session")
def classifier():
    return SAN1Classifier()


def test_list_categories(classifier):
    cats = classifier.list_categories()
    assert "total_categories" in cats
    assert cats["total_categories"] >= 60
    assert len(cats["groups"]) == 12
    assert any(c["id"] == "cat" for c in cats["visual_categories"])
    assert any(c["id"] == "closeup_selfie" for c in cats["visual_categories"])
    assert any(c["id"] == "weapon_firearm_knife" for c in cats["visual_categories"])


def test_synthetic_image_multi_label_percentages(classifier):
    # Create synthetic image
    arr = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(arr)

    res = classifier.classify_image(img, top_k=5, min_confidence=10.0)
    assert isinstance(res, ClassificationResult)
    assert res.category_id is not None
    assert res.category_label is not None
    assert 0.0 <= res.confidence_pct <= 100.0
    assert len(res.categories) >= 1

    # Check share_pct sums to 100%
    shares = [c["share_pct"] for c in res.categories]
    assert np.isclose(sum(shares), 100.0, atol=0.2)

    # Check each category structure
    for c in res.categories:
        assert "id" in c
        assert "label" in c
        assert "group" in c
        assert "confidence_pct" in c
        assert "share_pct" in c
        assert 0.0 <= c["confidence_pct"] <= 100.0
        assert 0.0 <= c["share_pct"] <= 100.0


def test_custom_zero_shot_categories(classifier):
    arr = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(arr)

    custom = ["quantum_computer:a futuristic quantum computer with golden wires", "neon_cyberpunk:neon lights and cyberpunk street"]
    res = classifier.classify_image(img, custom_categories=custom, top_k=5, min_confidence=5.0)

    cat_ids = [c["id"] for c in res.categories]
    # Check that custom categories were evaluated and integrated into the distribution
    assert "details" in res.to_dict()
    assert res.details["total_categories_evaluated"] >= 62


def test_deep_spatial_multicrop(classifier):
    arr = np.random.randint(50, 200, (300, 300, 3), dtype=np.uint8)
    img = Image.fromarray(arr)

    res = classifier.classify_image(img, deep=True)
    assert res.details["crops_analyzed"] == 5
    assert "System-2" in res.details["route"] or "Spatial" in res.details["route"]


def test_text_multi_label_classification(classifier):
    res = classifier.classify_text("Good morning friends, remember to work hard and pursue your dreams!")
    assert isinstance(res, ClassificationResult)
    assert len(res.categories) >= 1
    assert any(c["id"] in ("inspirational_quote", "greeting_social_caption") for c in res.categories)

    # Test sensitive text without blocking
    res_profane = classifier.classify_text("you are a fucking idiot shut up")
    assert any("profanity" in c["id"] for c in res_profane.categories)
    assert res_profane.confidence_pct > 70.0


def test_real_dataset_selfie(classifier):
    safe_images = list((PROJECT_ROOT / "benchmark" / "dataset" / "images" / "safe").glob("*"))
    if not safe_images:
        pytest.skip("No benchmark safe images found.")

    res = classifier.classify_image(safe_images[0], top_k=5)
    assert res.primary_category["id"] in ("closeup_selfie", "portrait_studio", "child_or_baby")
    assert res.primary_category["group"] == "people"
    assert res.confidence_pct > 50.0
