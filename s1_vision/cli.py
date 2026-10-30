"""
SAN-1 Command Line Interface (CLI)
==================================
Command-line tool to run SAN-1 multi-label content classification and management.
Usage:
    san1 classify-image <path/to/image.jpg> [--top-k 10] [--min-confidence 25.0] [--deep]
    san1 classify-text "Text to classify" [--top-k 5] [--min-confidence 25.0]
    san1 list-categories
    san1 serve [--host 0.0.0.0] [--port 8000]
    san1 verify-models
"""

import sys
import json
import argparse
from pathlib import Path

from .classifier import SAN1Classifier


def cmd_classify_image(args):
    img_path = Path(args.image_path)
    if not img_path.exists():
        print(f"[!] Error: file not found: {img_path}", file=sys.stderr)
        sys.exit(1)

    custom_cats = None
    if getattr(args, "custom", None):
        custom_cats = [c.strip() for c in args.custom.split(",") if c.strip()]

    clf = SAN1Classifier()
    res = clf.classify_image(
        img_path,
        top_k=args.top_k,
        min_confidence=args.min_confidence,
        deep=args.deep,
        custom_categories=custom_cats,
    )
    print(json.dumps(res.to_dict(), indent=2, ensure_ascii=False))


def cmd_classify_text(args):
    clf = SAN1Classifier()
    res = clf.classify_text(
        args.text,
        top_k=args.top_k,
        min_confidence=args.min_confidence,
    )
    print(json.dumps(res.to_dict(), indent=2, ensure_ascii=False))


def cmd_list_categories(args):
    clf = SAN1Classifier()
    cats = clf.list_categories()
    print(json.dumps(cats, indent=2, ensure_ascii=False))


def cmd_validate_image(args):
    """Compatibility alias for validate-image."""
    cmd_classify_image(args)


def cmd_validate_text(args):
    """Compatibility alias for validate-text."""
    cmd_classify_text(args)


def cmd_verify_models(args):
    manifest_path = Path(__file__).resolve().parent.parent / "models" / "manifest.json"
    if not manifest_path.exists():
        print(f"[!] Error: models/manifest.json not found at {manifest_path}", file=sys.stderr)
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print("=" * 70)
    print(f"SAN-1 ONNX MODEL SUITE INTEGRITY CHECK (v{manifest['version']})")
    print("=" * 70)
    all_ok = True
    base_dir = manifest_path.parent.parent

    for m_key, m_info in manifest["models"].items():
        p = base_dir / m_info["file"]
        status = "OK" if p.exists() else "MISSING"
        if not p.exists():
            all_ok = False
        print(f"  • [{status:<7}] {m_key:<22}: {m_info['file']} ({m_info['size_bytes'] / (1024*1024):.1f} MB)")

    print("-" * 70)
    for a_key, a_info in manifest["assets"].items():
        p = base_dir / a_info["file"]
        status = "OK" if p.exists() else "MISSING"
        if not p.exists():
            all_ok = False
        print(f"  • [{status:<7}] {a_key:<22}: {a_info['file']}")

    print("=" * 70)
    if all_ok:
        print("[+] All ONNX model files and assets are intact and present.")
    else:
        print("[!] Warning: one or more model files are missing.")
        sys.exit(1)


def cmd_serve(args):
    from .server import main as server_main
    import os
    if args.host:
        os.environ["SAN1_HOST"] = args.host
    if args.port:
        os.environ["SAN1_PORT"] = str(args.port)
    server_main()


def main():
    parser = argparse.ArgumentParser(
        prog="san1",
        description="SAN-1 Multimodal Content Classifier CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # classify-image
    p_cls_img = subparsers.add_parser("classify-image", help="Classify an image with multi-label percentages")
    p_cls_img.add_argument("image_path", type=str, help="Path to the image file to analyze")
    p_cls_img.add_argument("--top-k", type=int, default=10, help="Maximum number of top categories to return (default: 10)")
    p_cls_img.add_argument("--min-confidence", type=float, default=25.0, help="Minimum confidence percentage threshold (default: 25.0)")
    p_cls_img.add_argument("--deep", action="store_true", help="Enable deep 5-crop spatial inspection")
    p_cls_img.add_argument("--custom", type=str, default=None, help="Comma-separated zero-shot custom categories (e.g. 'cat:a pet cat,pizza:a slice of pizza')")
    p_cls_img.set_defaults(func=cmd_classify_image)

    # classify-text
    p_cls_txt = subparsers.add_parser("classify-text", help="Classify a text string with multi-label percentages")
    p_cls_txt.add_argument("text", type=str, help="Text string to analyze")
    p_cls_txt.add_argument("--top-k", type=int, default=5, help="Maximum number of top categories to return (default: 5)")
    p_cls_txt.add_argument("--min-confidence", type=float, default=25.0, help="Minimum confidence percentage threshold (default: 25.0)")
    p_cls_txt.set_defaults(func=cmd_classify_text)

    # list-categories
    p_list = subparsers.add_parser("list-categories", help="List all visual and textual categories")
    p_list.set_defaults(func=cmd_list_categories)

    # Compatibility: validate-image
    p_img = subparsers.add_parser("validate-image", help="Compatibility alias for classify-image")
    p_img.add_argument("image_path", type=str, help="Path to the image file to analyze")
    p_img.add_argument("--top-k", type=int, default=10, help="Maximum number of top categories")
    p_img.add_argument("--min-confidence", type=float, default=25.0, help="Minimum confidence percentage")
    p_img.add_argument("--deep", action="store_true", help="Enable deep 5-crop inspection")
    p_img.add_argument("--custom", type=str, default=None, help="Custom zero-shot categories")
    p_img.set_defaults(func=cmd_validate_image)

    # Compatibility: validate-text
    p_txt = subparsers.add_parser("validate-text", help="Compatibility alias for classify-text")
    p_txt.add_argument("text", type=str, help="Text string to analyze")
    p_txt.add_argument("--top-k", type=int, default=5, help="Maximum categories")
    p_txt.add_argument("--min-confidence", type=float, default=25.0, help="Minimum confidence percentage")
    p_txt.set_defaults(func=cmd_validate_text)

    # verify-models
    p_chk = subparsers.add_parser("verify-models", help="Verify integrity of ONNX models and assets")
    p_chk.set_defaults(func=cmd_verify_models)

    # serve
    p_srv = subparsers.add_parser("serve", help="Start the HTTP REST microservice")
    p_srv.add_argument("--host", type=str, default="0.0.0.0", help="Host binding address (default: 0.0.0.0)")
    p_srv.add_argument("--port", type=int, default=8000, help="TCP listen port (default: 8000)")
    p_srv.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
