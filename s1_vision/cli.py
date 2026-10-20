"""
SAN-1 Command Line Interface (CLI)
==================================
Command-line tool to run SAN-1 moderation and management.
Usage:
    san1 validate-image <path/to/image.jpg>
    san1 validate-text "Text to moderate"
    san1 serve [--host 0.0.0.0] [--port 8000]
    san1 verify-models
"""

import sys
import json
import argparse
from pathlib import Path

from .guardrail import SAN1Guardrail


def cmd_validate_image(args):
    img_path = Path(args.image_path)
    if not img_path.exists():
        print(f"[!] Error: file not found: {img_path}", file=sys.stderr)
        sys.exit(1)

    guardrail = SAN1Guardrail()
    res = guardrail.validate_image(img_path)
    print(json.dumps(res.to_dict(), indent=2, ensure_ascii=False))
    if not res.is_allowed:
        sys.exit(2)


def cmd_validate_text(args):
    guardrail = SAN1Guardrail()
    res = guardrail.validate_text(args.text)
    print(json.dumps(res.to_dict(), indent=2, ensure_ascii=False))
    if not res.is_allowed:
        sys.exit(2)


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
        description="SAN-1 Multimodal Content Guardrail CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # validate-image
    p_img = subparsers.add_parser("validate-image", help="Validate a local image with the guardrail")
    p_img.add_argument("image_path", type=str, help="Path to the image file to analyze")
    p_img.set_defaults(func=cmd_validate_image)

    # validate-text
    p_txt = subparsers.add_parser("validate-text", help="Validate a text string with the guardrail")
    p_txt.add_argument("text", type=str, help="Text string to analyze")
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
