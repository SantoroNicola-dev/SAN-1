"""
Worker process for Test/Image dataset classification and routing.
Executes Unified Vision + OCR Moderation Guardrail.
"""

import sys
import os

# Limit CPU threads per process to prevent thread thrashing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import time
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from s1_vision.selfie_guardrail import SelfieGuardrail
from s1_vision.ocr_guardrail import OCRTextGuardrail

IMAGE_DIR = REPO_ROOT / "Test" / "Image"
ACCEPTED_DIR = IMAGE_DIR / "accepted"
DENY_DIR = IMAGE_DIR / "deny"
CHECKPOINT_DIR = IMAGE_DIR / "checkpoints"

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--dir", type=str, default="Test/Image", help="Target dataset directory")
    args = parser.parse_args()

    w_id = args.worker_id
    num_w = args.num_workers

    target_dir = Path(args.dir)
    if not target_dir.is_absolute():
        target_dir = (REPO_ROOT / target_dir).resolve()

    image_dir = target_dir
    accepted_dir = image_dir / "accepted"
    deny_dir = image_dir / "deny"
    checkpoint_dir = image_dir / "checkpoints"

    accepted_dir.mkdir(parents=True, exist_ok=True)
    deny_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover all candidate images directly in target directory
    all_files = sorted([
        f.name for f in image_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
    ])

    # 2. Assign deterministic shard to this worker
    my_files = [fn for idx, fn in enumerate(all_files) if idx % num_w == w_id]
    total_assigned = len(my_files)

    checkpoint_file = checkpoint_dir / f"checkpoint_worker_{w_id}.json"
    evaluated = {}
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                evaluated = json.load(f)
        except Exception:
            evaluated = {}

    to_process = [fn for fn in my_files if fn not in evaluated]
    print(f"[Worker {w_id:02d}/{num_w}] Assigned: {total_assigned} | Already Done: {len(evaluated)} | Left: {len(to_process)}", flush=True)

    if not to_process:
        print(f"[Worker {w_id:02d}] All {total_assigned} images already completed!", flush=True)
        return

    # Initialize guardrails
    vision_guardrail = SelfieGuardrail(allow_benign_content=True)
    ocr_guardrail = OCRTextGuardrail()

    t0 = time.time()
    processed_count = 0
    blocked_count = sum(1 for r in evaluated.values() if not r.get("is_safe", False))
    accepted_count = sum(1 for r in evaluated.values() if r.get("is_safe", False))

    for fn in to_process:
        img_p = image_dir / fn
        if not img_p.exists():
            # Check if already moved
            if (deny_dir / fn).exists():
                evaluated[fn] = {"filename": fn, "is_safe": False, "status": "BLOCKED"}
                processed_count += 1
                blocked_count += 1
                continue
            elif (accepted_dir / fn).exists():
                evaluated[fn] = {"filename": fn, "is_safe": True, "status": "APPROVED"}
                processed_count += 1
                accepted_count += 1
                continue
            else:
                continue

        try:
            # Pass 1: Vision Guardrail (gesture, NSFW, weapon, hate symbol)
            t_v0 = time.perf_counter()
            v_res = vision_guardrail.validate(img_p)
            v_ms = (time.perf_counter() - t_v0) * 1000.0

            if not v_res.is_allowed:
                # Blocked by visual safety policy
                is_safe = False
                status = "BLOCKED"
                category_id = v_res.category_id
                category_label = v_res.category_label
                rejection_code = v_res.rejection_code
                rejection_reason = v_res.rejection_reason
                detected_text = ""
                offending_words = []
            else:
                # Pass 2: OCR Guardrail (Italian bad words, blasphemies, English offensive quotes)
                o_res = ocr_guardrail.validate_image(img_p)
                detected_text = o_res.detected_text
                offending_words = o_res.offending_words

                if not o_res.is_allowed:
                    is_safe = False
                    status = "BLOCKED"
                    category_id = o_res.category
                    category_label = "Testo o Citazione Volgare/Offensiva (OCR)"
                    rejection_code = o_res.rejection_code or "PROHIBITED_TEXT"
                    rejection_reason = o_res.rejection_reason or "Rilevato testo o citazione offensiva."
                else:
                    is_safe = True
                    status = "APPROVED"
                    category_id = v_res.category_id
                    category_label = v_res.category_label
                    rejection_code = None
                    rejection_reason = None
        except Exception as e:
            is_safe = False
            status = "BLOCKED"
            category_id = "BLOCKED_CORRUPT"
            category_label = "File Immagine Danneggiato o Non Decodificabile"
            rejection_code = "CORRUPT_IMAGE"
            rejection_reason = f"Errore durante l'apertura o decodifica dell'immagine: {e}"
            detected_text = ""
            offending_words = []
            v_ms = 0.0

        # Route physically
        dest_dir = accepted_dir if is_safe else deny_dir
        dest_p = dest_dir / fn
        try:
            img_p.replace(dest_p)
        except Exception as e:
            # Fallback to copy + remove
            import shutil
            shutil.copy2(img_p, dest_p)
            img_p.unlink(missing_ok=True)

        rec = {
            "filename": fn,
            "is_safe": is_safe,
            "status": status,
            "category_id": category_id,
            "category_label": category_label,
            "rejection_code": rejection_code,
            "rejection_reason": rejection_reason,
            "detected_text": detected_text[:200] if detected_text else "",
            "offending_words": offending_words,
            "vision_latency_ms": round(v_ms, 2),
            "dest_path": str(dest_p.relative_to(REPO_ROOT)),
        }
        evaluated[fn] = rec
        processed_count += 1
        if is_safe:
            accepted_count += 1
        else:
            blocked_count += 1

        if processed_count % 25 == 0 or processed_count == len(to_process):
            elapsed = time.time() - t0
            rate = processed_count / elapsed if elapsed > 0 else 0
            acc_pct = (accepted_count / len(evaluated) * 100.0) if evaluated else 0
            block_pct = (blocked_count / len(evaluated) * 100.0) if evaluated else 0
            print(
                f"[Worker {w_id:02d}] [{len(evaluated):04d}/{total_assigned}] "
                f"Blocked: {blocked_count} ({block_pct:.1f}%) | Acc: {accepted_count} ({acc_pct:.1f}%) | "
                f"Speed: {rate:.1f} img/s",
                flush=True
            )
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(evaluated, f)

    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(evaluated, f)

    elapsed_tot = time.time() - t0
    final_rate = processed_count / elapsed_tot if elapsed_tot > 0 else 0
    print(f"[Worker {w_id:02d}] Completed {processed_count} images in {elapsed_tot:.1f}s ({final_rate:.1f} img/s)!", flush=True)

if __name__ == "__main__":
    main()
