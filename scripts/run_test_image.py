"""
Master Coordinator & Orchestrator for Test/Image Dataset Moderation
===================================================================
Executes multi-process parallel evaluation of all images in Test/Image/
using the Unified Vision + OCR Moderation Guardrail.
Physical routing into Test/Image/accepted/ and Test/Image/deny/.

Usage:
    python run_test_image.py [--workers 10]
"""

import sys
import os
import json
import time
import argparse
import subprocess
from pathlib import Path
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = REPO_ROOT / "Test" / "Image"
ACCEPTED_DIR = IMAGE_DIR / "accepted"
DENY_DIR = IMAGE_DIR / "deny"
CHECKPOINT_DIR = IMAGE_DIR / "checkpoints"
LOGS_DIR = IMAGE_DIR / "logs"
MASTER_RESULTS_PATH = IMAGE_DIR / "benchmark_results.json"
REPORT_PATH = IMAGE_DIR / "moderation_report.md"

def get_python_exe():
    try:
        import onnxruntime  # noqa: F401
        return sys.executable
    except ImportError:
        candidate = r"C:\Users\nicol\AppData\Local\Programs\Python\Python312\python.exe"
        if os.path.exists(candidate):
            return candidate
        return sys.executable

PYTHON_EXE = get_python_exe()
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

def generate_report(target_dir, total_evaluated, safe_count, deny_count, categories, elapsed_sec, accepted_dir, deny_dir, image_dir, report_path, master_results_path, cat_safety_map=None):
    safe_pct = (safe_count / total_evaluated * 100.0) if total_evaluated > 0 else 0
    deny_pct = (deny_count / total_evaluated * 100.0) if total_evaluated > 0 else 0
    throughput = (total_evaluated / elapsed_sec) if elapsed_sec > 0 else 0

    dataset_name = target_dir.name
    cat_safety_map = cat_safety_map or {}

    md = f"""# S1stem Vision: Report Moderazione Dataset `{dataset_name}`

**Data e Ora Test:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Cartella di Lavoro:** `{target_dir}`  
**Engine di Moderazione:** S1stem Hybrid Vision (Contrastive Fast-Path) + RapidOCR ONNX  
**Architettura Esecuzione:** Multi-process Worker Pool  

---

## 1. Risultati Globali di Moderazione

| Esito Classificazione | Conteggio Immagini | Percentuale Quota | Cartella di Routing | Criterio Applicato |
| :--- | :---: | :---: | :--- | :--- |
| **ACCEPTED (Approvato)** | **{safe_count:,}** | **{safe_pct:.2f}%** | `{accepted_dir.name}\\` | Volti umani, selfie, pose amichevoli, oggetti o paesaggi innocui, testo pulito |
| **DENY (Rifiutato)** | **{deny_count:,}** | **{deny_pct:.2f}%** | `{deny_dir.name}\\` | Gesti volgari (dito medio), nudità/esplicito, armi, simboli d'odio, bestemmie, volgarità/citazioni offensive (OCR) |
| **TOTALE VALUTATO** | **{total_evaluated:,}** | **100.00%** | `{target_dir.name}\\` | Dataset completo classificato e smistato fisicamente |

- **Tempo Totale di Elaborazione:** **{elapsed_sec:.1f} s** ({elapsed_sec/60:.1f} min)
- **Throughput Medio di Sistema:** **{throughput:.1f} img/s**
- **Latenza Media per Immagine:** **{(elapsed_sec/total_evaluated*1000.0) if total_evaluated else 0:.1f} ms**

---

## 2. Ripartizione per Categoria di Moderazione

| Categoria Rilevata | Conteggio | Percentuale | Esito Moderazione | Livello Guardrail |
| :--- | :---: | :---: | :---: | :--- |
"""
    for cat_name, cnt in categories.most_common():
        pct = (cnt / total_evaluated * 100.0) if total_evaluated > 0 else 0
        is_safe = cat_safety_map.get(cat_name, False)
        status_tag = "✅ ACCEPTED" if is_safe else "❌ DENY"

        guard_type = "Vision Guardrail (S1stem)" if "OCR" not in cat_name else "OCR Guardrail (RapidOCR)"
        md += f"| **{cat_name}** | **{cnt:,}** | {pct:.1f}% | {status_tag} | {guard_type} |\n"

    md += f"""
---

## 3. Verifica Integrità del Routing Fisico

- File presenti in `{accepted_dir}`: **{len(list(accepted_dir.glob('*'))):,}**
- File presenti in `{deny_dir}`: **{len(list(deny_dir.glob('*'))):,}**
- Immagini residue in `{image_dir}`: **{len([f for f in image_dir.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS]):,}**
- Tutti i record dettagliati sono archiviati in formato JSON in `{master_results_path}`.
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\n[+] Report di moderazione salvato con successo in:\n    {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Test runner for dataset moderation")
    parser.add_argument("--dir", type=str, default="Test/Image", help="Target dataset directory containing candidate images to moderate")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel worker processes (default: 10)")
    args = parser.parse_args()

    num_workers = args.workers

    target_dir = Path(args.dir)
    if not target_dir.is_absolute():
        target_dir = (REPO_ROOT / target_dir).resolve()

    image_dir = target_dir
    accepted_dir = image_dir / "accepted"
    deny_dir = image_dir / "deny"
    checkpoint_dir = image_dir / "checkpoints"
    logs_dir = image_dir / "logs"
    master_results_path = image_dir / "benchmark_results.json"
    report_path = image_dir / "moderation_report.md"

    accepted_dir.mkdir(parents=True, exist_ok=True)
    deny_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Count images currently in target root
    images_in_root = [
        f for f in image_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
    ]
    total_images = len(images_in_root)

    print("=" * 80)
    print("S1STEM VISION: UNIFIED MODERATION TEST RUNNER")
    print(f"Target Directory:      {image_dir}")
    print(f"Total Candidate Images: {total_images:,}")
    print(f"Parallel Worker Pool:  {num_workers} processes")
    print(f"Routing Destinations:  {accepted_dir.name}/ | {deny_dir.name}/")
    print("=" * 80, flush=True)

    if total_images == 0:
        print(f"[!] Nessuna immagine trovata direttamente in {image_dir}. Verifica se sono già state smistate.")
        return

    # Launch parallel workers
    t0 = time.time()
    procs = []
    log_files = []

    for w_id in range(num_workers):
        log_f = open(logs_dir / f"worker_{w_id}.log", "w", encoding="utf-8")
        log_files.append(log_f)
        worker_script = str(REPO_ROOT / "scripts" / "run_test_image_worker.py")
        cmd = [
            PYTHON_EXE, "-u", worker_script,
            "--worker-id", str(w_id),
            "--num-workers", str(num_workers),
            "--dir", str(image_dir)
        ]
        p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, text=True)
        procs.append((w_id, p))

    print(f"[+] Avviati {num_workers} worker in parallelo. Monitoraggio live dei checkpoint...\n", flush=True)

    last_print = 0
    while any(p.poll() is None for _, p in procs):
        time.sleep(3)
        total_done = 0
        total_blocked = 0
        total_accepted = 0

        for w_id in range(num_workers):
            cp = checkpoint_dir / f"checkpoint_worker_{w_id}.json"
            if cp.exists():
                try:
                    with open(cp, "r", encoding="utf-8") as f:
                        d = json.load(f)
                        total_done += len(d)
                        total_blocked += sum(1 for r in d.values() if not r.get("is_safe", False))
                        total_accepted += sum(1 for r in d.values() if r.get("is_safe", False))
                except Exception:
                    pass

        now = time.time()
        elapsed = now - t0
        if now - last_print >= 6 or total_done == total_images:
            last_print = now
            speed = total_done / elapsed if elapsed > 0 else 0
            pct = (total_done / total_images * 100.0) if total_images > 0 else 0
            left = max(0, total_images - total_done)
            eta_sec = left / speed if speed > 0 else 0
            print(
                f"[PROGRESS] {total_done:5d}/{total_images} ({pct:5.1f}%) | "
                f"Blocked: {total_blocked:,} | Acc: {total_accepted:,} | "
                f"Speed: {speed:4.1f} img/s | ETA: {eta_sec/60:4.1f} min",
                flush=True
            )

    # Close worker logs
    for lf in log_files:
        lf.close()

    elapsed_total = time.time() - t0
    print(f"\n[+] Tutti i worker hanno completato l'elaborazione in {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)!")

    # Aggregate all checkpoints into master results
    print("[+] Aggregazione dei risultati di tutti i worker...")
    master_records = {}
    for w_id in range(num_workers):
        cp = checkpoint_dir / f"checkpoint_worker_{w_id}.json"
        if cp.exists():
            try:
                with open(cp, "r", encoding="utf-8") as f:
                    master_records.update(json.load(f))
            except Exception as e:
                print(f"Warning: errore lettura {cp.name}: {e}")

    with open(master_results_path, "w", encoding="utf-8") as f:
        json.dump(master_records, f)
    print(f"[+] Master JSON salvato in: {master_results_path}")

    safe_count = sum(1 for r in master_records.values() if r.get("is_safe", False))
    deny_count = sum(1 for r in master_records.values() if not r.get("is_safe", False))
    categories = Counter(r.get("category_label", "Non Definito") for r in master_records.values())
    cat_safety_map = {r.get("category_label"): r.get("is_safe", False) for r in master_records.values()}

    # Generate Markdown Report
    generate_report(target_dir, len(master_records), safe_count, deny_count, categories, elapsed_total, accepted_dir, deny_dir, image_dir, report_path, master_results_path, cat_safety_map=cat_safety_map)

    # Console Summary Table
    print("\n" + "=" * 80)
    print(f"RIEPILOGO FINALE TEST MODERAZIONE: {target_dir.name}")
    print("=" * 80)
    print(f"• Immagini Totali Valutate: {len(master_records):,}")
    print(f"• Approvate (in accepted/): {safe_count:,} ({(safe_count/len(master_records)*100.0) if master_records else 0:.2f}%)")
    print(f"• Rifiutate (in deny/):     {deny_count:,} ({(deny_count/len(master_records)*100.0) if master_records else 0:.2f}%)")
    print(f"• Residue in directory:    {len([f for f in image_dir.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS]):,}")
    print("=" * 80)

if __name__ == "__main__":
    main()
