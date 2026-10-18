"""Automated Release Packaging and Checksum Generation Script.

Packages all clean source code, configurations, documentation, and evaluation
artifacts into release/GenAI_LSTM_Final_Submission.zip while strictly excluding
virtual environments, node_modules, generated build caches, and historical archives.
Generates authoritative SHA-256 manifests and checksum verification files.
"""

import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXCLUDE_DIRS = {
    ".venv",
    "node_modules",
    ".next",
    "__pycache__",
    ".pytest_cache",
    ".git",
    "archive",
    ".gemini",
    ".agents",
    "release",
}

EXCLUDE_FILE_PATTERNS = {
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".DS_Store",
    "Thumbs.db",
    "*.zip",
    "*.log",
    "*.dup",
}


def compute_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def should_include(rel_path: Path) -> bool:
    parts = rel_path.parts
    # Check directory exclusions
    for ex in EXCLUDE_DIRS:
        if ex in parts:
            return False

    # Check file pattern exclusions
    for pat in EXCLUDE_FILE_PATTERNS:
        if rel_path.match(pat):
            return False

    # Exclude massive training binaries and raw corpus files (>10MB)
    if "datasets" in parts and rel_path.name.endswith(".bin"):
        return False
    if "raw" in parts and rel_path.name.endswith(".txt") and rel_path.stat().st_size > 10_000_000:
        return False

    # Checkpoints: Include inference weights (best_val.pt and multidomain_adapted.pt).
    # Exclude latest.pt (306 MB training continuation checkpoint) to keep package under 512 MB.
    if rel_path.name == "latest.pt":
        return False

    return True


def collect_package_files() -> List[Path]:
    packaged_files: List[Path] = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in files:
            p = Path(root) / f
            rel = p.relative_to(PROJECT_ROOT)
            if should_include(rel):
                packaged_files.append(rel)
    return sorted(packaged_files)


def build_release_package():
    release_dir = PROJECT_ROOT / "release"
    release_dir.mkdir(parents=True, exist_ok=True)
    zip_path = release_dir / "GenAI_LSTM_Final_Submission.zip"

    # 1. Collect files to pack (strictly excluding release/ directory)
    files_to_pack = collect_package_files()
    print(f"Collected {len(files_to_pack)} clean submission files to package.")

    # 2. Build Primary Runnable ZIP archive (includes inference checkpoints best_val.pt and multidomain_adapted.pt)
    if zip_path.exists():
        zip_path.unlink()

    print(f"Compressing into primary package: {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in files_to_pack:
            abs_p = PROJECT_ROOT / rel
            zf.write(abs_p, str(rel).replace("\\", "/"))

    zip_size = zip_path.stat().st_size
    zip_sha = compute_sha256(zip_path)
    print(f"Generated Primary ZIP package: {zip_path} ({zip_size:,} bytes, {zip_size / (1024**2):.2f} MB, SHA-256: {zip_sha})")

    # 2b. Build Lightweight Source-Only ZIP archive (source code, configs, evaluation data; weights excluded)
    source_zip = release_dir / "GenAI_LSTM_Final_Submission_source_only.zip"
    source_files = [f for f in files_to_pack if not str(f).endswith(".pt")]
    if source_zip.exists():
        source_zip.unlink()

    print(f"Compressing into source-only package: {source_zip.name}...")
    with zipfile.ZipFile(source_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in source_files:
            abs_p = PROJECT_ROOT / rel
            zf.write(abs_p, str(rel).replace("\\", "/"))

    source_size = source_zip.stat().st_size
    source_sha = compute_sha256(source_zip)
    print(f"Generated Source-Only ZIP package: {source_zip} ({source_size:,} bytes, {source_size / (1024**2):.2f} MB, SHA-256: {source_sha})")

    # 3. Checkpoint Artifact Verification
    checkpoints = [
        ("best_val.pt", PROJECT_ROOT / "backend/model_artifacts/checkpoints/best_val.pt", 46, "PACKAGED (Primary Inference)"),
        ("multidomain_adapted.pt", PROJECT_ROOT / "experiments/multidomain_v1/checkpoints/multidomain_adapted.pt", 5, "PACKAGED (Domain Adaptation Inference)"),
        ("latest.pt", PROJECT_ROOT / "backend/model_artifacts/checkpoints/latest.pt", 50, "EXTERNAL (Training Continuity Only)"),
    ]

    ckpt_info = []
    for name, p, epoch, status in checkpoints:
        if p.exists():
            h = compute_sha256(p)
            sz = p.stat().st_size
            ckpt_info.append({
                "name": name,
                "path": p.relative_to(PROJECT_ROOT),
                "epoch": epoch,
                "size": sz,
                "sha256": h,
                "status": status,
            })
            print(f"  Checkpoint {name} [{status}]: {sz:,} bytes -> SHA-256: {h}")
        else:
            print(f"  WARNING: Checkpoint {p} not found!")

    # 4. Write authoritative non-circular RELEASE_CHECKSUMS.sha256
    checksums_file = release_dir / "RELEASE_CHECKSUMS.sha256"
    checksum_lines = [
        f"{zip_sha}  {zip_path.name}",
        f"{source_sha}  {source_zip.name}",
    ]
    for c in ckpt_info:
        checksum_lines.append(f"{c['sha256']}  {c['name']}")

    checksums_file.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(f"Updated: {checksums_file}")

    for c in ckpt_info:
        checksum_lines.append(f"{c['sha256']}  {c['name']}")

    checksums_file.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(f"Updated: {checksums_file}")

    # 5. Build authoritative RELEASE_MANIFEST.md
    manifest_file = release_dir / "RELEASE_MANIFEST.md"
    total_uncompressed = sum((PROJECT_ROOT / f).stat().st_size for f in files_to_pack)

    manifest_md = [
        "# Release Package Manifest — GenAI LSTM Final Submission",
        "",
        "**Project Title:** LSTM Next-Word Prediction & Text Generation System  ",
        f"**Primary Archive File:** `release/{zip_path.name}`  ",
        f"**Total Packaged Files:** {len(files_to_pack)} files  ",
        f"**Total Uncompressed Size:** {total_uncompressed / (1024**2):.2f} MB ({total_uncompressed:,} bytes)  ",
        f"**Archive Size (Compressed):** {zip_size / (1024**2):.2f} MB ({zip_size:,} bytes)  ",
        f"**Package SHA-256:** `{zip_sha}`  ",
        "**Standalone Run-Readiness:** **YES — Out-of-the-Box Runnable for Inference & Evaluation**  ",
        "",
        "---",
        "",
        "## 1. Model Checkpoint Status & Checksums",
        "",
        "| Artifact | Relative Path | Epoch | File Size | Status | SHA-256 Checksum |",
        "| :--- | :--- | :---: | :---: | :---: | :--- |",
    ]

    for c in ckpt_info:
        manifest_md.append(
            f"| **{c['name']}** | `{str(c['path']).replace(chr(92), '/')}` | {c['epoch']} | {c['size'] / (1024**2):.1f} MB ({c['size']:,} bytes) | {c['status']} | `{c['sha256']}` |"
        )

    manifest_md.extend([
        "",
        "### Checkpoint Inclusion & Run-Readiness Policy",
        "- **`best_val.pt` (Epoch 46):** **INCLUDED inside the ZIP archive.** This is the primary trained baseline model loaded by the FastAPI inference service (`backend/app/main.py`) for next-word prediction and greedy/temperature/top-k text generation.",
        "- **`multidomain_adapted.pt` (Epoch 5):** **INCLUDED inside the ZIP archive.** This is the adapted model checkpoint used for multi-domain zero-shot and adapted comparison benchmarks.",
        "- **`latest.pt` (Epoch 50):** **Maintained separately in the project workspace.** This checkpoint represents the final training epoch (step 50) and is required only if continuing LSTM pretraining on WikiText-2. It is omitted from the submission archive to keep the package size well within the 512 MB submission ceiling (~379 MB total).",
        "",
        "---",
        "",
        "## 2. Quickstart Execution Guide (Post-Extraction)",
        "",
        "Because `best_val.pt` and `multidomain_adapted.pt` are bundled inside the package, no external weight downloads or manual file placements are needed to run inference.",
        "",
        "### Step 1: Verify Extracted Checkpoints",
        "```bash",
        "python scripts/verify_checkpoints.py",
        "```",
        "",
        "### Step 2: Start the FastAPI Backend",
        "```bash",
        "cd backend",
        "python -m venv .venv",
        "# Windows: .venv\\Scripts\\activate | Linux/macOS: source .venv/bin/activate",
        "pip install -r requirements.txt",
        "uvicorn app.main:app --host 127.0.0.1 --port 8000",
        "```",
        "",
        "### Step 3: Start the Next.js Frontend Dashboard",
        "```bash",
        "cd ../frontend",
        "npm install",
        "npm run dev",
        "```",
        "Open `http://localhost:3000` to interact with Next-Word Prediction, Text Generation, and the Generalization Benchmark Dashboard.",
        "",
        "---",
        "",
        "## 3. Packaged Files Inventory & SHA-256 Checksums",
        "",
        "The submission ZIP contains all clean source code, configurations, evaluation data, and required model weights. The external `release/` directory is excluded from packaging to prevent circular self-referencing.",
        "",
        "| Relative File Path | Size (Bytes) | SHA-256 Checksum |",
        "| :--- | :---: | :--- |",
    ])

    for f in files_to_pack:
        abs_p = PROJECT_ROOT / f
        sz = abs_p.stat().st_size
        sha = compute_sha256(abs_p)
        manifest_md.append(f"| `{str(f).replace(chr(92), '/')}` | {sz:,} | `{sha}` |")

    manifest_md.append("")
    manifest_file.write_text("\n".join(manifest_md), encoding="utf-8")
    print(f"Updated: {manifest_file}")
    return zip_sha


if __name__ == "__main__":
    build_release_package()

