"""Corpus Downloader for LSTM Language Model training.

Downloads and extracts the complete WikiText-103 corpus from the verified fast.ai
AWS S3 mirror (replacing the deprecated MetaMind S3 URL).

Converts the articles into clean, standardized raw text with `<|endoftext|>` document
delimiters directly inside `backend/datasets/raw/wikitext-103.txt`.

Usage:
    python -m backend.training.download_corpus --source wikitext
"""

import argparse
import csv
import io
import sys
import tarfile
import urllib.request
from pathlib import Path


# Fast.ai verified AWS S3 mirror for WikiText-103 (181.3 MB compressed .tgz)
FASTAI_WIKITEXT_URL = "https://s3.amazonaws.com/fast-ai-nlp/wikitext-103.tgz"


def reporthook(block_num: int, block_size: int, total_size: int) -> None:
    """Download progress hook."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, (downloaded / total_size) * 100.0)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\rDownloading WikiText-103: {percent:5.1f}% ({mb_down:6.1f} / {mb_total:6.1f} MB)")
        sys.stdout.flush()


def download_and_extract_wikitext(raw_dir: Path) -> Path:
    """Download WikiText-103 archive and extract articles into a raw text file."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    tgz_path = raw_dir / "wikitext-103.tgz"
    target_txt = raw_dir / "wikitext-103.txt"

    # Increase CSV field size limit for large Wikipedia articles
    csv.field_size_limit(2**30)

    # 1. Download archive if not already downloaded
    if not tgz_path.exists() and not target_txt.exists():
        print(f"Downloading WikiText-103 archive from: {FASTAI_WIKITEXT_URL}")
        tmp_tgz = tgz_path.with_suffix(".tmp")
        urllib.request.urlretrieve(FASTAI_WIKITEXT_URL, tmp_tgz, reporthook=reporthook)
        tmp_tgz.replace(tgz_path)
        print("\nDownload complete.")

    # 2. Extract articles from archive into wikitext-103.txt
    if not target_txt.exists():
        archive_file = tgz_path if tgz_path.exists() else Path("wikitext_test.tgz")
        if not archive_file.exists():
            raise FileNotFoundError(f"Neither {tgz_path} nor wikitext_test.tgz found.")

        print(f"Extracting articles from {archive_file} into {target_txt}...")
        total_articles = 0
        total_bytes = 0

        with tarfile.open(archive_file, "r:gz") as tar, open(target_txt, "w", encoding="utf-8") as out_f:
            for suffix in ["train.csv", "test.csv"]:
                member = next((m for m in tar.getmembers() if m.name.endswith(suffix)), None)
                if member is None:
                    continue
                f_stream = tar.extractfile(member)
                if f_stream is None:
                    continue

                text_stream = io.TextIOWrapper(f_stream, encoding="utf-8", errors="replace")
                reader = csv.reader(text_stream)

                for row in reader:
                    if not row or not row[0].strip():
                        continue
                    article_text = row[0].strip()
                    # Write article demarcated by <|endoftext|>
                    out_f.write(article_text + "\n\n<|endoftext|>\n\n")
                    total_articles += 1
                    total_bytes += len(article_text.encode("utf-8"))

                    if total_articles % 5000 == 0:
                        sys.stdout.write(f"\rExtracted {total_articles:,} articles ({total_bytes / (1024*1024):.1f} MB)...")
                        sys.stdout.flush()

        print(f"\nExtraction complete! {total_articles:,} articles written ({target_txt.stat().st_size / (1024*1024):.1f} MB).")

        # Cleanup compressed archive to conserve disk space
        if tgz_path.exists():
            tgz_path.unlink()
    else:
        print(f"WikiText-103 already exists at {target_txt} ({target_txt.stat().st_size / (1024*1024):.1f} MB).")

    return target_txt


def main():
    parser = argparse.ArgumentParser(description="Corpus Downloader for LSTM Language Model")
    parser.add_argument("--source", type=str, default="wikitext", choices=["wikitext"], help="Dataset source")
    parser.add_argument("--raw-dir", type=str, default="backend/datasets/raw", help="Target raw dataset directory")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    download_and_extract_wikitext(raw_dir)


if __name__ == "__main__":
    main()
