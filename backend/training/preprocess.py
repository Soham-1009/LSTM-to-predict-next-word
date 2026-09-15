"""Dataset loading and text preprocessing utilities used by training.

Three corpus sources are supported:

1. ``bundled``  -- the small public-domain sample that ships with the
   repo, so a fresh clone always trains end-to-end in seconds.
2. ``gutenberg`` -- one of several Project Gutenberg eBooks. The user
   picks the book by name; the file is downloaded once into
   ``backend/datasets/`` and cached for subsequent runs.
3. ``file`` -- an arbitrary plain-text file the user points at.

The CLI in ``train.py`` exposes all three via ``--corpus {bundled,
gutenberg, file}``.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

from app.utils.text_processing import normalize_text, tokenize

# Where downloaded corpora are cached.
DATASETS_DIR: Path = Path(__file__).resolve().parent.parent / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

# Default paths.
DEFAULT_LOCAL_CORPUS: Path = DATASETS_DIR / "sample_corpus.txt"

# A tiny curated corpus used as a last-resort fallback if the sample file
# is somehow missing.  Contains enough repetition to teach the model
# reasonable local patterns.
FALLBACK_CORPUS: str = (
    "the future of artificial intelligence is bright and full of promise. "
    "machine learning can transform the way we live and work. "
    "deep learning is a powerful tool for solving complex problems. "
    "neural networks learn hierarchical representations of data. "
    "natural language processing helps computers understand human language. "
    "the future of technology is shaped by artificial intelligence and machine learning. "
    "deep learning models require large amounts of data and compute. "
    "data scientists build models that learn from data. "
    "python is a popular programming language for data science. "
    "the rapid growth of artificial intelligence is changing every industry. "
) * 30


# ---------------------------------------------------------------------------
# Project Gutenberg catalog
# ---------------------------------------------------------------------------
# Public-domain English eBooks maintained by https://www.gutenberg.org.
# We hand-pick a small set so the training CLI can offer them by name
# without exposing a raw URL surface area.
GUTENBERG_BOOKS: Dict[str, Dict[str, str]] = {
    "alice": {
        "title": "Alice's Adventures in Wonderland",
        "author": "Lewis Carroll",
        # Plain-text UTF-8 mirror on gutenberg.org.
        "url": "https://www.gutenberg.org/cache/epub/11/pg11.txt",
        "filename": "gutenberg_alice.txt",
    },
    "frankenstein": {
        "title": "Frankenstein",
        "author": "Mary Shelley",
        "url": "https://www.gutenberg.org/cache/epub/84/pg84.txt",
        "filename": "gutenberg_frankenstein.txt",
    },
    "pride": {
        "title": "Pride and Prejudice",
        "author": "Jane Austen",
        "url": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
        "filename": "gutenberg_pride.txt",
    },
    "sherlock": {
        "title": "The Adventures of Sherlock Holmes",
        "author": "Arthur Conan Doyle",
        "url": "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
        "filename": "gutenberg_sherlock.txt",
    },
    "dorian": {
        "title": "The Picture of Dorian Gray",
        "author": "Oscar Wilde",
        "url": "https://www.gutenberg.org/cache/epub/174/pg174.txt",
        "filename": "gutenberg_dorian.txt",
    },
    "moby": {
        "title": "Moby-Dick",
        "author": "Herman Melville",
        "url": "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
        "filename": "gutenberg_moby.txt",
    },
    "warpeace": {
        "title": "War and Peace",
        "author": "Leo Tolstoy",
        "url": "https://www.gutenberg.org/cache/epub/2600/pg2600.txt",
        "filename": "gutenberg_warpeace.txt",
    },
    "ulysses": {
        "title": "Ulysses",
        "author": "James Joyce",
        "url": "https://www.gutenberg.org/cache/epub/4300/pg4300.txt",
        "filename": "gutenberg_ulysses.txt",
    },
}


# ---------------------------------------------------------------------------
# Bundled sample handling
# ---------------------------------------------------------------------------
def _ensure_sample_corpus() -> Path:
    """Make sure there is a non-empty `sample_corpus.txt` to train on."""
    if DEFAULT_LOCAL_CORPUS.exists() and DEFAULT_LOCAL_CORPUS.stat().st_size > 0:
        return DEFAULT_LOCAL_CORPUS

    DEFAULT_LOCAL_CORPUS.write_text(FALLBACK_CORPUS, encoding="utf-8")
    return DEFAULT_LOCAL_CORPUS


# ---------------------------------------------------------------------------
# Gutenberg download
# ---------------------------------------------------------------------------
def _strip_gutenberg_boilerplate(text: str) -> str:
    """Remove the standard Project Gutenberg header/footer.

    Gutenberg files contain explicit ``*** START OF ... ***`` and
    ``*** END OF ... ***`` markers around the actual book body. Keeping
    the boilerplate in training data adds a lot of noise like
    "project gutenberg", "license", "electronic works", etc.
    """
    start = text.find("*** START OF")
    end = text.find("*** END OF")
    if start != -1 and end != -1 and end > start:
        # Skip past the line that contains the START marker.
        newline = text.find("\n", start)
        if newline == -1:
            return text
        return text[newline + 1 : end].strip()
    return text


def download_gutenberg(book_key: str, *, force: bool = False) -> Path:
    """Download a Project Gutenberg book and cache it locally.

    Returns the path to the cached file. If the network is unavailable
    or the book key is unknown, falls back to the bundled sample.
    """
    info = GUTENBERG_BOOKS.get(book_key.lower())
    if info is None:
        raise KeyError(
            f"Unknown Gutenberg book '{book_key}'. "
            f"Available: {', '.join(sorted(GUTENBERG_BOOKS))}"
        )

    dest = DATASETS_DIR / info["filename"]
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return dest

    url = info["url"]
    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "LSTM-Next-Word-Predictor/1.0"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:  # nosec - trusted URL
            raw = response.read()
        text = raw.decode("utf-8", errors="ignore")
        text = _strip_gutenberg_boilerplate(text)
        if len(text) < 1_000:
            raise ValueError(f"Downloaded content from {url} looks too small ({len(text)} chars).")
        dest.write_text(text, encoding="utf-8")
        return dest
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        # Don't crash the demo if the user is offline; the bundled
        # sample is still good enough to run end-to-end.
        print(f"[preprocess] WARNING: could not download {info['title']} ({exc}). "
              f"Falling back to the bundled sample corpus.")
        return _ensure_sample_corpus()


# ---------------------------------------------------------------------------
# Corpus dispatch
# ---------------------------------------------------------------------------
def load_bundled_corpus() -> str:
    """Return the bundled `sample_corpus.txt` as a single string."""
    path = _ensure_sample_corpus()
    return path.read_text(encoding="utf-8")


def load_gutenberg_corpus(book_key: str, *, force: bool = False) -> str:
    """Download (if needed) and return a Project Gutenberg book."""
    path = download_gutenberg(book_key, force=force)
    return path.read_text(encoding="utf-8")


def load_corpus_from_file(path: str | os.PathLike) -> str:
    """Read an arbitrary plain-text file as the corpus."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Corpus file not found: {p}")
    return p.read_text(encoding="utf-8", errors="ignore")


def load_default_corpus() -> str:
    """Backwards-compatible alias for the bundled corpus."""
    return load_bundled_corpus()


def load_corpus_lines(path: str | os.PathLike) -> List[str]:
    """Read a plain-text file as a list of non-empty lines."""
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="ignore")
    return [line for line in text.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess_corpus(raw_text: str) -> List[List[str]]:
    """Normalize and tokenize a raw corpus into sentences-of-tokens.

    Sentences are detected by `.!?` and line breaks; everything else is
    collapsed into a single long sequence which is then split on those
    markers.
    """
    if not raw_text:
        return []

    normalized = normalize_text(raw_text)
    # Split on sentence punctuation + newlines, keeping small chunks.
    sentences: List[str] = []
    buffer: List[str] = []
    for ch in normalized:
        buffer.append(ch)
        if ch in {".", "!", "?", "\n"}:
            sentence = "".join(buffer).strip()
            if sentence:
                sentences.append(sentence)
            buffer = []
    tail = "".join(buffer).strip()
    if tail:
        sentences.append(tail)

    token_seqs: List[List[str]] = []
    for sent in sentences:
        tokens = tokenize(sent)
        if tokens:
            token_seqs.append(tokens)
    return token_seqs


def corpus_stats(raw_text: str, token_sequences: List[List[str]]) -> Dict[str, int]:
    """Return a small dict of corpus statistics for the training log."""
    return {
        "characters": len(raw_text),
        "sentences": len(token_sequences),
        "tokens": sum(len(seq) for seq in token_sequences),
        "unique_tokens": len({tok for seq in token_sequences for tok in seq}),
    }
