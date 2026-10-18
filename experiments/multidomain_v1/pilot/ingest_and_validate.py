"""Master Ingestion and Validation Pipeline for Multi-Domain Real-Data Pilot.

Phase 3B: Correct Split Leakage, Parent-Source Disjoint Partitioning & Licensing Verification.
Validates the end-to-end multi-domain ingestion and preprocessing pipeline
using real, legally vetted documents across 6 target domains:
1. WikiText-103 Training Split (Expository Replay)
2. Project Gutenberg US Public Domain (Literature)
3. IETF RFCs & Python Official Docs (Technical Documentation)
4. PubMed Central Open Access CC-BY (Scientific Writing)
5. Public Domain Dramatic Dialogue (Conversation)
6. Reuters-21578 Financial & Macroeconomic Journalism (News)

Core Invariants Enforced:
- Explicit parent_source_id grouping: Entire parent works (books, plays, RFCs,
  articles, PMC papers) are assigned to a single split before chunking.
- Zero parent-source overlap between train, val, and test partitions.
- Zero duplicate or normalized-identical hash leakage across splits.
- Full per-domain split coverage: Every domain has non-zero representation
  in train, validation, and test.
- Machine-readable license verification (PMC XML <license_ref> parsed).
- Exact SHA-256 deduplication and quality filtering.
- Production vocabulary (30k) compatibility and domain OOV rate measurement.
- uint16 binary serialization with uint64 document boundary offsets.
- PyTorch CausalLanguageModelDataset loading, causal shifting, and boundary isolation.
- Strict bit-for-bit reproducibility (seed 42).
"""

import gc
import hashlib
import io
import json
from pathlib import Path
import random
import re
import sys
import time
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

# Ensure project root is at sys.path[0] and backend/training is NOT in sys.path
# to avoid shadowing Python's standard library 'tokenize' module.
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.model.tokenizer import Tokenizer
from backend.training.preprocess import (
    compute_doc_hash,
    filter_document,
    normalize_document,
)
from backend.training.serialize import (
    compute_file_sha256,
    get_token_dtype,
)
from backend.training.dataset import CausalLanguageModelDataset


# Domain definitions with legal and provenance metadata
DOMAIN_METADATA = {
    "wikitext103_replay": {
        "domain_name": "Expository Replay",
        "source_name": "WikiText-103 Training Split (wiki.train.tokens)",
        "canonical_url": "https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/",
        "license": "CC-BY-SA 3.0",
        "license_status": "APPROVED",
        "attribution": "Stephen Merity et al. (Salesforce Research) / Wikipedia contributors",
        "evidence_url": "https://creativecommons.org/licenses/by-sa/3.0/",
        "terms_notes": "Exclusively sampled from the training partition; zero test tokens ingested."
    },
    "literature": {
        "domain_name": "Literature / Prose",
        "source_name": "Project Gutenberg US Public Domain Fiction (Pre-1929)",
        "canonical_url": "https://www.gutenberg.org/",
        "license": "Public Domain in the United States",
        "license_status": "APPROVED",
        "attribution": "Project Gutenberg Literary Archive Foundation",
        "evidence_url": "https://www.gutenberg.org/policy/license.html",
        "jurisdiction_notes": "Works published before 1929 are US public domain. Authors (Shelley, Austen, Dickens, Stoker, Doyle) verified against EU/UK life+70 rules; all authors deceased >70 years. Standard Gutenberg license headers/footers stripped."
    },
    "technical_docs": {
        "domain_name": "Technical Documentation",
        "source_name": "IETF RFC Standards & Python Official Documentation",
        "canonical_url": "https://www.rfc-editor.org/ / https://github.com/python/cpython/tree/main/Doc",
        "license": "IETF Trust Legal Provisions (RFC 5378) / PSF License v2",
        "license_status": "APPROVED",
        "attribution": "IETF Trust / Python Software Foundation",
        "evidence_url": "https://trustee.ietf.org/license-info / https://docs.python.org/3/license.html",
        "terms_notes": "Standard specifications and open-source documentation. ASCII bit diagrams filtered."
    },
    "scientific": {
        "domain_name": "Scientific Writing",
        "source_name": "PubMed Central (PMC) Open Access Verified CC-BY Subset",
        "canonical_url": "https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/",
        "license": "CC-BY 4.0 International",
        "license_status": "APPROVED",
        "attribution": "National Library of Medicine (NLM) / Original Article Authors / PLoS",
        "evidence_url": "https://creativecommons.org/licenses/by/4.0/",
        "terms_notes": "Machine-readable XML <license_ref> parsed directly from article JATS XML. All articles verified CC-BY 4.0."
    },
    "conversation": {
        "domain_name": "Conversational Dialogue",
        "source_name": "Public Domain Dramatic Dialogue Plays (Wilde, Shaw)",
        "canonical_url": "https://www.gutenberg.org/cache/epub/844/ / https://www.gutenberg.org/cache/epub/3825/",
        "license": "Public Domain in the United States",
        "license_status": "APPROVED",
        "attribution": "Project Gutenberg Literary Archive Foundation",
        "evidence_url": "https://www.gutenberg.org/policy/license.html",
        "jurisdiction_notes": "Pre-1929 published dramatic works. Wilde (d. 1900, life+126y) and Shaw (Pygmalion 1913, Shaw d. 1950, life+75y) verified. Multi-turn theatrical conversations with distinct speaker turns."
    },
    "news": {
        "domain_name": "News / Journalism",
        "source_name": "Reuters-21578 Financial & Macroeconomic Journalism Archive",
        "canonical_url": "https://kdd.ics.uci.edu/databases/reuters21578/reuters21578.html",
        "license": "Open Academic & Research Use (Reuters Ltd. / Carnegie Group)",
        "license_status": "CONDITIONAL",
        "attribution": "David D. Lewis et al. / Reuters Ltd.",
        "evidence_url": "https://kdd.ics.uci.edu/databases/reuters21578/README.txt",
        "terms_notes": "UCI repository records that Reuters Ltd. and Carnegie Group made articles freely available for text categorization research. NLTK package metadata restricts use to research. Labeled CONDITIONAL: permitted for non-commercial research, benchmarking, and language model adaptation; commercial redistribution of the raw text corpus is prohibited."
    }
}


def http_get(url: str, timeout: int = 25, retries: int = 2) -> str:
    """Perform robust HTTP GET request with retry handling and custom User-Agent."""
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) RealDataPilot/1.0"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                content = resp.read()
                return content.decode("utf-8", errors="replace")
        except Exception as e:
            last_err = e
            time.sleep(1.0)
    raise last_err if last_err is not None else RuntimeError(f"Failed to fetch {url}")


def http_get_bytes(url: str, timeout: int = 35, retries: int = 2) -> bytes:
    """Perform robust HTTP GET request returning raw bytes with retry handling."""
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) RealDataPilot/1.0"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as e:
            last_err = e
            time.sleep(1.0)
    raise last_err if last_err is not None else RuntimeError(f"Failed to fetch bytes {url}")


def strip_gutenberg_envelope(text: str) -> str:
    """Strip standard Project Gutenberg header and footer boilerplates."""
    start_markers = [
        r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK[^\*]*\*\*\*",
        r"\*\*\* START OF THIS PROJECT GUTENBERG EBOOK[^\*]*\*\*\*",
    ]
    end_markers = [
        r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK[^\*]*\*\*\*",
        r"\*\*\* END OF THIS PROJECT GUTENBERG EBOOK[^\*]*\*\*\*",
    ]

    for sm in start_markers:
        m = re.search(sm, text, re.IGNORECASE)
        if m:
            text = text[m.end():]
            break

    for em in end_markers:
        m = re.search(em, text, re.IGNORECASE)
        if m:
            text = text[:m.start()]
            break

    return text.strip()


def normalize_license_url(url: str) -> str:
    """Normalize Creative Commons URL by stripping queries/fragments/trailing spaces."""
    return re.sub(r"[?#].*$", "", url.strip())


def verify_pmc_license_metadata(
    license_refs: List[str],
    license_text: str = ""
) -> Tuple[bool, Optional[str], str]:
    """Conservatively verify whether PMC metadata specifies an exact CC-BY 4.0 license.

    Requirements:
    1. Must contain at least one machine-readable <license_ref> (or equivalent URI).
    2. The machine-readable URI must resolve strictly to Creative Commons Attribution 4.0 International (CC-BY 4.0).
    3. Incompatible variants (CC-BY-NC, CC-BY-SA, CC-BY-ND, CC-BY 3.0/2.0) are strictly rejected.
    4. Substrings like '/licenses/by' are NOT sufficient.
    5. Generic prose like 'Creative Commons Attribution' without valid machine-readable URI is rejected.
    6. Multiple references: If any reference is incompatible or conflicting, reject due to ambiguity.
       All non-empty references must consistently resolve to CC-BY 4.0.

    Returns:
        (is_valid, verified_uri, decision_code)
    """
    if not license_refs:
        if license_text and "attribution" in license_text.lower():
            return (False, None, "REJECTED_GENERIC_PROSE_WITHOUT_MACHINE_READABLE_REF")
        return (False, None, "REJECTED_MISSING_MACHINE_READABLE_REF")

    # Regular expressions for strict CC-BY 4.0 vs variants
    cc_by_4_pattern = re.compile(
        r"^https?://(?:www\.)?creativecommons\.org/licenses/by/4\.0(?:/(?:legalcode|deed(?:\.[a-z]{2})?)?)?/?$",
        re.IGNORECASE
    )
    nc_pattern = re.compile(r"licenses/by-nc", re.IGNORECASE)
    sa_pattern = re.compile(r"licenses/by-(?:[a-z]+-)?sa", re.IGNORECASE)
    nd_pattern = re.compile(r"licenses/by-(?:[a-z]+-)?nd", re.IGNORECASE)
    older_ver_pattern = re.compile(r"licenses/by/(?:1\.0|2\.0|2\.5|3\.0)", re.IGNORECASE)

    clean_refs = [normalize_license_url(ref) for ref in license_refs if ref and ref.strip()]
    if not clean_refs:
        return (False, None, "REJECTED_MISSING_MACHINE_READABLE_REF")

    verified_uri = None
    for ref in clean_refs:
        if nc_pattern.search(ref):
            return (False, None, "REJECTED_NON_COMMERCIAL_VARIANT")
        if sa_pattern.search(ref):
            return (False, None, "REJECTED_SHARE_ALIKE_VARIANT")
        if nd_pattern.search(ref):
            return (False, None, "REJECTED_NO_DERIVATIVES_VARIANT")
        if older_ver_pattern.search(ref):
            return (False, None, "REJECTED_INCOMPATIBLE_VERSION_NOT_4_0")

        if cc_by_4_pattern.match(ref):
            if verified_uri is None:
                verified_uri = ref
        else:
            return (False, None, f"REJECTED_INCOMPATIBLE_OR_UNKNOWN_LICENSE_REF: {ref}")

    if verified_uri is not None:
        return (True, verified_uri, "VERIFIED_CC_BY_4_0")

    return (False, None, "REJECTED_UNVERIFIED_LICENSE_TERMS")


def extract_pmc_xml_data(xml_content: str) -> Tuple[str, List[str], str]:
    """Extract abstract/body prose and machine-readable license info from PMC XML.

    Returns:
        prose: Clean extracted paragraphs
        license_refs: List of license URLs extracted from <license_ref> and <license> attributes
        license_text: Description from <license-p>
    """
    try:
        root = ET.fromstring(xml_content)
    except Exception:
        return "", [], ""

    # Parse license info
    license_refs: List[str] = []
    for elem in root.iter():
        if elem.tag.endswith("license_ref") and elem.text and elem.text.strip():
            ref = elem.text.strip()
            if ref not in license_refs:
                license_refs.append(ref)
        if elem.tag.endswith("license") or elem.tag.endswith("license_ref"):
            for k, v in elem.attrib.items():
                if ("href" in k.lower() or "license_ref" in k.lower()) and v and v.strip():
                    if "creativecommons.org" in v.lower() and v.strip() not in license_refs:
                        license_refs.append(v.strip())

    lic_p_elements = root.findall(".//license-p")
    license_text = " ".join("".join(p.itertext()).strip() for p in lic_p_elements) if lic_p_elements else ""

    paragraphs = []
    # Extract abstract paragraphs
    for abs_elem in root.findall(".//abstract"):
        for p in abs_elem.findall(".//p"):
            text = "".join(p.itertext()).strip()
            if text:
                paragraphs.append(text)

    # Extract body paragraphs
    for body_elem in root.findall(".//body"):
        for p in body_elem.findall(".//p"):
            text = "".join(p.itertext()).strip()
            if text:
                paragraphs.append(text)

    prose = "\n\n".join(paragraphs)
    return prose, license_refs, license_text



def fetch_raw_pilot_documents() -> List[Dict[str, Any]]:
    """Fetch real-world documents across all 6 verified domains with explicit parent_source_id."""
    print("=" * 75)
    print("INGESTING CANDIDATE DOCUMENTS WITH EXPLICIT PARENT-SOURCE GROUPING")
    print("=" * 75)

    raw_docs: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # 1. WikiText-103 Training Split (Expository Replay) - 15 Parent Articles
    # -------------------------------------------------------------------------
    print("[1/6] Ingesting WikiText-103 Training Split (15 Independent Parent Articles)...")
    wikitext_path = project_root / "backend/datasets/raw/wikitext-103.txt"
    if wikitext_path.exists():
        articles_collected = 0
        current_article: List[str] = []
        with open(wikitext_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith(" = ") and not line.startswith(" = = ") and current_article:
                    art_text = "".join(current_article).strip()
                    if len(art_text) >= 1500:
                        parent_id = f"wiki_art_{articles_collected + 1}"
                        header_title = current_article[0].strip(" = \n")
                        # Bound article to ~12000 chars (~2500 tokens)
                        raw_docs.append({
                            "domain": "wikitext103_replay",
                            "parent_source_id": parent_id,
                            "doc_id": f"{parent_id}_seg1",
                            "source_title": f"Wikipedia Article: {header_title}",
                            "author": "Wikipedia Contributors",
                            "source_name": DOMAIN_METADATA["wikitext103_replay"]["source_name"],
                            "canonical_url": DOMAIN_METADATA["wikitext103_replay"]["canonical_url"],
                            "license": DOMAIN_METADATA["wikitext103_replay"]["license"],
                            "license_evidence_url": DOMAIN_METADATA["wikitext103_replay"]["evidence_url"],
                            "license_status": DOMAIN_METADATA["wikitext103_replay"]["license_status"],
                            "attribution": DOMAIN_METADATA["wikitext103_replay"]["attribution"],
                            "jurisdiction_notes": DOMAIN_METADATA["wikitext103_replay"]["terms_notes"],
                            "raw_text": art_text[:12000]
                        })
                        articles_collected += 1
                        if articles_collected >= 15:
                            break
                    current_article = [line]
                else:
                    current_article.append(line)
        print(f"  -> Ingested {articles_collected} independent WikiText-103 parent articles.")
    else:
        print("  -> Warning: wikitext-103.txt not found!")

    # -------------------------------------------------------------------------
    # 2. Literature (Project Gutenberg US Public Domain) - 5 Independent Books
    # -------------------------------------------------------------------------
    print("[2/6] Ingesting Project Gutenberg Prose (5 Independent Parent Books)...")
    gutenberg_books = [
        ("Frankenstein", "Mary Wollstonecraft Shelley", "1818", "d. 1851 (life+175y)", "https://www.gutenberg.org/cache/epub/84/pg84.txt", "gutenberg_pg84"),
        ("Pride and Prejudice", "Jane Austen", "1813", "d. 1817 (life+209y)", "https://www.gutenberg.org/cache/epub/1342/pg1342.txt", "gutenberg_pg1342"),
        ("The Adventures of Sherlock Holmes", "Arthur Conan Doyle", "1892", "d. 1930 (life+96y)", "https://www.gutenberg.org/cache/epub/1661/pg1661.txt", "gutenberg_pg1661"),
        ("A Tale of Two Cities", "Charles Dickens", "1859", "d. 1870 (life+156y)", "https://www.gutenberg.org/cache/epub/98/pg98.txt", "gutenberg_pg98"),
        ("Dracula", "Bram Stoker", "1897", "d. 1912 (life+114y)", "https://www.gutenberg.org/cache/epub/345/pg345.txt", "gutenberg_pg345"),
    ]
    for title, author, pub_yr, death_info, url, parent_id in gutenberg_books:
        try:
            raw_text = http_get(url)
            clean_book = strip_gutenberg_envelope(raw_text)
            chapters = re.split(r"\n\s*(?:CHAPTER|Chapter|LETTER|Letter|STORY|Story)\s+[IVXLCDM0-9]+", clean_book)
            ch_count = 0
            for idx, ch in enumerate(chapters):
                ch_text = ch.strip()
                if len(ch_text) >= 2000:
                    raw_docs.append({
                        "domain": "literature",
                        "parent_source_id": parent_id,
                        "doc_id": f"{parent_id}_ch{idx + 1}",
                        "source_title": f"{title} (Pub. {pub_yr})",
                        "author": author,
                        "source_name": f"Project Gutenberg: {title} by {author}",
                        "canonical_url": url,
                        "license": "Public Domain in the United States (Pre-1929)",
                        "license_evidence_url": "https://www.gutenberg.org/policy/license.html",
                        "license_status": "APPROVED",
                        "attribution": DOMAIN_METADATA["literature"]["attribution"],
                        "jurisdiction_notes": f"Published {pub_yr} in US public domain. Author {death_info}; public domain globally under EU/UK life+70 rules. PG header/footer stripped.",
                        "raw_text": ch_text[:12000]
                    })
                    ch_count += 1
                    if ch_count >= 3:  # 3 chapters per parent book
                        break
            print(f"  -> Ingested {ch_count} chapters from parent work '{title}' ({parent_id}).")
        except Exception as e:
            print(f"  -> Error fetching {title}: {e}")

    # -------------------------------------------------------------------------
    # 3. Technical Documentation (6 Independent Parent Documents/RFCs)
    # -------------------------------------------------------------------------
    print("[3/6] Ingesting Technical Documentation (6 Independent Parent Works)...")
    tech_sources = [
        ("IETF RFC 8446: The Transport Layer Security (TLS) Protocol 1.3", "IETF Trust", "https://www.rfc-editor.org/rfc/rfc8446.txt", "ietf_rfc8446", "IETF Trust Provisions (RFC 5378)", "https://trustee.ietf.org/license-info"),
        ("IETF RFC 7230: Hypertext Transfer Protocol (HTTP/1.1): Message Syntax", "IETF Trust", "https://www.rfc-editor.org/rfc/rfc7230.txt", "ietf_rfc7230", "IETF Trust Provisions (RFC 5378)", "https://trustee.ietf.org/license-info"),
        ("IETF RFC 7540: Hypertext Transfer Protocol Version 2 (HTTP/2)", "IETF Trust", "https://www.rfc-editor.org/rfc/rfc7540.txt", "ietf_rfc7540", "IETF Trust Provisions (RFC 5378)", "https://trustee.ietf.org/license-info"),
        ("Python 3 Documentation: Data Structures Tutorial", "Python Software Foundation", "https://raw.githubusercontent.com/python/cpython/main/Doc/tutorial/datastructures.rst", "cpython_datastruct", "PSF License v2", "https://docs.python.org/3/license.html"),
        ("Python 3 Documentation: Modules Tutorial", "Python Software Foundation", "https://raw.githubusercontent.com/python/cpython/main/Doc/tutorial/modules.rst", "cpython_modules", "PSF License v2", "https://docs.python.org/3/license.html"),
        ("Python 3 Documentation: Errors and Exceptions Tutorial", "Python Software Foundation", "https://raw.githubusercontent.com/python/cpython/main/Doc/tutorial/errors.rst", "cpython_errors", "PSF License v2", "https://docs.python.org/3/license.html"),
    ]
    for title, author, url, parent_id, lic_str, evid_url in tech_sources:
        try:
            content = http_get(url)
            sections = re.split(r"\n(?=[0-9]+\.[0-9]*\s+[A-Z])", content)
            if len(sections) < 3:
                sections = re.split(r"\n\n\n+", content)
            sec_count = 0
            for idx, sec in enumerate(sections):
                sec_text = sec.strip()
                # Discard ASCII packet diagrams and short headers
                if len(sec_text) >= 1500 and not sec_text.count("+-+") > 5:
                    raw_docs.append({
                        "domain": "technical_docs",
                        "parent_source_id": parent_id,
                        "doc_id": f"{parent_id}_sec{idx + 1}",
                        "source_title": title,
                        "author": author,
                        "source_name": title,
                        "canonical_url": url,
                        "license": lic_str,
                        "license_evidence_url": evid_url,
                        "license_status": "APPROVED",
                        "attribution": DOMAIN_METADATA["technical_docs"]["attribution"],
                        "jurisdiction_notes": "Open standards / permissive open source documentation. Packet bit diagrams filtered.",
                        "raw_text": sec_text[:12000]
                    })
                    sec_count += 1
                    if sec_count >= 3:  # 3 sections per parent doc
                        break
            print(f"  -> Ingested {sec_count} sections from parent work '{title}' ({parent_id}).")
        except Exception as e:
            print(f"  -> Error fetching {title}: {e}")

    # -------------------------------------------------------------------------
    # 4. Scientific Writing (6 Independent PMC Open Access Articles - CC-BY Verified)
    # -------------------------------------------------------------------------
    print("[4/6] Ingesting PubMed Central Open Access Papers (6 CC-BY Verified Articles)...")
    pmc_candidates = [
        ("PMC7095448", "Nature / Springer Open", "SARS-CoV-2 and coronavirus structural biology review"),
        ("PMC13660980", "PLoS ONE", "Genomic diversity and molecular epidemiology study"),
        ("PMC13660983", "PLoS ONE", "Cellular biomechanics and fluorescence microscopy analysis"),
        ("PMC13660993", "PLoS ONE", "High-throughput computational protein screening"),
        ("PMC13660882", "PLoS ONE", "Macromolecular dynamics and crystallographic analysis"),
        ("PMC13660988", "PLoS ONE", "Evolutionary genetics and phylogenetic tree reconstruction")
    ]
    for pmcid, journal, topic in pmc_candidates:
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmcid}&retmode=xml"
        try:
            xml_text = http_get(url)
            prose, lic_refs, lic_text = extract_pmc_xml_data(xml_text)

            # Strict, exact machine-readable license verification
            is_valid, verified_uri, decision_code = verify_pmc_license_metadata(lic_refs, lic_text)
            if not is_valid or verified_uri is None:
                print(f"  -> REJECTED {pmcid}: {decision_code} (refs={lic_refs})")
                continue

            evidence_url = verified_uri
            paragraphs = prose.split("\n\n")
            chunk: List[str] = []
            chunk_len = 0
            sec_idx = 1
            for p in paragraphs:
                chunk.append(p)
                chunk_len += len(p)
                if chunk_len >= 3000:
                    doc_body = "\n\n".join(chunk).strip()
                    parent_id = f"pmc_{pmcid.lower()}"
                    raw_docs.append({
                        "domain": "scientific",
                        "parent_source_id": parent_id,
                        "doc_id": f"{parent_id}_part{sec_idx}",
                        "source_title": f"{pmcid}: {topic} ({journal})",
                        "author": f"{journal} Contributing Authors",
                        "source_name": f"PubMed Central Open Access: {pmcid}",
                        "canonical_url": url,
                        "license": "Creative Commons Attribution 4.0 International (CC-BY 4.0)",
                        "license_evidence_url": evidence_url,
                        "license_status": "APPROVED",
                        "attribution": f"National Library of Medicine (NLM) / {journal} Authors",
                        "jurisdiction_notes": f"Machine-readable XML license confirmed via <license_ref>: {evidence_url}. Decision: {decision_code}. Reference lists and XML tags stripped.",
                        "raw_text": doc_body[:12000]
                    })
                    sec_idx += 1
                    chunk = []
                    chunk_len = 0
                    if sec_idx > 3:  # 3 segments per PMC article
                        break
            print(f"  -> Ingested {sec_idx - 1} sections from parent work '{pmcid}' (CC-BY 4.0 Verified).")
        except Exception as e:
            print(f"  -> Error fetching {pmcid}: {e}")

    # -------------------------------------------------------------------------
    # 5. Conversational Dialogue (4 Independent Dramatic Plays)
    # -------------------------------------------------------------------------
    print("[5/6] Ingesting Conversational Dialogue (4 Independent Dramatic Works)...")
    dialogue_plays = [
        ("The Importance of Being Earnest", "Oscar Wilde", "1895", "d. 1900 (life+126y)", "https://www.gutenberg.org/cache/epub/844/pg844.txt", "gutenberg_pg844"),
        ("Pygmalion", "George Bernard Shaw", "1913", "d. 1950 (life+76y)", "https://www.gutenberg.org/cache/epub/3825/pg3825.txt", "gutenberg_pg3825"),
        ("Lady Windermere's Fan", "Oscar Wilde", "1892", "d. 1900 (life+126y)", "https://www.gutenberg.org/cache/epub/790/pg790.txt", "gutenberg_pg790"),
        ("An Ideal Husband", "Oscar Wilde", "1895", "d. 1900 (life+126y)", "https://www.gutenberg.org/cache/epub/882/pg882.txt", "gutenberg_pg882")
    ]
    for title, author, pub_yr, death_info, url, parent_id in dialogue_plays:
        try:
            play_text = http_get(url)
            clean_play = strip_gutenberg_envelope(play_text)
            scenes = re.split(r"\n\s*(?:ACT|Act|SCENE|Scene)\s+[IVXLCDM0-9]+", clean_play)
            bouts_collected = 0
            for idx, scene in enumerate(scenes):
                scene_text = scene.strip()
                if len(scene_text) >= 2000:
                    chunks = [scene_text[i:i+4000] for i in range(0, len(scene_text), 4000)]
                    for c_idx, chunk in enumerate(chunks):
                        if len(chunk) >= 1500:
                            raw_docs.append({
                                "domain": "conversation",
                                "parent_source_id": parent_id,
                                "doc_id": f"{parent_id}_s{idx+1}_c{c_idx+1}",
                                "source_title": f"{title} (Pub. {pub_yr})",
                                "author": author,
                                "source_name": f"{title} by {author}",
                                "canonical_url": url,
                                "license": "Public Domain in the United States (Pre-1929)",
                                "license_evidence_url": "https://www.gutenberg.org/policy/license.html",
                                "license_status": "APPROVED",
                                "attribution": DOMAIN_METADATA["conversation"]["attribution"],
                                "jurisdiction_notes": f"Pre-1929 dramatic work. Author {death_info}; public domain in US and EU/UK. Formatted into multi-turn character dialogues.",
                                "raw_text": chunk[:12000]
                            })
                            bouts_collected += 1
                            if bouts_collected >= 3:  # 3 dialogue bouts per play
                                break
                    if bouts_collected >= 3:
                        break
            print(f"  -> Ingested {bouts_collected} dialogue bouts from parent work '{title}' ({parent_id}).")
        except Exception as e:
            print(f"  -> Error fetching {title}: {e}")

    # -------------------------------------------------------------------------
    # 6. News / Journalism (Reuters-21578 Training Split) - 50 Independent Articles
    # -------------------------------------------------------------------------
    print("[6/6] Ingesting News (50 Independent Reuters-21578 Story Articles)...")
    reuters_url = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/reuters.zip"
    try:
        zip_bytes = http_get_bytes(reuters_url)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            namelist = [n for n in z.namelist() if n.startswith("reuters/training/") and not n.endswith("/")]
            namelist.sort()
            news_count = 0
            for name in namelist[:120]:
                art_bytes = z.read(name)
                art_text = art_bytes.decode("latin-1", errors="replace").strip()
                if len(art_text) >= 400:
                    art_id = Path(name).name
                    parent_id = f"reuters_{art_id}"
                    raw_docs.append({
                        "domain": "news",
                        "parent_source_id": parent_id,
                        "doc_id": parent_id,
                        "source_title": f"Reuters News Dispatch #{art_id}",
                        "author": "Reuters Ltd. Financial Wire",
                        "source_name": "Reuters-21578 Financial News Archive",
                        "canonical_url": reuters_url,
                        "license": "Open Academic & Research Use (Reuters Ltd. / Carnegie Group)",
                        "license_evidence_url": "https://kdd.ics.uci.edu/databases/reuters21578/README.txt",
                        "license_status": "CONDITIONAL",
                        "attribution": DOMAIN_METADATA["news"]["attribution"],
                        "jurisdiction_notes": "UCI repository README records that Reuters Ltd. made articles freely available for text categorization research. NLTK package metadata restricts use to research. Labeled CONDITIONAL: permitted for non-commercial research and benchmarking; commercial redistribution prohibited.",
                        "raw_text": art_text[:12000]
                    })
                    news_count += 1
                    if news_count >= 50:
                        break
            print(f"  -> Ingested {news_count} independent Reuters news parent articles.")
    except Exception as e:
        print(f"  -> Error fetching Reuters: {e}")

    # -------------------------------------------------------------------------
    # Intentional Fixture Injections for Deduplication and Filter Auditing
    # -------------------------------------------------------------------------
    if raw_docs:
        print("\nInjecting test fixtures for deduplication and quality filter verification...")
        # Exact duplicate of doc 0
        raw_docs.append({
            "domain": raw_docs[0]["domain"],
            "parent_source_id": raw_docs[0]["parent_source_id"],
            "doc_id": "test_duplicate_doc_1",
            "source_title": "Intentionally Duplicated Ingestion Fixture #1",
            "author": "Test Fixture",
            "source_name": "Test Duplicate",
            "canonical_url": raw_docs[0]["canonical_url"],
            "license": raw_docs[0]["license"],
            "license_evidence_url": raw_docs[0]["license_evidence_url"],
            "license_status": raw_docs[0]["license_status"],
            "attribution": raw_docs[0]["attribution"],
            "jurisdiction_notes": "Test fixture",
            "raw_text": raw_docs[0]["raw_text"]
        })
        # Exact duplicate of doc 10 with superficial carriage return alterations
        raw_docs.append({
            "domain": raw_docs[10]["domain"],
            "parent_source_id": raw_docs[10]["parent_source_id"],
            "doc_id": "test_duplicate_doc_2",
            "source_title": "Intentionally Duplicated Ingestion Fixture #2",
            "author": "Test Fixture",
            "source_name": "Test Duplicate",
            "canonical_url": raw_docs[10]["canonical_url"],
            "license": raw_docs[10]["license"],
            "license_evidence_url": raw_docs[10]["license_evidence_url"],
            "license_status": raw_docs[10]["license_status"],
            "attribution": raw_docs[10]["attribution"],
            "jurisdiction_notes": "Test fixture",
            "raw_text": raw_docs[10]["raw_text"].replace("\n", "\r\n") + "   "
        })
        # Corrupted / non-printable document (should be rejected by filter)
        raw_docs.append({
            "domain": "news",
            "parent_source_id": "test_corrupt_parent",
            "doc_id": "test_corrupt_doc_rejected",
            "source_title": "Intentionally Corrupted Test Fixture",
            "author": "Test Fixture",
            "source_name": "Test Corrupt",
            "canonical_url": "test://corrupt",
            "license": "Test",
            "license_evidence_url": "test://corrupt",
            "license_status": "EXCLUDED",
            "attribution": "Test",
            "jurisdiction_notes": "Test fixture",
            "raw_text": "\x00\x01\x02\x03\x04\x05\x06\x07 Corrupt binary noise \x08\x0b\x0c\x0e\x0f"
        })

    print(f"Total raw document records assembled: {len(raw_docs)}")
    return raw_docs


def split_parent_sources_stratified(
    documents: List[Dict[str, Any]],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Dict[str, str]:
    """Assign entire parent sources to train, val, or test stratified by domain.

    Guarantees:
    1. Every parent source is assigned to exactly ONE split.
    2. All chunks/segments of a parent source inherit that parent's split.
    3. Every domain receives at least 1 parent source in train, 1 in val, and 1 in test
       (provided the domain has >= 3 parent sources).
    """
    rng = random.Random(seed)
    parent_to_split: Dict[str, str] = {}

    # Group parent sources by domain
    domain_to_parents: Dict[str, List[str]] = {}
    for d in documents:
        dom = d["domain"]
        p_id = d["parent_source_id"]
        if dom not in domain_to_parents:
            domain_to_parents[dom] = []
        if p_id not in domain_to_parents[dom]:
            domain_to_parents[dom].append(p_id)

    print("\nParent-Source Stratified Splitting:")
    for dom, parents in sorted(domain_to_parents.items()):
        # Deterministically sort then shuffle parents
        parents_sorted = sorted(parents)
        rng.shuffle(parents_sorted)
        n = len(parents_sorted)

        if n >= 5:
            n_val = max(1, int(round(n * val_ratio)))
            n_test = max(1, int(round(n * test_ratio)))
            n_train = n - n_val - n_test
            # Ensure at least 1 in each
            train_p = parents_sorted[:n_train]
            val_p = parents_sorted[n_train : n_train + n_val]
            test_p = parents_sorted[n_train + n_val :]
        elif n == 4:
            # 2 train, 1 val, 1 test
            train_p = parents_sorted[:2]
            val_p = [parents_sorted[2]]
            test_p = [parents_sorted[3]]
        elif n == 3:
            # 1 train, 1 val, 1 test
            train_p = [parents_sorted[0]]
            val_p = [parents_sorted[1]]
            test_p = [parents_sorted[2]]
        else:
            train_p = parents_sorted
            val_p = []
            test_p = []

        for p in train_p: parent_to_split[p] = "train"
        for p in val_p: parent_to_split[p] = "val"
        for p in test_p: parent_to_split[p] = "test"

        print(f"  [{dom:<20}] Total Parents: {n:<2} -> Train: {len(train_p)}, Val: {len(val_p)}, Test: {len(test_p)}")

    return parent_to_split


def run_real_data_pilot() -> Dict[str, Any]:
    """Execute complete Phase 3B real-data pilot validation with parent-source separation."""
    start_time = time.time()
    print("=" * 75)
    print("STARTING PHASE 3B REAL-DATA PILOT VALIDATION FOR MULTI-DOMAIN LSTM")
    print("=" * 75)

    pilot_dir = project_root / "experiments/multidomain_v1/pilot"
    processed_dir = pilot_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = pilot_dir / "pilot_manifest.jsonl"
    metadata_path = processed_dir / "metadata.json"

    # Step 1: Fetch raw documents
    raw_records = fetch_raw_pilot_documents()
    assert len(raw_records) > 0, "Failed to ingest raw pilot documents!"

    # Step 2: Unicode normalization, quality filtering & deduplication
    print("\n" + "=" * 75)
    print("STEP 2: UNICODE NORMALIZATION, QUALITY FILTERING & EXACT DEDUPLICATION")
    print("=" * 75)

    seen_hashes: Set[str] = set()
    manifest_entries: List[Dict[str, Any]] = []
    unique_docs: List[Dict[str, Any]] = []

    rejected_count = 0
    duplicate_count = 0

    for rec in raw_records:
        raw_text = rec["raw_text"]
        raw_hash = hashlib.sha256(raw_text.encode("utf-8", errors="replace")).hexdigest()
        raw_len = len(raw_text)

        # Unicode NFKC normalization and whitespace standardization
        norm_text = unicodedata.normalize("NFKC", raw_text)
        cleaned = normalize_document(norm_text)
        clean_len = len(cleaned)
        clean_hash = compute_doc_hash(cleaned)

        base_manifest_entry = {
            "doc_id": rec["doc_id"],
            "parent_source_id": rec["parent_source_id"],
            "domain": rec["domain"],
            "source_title": rec["source_title"],
            "author": rec["author"],
            "source_name": rec["source_name"],
            "canonical_url": rec["canonical_url"],
            "license": rec["license"],
            "license_evidence_url": rec["license_evidence_url"],
            "license_status": rec["license_status"],
            "attribution": rec["attribution"],
            "jurisdiction_notes": rec["jurisdiction_notes"],
            "raw_sha256": raw_hash,
            "clean_sha256": clean_hash,
            "raw_char_len": raw_len,
            "clean_char_len": clean_len,
            "token_count": 0,
            "oov_token_count": 0,
            "oov_rate_pct": 0.0,
        }

        # Quality filtering (min 100 characters, high printable ratio)
        if not filter_document(cleaned, min_chars=100, max_chars=5_000_000, min_printable_ratio=0.95):
            entry = dict(base_manifest_entry)
            entry["assigned_split"] = "REJECTED"
            entry["filter_decision"] = "REJECTED_QUALITY_FILTER"
            manifest_entries.append(entry)
            rejected_count += 1
            print(f"  [REJECTED] {rec['doc_id']} failed quality filter (len={clean_len})")
            continue

        # Exact SHA-256 deduplication
        if clean_hash in seen_hashes:
            entry = dict(base_manifest_entry)
            entry["assigned_split"] = "DEDUPLICATED"
            entry["filter_decision"] = "REJECTED_EXACT_DUPLICATE"
            manifest_entries.append(entry)
            duplicate_count += 1
            print(f"  [DEDUPLICATED] {rec['doc_id']} matched existing SHA-256: {clean_hash[:12]}...")
            continue

        seen_hashes.add(clean_hash)
        rec_copy = dict(rec)
        rec_copy["raw_sha256"] = raw_hash
        rec_copy["clean_sha256"] = clean_hash
        rec_copy["raw_char_len"] = raw_len
        rec_copy["clean_char_len"] = clean_len
        rec_copy["cleaned_text"] = cleaned
        unique_docs.append(rec_copy)

    print(f"\nDeduplication & Quality Filter Summary:")
    print(f"  Raw Ingested: {len(raw_records)}")
    print(f"  Rejected Quality Filter: {rejected_count}")
    print(f"  Duplicates Purged: {duplicate_count}")
    print(f"  Unique Documents Retained: {len(unique_docs)}")
    assert duplicate_count >= 2, f"Expected at least 2 duplicates purged, got {duplicate_count}"
    assert rejected_count >= 1, f"Expected at least 1 rejected corrupt doc, got {rejected_count}"

    # Step 3: Stratified Parent-Source Level Splitting
    print("\n" + "=" * 75)
    print("STEP 3: STRATIFIED PARENT-SOURCE LEVEL SPLITTING (ZERO PARENT OVERLAP)")
    print("=" * 75)

    parent_to_split = split_parent_sources_stratified(
        unique_docs, train_ratio=0.80, val_ratio=0.10, test_ratio=0.10, seed=42
    )

    # Assign split to all unique documents based on parent_source_id
    for d in unique_docs:
        p_id = d["parent_source_id"]
        d["assigned_split"] = parent_to_split[p_id]

    # Verify zero parent-source leakage across splits
    train_parents = {d["parent_source_id"] for d in unique_docs if d["assigned_split"] == "train"}
    val_parents = {d["parent_source_id"] for d in unique_docs if d["assigned_split"] == "val"}
    test_parents = {d["parent_source_id"] for d in unique_docs if d["assigned_split"] == "test"}

    p_tv_leak = train_parents.intersection(val_parents)
    p_tt_leak = train_parents.intersection(test_parents)
    p_vt_leak = val_parents.intersection(test_parents)

    assert len(p_tv_leak) == 0, f"PARENT-SOURCE LEAKAGE! Train/Val overlap: {p_tv_leak}"
    assert len(p_tt_leak) == 0, f"PARENT-SOURCE LEAKAGE! Train/Test overlap: {p_tt_leak}"
    assert len(p_vt_leak) == 0, f"PARENT-SOURCE LEAKAGE! Val/Test overlap: {p_vt_leak}"
    print(f"Parent Source Counts: Train={len(train_parents)}, Val={len(val_parents)}, Test={len(test_parents)}")
    print("  -> PASSED: Strict zero parent-source overlap verified across all partitions.")

    # Verify zero clean_hash leakage across splits
    train_hashes = {d["clean_sha256"] for d in unique_docs if d["assigned_split"] == "train"}
    val_hashes = {d["clean_sha256"] for d in unique_docs if d["assigned_split"] == "val"}
    test_hashes = {d["clean_sha256"] for d in unique_docs if d["assigned_split"] == "test"}

    assert len(train_hashes.intersection(val_hashes)) == 0, "Hash leakage between train and val!"
    assert len(train_hashes.intersection(test_hashes)) == 0, "Hash leakage between train and test!"
    assert len(val_hashes.intersection(test_hashes)) == 0, "Hash leakage between val and test!"
    print("  -> PASSED: Strict zero-leakage hash disjointness verified across all partitions.")

    # Step 4: Vocabulary Compatibility & OOV Analysis
    print("\n" + "=" * 75)
    print("STEP 4: VOCABULARY COMPATIBILITY & MULTI-DOMAIN OOV RATE AUDIT")
    print("=" * 75)

    vocab_path = project_root / "backend/model_artifacts/tokenizer/vocab.json"
    assert vocab_path.exists(), f"Missing production vocabulary at {vocab_path}!"
    tokenizer = Tokenizer.load(vocab_path)
    assert len(tokenizer) == 30000, f"Expected 30,000 vocab, got {len(tokenizer)}"
    assert tokenizer.pad_id == 0, f"Expected <PAD>=0, got {tokenizer.pad_id}"
    assert tokenizer.unk_id == 1, f"Expected <UNK>=1, got {tokenizer.unk_id}"
    assert tokenizer.bos_id == 2, f"Expected <BOS>=2, got {tokenizer.bos_id}"
    assert tokenizer.eos_id == 3, f"Expected <EOS>=3, got {tokenizer.eos_id}"
    print(f"Loaded production vocabulary: {len(tokenizer)} tokens from {vocab_path}")
    print(f"Locked Special Tokens: <PAD>={tokenizer.pad_id}, <UNK>={tokenizer.unk_id}, <BOS>={tokenizer.bos_id}, <EOS>={tokenizer.eos_id}")

    domain_stats: Dict[str, Dict[str, Any]] = {
        dom: {
            "parents": {"train": set(), "val": set(), "test": set()},
            "docs": {"train": 0, "val": 0, "test": 0, "total": 0},
            "tokens": {"train": 0, "val": 0, "test": 0, "total": 0},
            "unk_tokens": 0
        }
        for dom in DOMAIN_METADATA.keys()
    }
    split_stats: Dict[str, Dict[str, int]] = {
        "train": {"parents": len(train_parents), "docs": 0, "tokens": 0, "unk": 0},
        "val": {"parents": len(val_parents), "docs": 0, "tokens": 0, "unk": 0},
        "test": {"parents": len(test_parents), "docs": 0, "tokens": 0, "unk": 0}
    }

    # Encode documents and track metrics
    for d in unique_docs:
        tokens = tokenizer.encode(d["cleaned_text"], add_bos=True, add_eos=True)
        assert all(0 <= tid < 30000 for tid in tokens), f"Invalid token ID in {d['doc_id']}!"

        tok_count = len(tokens)
        unk_count = sum(1 for tid in tokens if tid == tokenizer.unk_id)
        oov_rate = (unk_count / tok_count * 100) if tok_count > 0 else 0.0

        d["token_count"] = tok_count
        d["oov_token_count"] = unk_count
        d["oov_rate_pct"] = round(oov_rate, 2)
        d["token_ids"] = tokens

        dom = d["domain"]
        split = d["assigned_split"]
        p_id = d["parent_source_id"]

        domain_stats[dom]["parents"][split].add(p_id)
        domain_stats[dom]["docs"][split] += 1
        domain_stats[dom]["docs"]["total"] += 1
        domain_stats[dom]["tokens"][split] += tok_count
        domain_stats[dom]["tokens"]["total"] += tok_count
        domain_stats[dom]["unk_tokens"] += unk_count

        split_stats[split]["docs"] += 1
        split_stats[split]["tokens"] += tok_count
        split_stats[split]["unk"] += unk_count

        manifest_entries.append({
            "doc_id": d["doc_id"],
            "parent_source_id": d["parent_source_id"],
            "domain": d["domain"],
            "source_title": d["source_title"],
            "author": d["author"],
            "source_name": d["source_name"],
            "canonical_url": d["canonical_url"],
            "license": d["license"],
            "license_evidence_url": d["license_evidence_url"],
            "license_status": d["license_status"],
            "attribution": d["attribution"],
            "jurisdiction_notes": d["jurisdiction_notes"],
            "raw_sha256": d["raw_sha256"],
            "clean_sha256": d["clean_sha256"],
            "assigned_split": d["assigned_split"],
            "raw_char_len": d["raw_char_len"],
            "clean_char_len": d["clean_char_len"],
            "token_count": tok_count,
            "oov_token_count": unk_count,
            "oov_rate_pct": round(oov_rate, 2),
            "filter_decision": "RETAINED"
        })

    # Print Domain Statistics Table
    print("\nMulti-Domain Pilot Lexical & OOV Statistics (Frozen 30,000 Vocab):")
    print(f"{'Domain':<20} | {'Parents':<7} | {'Docs':<5} | {'Tokens':<8} | {'OOV%':<6} | {'Train / Val / Test Tokens':<26} | {'Parent Distribution (Tr/V/Te)'}")
    print("-" * 115)
    for dom, s in sorted(domain_stats.items()):
        total_tok = s["tokens"]["total"]
        oov = (s["unk_tokens"] / total_tok * 100) if total_tok > 0 else 0.0
        tr_tok = s["tokens"]["train"]
        val_tok = s["tokens"]["val"]
        te_tok = s["tokens"]["test"]
        p_tr = len(s["parents"]["train"])
        p_val = len(s["parents"]["val"])
        p_te = len(s["parents"]["test"])
        total_p = p_tr + p_val + p_te
        tok_str = f"{tr_tok} / {val_tok} / {te_tok}"
        p_str = f"{p_tr} / {p_val} / {p_te} ({total_p} parents)"
        print(f"{dom:<20} | {total_p:<7} | {s['docs']['total']:<5} | {total_tok:<8} | {oov:>5.2f}% | {tok_str:<26} | {p_str}")

        # Assert every domain has non-zero tokens in all three splits
        assert tr_tok > 0, f"Domain {dom} has 0 train tokens!"
        assert val_tok > 0, f"Domain {dom} has 0 val tokens!"
        assert te_tok > 0, f"Domain {dom} has 0 test tokens!"

    total_pilot_tokens = sum(s["tokens"] for s in split_stats.values())
    print("-" * 115)
    print(f"TOTAL PILOT TOKENS: {total_pilot_tokens:,} tokens across {len(unique_docs)} unique docs and {len(parent_to_split)} parent sources.")
    print(f"Pilot token limit (1,000,000) respected: {total_pilot_tokens <= 1_000_000}")
    assert total_pilot_tokens <= 1_000_000, f"Pilot exceeded 1M token limit: {total_pilot_tokens}"
    print("  -> PASSED: All 6 domains have non-zero token coverage across Train, Val, and Test splits.")

    # Step 5: Binary Serialization & Document Boundary Offsets
    print("\n" + "=" * 75)
    print("STEP 5: UINT16 BINARY SERIALIZATION & 64-BIT DOCUMENT OFFSETS")
    print("=" * 75)

    dtype, dtype_str = get_token_dtype(len(tokenizer))
    assert dtype == np.uint16, f"Expected uint16 dtype, got {dtype}"

    split_bins: Dict[str, Dict[str, Any]] = {}
    file_checksums: Dict[str, str] = {}

    for split_name in ["train", "val", "test"]:
        split_docs = [d for d in unique_docs if d["assigned_split"] == split_name]
        bin_path = processed_dir / f"{split_name}.bin"
        offsets_path = processed_dir / f"{split_name}_offsets.bin"

        tokens_written = 0
        offsets: List[int] = [0]

        with open(bin_path, "wb") as f_bin:
            for d in split_docs:
                arr = np.array(d["token_ids"], dtype=dtype)
                arr.tofile(f_bin)
                tokens_written += len(d["token_ids"])
                offsets.append(tokens_written)

        offsets_arr = np.array(offsets, dtype=np.uint64)
        with open(offsets_path, "wb") as f_off:
            offsets_arr.tofile(f_off)

        bin_hash = compute_file_sha256(bin_path)
        off_hash = compute_file_sha256(offsets_path)
        file_checksums[f"{split_name}.bin"] = bin_hash
        file_checksums[f"{split_name}_offsets.bin"] = off_hash

        split_bins[split_name] = {
            "parents": split_stats[split_name]["parents"],
            "documents": len(split_docs),
            "tokens": tokens_written,
            "bytes": bin_path.stat().st_size,
            "offsets_count": len(offsets),
            "sha256": bin_hash
        }
        print(f"  [{split_name.upper()}] Serialized {len(split_docs)} docs ({split_stats[split_name]['parents']} parents) -> {tokens_written:,} tokens ({bin_path.stat().st_size:,} bytes). SHA-256: {bin_hash[:16]}...")

    # Write metadata.json
    metadata = {
        "dataset_version": "1.1.0-pilot-parent-disjoint",
        "experiment_id": "multidomain_v1_pilot",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "random_seed": 42,
        "token_dtype": dtype_str,
        "vocab_size": len(tokenizer),
        "special_token_ids": {
            "<PAD>": tokenizer.pad_id,
            "<UNK>": tokenizer.unk_id,
            "<BOS>": tokenizer.bos_id,
            "<EOS>": tokenizer.eos_id
        },
        "split_ratios_target": {"train": 0.80, "val": 0.10, "test": 0.10},
        "parent_source_splitting": {
            "enabled": True,
            "strategy": "stratified_by_domain",
            "total_parent_sources": len(parent_to_split),
            "zero_parent_overlap_verified": True,
            "zero_hash_overlap_verified": True,
            "per_domain_split_coverage_verified": True
        },
        "statistics": {
            "raw_document_count": len(raw_records),
            "rejected_document_count": rejected_count,
            "duplicate_document_count": duplicate_count,
            "unique_document_count": len(unique_docs),
            "total_parent_sources": len(parent_to_split),
            "train": split_bins["train"],
            "val": split_bins["val"],
            "test": split_bins["test"],
            "total_tokens": total_pilot_tokens
        },
        "domain_breakdown": {
            dom: {
                "parent_sources": {
                    "train": len(s["parents"]["train"]),
                    "val": len(s["parents"]["val"]),
                    "test": len(s["parents"]["test"]),
                    "total": len(s["parents"]["train"]) + len(s["parents"]["val"]) + len(s["parents"]["test"])
                },
                "documents": s["docs"],
                "tokens": s["tokens"],
                "unk_tokens": s["unk_tokens"],
                "oov_rate_pct": round((s["unk_tokens"] / s["tokens"]["total"] * 100), 2) if s["tokens"]["total"] > 0 else 0.0,
            }
            for dom, s in sorted(domain_stats.items())
        },
        "file_checksums": file_checksums
    }

    with open(metadata_path, "w", encoding="utf-8") as f_meta:
        json.dump(metadata, f_meta, indent=2)
    print(f"Saved pilot metadata to {metadata_path}")

    # Write pilot_manifest.jsonl
    with open(manifest_path, "w", encoding="utf-8") as f_man:
        for entry in manifest_entries:
            f_man.write(json.dumps(entry) + "\n")
    print(f"Saved machine-readable pilot manifest ({len(manifest_entries)} entries) to {manifest_path}")

    # Step 6: Dataset Causal Shift & Boundary Integrity Verification
    print("\n" + "=" * 75)
    print("STEP 6: CAUSAL SHIFT, DOCUMENT-BOUNDARY ISOLATION & EXCEPTION HANDLING")
    print("=" * 75)

    train_bin = processed_dir / "train.bin"
    context_length = 35
    memmap_data = np.memmap(train_bin, dtype=np.uint16, mode="r")
    train_offsets = np.fromfile(processed_dir / "train_offsets.bin", dtype=np.uint64)
    dataset = CausalLanguageModelDataset(
        token_data=memmap_data,
        seq_len=context_length,
        stride=context_length,
        doc_offsets=train_offsets
    )

    print(f"Loaded train.bin into offset-aware CausalLanguageModelDataset ({len(dataset):,} training sequences).")
    assert len(dataset) > 0, "Empty dataset!"

    # 1. Causal Shift Check: x[t] aligns with target y[t+1]
    x_sample, y_sample = dataset[0]
    assert x_sample.shape == (context_length,), f"Expected shape ({context_length},), got {x_sample.shape}"
    assert y_sample.shape == (context_length,), f"Expected shape ({context_length},), got {y_sample.shape}"

    for t in range(context_length - 1):
        assert x_sample[t + 1].item() == y_sample[t].item(), (
            f"Causal shift mismatch at step {t}: x[t+1]={x_sample[t+1].item()} vs y[t]={y_sample[t].item()}"
        )
    print("  -> PASSED: Verified causal shifted framing: x[t] aligns with target y[t+1].")

    # 2. Boundary Framing Check: First token is <BOS>
    assert x_sample[0].item() == tokenizer.bos_id, f"Expected first token to be <BOS> (2), got {x_sample[0].item()}"
    print(f"  -> PASSED: Initial sequence begins cleanly with <BOS> token ({tokenizer.bos_id}).")

    # 3. Document Boundary Verification: sample sequences across document boundaries
    offsets = train_offsets
    for doc_idx in range(min(5, len(offsets) - 1)):
        start = int(offsets[doc_idx])
        end = int(offsets[doc_idx + 1])
        doc_slice = memmap_data[start:end]
        assert doc_slice[0] == tokenizer.bos_id, f"Doc {doc_idx} does not start with <BOS>!"
        assert doc_slice[-1] == tokenizer.eos_id, f"Doc {doc_idx} does not end with <EOS>!"
    print("  -> PASSED: Document boundary integrity confirmed with <BOS> and <EOS> markers across offsets.")

    # 4. Intra-document sequence boundary audit: verify every generated sample is strictly within its parent document
    for s_idx in range(len(dataset)):
        start_idx = int(dataset.sample_indices[s_idx])
        end_idx = start_idx + context_length + 1
        doc_idx = int(np.searchsorted(offsets, start_idx, side="right") - 1)
        doc_start = int(offsets[doc_idx])
        doc_end = int(offsets[doc_idx + 1])
        assert doc_start <= start_idx, f"Sample {s_idx} starts before document {doc_idx}"
        assert end_idx <= doc_end, f"Sample {s_idx} [start={start_idx}, end={end_idx}] crosses document boundary {doc_end}"
    print(f"  -> PASSED: Verified all {len(dataset):,} training sequences are strictly contained within document boundaries.")

    # 5. Out-of-Bounds Exception Handling
    try:
        _ = dataset[len(dataset) + 100]
        assert False, "Failed to raise IndexError on out-of-range index!"
    except IndexError:
        pass

    try:
        _ = dataset[-1]
        assert False, "Failed to raise IndexError on negative index!"
    except IndexError:
        pass

    # 6. Invalid Document Offsets Exception Handling
    try:
        _ = CausalLanguageModelDataset(memmap_data, seq_len=context_length, doc_offsets=np.array([10, 50]))
        assert False, "Failed to raise ValueError when offsets don't start at 0"
    except ValueError:
        pass

    try:
        _ = CausalLanguageModelDataset(memmap_data, seq_len=context_length, doc_offsets=np.array([0, 50, 40, len(memmap_data)]))
        assert False, "Failed to raise ValueError when offsets are non-monotonic"
    except ValueError:
        pass
    print("  -> PASSED: IndexError and ValueError properly raised for invalid accesses and offset tables.")

    # Explicitly close memory map to prevent Windows file locking
    if hasattr(memmap_data, "_mmap") and memmap_data._mmap is not None:
        memmap_data._mmap.close()
    del memmap_data
    del dataset
    gc.collect()

    # Step 7: Reproducibility Verification
    print("\n" + "=" * 75)
    print("STEP 7: BIT-FOR-BIT REPRODUCIBILITY VERIFICATION (REPEAT EXECUTION)")
    print("=" * 75)

    parent_to_split_2 = split_parent_sources_stratified(
        unique_docs, train_ratio=0.80, val_ratio=0.10, test_ratio=0.10, seed=42
    )
    assert parent_to_split == parent_to_split_2, "Non-deterministic parent split detected!"
    print("  -> PASSED: Exact bit-for-bit parent-source split reproducibility verified under seed 42.")

    # Step 8: Authoritative Artifact Reconciliation Audit
    print("\n" + "=" * 75)
    print("STEP 8: AUTHORITATIVE ARTIFACT RECONCILIATION AUDIT")
    print("=" * 75)
    from experiments.multidomain_v1.pilot.audit_pilot import audit_pilot_artifacts
    audit_results = audit_pilot_artifacts(pilot_dir)
    print(f"Authoritative Audit Status: {audit_results['status']}")
    assert audit_results["status"] == "AUDIT_PASSED"
    print("  -> PASSED: 100% mathematical reconciliation across manifest, metadata, binaries, and offsets.")

    # Storage and scaling metrics
    elapsed_time = time.time() - start_time
    total_bytes = sum(b["bytes"] for b in split_bins.values())
    total_kb = total_bytes / 1024

    print("\n" + "=" * 75)
    print("PHASE 3B REAL-DATA PILOT COMPLETED SUCCESSFULLY")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")
    print(f"Storage Used: {total_kb:.2f} KB ({total_bytes:,} bytes)")
    print("=" * 75)

    return {
        "status": "SUCCESS",
        "elapsed_seconds": round(elapsed_time, 2),
        "total_documents_ingested": len(raw_records),
        "rejected_count": rejected_count,
        "duplicate_count": duplicate_count,
        "unique_documents_retained": len(unique_docs),
        "total_parent_sources": len(parent_to_split),
        "total_pilot_tokens": total_pilot_tokens,
        "split_stats": split_stats,
        "domain_stats": {
            dom: {
                "parents": {k: len(v) if isinstance(v, set) else v for k, v in s["parents"].items()},
                "docs": s["docs"],
                "tokens": s["tokens"],
                "unk_tokens": s["unk_tokens"]
            }
            for dom, s in domain_stats.items()
        },
        "file_checksums": file_checksums,
        "total_bytes": total_bytes
    }


if __name__ == "__main__":
    results = run_real_data_pilot()
    print("\nExecution Results Summary:")
    print(json.dumps(results, indent=2))
