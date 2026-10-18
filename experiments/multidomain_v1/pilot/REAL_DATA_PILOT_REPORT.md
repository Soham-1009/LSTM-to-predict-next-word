# Multi-Domain Real-Data Pilot Final Audit Report (Phase 3C)

**Project:** LSTM Next-Word Prediction & Text Generation System  
**Phase:** **Phase 3C: Final Pilot Integrity Audit and Readiness Gate**  
**Role:** Senior NLP Engineer, PyTorch Engineer, Data Governance Engineer, Independent Code Auditor  
**Project Root:** `C:\Users\soham\Desktop\Soham\GenAI`  
**Execution Date:** October 15, 2026  
**Final Verdict:** **READY WITH RESTRICTIONS**  
**Executive Guardrail:** Baseline Epoch 46 checkpoint (`best_val.pt`) and Epoch 50 checkpoint (`latest.pt`) strictly preserved bit-for-bit. Model weights unchanged. No model training launched.

---

## 1. Executive Summary & Objective

This report documents the independent integrity audit and architectural gate executed under **Phase 3C: Final Pilot Integrity Audit and Readiness Gate**.

Rather than accepting prior report claims at face value, every assertion, count, licensing declaration, and sequence construction was audited directly against the source code, machine-readable manifest (`pilot_manifest.jsonl`), metadata (`processed/metadata.json`), serialized binary memory maps, and executed regression tests.

### Key Architectural & Governance Enhancements in Phase 3C:
1. **Conservative, Exact Machine-Readable PMC License Verification:**
   - Audited the XML ingestion pipeline and eliminated substring matching vulnerabilities (`creativecommons.org/licenses/by` previously matched non-commercial and share-alike licenses).
   - Implemented `verify_pmc_license_metadata()`, enforcing exact regex/URI normalization matching CC-BY 4.0 International (`https://creativecommons.org/licenses/by/4.0/`).
   - Strictly rejected all non-commercial (CC-BY-NC), share-alike (CC-BY-SA), no-derivatives (CC-BY-ND), and legacy versions (CC-BY 3.0/2.0).
   - Eliminated inference from generic prose (e.g. "Creative Commons Attribution") when no machine-readable reference is present.
   - Prohibited inventing license evidence URLs; stored the exact verified machine-readable URI and decision code for every retained paper.
2. **Authoritative Mathematical Reconciliation Audit:**
   - Resolved prior report count discrepancies (where parent counts and document counts were conflated).
   - Implemented an authoritative audit routine (`audit_pilot.py`) enforcing 11 strict mathematical invariants across manifest records, metadata, binary files, and 64-bit document offset tables.
   - Verified that all report tables are generated directly from validated metadata rather than manual hardcoding.
3. **Document-Boundary-Safe Causal Sequence Generation:**
   - Audited PyTorch `CausalLanguageModelDataset` and identified that flat sliding windows crossed document boundaries, causing targets to be sourced across distinct works.
   - Refactored `CausalLanguageModelDataset` to be fully offset-aware (`doc_offsets`), constructing input/target sequences strictly within document limits.
   - Guaranteed that no input/target window crosses any document boundary and no target is sourced from an adjacent document.
   - Validated that all 2,301 training sequences from `train.bin` strictly reside within their respective document boundaries.
4. **Rigorous Evaluation Strength Classification:**
   - Distinguished engineering validation of the data pipeline from statistical evaluation power.
   - Categorized domain evaluations into **Standard Evaluation** (adequate independent parent sources, e.g. news with 40/5/5) versus **Limited (Smoke-Test Only)** (e.g. literature 3/1/1, scientific 4/1/1, technical docs 4/1/1, conversation 2/1/1).
5. **Jurisdictional Copyright Qualification & Corpus Rights Accounting:**
   - Reconciled Reuters-21578 non-commercial research scope (NLTK vs. UCI).
   - Qualified Project Gutenberg copyright status: verified US pre-1929 status and EU/UK life+70 status, while noting that extended post-mortem terms in Spain (life+80) and Mexico (life+100) maintain active rights for Doyle and Shaw until 2031 and 2051.
   - Structured the planned 100M-token corpus into Approved, Conditional, Excluded, and Unresolved tiers.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                  Phase 3C Final Audit Summary                                          │
├───────────────────────────────────┬────────────────────────────────────────────────────────────────────┤
│ Total Documents Ingested          │ 129 raw document records across 86 parent sources                  │
│ Filtered / Rejected Documents     │ 1 corrupt test fixture rejected by quality filter (0.78%)          │
│ Duplicates Detected & Purged      │ 2 exact SHA-256 duplicate fixtures purged (1.55%)                  │
│ Unique Retained Documents         │ 126 clean, unique documents across 86 parent sources (97.67%)      │
│ Total Pilot Tokens Ingested       │ 123,983 tokens (strictly respects <= 1,000,000 token ceiling)      │
│ Parent-Source Partitioning        │ Train: 64 parents (74.4%) | Val: 11 (12.8%) | Test: 11 (12.8%)     │
│ Document Split Distribution       │ Train: 90 docs (71.4%) | Val: 19 docs (15.1%) | Test: 17 (13.5%)   │
│ Token Split Distribution          │ Train: 82,134 tokens | Val: 20,354 tokens | Test: 21,495 tokens   │
│ Parent-Source Split Leakage       │ 0.00% (Strict zero parent-source overlap verified)                 │
│ Document Hash Split Leakage       │ 0.00% (Strict zero clean_sha256 overlap verified)                  │
│ Per-Domain 3-Split Coverage       │ 100% (All 6 domains have non-zero train, val, and test tokens)     │
│ Machine-Readable License Check    │ 100% of PMC papers parsed from XML <license_ref> as CC-BY 4.0      │
│ Document Boundary Safety          │ 100% of 2,301 training sequences strictly inside doc boundaries   │
│ Authoritative Mathematical Audit  │ AUDIT_PASSED (11/11 mathematical invariants verified)              │
│ Checkpoint SHA-256 Integrity      │ 100% bit-for-bit match on best_val.pt and latest.pt               │
│ Overall Pilot Recommendation      │ READY WITH RESTRICTIONS (Full corpus ingestion conditioned)        │
└───────────────────────────────────┴────────────────────────────────────────────────────────────────────┘
```

---

## 2. Source-Level Licensing & Rights Verification Audit

Every retained parent work was audited individually. Automated metadata verification outcomes and legal parameters were recorded in `pilot_manifest.jsonl` and `pilot_config.json`.

### 2.1 Authoritative Source-Level Audit Table

| Domain | Parent Source ID | Title & Work Identity | Author / Entity | Canonical URL | Exact Verified License | Evidence URL | Verification Outcome | Terms & Jurisdiction Notes | Licensing Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| **Replay** | `wiki_art_1` .. `15` | 15 Independent Wikipedia Articles | Wikipedia Contributors | [Salesforce HuggingFace](https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/) | CC-BY-SA 3.0 | [Creative Commons](https://creativecommons.org/licenses/by-sa/3.0/) | Verified Salesforce release metadata | Sampled exclusively from `wiki.train.tokens`. Zero test split overlap. | **APPROVED** |
| **Literature** | `gutenberg_pg84` | *Frankenstein* (Pub. 1818) | Mary Wollstonecraft Shelley | [Gutenberg #84](https://www.gutenberg.org/cache/epub/84/pg84.txt) | US Public Domain (Pre-1929) | [PG Policy](https://www.gutenberg.org/policy/license.html) | Verified publication 1818; author d. 1851 | Author d. 1851 (life+175y). Public domain in US (17 U.S.C. § 304), UK, EU, and worldwide. PG envelope stripped. | **APPROVED** |
| **Literature** | `gutenberg_pg1342` | *Pride and Prejudice* (Pub. 1813) | Jane Austen | [Gutenberg #1342](https://www.gutenberg.org/cache/epub/1342/pg1342.txt) | US Public Domain (Pre-1929) | [PG Policy](https://www.gutenberg.org/policy/license.html) | Verified publication 1813; author d. 1817 | Author d. 1817 (life+209y). Public domain in US, UK, EU, and worldwide. PG envelope stripped. | **APPROVED** |
| **Literature** | `gutenberg_pg1661` | *Adventures of Sherlock Holmes* (1892) | Arthur Conan Doyle | [Gutenberg #1661](https://www.gutenberg.org/cache/epub/1661/pg1661.txt) | US Public Domain (Pre-1929) | [PG Policy](https://www.gutenberg.org/policy/license.html) | Verified publication 1892; author d. 1930 | Author d. 1930 (life+96y). US public domain (pre-1929); EU/UK public domain since 2000. Protected in Mexico (life+100) until 2031. | **APPROVED (QUALIFIED)** |
| **Literature** | `gutenberg_pg98` | *A Tale of Two Cities* (Pub. 1859) | Charles Dickens | [Gutenberg #98](https://www.gutenberg.org/cache/epub/98/pg98.txt) | US Public Domain (Pre-1929) | [PG Policy](https://www.gutenberg.org/policy/license.html) | Verified publication 1859; author d. 1870 | Author d. 1870 (life+156y). Public domain worldwide. PG envelope stripped. | **APPROVED** |
| **Literature** | `gutenberg_pg345` | *Dracula* (Pub. 1897) | Bram Stoker | [Gutenberg #345](https://www.gutenberg.org/cache/epub/345/pg345.txt) | US Public Domain (Pre-1929) | [PG Policy](https://www.gutenberg.org/policy/license.html) | Verified publication 1897; author d. 1912 | Author d. 1912 (life+114y). Public domain worldwide. PG envelope stripped. | **APPROVED** |
| **Tech Docs** | `ietf_rfc8446` | RFC 8446: The TLS 1.3 Protocol | IETF Trust | [RFC Editor](https://www.rfc-editor.org/rfc/rfc8446.txt) | IETF Trust Provisions (RFC 5378) | [IETF Trust](https://trustee.ietf.org/license-info) | Verified standard publication | Open standards specification for technical review and implementation. ASCII diagrams filtered. | **APPROVED** |
| **Tech Docs** | `ietf_rfc7230` | RFC 7230: HTTP/1.1 Message Syntax | IETF Trust | [RFC Editor](https://www.rfc-editor.org/rfc/rfc7230.txt) | IETF Trust Provisions (RFC 5378) | [IETF Trust](https://trustee.ietf.org/license-info) | Verified standard publication | Open standards specification for technical review and implementation. ASCII diagrams filtered. | **APPROVED** |
| **Tech Docs** | `ietf_rfc7540` | RFC 7540: HTTP/2 Protocol | IETF Trust | [RFC Editor](https://www.rfc-editor.org/rfc/rfc7540.txt) | IETF Trust Provisions (RFC 5378) | [IETF Trust](https://trustee.ietf.org/license-info) | Verified standard publication | Open standards specification for technical review and implementation. ASCII diagrams filtered. | **APPROVED** |
| **Tech Docs** | `cpython_datastruct` | Python Tutorial: Data Structures | Python Software Foundation | [CPython GitHub](https://raw.githubusercontent.com/python/cpython/main/Doc/tutorial/datastructures.rst) | PSF License v2 | [Python Docs](https://docs.python.org/3/license.html) | Verified CPython Doc tree | Permissive open-source documentation. Code blocks and prose preserved. | **APPROVED** |
| **Tech Docs** | `cpython_modules` | Python Tutorial: Modules | Python Software Foundation | [CPython GitHub](https://raw.githubusercontent.com/python/cpython/main/Doc/tutorial/modules.rst) | PSF License v2 | [Python Docs](https://docs.python.org/3/license.html) | Verified CPython Doc tree | Permissive open-source documentation. Code blocks and prose preserved. | **APPROVED** |
| **Tech Docs** | `cpython_errors` | Python Tutorial: Errors & Exceptions | Python Software Foundation | [CPython GitHub](https://raw.githubusercontent.com/python/cpython/main/Doc/tutorial/errors.rst) | PSF License v2 | [Python Docs](https://docs.python.org/3/license.html) | Verified CPython Doc tree | Permissive open-source documentation. Code blocks and prose preserved. | **APPROVED** |
| **Scientific** | `pmc_7095448` | SARS-CoV-2 structural biology review | Springer Nature / PMC OA | [NCBI E-utilities](https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=PMC7095448&retmode=xml) | CC-BY 4.0 International | `https://creativecommons.org/licenses/by/4.0/` | Exact XML `<license_ref>` parsed: `VERIFIED_CC_BY_4_0` | Unrestricted commercial/non-commercial redistribution and adaptation. Reference lists stripped. | **APPROVED** |
| **Scientific** | `pmc_13660980` | Genomic diversity & epidemiology | PLoS ONE | [NCBI E-utilities](https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=PMC13660980&retmode=xml) | CC-BY 4.0 International | `https://creativecommons.org/licenses/by/4.0/` | Exact XML `<license_ref>` parsed: `VERIFIED_CC_BY_4_0` | Full PLoS ONE open access paper. Unrestricted reuse. Reference lists stripped. | **APPROVED** |
| **Scientific** | `pmc_13660983` | Cellular biomechanics & microscopy | PLoS ONE | [NCBI E-utilities](https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=PMC13660983&retmode=xml) | CC-BY 4.0 International | `https://creativecommons.org/licenses/by/4.0/` | Exact XML `<license_ref>` parsed: `VERIFIED_CC_BY_4_0` | Full PLoS ONE open access paper. Unrestricted reuse. Reference lists stripped. | **APPROVED** |
| **Scientific** | `pmc_13660993` | High-throughput protein screening | PLoS ONE | [NCBI E-utilities](https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=PMC13660993&retmode=xml) | CC-BY 4.0 International | `https://creativecommons.org/licenses/by/4.0/` | Exact XML `<license_ref>` parsed: `VERIFIED_CC_BY_4_0` | Full PLoS ONE open access paper. Unrestricted reuse. Reference lists stripped. | **APPROVED** |
| **Scientific** | `pmc_13660882` | Macromolecular crystallography | PLoS ONE | [NCBI E-utilities](https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=PMC13660882&retmode=xml) | CC-BY 4.0 International | `https://creativecommons.org/licenses/by/4.0/` | Exact XML `<license_ref>` parsed: `VERIFIED_CC_BY_4_0` | Full PLoS ONE open access paper. Unrestricted reuse. Reference lists stripped. | **APPROVED** |
| **Scientific** | `pmc_13660988` | Evolutionary genetics & phylogenetics | PLoS ONE | [NCBI E-utilities](https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=PMC13660988&retmode=xml) | CC-BY 4.0 International | `https://creativecommons.org/licenses/by/4.0/` | Exact XML `<license_ref>` parsed: `VERIFIED_CC_BY_4_0` | Full PLoS ONE open access paper. Unrestricted reuse. Reference lists stripped. | **APPROVED** |
| **Conversation** | `gutenberg_pg844` | *The Importance of Being Earnest* | Oscar Wilde (Pub. 1895) | [Gutenberg #844](https://www.gutenberg.org/cache/epub/844/pg844.txt) | US Public Domain (Pre-1929) | [PG Policy](https://www.gutenberg.org/policy/license.html) | Verified publication 1895; author d. 1900 | Author d. 1900 (life+126y). Worldwide public domain. Character speaker turns preserved. | **APPROVED** |
| **Conversation** | `gutenberg_pg3825` | *Pygmalion* (Pub. 1913) | George Bernard Shaw | [Gutenberg #3825](https://www.gutenberg.org/cache/epub/3825/pg3825.txt) | US Public Domain (Pre-1929) | [PG Policy](https://www.gutenberg.org/policy/license.html) | Verified publication 1913; author d. 1950 | US public domain (1913). UK/EU public domain since Jan 1, 2021. Protected in Spain (life+80) until 2031 and Mexico (life+100) until 2051. | **APPROVED (QUALIFIED)** |
| **Conversation** | `gutenberg_pg790` | *Lady Windermere's Fan* (Pub. 1892) | Oscar Wilde | [Gutenberg #790](https://www.gutenberg.org/cache/epub/790/pg790.txt) | US Public Domain (Pre-1929) | [PG Policy](https://www.gutenberg.org/policy/license.html) | Verified publication 1892; author d. 1900 | Author d. 1900 (life+126y). Worldwide public domain. Character speaker turns preserved. | **APPROVED** |
| **Conversation** | `gutenberg_pg882` | *An Ideal Husband* (Pub. 1895) | Oscar Wilde | [Gutenberg #882](https://www.gutenberg.org/cache/epub/882/pg882.txt) | US Public Domain (Pre-1929) | [PG Policy](https://www.gutenberg.org/policy/license.html) | Verified publication 1895; author d. 1900 | Author d. 1900 (life+126y). Worldwide public domain. Character speaker turns preserved. | **APPROVED** |
| **News** | `reuters_1` .. `50` | 50 Independent Reuters News Stories | Reuters Wire Correspondents | [UCI Repository](https://kdd.ics.uci.edu/databases/reuters21578/reuters21578.html) | Open Academic & Research Use | [UCI README](https://kdd.ics.uci.edu/databases/reuters21578/README.txt) | Reconciled UCI terms vs. NLTK distribution | Permitted for non-commercial research, benchmarking, and internal adaptation. Commercial raw corpus redistribution prohibited. | **CONDITIONAL** |

### 2.2 Deep Legal Reconciliation & Jurisdictional Analysis

1. **Exact Automated PMC License Verification:**
   - *Audit Finding:* In Phase 3B, substring matching on `/licenses/by` created a latent defect where articles licensed under CC-BY-NC (Non-Commercial) or CC-BY-SA (Share-Alike) could pass verification undetected.
   - *Resolution:* Implemented strict regex parsing in `verify_pmc_license_metadata()`. All 6 selected PMC articles (PLoS ONE and Nature) contain explicit `<license_ref>` matching `https://creativecommons.org/licenses/by/4.0/`.
   - *Legal Disclaimer:* This automated check verifies machine-readable XML declarations provided by NLM. It does not constitute formal legal counsel. Articles containing ambiguous, third-party embedded imagery or conflicting data licenses are quarantined.
2. **Reuters-21578 Reconciliation (NLTK vs. UCI):**
   - *Investigation:* UCI documentation indicates Reuters Ltd. and Carnegie Group made the text available for text categorization research in 1987. NLTK metadata designates it for research only.
   - *Resolution:* Neither source grants commercial redistribution rights for the underlying news articles. Reuters-21578 is designated **CONDITIONAL**: approved for internal experimentation, benchmarking, and domain adaptation; commercial distribution of the raw text dataset is strictly prohibited. For a commercial deployment, open CC-BY journalism (e.g. WikiNews) must replace Reuters.
3. **Project Gutenberg Jurisdictional Copyright Nuances:**
   - *Investigation:* Works published prior to 1929 are in the US public domain under 17 U.S.C. § 304. However, under the Berne Convention, local copyright terms apply:
     - European Union & UK: Life of author + 70 years.
     - Spain: Life + 80 years for authors dying prior to December 7, 1987.
     - Mexico: Life + 100 years.
   - *Author-by-Author Audit:*
     - Mary Shelley (d. 1851), Jane Austen (d. 1817), Charles Dickens (d. 1870), Bram Stoker (d. 1912), Oscar Wilde (d. 1900): All deceased $>110$ years $\to$ Worldwide public domain.
     - Arthur Conan Doyle (d. 1930): US public domain (pre-1929); UK/EU public domain since 2000. Under Mexican copyright law, protected until 2031.
     - George Bernard Shaw (d. 1950): *Pygmalion* (1913) is US public domain. In UK/EU, entered public domain on Jan 1, 2021 (70 years post-mortem). However, under Spanish law (life+80), protected until 2031; under Mexican law (life+100), protected until 2051.
   - *Conclusion:* Public domain status is **unconditional** in the US and EU/UK, but **qualified** in territories with extended terms (Spain, Mexico). Raw text distribution in those countries is restricted.

---

## 3. Data Intake, Quality Filtering & Deduplication Accounting

Every candidate record was tracked from initial ingestion through normalization, filtering, and splitting.

```
Total Ingested Document Records:            129 (100.00%)
  ├── Clean, Unique Documents Retained:     126 ( 97.67%)
  ├── Duplicates Purged by Exact SHA-256:     2 (  1.55%)
  └── Corrupt / Filter Rejections:            1 (  0.78%)
```

1. **Exact SHA-256 Deduplication:**
   - Injected fixtures `test_duplicate_doc_1` and `test_duplicate_doc_2` were purged.
   - Whitespace normalization (NFKC + standardized newlines) matched identical clean hashes (`dcf860b6cb00...` and `51f271a9c0f9...`).
   - Both were marked `REJECTED_EXACT_DUPLICATE` in `pilot_manifest.jsonl` with `token_count = 0` and `assigned_split = "DEDUPLICATED"`.
2. **Quality Filtering:**
   - Fixture `test_corrupt_doc_rejected` (non-printable control chars, printable ratio 0.44) was rejected by `filter_document()`.
   - Logged as `REJECTED_QUALITY_FILTER` with `token_count = 0` and `assigned_split = "REJECTED"`.
   - All 126 retained documents met minimum length ($\ge 100$ characters) and printable ratio ($\ge 0.95$) requirements.
3. **Zero Token Leakage from Rejected Records:**
   - Authoritative audit confirmed that rejected and deduplicated records contribute exactly 0 tokens to any split or domain statistic.

---

## 4. Authoritative Reconciled Corpus Accounting

All figures below were generated directly by `audit_pilot.py` and mathematically verified against `pilot_manifest.jsonl`, `processed/metadata.json`, and the 6 serialized binary files.

### 4.1 Reconciled Multi-Domain Pilot Corpus Accounting

| Domain Identifier | Parent Sources (Tr/V/Te) | Documents (Tr/V/Te) | Train Tokens | Val Tokens | Test Tokens | Total Tokens | OOV Rate | Evaluation Strength |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`conversation`** | 2 / 1 / 1 (4) | 6 / 3 / 3 (12) | 4,750 | 2,468 | 2,455 | **9,673** | 6.53% | Pipeline Integrity Smoke-Test (1 test parent; limited) |
| **`literature`** | 3 / 1 / 1 (5) | 9 / 3 / 1 (13) | 17,626 | 5,787 | 2,613 | **26,026** | 3.74% | Pipeline Integrity Smoke-Test (1 test parent; limited) |
| **`news`** | 40 / 5 / 5 (50) | 40 / 5 / 5 (50) | 7,910 | 1,599 | 853 | **10,362** | 7.37% | Preliminary Domain Evaluation (5 test parents, 853 toks; small sample) |
| **`scientific`** | 4 / 1 / 1 (6) | 12 / 3 / 3 (18) | 6,895 | 2,007 | 1,946 | **10,848** | 7.46% | Pipeline Integrity Smoke-Test (1 test parent; limited) |
| **`technical_docs`** | 4 / 1 / 1 (6) | 12 / 3 / 3 (18) | 18,726 | 3,715 | 8,990 | **31,431** | 5.98% | Pipeline Integrity Smoke-Test (1 test parent; limited) |
| **`wikitext103_replay`** | 11 / 2 / 2 (15) | 11 / 2 / 2 (15) | 26,227 | 4,778 | 4,638 | **35,643** | 3.67% | Pipeline Integrity Smoke-Test (2 test parent; limited) |
| **Total Reconciled Corpus** | **64 / 11 / 11 (86)** | **90 / 19 / 17 (126)** | **82,134** | **20,354** | **21,495** | **123,983** | **5.14%** | Multi-Domain Pipeline Verified |

### 4.2 Split Proportions: Indivisible Parent-Source Granularity

- **Target Ratios:** 80.0% Train / 10.0% Val / 10.0% Test.
- **Observed Parent Works:** 74.4% Train (64) / 12.8% Val (11) / 12.8% Test (11).
- **Observed Token Volumes:** 66.25% Train (82,134) / 16.42% Val (20,354) / 17.34% Test (21,495).
- **Engineering Rationale:** Preserving parent works as indivisible atomic units (e.g. RFC 7230 with 8,990 tokens or *Frankenstein* chapters totaling 5,787 tokens) prevents split leakage at the cost of token-level coarseness. Sub-document chunking across splits to force an exact 80/10/10 ratio would re-introduce cross-partition contamination.

### 4.3 Evaluation Strength: Pipeline Integrity vs. Preliminary Evaluation vs. Statistical Adequacy

- **Tier 1: Pipeline Integrity Validation (100% Passed):**
  Ingestion, cleaning, tokenization, serialization, offset indexing, and boundary-safe dataset creation function flawlessly across all 6 domains. Zero parent-source leakage and zero hash leakage are verified mathematically.
- **Tier 2: Preliminary Domain Evaluation (Smoke-Test Scale):**
  - **`news`:** Contains 5 test parent articles and 853 tokens. While providing preliminary directional validation, five test articles do not constitute a statistically adequate or broadly representative benchmark.
  - **`wikitext103_replay`, `literature`, `technical_docs`, `scientific`, `conversation`:** Contain only 1 or 2 parent works in validation and test. This proves the pipeline architecture can isolate and evaluate held-out parent works, but represents a limited smoke-test.
- **Tier 3: Statistically Adequate Model Evaluation (Required for Full Study):**
  A defensible domain benchmark requires $\ge 20$ independent validation parent sources and $\ge 20$ independent test parent sources per domain ($\ge 40$ held-out works per domain) to ensure statistical significance across stylistic, thematic, and syntactic variations. This expansion is designed in Phase 3D.

---

## 5. Document-Boundary-Safe Causal Sequence Generation

In causal language modeling ($P(w_t \mid w_{<t})$), training sequences are framed as shifted input/target pairs:
$$\mathbf{x} = [w_1, w_2, \dots, w_T], \quad \mathbf{y} = [w_2, w_3, \dots, w_{T+1}]$$

### 5.1 The Boundary-Crossing Defect in Flat Concatenation

When documents are concatenated into a flat binary file separated by `<EOS> <BOS>`, an unconstrained sliding window overlapping the document boundary forms sequences where:
- $\mathbf{x}$ ends with the trailing tokens of Document $A$ and `<EOS>`.
- $\mathbf{y}$ contains `<BOS>` and the opening tokens of Document $B$.

This erroneously trains the model to predict the beginning of an unrelated document from the conclusion of a preceding one.

### 5.2 Offset-Aware Sequence Construction

In Phase 3C, `CausalLanguageModelDataset` was refactored with explicit document offset awareness (`doc_offsets`):
1. **Intra-Document Windowing:**
   For document $i$ spanning tokens $[\text{offset}[i], \text{offset}[i+1])$, candidate start positions $s$ are constructed strictly such that:
   $$\text{offset}[i] \le s \quad \text{and} \quad s + T + 1 \le \text{offset}[i+1]$$
2. **Sub-Context Handling:**
   Documents shorter than $T + 1 = 36$ tokens produce 0 sequences, avoiding padding artifacts or cross-document splicing.
3. **Validation & Exception Handling:**
   - Offsets not starting at 0 raise `ValueError`.
   - Non-monotonic offsets raise `ValueError`.
   - Offsets where $\text{offset}[-1] \neq \text{len}(\text{data})$ raise `ValueError`.
   - Out-of-range indices raise `IndexError`.

### 5.3 Real Pilot Verification Results

Using context length $T = 35$ and stride $S = 35$ on `train.bin` (82,134 tokens across 90 documents):
- **Total Boundary-Safe Training Sequences Generated:** **2,301 sequences**.
- **Document Boundary Containment:** **100.00%**. Every single sample $[s, s + 36)$ was verified to lie strictly within a single document.
- **Zero Cross-Document Targets:** No target token in $\mathbf{y}$ was sourced from a subsequent document.

---

## 6. Binary File Checksums and Offset Table Integrity

The authoritative audit verified all 6 binary files, matching checksums, token sizes, and offset counts:

| Split Name | Documents | Offsets Count | Tokens | Binary Size (bytes) | Offset Binary Size (bytes) | SHA-256 Checksum |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `train` | 90 | 91 | 82,134 | 164,268 | 728 | `06b1b38f53cbdeb9b5f05dba0547b1e62f9eef6abe7af596ff2dfc3772213109` |
| `val` | 19 | 20 | 20,354 | 40,708 | 160 | `cdcd6ad7983bb6df0a96e5a7385058e39784e50a56bf3f587178432b64e704d9` |
| `test` | 17 | 18 | 21,495 | 42,990 | 144 | `873f9593293c1667dd94f9de07177bab0b10ac704a63aa685f64ddd225940288` |

### Offset File Integrity Assertions
- `train_offsets.bin`: Exactly 91 offsets (90 docs + 1). Offset[0] = 0, Offset[90] = 82,134. SHA-256: `f70b331763f8b4b58dd8fb4bb285d8e18ba929a81df5a5c7c163176f5dac5d0e`.
- `val_offsets.bin`: Exactly 20 offsets (19 docs + 1). Offset[0] = 0, Offset[19] = 20,354. SHA-256: `719e91883ff7864fc59f0a62b6c2a33099b406f45e5f81645f4586ad9d13ead9`.
- `test_offsets.bin`: Exactly 18 offsets (17 docs + 1). Offset[0] = 0, Offset[17] = 21,495. SHA-256: `ddfb71e8761bf3aa7ca184bd6be7bc95df2d11438dc934d6518a3085d9b2a850`.
- All offset tables are strictly increasing; every offset delta equals the document token count in `pilot_manifest.jsonl`.

---

## 7. Multi-Domain OOV & Lexical Audit (Frozen 30k Vocabulary)

Tokenization used the frozen production word-level vocabulary (`vocab.json`, 30,000 tokens):

```
Multi-Domain Lexical OOV Rate Breakdown (Frozen 30,000 Vocabulary)
┌──────────────────────┬─────────────┬──────────────┬──────────────┬──────────────┐
│ Domain               │ Total Tokens│ Valid Tokens │ UNK Tokens   │ OOV Rate (%) │
├──────────────────────┼─────────────┼──────────────┼──────────────┼──────────────┤
│ wikitext103_replay   │      35,643 │       34,334 │        1,309 │        3.67% │
│ literature           │      26,026 │       25,052 │          974 │        3.74% │
│ technical_docs       │      31,431 │       29,551 │        1,880 │        5.98% │
│ conversation         │       9,673 │        9,041 │          632 │        6.53% │
│ news                 │      10,362 │        9,598 │          764 │        7.37% │
│ scientific           │      10,848 │       10,039 │          809 │        7.46% │
├──────────────────────┼─────────────┼──────────────┼──────────────┼──────────────┤
│ Total Pilot Corpus   │     123,983 │      117,615 │        6,368 │        5.14% │
└──────────────────────┴─────────────┴──────────────┴──────────────┴──────────────┘
```

The overall OOV rate of **5.14%** confirms that word-level domain adaptation under Approach A is viable. Lexical elevation in `scientific` (7.46%) and `news` (7.37%) reflects specialized terminology (e.g. *phylogenetic*, *crystallography*, *macroeconomic*).

---

## 8. Cryptographic Disjointness & Reproducibility Audit

### 8.1 Partition Disjointness
1. **Parent-Source Disjointness:**
   - $\text{Parents}(\text{Train}) \cap \text{Parents}(\text{Val}) = \emptyset$ (0 intersecting parent works).
   - $\text{Parents}(\text{Train}) \cap \text{Parents}(\text{Test}) = \emptyset$ (0 intersecting parent works).
   - $\text{Parents}(\text{Val}) \cap \text{Parents}(\text{Test}) = \emptyset$ (0 intersecting parent works).
2. **Document Hash Disjointness:**
   - $\text{Hashes}(\text{Train}) \cap \text{Hashes}(\text{Val}) = \emptyset$ (0 intersecting clean document hashes).
   - $\text{Hashes}(\text{Train}) \cap \text{Hashes}(\text{Test}) = \emptyset$ (0 intersecting clean document hashes).
   - $\text{Hashes}(\text{Val}) \cap \text{Hashes}(\text{Test}) = \emptyset$ (0 intersecting clean document hashes).

### 8.2 Seed 42 Determinism
Rerunning `split_parent_sources_stratified(seed=42)` yields 100% bit-for-bit identical parent assignments, split assignments, and token distributions.

---

## 9. Full 100-Million-Token Corpus Plan & Rights Accounting

The target 100M-token corpus must not be treated as fully cleared or immediately available. Candidate data sources are classified into 4 distinct governance tiers:

| Governance Category | Volume Target | Source Candidates | Licensing Framework | Usage Scope & Restrictions |
| :--- | :---: | :--- | :--- | :--- |
| **APPROVED** | ~50M tokens | WikiText-103 training set (~25M), verified CC-BY 4.0 PMC papers (~15M), pre-1929 Gutenberg books (~5M), IETF RFCs & Python docs (~5M) | CC-BY-SA 3.0, CC-BY 4.0, US Public Domain, IETF TLP 5.0, PSFL v2 | Unrestricted research and model adaptation. Commercial raw redistribution permitted where CC-BY/PSF applies. |
| **CONDITIONAL** | ~15M tokens | Reuters-21578 Financial News Archive | Open Academic Research Use (Reuters Ltd. / Carnegie Group) | Non-commercial research and internal adaptation only. Prohibits commercial redistribution of raw news wire. |
| **EXCLUDED** | N/A (Filtered) | PMC articles with CC-BY-NC, CC-BY-ND, or emergency COVID-19 text-mining terms; non-open web pages | Non-Commercial / No-Derivatives / Proprietary | Excluded by automated XML verification. Incompatible with open model adaptation. |
| **UNRESOLVED** | ~35M tokens | Open web news feeds (CC-News / Common Crawl News); large-scale arXiv / bioRxiv dumps | To be audited at URL/article level | Requires domain-level robots.txt checking, publisher rights screening, and automated license metadata verification before ingestion. |

---

## 10. Regression Test Suite Execution & Baseline Checkpoint Verification

### 10.1 Regression Test Suite Results (`pytest backend/tests`)
Executed test command: `backend\.venv\Scripts\pytest backend\tests -v`
```
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0
collecting ... collected 49 items

backend/tests/test_api.py (26 tests) .......................... PASSED
backend/tests/test_evaluation.py (5 tests) .....                 PASSED
backend/tests/test_pipeline.py (18 tests) ....................   PASSED

======================= 49 passed, 2 warnings in 18.56s =======================
```
- **100% Pass Rate (49/49 passed in 18.56s)**.
- Specific Phase 3C tests verified:
  - `test_pmc_license_verification`: Valid CC-BY 4.0, CC-BY-NC rejection, CC-BY-SA rejection, CC-BY-ND rejection, missing ref rejection, generic text rejection, conflicting ref rejection.
  - `test_document_boundary_safe_causal_dataset`: Synthetic documents, intra-document containment, invalid offset exceptions.
  - `test_pilot_authoritative_audit_passes`: Mathematical reconciliation on live pilot artifacts.
  - `test_pilot_authoritative_audit_detects_inconsistencies`: Tampered token counts, bad checksums, tampered document counts caught by audit assertions.

### 10.2 Baseline Checkpoint Integrity & Inference Verification
Executed verification script: `scripts/verify_checkpoints.py`
```
======================================================================
STARTING LSTM CHECKPOINT INTEGRITY & INFERENCE VERIFICATION
======================================================================
Verifying best_val.pt...
[best_val.pt] Reported Epoch: 46, Val Loss: 3.9470
  -> PASSED: Epoch correctly recorded (46).
  -> PASSED: Zero 'module.' prefixes found across 11 tensors.
  -> PASSED: Loaded and evaluated in non-DDP LSTMLanguageModel on CPU.
  -> PASSED: Loaded and evaluated in non-DDP LSTMLanguageModel on CUDA (RTX 3050 Laptop GPU).
--------------------------------------------------
Verifying latest.pt...
[latest.pt] Reported Epoch: 50, Val Loss: 3.9693
  -> PASSED: Epoch correctly recorded (50).
  -> PASSED: Zero 'module.' prefixes found across 11 tensors.
  -> PASSED: Loaded and evaluated in non-DDP LSTMLanguageModel on CPU.
  -> PASSED: Loaded and evaluated in non-DDP LSTMLanguageModel on CUDA (RTX 3050 Laptop GPU).
--------------------------------------------------
Testing FastAPI InferenceService integration with best_val.pt...
InferenceService initialized successfully (Device: cuda, Checkpoint Loaded: True).
  -> PASSED: Autocomplete predict_next_word produced 5 valid candidates without structural tokens.
  -> PASSED: Greedy generation deterministic (10 tokens).
  -> PASSED: Sampled generation produced valid text (15 tokens).
  -> PASSED: Unicode/emoji input handled safely without tokenizer crash.
  -> PASSED: Context length truncation handled correctly (>35 tokens).
======================================================================
ALL VERIFICATIONS COMPLETED SUCCESSFULLY!
======================================================================
```

### 10.3 Checkpoint SHA-256 Bit-for-Bit Hash Verification
Direct cryptographic verification via Python `hashlib.sha256`:
- `backend/model_artifacts/checkpoints/best_val.pt`:
  `2AD9F4B1FAC120846A652406B45357515486838D85D7450EE91D96262B23185D` (**EXACT MATCH**)
- `backend/model_artifacts/checkpoints/latest.pt`:
  `C29B92F0B7E2380DF6A74EFBDE89EA7AD77FF64B7AF440773F33A22AF907629D` (**EXACT MATCH**)

---

## 11. Outstanding Risks and Open Issues

1. **Evaluation Statistical Power in Small Pilot:**
   Validation and test sets in literature, science, tech docs, and conversation contain 1 parent source each. While sufficient for data pipeline engineering verification, statistical generalizability requires expanding evaluation sets during full ingestion.
2. **Long-Tail Jurisdictional Copyright for Doyle & Shaw:**
   Doyle (d. 1930) and Shaw (d. 1950) remain under copyright in Mexico (life+100) until 2031 and 2051 respectively, and Shaw remains protected in Spain (life+80) until 2031. Any international release of raw corpora must exclude these authors in those specific jurisdictions.
3. **Reuters-21578 Commercial Limitation:**
   Approved strictly for non-commercial research and benchmarking. Any commercial product deployment must replace Reuters with open-licensed journalism.

---

## 12. Final Readiness Gate Verdict

$$\mathbf{READY\ WITH\ RESTRICTIONS}$$

### Gate Conditions Satisfied:
- [x] Zero parent-source split leakage.
- [x] Zero document hash split leakage.
- [x] 100% per-domain split coverage.
- [x] Exact, conservative machine-readable PMC XML license verification (CC-BY 4.0).
- [x] Document-boundary-safe `CausalLanguageModelDataset` with zero cross-document targets.
- [x] 100% mathematical reconciliation across manifest, metadata, binaries, and offset tables.
- [x] 49/49 regression tests passing.
- [x] Baseline checkpoints preserved bit-for-bit.

### Gate Restrictions Enforced:
1. **No Training Launched:** Model fine-tuning or training remains frozen pending user sign-off.
2. **Reuters-21578 Non-Commercial Constraint:** Restricted to academic research and internal model adaptation.
3. **Full Corpus Ingestion Gate:** Full-scale 100M-token corpus ingestion requires automated license filtering on all additional PMC papers and resolving open web news sources.
