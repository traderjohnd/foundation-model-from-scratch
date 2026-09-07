# Decision Register — 01 Data Preparation & Corpus Audit

Notebook 01 establishes the immutable data source and the structure-aware normalization/reconstruction contract consumed by later phases.

---

## D-006 — Source dataset
**Decision:** choose and freeze a meaningful general-purpose text corpus.  
**Selected choice:** Hugging Face `Salesforce/wikitext`, config `wikitext-103-raw-v1`, pinned to Hub commit `b08601e04326c79dfdd32d625aee71d232d685c3`.  
**Why:** Real Wikipedia prose is more linguistically complex and presentation-relevant than simplified synthetic corpora. Pinning the immutable upstream revision prevents moving-source drift. Local `datasets` fingerprints are treated only as cache-state diagnostics, not upstream version IDs.  
**Alternatives considered:** moving Hub `main`; local fingerprint as sole identity; TinyStories; other corpora.  
**Evidence:** 1,801,350 train rows, 3,760 validation rows, 4,358 test rows; 28,472 structurally reconstructed train documents and 60 validation documents.  
**Presentation relevance:** realistic-text tradeoff and upstream provenance vs local cache fingerprints.

## D-051 — WikiText normalization policy
**Decision:** define one auditable cleanup policy before tokenizer training/evaluation.  
**Selected choice:** one explicit, unit-tested `normalize_wikitext_text` function for train/validation/later test. Restore `@-@`, `@,@`, `@.@`; repair common punctuation/bracket spacing, apostrophe suffixes, currency, clock-time and numeric en-dash spacing; pair spaced double quotes deterministically. Detect generic heading rows before prose normalization, normalize only the inner title, then reconstruct equals framing at the original level. Preserve case, Unicode, row order, and article structure. Ambiguous spaced single quotes/bare plural possessives remain documented residue.  
**Why:** The tokenizer should learn natural punctuation, but normalization must not destroy the structural markup required to reconstruct articles. The same transformation must later apply to held-out data for comparable perplexity.  
**Alternatives considered:** normalize headings as prose; leave raw WikiText artifacts; placeholder-only cleanup; broad single-quote regex; split-specific cleanup.  
**Evidence:** 17/17 normalization tests passed; zero heading-level classification changes across train/validation; all 60 validation titles restored; punctuation-leading headings and numeric en-dash ranges handled correctly. Notebook 02 independently reran the same tests through pinned `src/data.py`.  
**Presentation relevance:** strong example of why text cleanup and structural parsing must be separated.

## Notebook 01 handoff
Notebook 01 freezes the data/normalization/reconstruction contract. Notebook 02 consumes that contract to train the tokenizer and construct the exact model-training corpus. Article sampling/corpus-manifest decisions that depend on the trained tokenizer are therefore recorded in the Notebook 02 register rather than duplicated here.
