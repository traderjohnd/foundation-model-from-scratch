# Decision Register — 02 Tokenizer Training & Corpus Construction

Notebook 02 consumes the frozen Notebook 01 text contract, trains the project tokenizer from scratch, and constructs the exact controlled 20M-token model-training corpus.

---

## D-007 — Controlled training-token budget
**Selected choice:** exactly **20,000,000 tokenizer-produced training tokens**.  
**Why:** Keeps compute affordable while intentionally creating a regime where larger models may become data/compute constrained, which directly supports the diminishing-returns research question.  
**Alternatives considered:** all ~103M training tokens; 10M; 50M.  
**Evidence:** final corpus contains 19,995,397 text tokens + 4,603 boundaries = exactly 20,000,000.  
**Presentation relevance:** model size vs token budget vs compute-optimality.

## D-008 — Hold dataset constant across models
**Selected choice:** Models A/B/C use the identical controlled 20M-token corpus.  
**Why:** If data volume/content changed with model size, attribution of performance differences to capacity would weaken.  
**Evidence:** one canonical corpus stream/checksum is used by all three models.  
**Presentation relevance:** controlled-variable experimental design.

## D-009 — Tokenizer ownership
**Selected choice:** train the tokenizer from scratch.  
**Why:** The project should reproduce the complete pretraining pipeline instead of inheriting a pretrained model's linguistic interface.  
**Alternatives considered:** reuse GPT/BERT/other open-model tokenizers.  
**Presentation relevance:** makes the pipeline genuinely start from raw text.

## D-010 — Tokenizer algorithm
**Selected choice:** **byte-level BPE**.  
**Why:** Efficient subword compression plus robust coverage of arbitrary byte sequences; representative of GPT-style tokenization.  
**Alternatives considered:** word-level, character-level, WordPiece, Unigram/SentencePiece, conventional BPE without byte fallback.  
**Evidence:** all 256 byte symbols present; 16,127 learned merges.  
**Presentation relevance:** direct comparison with BERT/WordPiece and classic NLP tokenization.

## D-011 — Vocabulary size
**Selected choice:** **16,384 total tokens**.  
**Why:** Balances embedding/output parameter cost, sequence compression, and compute. A 32K+ vocabulary would consume too much of the smallest model's parameter budget.  
**Alternatives considered:** 8K, 32K, 50K+, 100K+.  
**Evidence:** production tokenizer achieved exactly 16,384 entries including the sole registered special token.  
**Presentation relevance:** larger vocabularies trade larger embedding matrices for potentially shorter sequences.

## D-052 — Tokenizer training corpus
**Selected choice:** train the 16,384-token BPE on the **entire normalized official training split only**; validation/test are excluded. Language-model pretraining still uses the fixed 20M-token subset.  
**Why:** The tokenizer must exist before exact tokenizer-produced article lengths can define the 20M subset, so training the tokenizer on that eventual subset would be circular. Full-train exposure improves vocabulary coverage without held-out leakage.  
**Alternatives considered:** train on eventual 20M subset; include validation; use pretrained tokenizer.  
**Evidence:** trained on all 28,472 normalized/reconstructed training articles with `tokenizers==0.23.1`; canonical tokenizer SHA-256 `6ec601a267cec7c843df47927f53c4dd108c85a1d059318aeec4442c7274604f`.  
**Presentation relevance:** separates tokenizer-vocabulary learning from the controlled model-training compute budget.

## D-053 — Document-boundary token accounting
**Selected choice:** append one boundary token after each complete selected article and count it **inside** the exact 20M budget. If the final article is truncated before its boundary, omit that boundary.  
**Why:** “20M tokenizer-produced tokens” should describe the exact sequence consumed by every model, including structural tokens.  
**Alternatives considered:** 20M lexical tokens plus boundaries outside budget; exclude special tokens from accounting.  
**Evidence:** 4,603 boundaries; 4,604 selected records; final article truncated after 1,312 of 4,410 text tokens and therefore has no boundary.  
**Presentation relevance:** makes the training-token claim exact.

## D-054 — Article-boundary reconstruction and sampling manifest
**Selected choice:** identify level-1 article starts from raw rows using strict blank-line padding around headings; normalize content after structural detection. Hard-assert 28,472 train / 60 validation reconstructed articles. Immediately before sampling, create `np.random.default_rng(42)`, deterministically permute verified training-article indices, encode whole articles, append boundaries, and truncate only the final selected sequence to hit exactly 20M tokens. Persist an ordered provenance manifest.  
**Why:** Raw structural detection prevents cleanup from altering boundaries; strict blank padding avoids promoting equals-shaped table/equation/reference rows; a deterministic permutation and manifest make corpus membership replayable.  
**Alternatives considered:** shape-only headings; heuristic exceptions; looser padding; unrecorded/random sampling.  
**Evidence:** shape-only matching found 29,444 training candidates; strict rule removed 969 internal table/equation fragments and yielded 28,472 credible train documents. Seed-42 sampling selected 4,604 records. Article-permutation SHA-256 `d4e368c0c22c1ea044133f7648466201450e66dc170da8ba67235fc1cd3b836c`; manifest SHA-256 `4a00196b39311a6c2e2790780e8fc43316f24a014d3d3649028b10a671f8d3fe`.  
**Presentation relevance:** compact audit story: normalization-damaged headings, shape-only false positives, published-summary vs released-corpus discrepancy.

## D-055 — Tokenizer special-token contract
**Selected choice:** exactly one registered special token: `<|endoftext|>` as boundary/EOS, reserved within the 16,384 vocabulary and assigned ID **0**. No PAD, BOS, or UNK. Ordinary encoding does not automatically insert the boundary.  
**Why:** One explicit boundary signal is sufficient for contiguous causal training; byte-level coverage removes the need for UNK; explicit insertion keeps corpus accounting auditable.  
**Alternatives considered:** separate BOS/EOS; PAD; UNK; automatic EOS post-processing; another boundary name.  
**Evidence:** zero literal boundary-token collisions in train/visible validation; production tokenizer has ID0 boundary, no padding/truncation/post-processor, and recognizes explicit boundary text atomically.  
**Presentation relevance:** familiar GPT-style special-token naming does not imply inherited pretrained IDs/weights/vocabulary.

## D-056 — BPE `min_frequency`
**Selected choice:** `min_frequency=2`. Operationally, stop merge training when the best remaining pair occurs fewer than 2 times.  
**Why:** Smallest threshold that rejects singleton pair evidence from consuming vocabulary capacity while still permitting common structure to fill the vocabulary.  
**Alternatives considered:** 1; larger thresholds; no explicit threshold.  
**Evidence:** vocabulary filled completely with 16,127 learned merges; repeated same-environment training produced identical compact serialization.  
**Presentation relevance:** vocab size = allowed compression capacity; min frequency = evidence required to earn capacity.

## D-057 — Corpus artifact persistence and reproducibility
**Selected choice:** materialize the 20M-token stream as raw little-endian `uint16`, hash the exact bytes, but do **not** commit the 40MB binary. Commit compact provenance (`corpus_summary.json`, `corpus_manifest.jsonl`) sufficient to regenerate it.  
**Why:** Every token ID fits in uint16, and the stream is exactly reproducible from pinned data, preprocessing, tokenizer checksum, seed, and manifest. The large derived binary adds repository weight but no unique information.  
**Alternatives considered:** commit 40MB uint16 binary; 80MB int32; checksum without manifest; compressed container as canonical identity.  
**Evidence:** raw bytes = 40,000,000; stream SHA-256 `4101d5b18c38558a58110f54a161763186ab5318111366486ebbfa0a3fe584fa`; manifest replay reproduced the identical stream/checksum.  
**Presentation relevance:** distinction between canonical provenance and large generated artifacts.

## D-058 — 512-token input/target packing edge policy — handoff decision
**Decision at the time:** Notebook 02 should preserve the complete 20M-token corpus and **defer** the exact mapping to 512-token causal examples to Notebook 04. `20,000,000 mod 512 = 256` alone does not determine the number of dropped prediction targets because causal labels are shifted.  
**Why:** Corpus membership and training-example packing are separate contracts. Conflating them would make the “20M-token corpus” claim ambiguous.  
**Alternatives left for Notebook 04:** drop incomplete block; shorter masked final example; overlapping/shift-aware scheme; another explicit deterministic rule.  
**Historical outcome:** resolved by **D-065**, which selected the final stride-512 full-example packing rule and established that 255 potential next-token targets, not 256, are omitted.  
**Presentation relevance:** example of how a small indexing detail changes the true number of optimization targets.

## Notebook 02 closure
Canonical results: tokenizer vocab 16,384; 256/256 byte alphabet; 16,127 merges; `<|endoftext|>` ID0; exact 20M corpus; 4,604 selected records; 4,603 boundaries; tokenizer/corpus/manifest fingerprints frozen; official test text uninspected/unencoded.
