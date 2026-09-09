# Executive Narrative

Draft 1 | Prepared 2026-09-09 | STE style | Evidence contract D-101

## The question

Many small language-model projects show that a model can learn. This project asked a more useful question.

How does model capacity affect predictive quality and resource cost when three models use the same corpus and training-exposure budget?

The experiment compared three small decoder-only Transformers with about 7 million, 17 million, and 34 million parameters. Each model started from random weights.

## What I built

I built the model architecture and training loop with PyTorch. I also trained a 16,384-token byte-level BPE tokenizer from scratch.

The architecture used causal attention, RoPE, RMSNorm, SwiGLU, pre-norm residual blocks, and tied input and output weights.

Hugging Face Datasets and Tokenizers supplied the data and tokenizer tools. I did not use pretrained weights, a pretrained tokenizer, or HF Trainer.

The source data came from a fixed revision of WikiText-103. The data process normalized the text and reconstructed 28,472 training articles.

The tokenizer used the full normalized training split. The model corpus contained exactly 20,000,000 tokens from that split.

Each model used a 512-token context and the same causal packing rule. Each model processed 59,999,232 target exposures during three epochs.

The official validation split controlled checkpoint selection. The official test split remained sealed until the original experiment froze all training and selection decisions.

## What the original comparison found

Predictive quality improved at every tested size. Validation perplexity decreased from 53.09 to 43.66 to 39.83 across Models A, B, and C.

The final test results kept the same order. Test perplexity decreased from 52.21 to 43.47 to 39.67.

Resource cost also increased. Recorded training time increased from 11.78 to 23.55 to 42.22 minutes on the same Tesla T4 hardware.

Peak GPU allocation increased from 4.94 to 7.29 to 10.62 GiB.

Model C produced the best absolute likelihood. However, the B-to-C step produced less validation-loss improvement per added parameter, minute, and GiB.

For validation loss, B-to-C kept only 26.9 percent of the A-to-B efficiency per added parameter. The comparable time figure was 29.6 percent.

The comparable peak-memory figure was 33.2 percent.

These intervals show a local pattern, not a law for larger models. Published research often represents language-model loss with power laws, not linear relationships.

The rate depends on model size, data, compute, optimization, and architecture. Extrapolation needs more sizes and balanced budgets.

## What the Model C extension clarified

Model C ended the original budget at its best validation point. The project then examined its training-duration limit.

A separate experiment resumed Model C from the exact update-3,663 state. The resume gate examined the model, optimizer, scaler, counters, data order, and recorded hashes.

The extension used the same corpus and a constant learning rate of 2e-4. Validation loss selected the best checkpoint at update 12,210.

Validation loss decreased from 3.684501 to 3.599946. The precommitted exploratory test then measured only the selected extension checkpoint.

Test perplexity decreased from 39.67 to 36.85.

Model C was training-duration constrained under the specified continuation policy. The experiment does not isolate the value of more unique data or another learning-rate policy.

The run later stopped at update 13,263 under the fixed patience rule. The final pattern showed continued improvement and then a noisy validation plateau.

## What the generated text showed

The fixed generation probes did not produce a stable quality order across Models A, B, and C.

Lower perplexity did not make every sampled continuation better. Repetition, topic drift, invented entities, and factual errors remained visible.

Extended Model C improved topical continuity in two fixed prompts. The third prompt still fabricated biographical and achievement details.

This small probe does not measure factual accuracy. Likelihood, sampled text quality, and factual reliability remain different evaluation questions.

## Why the process matters

The project recorded 100 technical and experimental decisions before publication work started. Each major decision includes its rationale, alternatives, and later evidence.

A controlled learning-rate probe changed the provisional choice from 3e-4 to 2e-3.

A fail-closed evidence gate also found missing repository outputs after Notebook 05. The recovery process reproduced the frozen results and recorded the test re-measurement.

The Model C checkpoint then had to reproduce its recorded validation loss before one more update.

I directed the experiment, selected its scope, approved the material decisions, ran the Colab work, and examined the evidence. AI tools helped draft code, documentation, and analysis.

The repository records the implementation, decisions, tests, metrics, figures, and provenance. Readers can examine the evidence behind each public claim.

## The conclusion

Across these three tested models, more capacity improved predictive quality and reduced marginal resource efficiency. The separate extension also exposed a duration constraint.

More training improved Model C before the validation plateau. Better likelihood still did not produce reliable factual text or uniformly better sampled text.

The project therefore shows more than a successful training run. It shows how controlled decisions, evidence boundaries, and explicit limits make an AI experiment credible.

## Evidence links

- [Publication claim-to-evidence map](CLAIM_EVIDENCE_MAP.md)
- [Project context](../PROJECT_CONTEXT.md)
- [Decision index](../DECISION_INDEX.md)
- [Notebook 05 decision register](../decisions/05_evaluation_and_scaling.md)
- [Notebook 06A decision register](../decisions/06a_model_c_extended_training_probe.md)
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
- [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264)
