# Executive Narrative

Draft 2 | Prepared 2026-09-09 | STE style | Evidence contract D-101

## 1. The question

Many small language-model projects show that a model can learn. This project asked a more useful question.

How does model capacity change learning and computational cost when three models use one corpus and one training-exposure budget?

The experiment compared decoder-only Transformers with about 7 million, 17 million, and 34 million parameters. Each model started from random weights.

The models used the same 20,000,000-token corpus, tokenizer, 512-token context, packing rule, and three-epoch protocol. Each model processed 59,999,232 target exposures.

## 2. What we made

I built three small decoder-only Transformers, a 16,384-token byte-level BPE tokenizer, and an explicit PyTorch training pipeline.

The architecture used causal attention, RoPE, RMSNorm, SwiGLU, pre-norm residual blocks, and tied input and output weights.

The project used PyTorch primitives and Hugging Face Datasets and Tokenizers. It used no Trainer, pretrained tokenizer, or pretrained weights.

This choice exposed the main training mechanics instead of placing them behind a high-level training framework.

The source data came from a fixed revision of WikiText-103. The process normalized the text and reconstructed 28,472 training articles.

The official validation split controlled checkpoint selection. The official test split remained sealed until the original experiment froze all training and selection decisions.

## 3. What we found

Predictive quality improved at every tested size. Validation perplexity decreased from 53.09 to 43.66 to 39.83 across Models A, B, and C.

Test perplexity decreased from 52.21 to 43.47 to 39.67. Recorded training time increased from 11.78 to 23.55 to 42.22 minutes.

Peak GPU allocation increased from 4.94 to 7.29 to 10.62 GiB on the same Tesla T4 hardware.

The first finding was diminishing marginal efficiency. Across the two approximate doubling intervals, the second increase delivered a smaller validation-loss gain.

For B-to-C, efficiency retained only 26.9 percent per added parameter, 29.6 percent per minute, and 33.2 percent per GiB.

The second finding was that all three models still improved at the end of the budget. Data exposure and training duration constrained learning, not capacity.

The third finding was that the model with the best perplexity did not produce the best text in every fixed generation probe.

These results show a local pattern, not a law for larger models. Language-model loss often follows a power law instead of a linear relationship.

The rate depends on model size, data, compute, optimization, and architecture. Extrapolation requires more sizes, multiple seeds, and balanced budgets.

## 4. What Notebook 06A showed

Notebook 06A resumed Model C from update 3,663 and continued through update 13,263 under a fixed experimental contract.

The exact-resume gate examined the model, optimizer, scaler, counters, data order, and recorded hashes before training continued.

The best checkpoint occurred at update 12,210. Validation loss decreased from 3.684501 to 3.599946.

The precommitted one-time test produced a loss of 3.606928 and perplexity of 36.85. Frozen Model C had 3.680554 and 39.67.

Model C improved with more training, then reached a validation plateau, and the precommitted stop rule ended the run.

D-099 and D-100 classify the result as continued improvement followed by saturation / noisy validation plateau.

The extension used the same corpus and a constant learning rate of 2e-4. It does not isolate unique-data value or another learning-rate policy.

## 5. What the models can and cannot do

The models produce text continuations that read like passages from a Wikipedia article. They do not answer questions because they did not receive instruction training.

## 6. What the work shows

The work demonstrates technical judgment through explicit architecture, data, optimization, evaluation, and stopping decisions.

It demonstrates controlled experimentation through shared budgets, fixed evaluation streams, sealed test use, deterministic resume checks, and precommitted selection rules.

It also demonstrates traceable evidence. The repository connects each major claim to decisions, metrics, figures, tests, and provenance records.

A controlled learning-rate probe changed the provisional choice from 3e-4 to 2e-3. Evidence changed the implementation before the primary runs.

A fail-closed evidence gate later found missing repository outputs. The recovery process reproduced the frozen results and disclosed the test re-measurement.

The project recorded 100 technical and experimental decisions before publication work started. Each major decision includes its rationale, alternatives, and evidence.

I directed the experiment, selected its scope, approved the material decisions, ran the Colab work, and examined the evidence. AI tools helped draft code, documentation, and analysis.

The evidence supports a bounded conclusion. More capacity improved likelihood within the tested range, but marginal resource efficiency declined.

More training improved Model C before saturation. Better likelihood still did not produce reliable factual text or uniformly better sampled text.

## Evidence links

- [Publication claim-to-evidence map](CLAIM_EVIDENCE_MAP.md)
- [Project context](../PROJECT_CONTEXT.md)
- [Decision index](../DECISION_INDEX.md)
- [Notebook 05 decision register](../decisions/05_evaluation_and_scaling.md)
- [Notebook 06A decision register](../decisions/06a_model_c_extended_training_probe.md)
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
- [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264)
