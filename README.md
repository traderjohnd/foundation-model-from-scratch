# foundation-model-from-scratch

Building a Foundation Model from Scratch — a controlled small-model scaling experiment implemented explicitly in PyTorch.

## Documentation

Start here:

- [`docs/PROJECT_CONTEXT.md`](docs/PROJECT_CONTEXT.md) — current project state and next step
- [`docs/DECISION_INDEX.md`](docs/DECISION_INDEX.md) — chronological, project-wide decision index
- [`docs/decisions/`](docs/decisions/) — detailed decision register for each project phase/notebook

### Decision-record convention

Decision IDs are globally unique across the entire project and never restart at notebook boundaries. The decision record is append-only: if later evidence changes or refines an earlier choice, a new decision ID explicitly links to the earlier decision rather than rewriting history.

The detailed registers preserve **why** each decision was made, alternatives considered, evidence, and presentation relevance so the engineering history can feed directly into the final report/presentation.
