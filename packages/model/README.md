# Model artifacts

`feature-spec.json` freezes deterministic feature serialization. Shared Python-generated features are in `fixtures/features.json`; TypeScript parity tests verify the contract.

There is **no released semantic model**. `experimental/` contains an explicitly unvalidated int8 exporter artifact with a null validated cutoff. The default library does not load it. The initial website loaded it using `runtime.ts`, verifies its manifest/hash, and runs CPU inference to expose the experimental model's real raw cosine scores. No validated cutoff is applied. Float experimental checkpoints and real development evidence are in `runs/` and `eval/feasibility.json`. See `docs/model-card.md` for limitations and the stop decision.

Current demo artifact: `candidate/`, model `expanded-more-data-pooled-seed17`. It improves the weak-label development benchmark and preserves 32KiB int8 storage, but has no independently validated cutoff. The original `experimental/` artifact is preserved for regression comparisons. Training source attribution/licenses: CLINC150 (CC BY 3.0), BANKING77 (CC BY 4.0), VS Code (Microsoft, MIT); see `docs/data-sources.md`.
