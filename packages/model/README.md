# Model artifacts

`feature-spec.json` freezes deterministic feature serialization. Shared Python-generated features are in `fixtures/features.json`; TypeScript parity tests verify the contract.

There is **no released semantic model**. `experimental/` contains an explicitly unvalidated int8 exporter artifact with a null validated cutoff. The default library does not load it. The website explicitly loads it using `runtime.ts`, verifies its manifest/hash, and runs CPU inference to expose the experimental model's real raw cosine scores. No validated cutoff is applied. Float experimental checkpoints and real development evidence are in `runs/` and `eval/feasibility.json`. See `docs/model-card.md` for limitations and the stop decision.
