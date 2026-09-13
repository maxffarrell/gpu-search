# Implementation decisions

- 2026-09-12: The attached v1.1 specification is the design input. The user separately requested public GitHub publication and a Vercel Hobby demo. No npm publication, paid teacher calls, or paid hosting features are authorized or used.
- Preserve the normative API names. Until a semantic model meets release gates, the main entry returns `backend: 'lexical', degraded: true` when semantics are requested. The lexical entry/default demo intentionally disables semantics and reports no degradation. Context is validated but has no lexical effect.
- Keep normalized label equality above alias equality. Lexical subclass precedence is independent of score. Ordered token prefix scores are capped at one: token separators can otherwise make the numerator larger than the original field length (for example `foo-bar` and `fooBar`).
- Freeze feature serialization before synthetic generation; see `packages/model/feature-spec.json`. ASCII case rules and explicit Unicode whitespace prevent Python casefold divergence.
- Synthetic data is an experiment, not reviewed relevance ground truth. A synthetic positive result cannot satisfy the required unseen-product quality gate. The sealed synthetic set is not used to select a model; independent reviewed final evaluation remains unavailable.
- Do not build or advertise a GPU inference backend before semantic feasibility. `auto` remains lexical in this baseline release. The project name describes the research direction, not a claim of shipped acceleration.
- Ship a static Vite demo without telemetry, remote fonts, server functions, databases, or inference endpoints. All query processing remains in the browser. Run local browser validation before deployment.

## Interactive model demo revision

The user explicitly requested a real model-backed demo despite the failed release gate. The demo now evaluates the existing experimental pooled16 int8 weights on CPU and displays raw cosine rankings alongside the unchanged lexical library. It does not impose a made-up validated cutoff or present model scores as confidence. Initial candidates have no aliases or contexts. Arbitrary edited candidates use the same encoder and composition, and model load failure is shown as an error rather than replaced with baseline results. The stable library still defaults to explicit lexical degradation when semantics are requested. No WebGPU claim is made.

## 2026-09-13: expanded-data experiment

Freeze source revisions and split hashes before training. Add CLINC150/BANKING77 intents and VS Code setting descriptions with explicit source licenses and weak-label provenance. Keep original dev out of training/checkpoint selection. Compare three seeds against repeated old-data controls with equal optimizer updates at epoch1/10. Calibrate abstention on separate calibration rows. Among development-selected checkpoints, choose seed17 (epoch10) because it is the only one meeting the pre-existing 5% no-match gate; reject higher-nDCG checkpoints that exceed that budget. Preserve pilot artifacts, deploy the improved experimental int8 artifact, and retain raw-score/limited-generalization disclosure. Quality regression thresholds freeze measured development floors; they are not independent test evidence.
