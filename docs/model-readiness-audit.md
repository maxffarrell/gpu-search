# Model readiness audit

The evidence does not support a production-ready or best-in-class learned search claim. The current lexical engine is useful, and the learned models are small, but the measured task mostly rewards recognizing long utterances for previously seen intent labels. The intended product is short-query search over arbitrary software menus. More optimization on the existing development set cannot establish that capability.

This audit is read-only. No dataset, weights, cutoff, or runtime changed. The previously opened 40-row synthetic test is now explicitly a consumed regression set. Its examples were inspected to diagnose failures, so it cannot be reused as independent final evidence. The primary data are in `eval/readiness-audit.json`; the Node CPU diagnostic is in `eval/readiness-cpu.json`.

Follow-up: the caching gap identified in section 4 has since been addressed by an experimental prepared-model index. [Prepared-index documentation](prepared-model-index.md) records its API, browser measurements, and unchanged weights. The audit measurements below describe the original uncached reference path; the quality findings and missing integrated semantic-search evidence remain applicable.

## 1. The failure is representation shift as well as abstention

| Measurement | Expanded development | Consumed software-menu regression |
| --- | ---: | ---: |
| Median query length | 9 tokens | 2 tokens |
| Candidate count | 10 | 6 |
| Positive rows whose target label appeared in training | 876 / 1,120 (78.2%) | 0 / 30 |
| Query-token occurrences absent from training vocabulary | 804 / 14,469 (5.6%) | 42 / 80 (52.5%) |
| Bigram model median relevant cosine, semantic queries | 0.6751 | -0.0336 |
| Bigram raw top-one correct, semantic queries | 828 / 1,118 | 0 / 18 |

The rejected bigram model's frozen expanded-calibration cutoff is 0.6562. Its maximum relevant score on the consumed regression semantic slice is only 0.3286; every relevant candidate is below cutoff. The current deployed model similarly places every relevant regression semantic candidate below its 0.7004 cutoff, with raw top-one correctness only 2/18.

Lowering the threshold cannot repair incorrect raw ordering: the rejected model is wrong at rank one for all 18 semantic queries. This is not evidence that no semantic signal exists—lower ranks give nonzero nDCG—but it disproves a cutoff-only explanation. Those 18 correlated examples cover one publishing menu with six target labels, so they also cannot quantify broad product performance. The actionable response is independently judged short-query menu data across several software products, not threshold tuning against these examples.

The same symptom appears in the original 80-row development transfer probe: short queries, zero positive target labels seen in training, and much lower relevant scores. These are direct observed distribution differences. They do not isolate which causal factor—query length, domain, unseen labels, or menu composition—dominates. A new study should cross these factors while keeping judgments fixed.

## 2. Calibration negatives need retrieval judgment, not class deletion

Every one of the 352 calibration no-match menus and 352 expanded-development no-match menus was constructed by removing the source's single positive intent. The remaining labels were assigned zero relevance without a human retrieval review.

A conservative lexical review flag—at least two shared label tokens and token-set Jaccard at least 0.5 between the removed label and a remaining label—identifies **106/352 calibration no-match menus (30.1%)**, **77/352 development menus (21.9%)**, and 161 training menus. Examples include:

- Removed **Schedule meeting**, remaining **Meeting schedule**, query “schedule my meeting with jim at 3pm.”
- Removed **Card not working**, remaining **Virtual card not working**, query “Please help. The card won't work.”

These flags are not proof that the remaining destination is relevant. They show why source intent classification does not determine whether an interface destination is useful. Some labels encode opposite actions; others may be ambiguous without context. Treating every alternative as a trusted negative can punish sensible retrieval and force a high abstention threshold.

There are no multi-positive menus in the training, calibration, or expanded-development data. There are also no exact-label contradictions within a menu. The problem is unmeasured semantic ambiguity, not a detected duplicate-ID or literal-label bug. Review flagged menus, permit multiple positive judgments, and leave genuinely unjudged candidates out of training loss and headline quality metrics. Do not automatically relabel the flags as positive or selectively remove difficult negatives after seeing model scores.

## 3. The byte budget creates collision pressure, but causality remains unproven

Across distinct training query/label features:

| Feature table | Distinct feature preimages | Occupied buckets | Buckets with multiple preimages | Largest bucket |
| --- | ---: | ---: | ---: | ---: |
| Words | 5,179 | 1,019 / 1,024 | 983 | 14 |
| Character grams | 15,710 | 1,024 / 1,024 | 1,024 | 27 |
| Shared words plus bigrams | 27,282 | 1,024 / 1,024 | 1,024 | 47 |

The bigram variant adds order information while forcing substantially more distinct features into the same word table. A collision does not by itself establish a retrieval error; subword pooling can tolerate collisions. These counts justify matched-byte bucket/dimension ablations and error slices for unseen tokens. They do not justify claiming that larger tables will solve the domain problem, nor increasing the 32,768-byte weight budget without authorization. Existing same-size collision experiments should be judged on independent software-menu data rather than this consumed regression set.

## 4. A production semantic index and its latency evidence are missing

At the time of the initial audit, the stable `createIndex` was lexical and the separate experimental `Model.score` recomputed label, alias, and context embeddings for every candidate on every query. The subsequent prepared API resolves candidate caching and per-index disposal. The stable `createIndex` still reports lexical degradation when semantics are requested; there is no integrated exact/lexical/semantic result API or validated semantic cutoff. The demo continues to expose raw scores.

A direct local Node diagnostic of the current trained CPU runtime on Apple M4 Max, Node 26.8.1, macOS 27.0.0 measured:

| Candidates, each with an alias and context | Raw semantic scoring p95 |
| --- | ---: |
| 10 | 0.485 ms |
| 100 | 4.403 ms |
| 1,000 | 43.987 ms |

The same candidate list was reused, with 30 warmups and 200 measured queries per size. Power state was uncontrolled. This is **Node diagnostic evidence**, not browser proof or a full hybrid-search benchmark; it excludes final lexical ranking. Even this partial operation is about 5.5 times the proposed 8 ms target at 1,000 candidates. Cached immutable candidate vectors are an actionable improvement before GPU work. Existing browser benchmarks measure lexical search and cannot be cited as learned-runtime latency proof.

The research build fits the size target: the bigram artifact has 32,768 raw weight bytes and its complete separately compressed demo resources total 36,243 Brotli bytes. That is size evidence for the recorded research build, not a quality or live deployment guarantee. WebGPU remains unimplemented.

## New independent software-menu evaluation, before further training

Create a new versioned source manifest and register the protocol before training on any new extraction. The already consumed synthetic test and any inspected query/label families become regression material. No convenient relabeling of those records can make them fresh.

Candidate permissively licensed source projects include [Excalidraw (MIT)](https://github.com/excalidraw/excalidraw/blob/master/LICENSE), [JupyterLab (BSD)](https://github.com/jupyterlab/jupyterlab/blob/main/LICENSE), [Apache Airflow (Apache-2.0)](https://github.com/apache/airflow), and [Ghost (MIT)](https://github.com/TryGhost/Ghost/blob/main/LICENSE). License locations were checked for this proposal; exact commit SHAs, extracted paths, notice retention, and source-specific permissions must be recorded before extraction. These are proposals, not sources already ingested by this audit. Ghost's publishing domain overlaps the consumed regression concept family, so it should be development or an explicitly overlapping slice, not the headline independent holdout.

1. **Capture actual menus and commands.** Preserve candidate ID, displayed short label, application section, contextual description, and source path at a pinned revision. Exclude inaccessible destinations before constructing the menu. Use real competing actions from the same interface, including import/export and enable/disable, rather than randomly combining unrelated labels.
2. **Split by product and task family before authoring queries.** Keep all wording variants, aliases, and derived menus of a task together. Reserve at least two whole application families for final evaluation. Cross-check near-duplicate task descriptions and normalized query/token overlap with every prior dataset, including the consumed synthetic regression set. Generic shared words are unavoidable; report them rather than claiming perfect conceptual independence.
3. **Target the product's queries.** Proposed primary distribution: 70% one-to-four tokens, 25% five-to-eight, 5% nine-to-twelve. Include exact, typo, prefix/acronym, low-overlap semantic, ambiguous/multi-positive, opposite-action, and genuine no-match cases. Keep representative 10-, 50-, 250-, and 1,000-candidate settings; report menu-size calibration transfer rather than fitting it from only ten-candidate menus.
4. **Judge every candidate for each evaluation query.** Two reviewers independently assign grades 0–3, then resolve disagreements with application context. At least 15% of relevant queries should exercise legitimate multiple answers, and include at least 200 explicitly reviewed no-match menus. An annotation generated from a source class or a language model is a proposal until reviewed. Queries should describe user intent before showing the exact destination wording where feasible.
5. **Keep fit, calibration, development, and final roles distinct.** Fit weights and aliases only on training. Fit abstention only on calibration. Use development for the registered finite model search. Freeze artifact, feature contract, aliases, threshold, and source hashes before opening the final judgments once. A failed final set becomes regression material; another round needs a genuinely new independent set.

If no human judgment resource is available, a source-derived benchmark still helps engineering, but the production-quality gate remains unresolved. Do not disguise weak labels as a reviewed test to meet a schedule.

## Measurable readiness gates

The existing PRD efficacy and size gates should remain unchanged: at least +0.05 semantic nDCG@5 over the strongest fair alias/lexical baseline, paired interval excluding zero, no more than 0.01 overall regression, at most 5% semantic no-match returns with coverage reported, at most 0.01 quantization loss, weights no larger than the current 32,768 bytes, and complete CPU deployment at most 50 KiB Brotli. Compare nearest training paraphrase retrieval, train-fitted TF-IDF/BM25, and the pinned offline embedding reference on identical candidates and metadata. “Best in class” additionally needs a named comparison class, shared corpus, size accounting, and reproducible results; a development win alone is insufficient.

For the proposed new corpus, preregister additional anti-abstention and domain checks before scoring: for example at least 60% relevant semantic coverage overall, at least 50% in each sufficiently populated held-product slice, and explicit multi-positive recall. These coverage numbers are proposed prospective gates, not retroactive changes or measured successes. Report no-match confidence intervals and source-group bootstrap intervals; 0/10 observed errors is weak evidence, not a demonstrated universal 5% false-positive bound.

Before production, also require:

- One typed reusable semantic `SearchIndex` with cached candidate vectors, hard-tier invariants, reasons and raw score components, validation, immutable snapshots, abort, concurrency, and disposal; failure to load a model must explicitly degrade.
- Browser CPU end-to-end p95 at most 8 ms for 1,000 and 16 ms for 5,000 candidates on named hardware, using 30 warmups and 200 measured queries, including aliases/context, indexing cost, memory, and cold asset loading. Verify another browser and constrained device; do not infer this from Node or lexical-only timings.
- Actual exported-asset numerical parity and malformed-asset tests, query-privacy network audit, and separately compressed model/runtime/manifest size reports. WebGPU is optional until real end-to-end benefit is demonstrated; never claim acceleration without it.
- A documented supported domain, calibration limitations, source licenses, model/data hashes, and independently reviewed final evidence. Until these pass, retain the experimental label and the dependable lexical/alias path.

Reproduce the read-only diagnostics:

```sh
uv run python -m eval.readiness-audit
node --import tsx eval/readiness-cpu.ts
```

These commands intentionally inspect the **already consumed** synthetic regression data. They do not train weights, select a new model, alter data, or produce a new sealed-test claim.
