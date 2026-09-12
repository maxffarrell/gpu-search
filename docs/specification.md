# gpu search PRD and Technical Specification

Version 1.1 • 12 September 2026 • Owner Max Farrell • Audience lead implementation agent and delegated engineering agents

## 1 Executive decision

Build a tiny, local, English-first search library for software interfaces. It must rank normalized exact matches first, strong lexical matches second, and learned semantic matches third. Train the semantic encoder with MLX on Apple Silicon. Deploy a custom TypeScript CPU implementation and optional specialized WebGPU kernels, with no general-purpose inference runtime in the browser.

Project name: `gpu-search`. Package-name availability is unverified. GPU acceleration is an implementation option, not the product's value proposition. The product is predictable, private retrieval of software concepts in a small download.

The lead agent should execute this specification, delegate independent work after fixing interfaces, integrate the results, and deliver a reproducible repository and evidence. This document authorizes implementation work; it does not authorize paid teacher API calls, publishing packages, or deploying public services. Use local or already authorized resources. Do not represent unrun training or benchmarks as completed.

**Central hypothesis:** a small domain-trained encoder can improve retrieval for unseen software interfaces enough to justify its bytes over lexical matching plus a curated alias dictionary. This is unproven. Prove it before optimizing shaders.

## 2 Product requirements

### Problem and intended users

Command palettes, settings search, navigation menus, and local action pickers often require users to know the application's exact terminology. A user may type “coworkers” when the destination is “Members,” or “my information” when it is “Profile.” Conventional spelling similarity cannot reliably bridge these expressions. General embedding runtimes may be disproportionate for a small menu.

The primary customer is a frontend developer supplying a local candidate list. The end user searches that list without sending queries to a server. The first release supports short English queries and labels, with optional developer-provided aliases and context.

### Required behavior

| ID | Requirement | Acceptance evidence |
| --- | --- | --- |
| P1 | Exact label matches always precede eligible lexical matches; lexical matches always precede semantic matches | Comparator and adversarial ordering suite has zero violations |
| P2 | Correct common typos without requiring a model | Lexical-only benchmark and short-query safeguards |
| P3 | Retrieve related UI concepts with little character overlap | Held-out semantic evaluation against all baselines |
| P4 | Accept arbitrary candidate strings at runtime | No fixed concept ID or canonical-label inventory required |
| P5 | Run locally after assets load | Network audit finds no query transmission or telemetry |
| P6 | Work without WebGPU | CPU path returns the same tier decisions and numerically equivalent semantic results |
| P7 | Explain returned matches | Typed match reason, matched field, and separate score components |
| P8 | Avoid filling the list with unrelated results | Validated semantic cutoff and explicit no-match coverage |
| P9 | Support repeated interactive queries | Reusable immutable index, cached candidate embeddings, safe concurrent calls |
| P10 | Remain genuinely small | Reproducible complete transfer-size report, including model and optional runtime assets |

Initial scope is 10–5,000 candidates. Benchmark 50,000 as a stretch workload. Candidate labels are typically 1–8 words and queries 1–12 words. Support longer bounded inputs mechanically, but do not imply general document retrieval quality.

### Examples and ambiguity

| Query | Candidates | Required relationship |
| --- | --- | --- |
| `profile` | Profile, Profiles, Account, Billing | Profile exact first; Profiles eligible lexical above Account semantic |
| `profle` | Profile, Account, Billing | Profile eligible typo above semantic results |
| `coworkers` | Members, Invoices, API Keys | Members relevant without relying on character overlap |
| `my information` | Profile, Personal Information, Billing | Multiple relevant answers allowed; precise order is evaluated |
| `plan` | Plan, Subscription, Billing | Exact Plan always first |
| `account` | Personal Account, Customer Accounts | Context-sensitive; do not declare either universally correct |
| `delete account` | Create Account, Delete Account | Exact destination first; opposite actions are difficult negatives |
| empty or whitespace | Any list | Return an empty result list |
| `xqzv` | Ordinary settings menu | Prefer no results over indiscriminate semantic matches |

“Profile ≈ account,” “organization ≈ workspace,” and “roles ≈ permissions” are context-dependent relationships, not universal equivalence classes. A model must distinguish “my profile” from “customer account” where supplied context permits it. Retrieval never executes actions or grants access. Applications must filter unauthorized candidates before indexing.

### Non-goals

No generation, chatbot, autonomous navigation, document search, multilingual quality guarantee, automatic personalization, cloud inference, online learning, ANN dependency, or fixed database of all SaaS concepts. A Transformer is an experimental alternative only if simpler models fail a documented capability requirement. No universal semantic confidence probability is claimed.

## 3 Success criteria and decision gates

All numbers below are proposed acceptance targets, not measured results. Register them before opening the final test set. Changes require a recorded decision, not silent threshold adjustment.

| Dimension | Initial release gate |
| --- | --- |
| Ordering | 100% exact and lexical tier invariants |
| Semantic improvement | At least +0.05 absolute nDCG@5 over the strongest lexical-plus-alias baseline on semantic-only held-out queries; paired bootstrap 95% interval for improvement excludes zero |
| Overall usefulness | No more than 0.01 absolute nDCG@5 regression on the complete held-out set |
| Typo quality | No degradation in lexical decisions relative to the same lexical engine alone |
| Abstention | At most 5% of no-match queries return any semantic result; report relevant-query coverage to prevent gaming by returning nothing |
| Quantization | At most 0.01 absolute nDCG@5 loss versus the selected float model |
| Download | Standard CPU semantic entry at most 50 KiB Brotli; CPU plus WebGPU total at most 64 KiB; lexical-only entry at most 8 KiB |
| Stretch download | Complete CPU plus WebGPU deployment at most 40 KiB Brotli |
| Warm latency | Reference Apple Silicon desktop: p95 at most 8 ms for 1,000 candidates and 16 ms for 5,000, full search including readback and ranking |
| Portability | Real Chromium WebGPU tested; another available browser/device tested; CPU fallback tested with unavailable and lost GPU |
| Reproducibility | Pinned dependencies, seeds, source manifests, dataset splits, export hashes, commands, and raw metrics |

Name the exact reference hardware and browser before collecting latency results. Cold initialization, indexing, shader compilation, and memory have mandatory reporting even though their first-release limits are not fixed here. A missed target must be reported. If no model beats the alias baseline, deliver that finding and the working baseline rather than advertise a successful semantic model.

## 4 Public API contract

Use ESM and TypeScript declarations. Keep lexical-only imports independent of model assets. The API below is normative at the behavioral level; final names may change once before implementation with a recorded interface decision.

```ts
type Backend = 'cpu' | 'webgpu' | 'auto';
type Candidate = {
  id: string;
  label: string;
  aliases?: readonly string[];
  context?: string; // e.g. "personal settings", not instructions
};
type Match = 'exact' | 'lexical' | 'semantic';
type SearchResult = {
  id: string;
  label: string;
  match: Match;
  reason: 'label-equality' | 'alias-equality' | 'prefix' |
          'token-prefix' | 'acronym' | 'typo' | 'concept';
  matchedField: 'label' | 'alias' | 'semantic';
  lexicalScore?: number; // [0, 1], meaningful only within lexical ranking
  semanticScore?: number; // raw cosine [-1, 1], not probability
};
type SearchResponse = {
  results: SearchResult[];
  backend: 'cpu' | 'webgpu' | 'lexical';
  degraded: boolean;
};
type SearchIndex = {
  search(query: string, options?: {
    limit?: number; signal?: AbortSignal;
  }): Promise<SearchResponse>;
  dispose(): void;
};
declare function createIndex(
  candidates: readonly Candidate[],
  options?: { backend?: Backend; semantic?: boolean }
): Promise<SearchIndex>;
```

Default limit is 10; limit zero returns no results; negative or noninteger limits throw. Duplicate IDs, empty labels, malformed records, and oversize inputs throw typed input errors. Duplicate labels with distinct IDs remain distinct and use insertion order as final tie breaker. Snapshot and validate inputs during construction. An empty candidate list is valid.

Default limits: 50,000 candidates; 256 Unicode scalar values per query or label; 8 aliases per candidate, each at most 128 scalars; context at most 256 scalars. Reject overflow rather than silently changing text. Document limits as configurable build constants, not training-performance guarantees.

Indexes are immutable in v1. Rebuild to change candidates. Abort must prevent stale responses from being delivered even when submitted GPU work cannot be cancelled. Queue GPU buffer use or provide per-call storage; concurrent searches must never overwrite one another. `dispose()` is idempotent; future searches reject. A lost GPU retries on CPU with the same weights and sets `degraded: true`. Failure to load valid model assets falls back to lexical with explicit degradation metadata. No unhandled async initialization races.

## 5 Deterministic ranking specification

### Normalization

Maintain the original text, a strict normalized key, and lexical/semantic tokens separately. Strict equality uses NFKC, ASCII A–Z lowercasing, Unicode whitespace collapse, and trim. Retain punctuation, digits, and diacritics. This English-first rule avoids Python versus JavaScript case-fold differences; expanding Unicode case behavior requires a feature-version change. Validate normalization with shared fixtures because Unicode versions can differ across runtimes.

For lexical and semantic tokens, split ASCII camelCase boundaries before lowercasing, then split on whitespace, underscore, and hyphen. Preserve meaningful symbols such as `+`, `#`, `@`, and digits. Exact comparison is never performed on the more aggressively split tokens. `C++`, `C#`, and `C` must remain distinct. Never stem or remove stopwords from the exact key.

### Eligibility and sort key

Tier 3 is normalized equality with the label only. Alias equality is tier 2, so a developer's alias cannot displace another candidate's literal label. Context is never an exact or fuzzy matching field.

Tier 2 uses these initial lexical subclasses, in descending precedence:

1. Alias equality.
2. Full normalized field prefix, with at least two query scalars.
3. Ordered token prefixes: each query token prefixes a distinct candidate token in order; every query token must be consumed, and the query has at least three non-space scalars.
4. Acronym equality over token initials for a 2–6 character query and at least two candidate tokens.
5. Strong typo match over the whole normalized label or alias.

For typo matching use optimal string alignment edit distance with adjacent transpositions, explicitly named OSA rather than claiming unrestricted Damerau–Levenshtein. Let L be the larger field/query scalar length. Require both lengths at least four, distance at most 1 when L is 4–7 or at most 2 when L is at least 8, and distance/L at most 0.25. Score is `1 - distance/L`. Do not use raw substring, unbounded subsequence, or n-gram similarity alone to elevate an item above semantic matches in v1. These create false lexical promotions for short queries.

For prefix subclasses use query non-space length divided by field non-space length as the score; acronym and alias equality score 1. Pick the best qualifying field/subclass per candidate, preferring label over alias on ties. Sort by descending tier, descending lexical subclass, descending lexical score, then insertion order. Exact ties use insertion order. Semantic-only candidates sort by descending cosine, then insertion order. Do not use a non-transitive epsilon comparator for near ties.

Tier 1 requires an available, nonzero embedding and cosine at or above a frozen model-specific cutoff. Below-cutoff candidates are omitted. For queries shorter than three scalars, disable learned semantic results in v1. Empty normalized queries return nothing. Only after all candidates have their best tier, apply top-k. Never prune to a lexical shortlist before semantic scoring; that would eliminate the intended synonym matches.

Semantic evidence must never promote a candidate into a lexical tier or change exact ordering. Scores across tiers are not comparable. Tighten lexical eligibility only through development evaluation and regression review; a falsely eligible lexical candidate will outrank a relevant semantic result by design.

## 6 Model and feature contract

### Architectural lineage and implementation decision

fastText informs the character-subword representation [S1]. StarSpace is the closer precedent for the retrieval task: representing entities through discrete features and learning task-dependent similarity in a shared embedding space, including information retrieval [S6]. Credit both; neither establishes that this project will meet its byte or quality targets.

The implementation is a custom MLX model inspired by these approaches. It does not require shipping or porting either original runtime. The proposed projections, multi-positive contrastive objective, optional teacher distillation, quantization, and deterministic ranking policy are project choices, not claims about the original StarSpace implementation. Verify original source and license before borrowing code.

Establish a StarSpace-style pooled shared-embedding baseline before adding nonlinear projections or distillation. Use the same hashed word/subword inputs, candidate composition, splits, and negative masks as the projected model; L2-normalize the pooled vector directly. First use the same contrastive loss to isolate the value of the projection. Then compare a sampled pairwise margin loss, `max(0, margin - sim(q,p) + sim(q,n))`, over judged positives and negatives, selecting margin on development. Label this an inspired baseline rather than an exact StarSpace reproduction. Reuse the pooled-model ablation already required below; do not create a duplicate workstream.

Select the simplest candidate that passes the quality and size gates. A projection or teacher is retained only with measured benefit. StarSpace is prior art and a baseline, not a new deployment dependency.

### Starting architecture

Use a shared encoder for query and candidate strings. This is a small embedding network, not a Transformer. Hashed features avoid a shipped vocabulary, but collisions remain and unknown terms are not automatically understood. Character subword representations are established prior art; fastText documents representation through substrings [S1].

| Tensor | Shape | Parameters |
| --- | --- | ---: |
| Word embeddings | 1024 × 16 | 16,384 |
| Character embeddings | 1024 × 16 | 16,384 |
| First projection weights and bias | 24 × 16 plus 24 | 408 |
| Second projection weights and bias | 16 × 24 plus 16 | 400 |
| Total | | 33,576 |

Start without shape embeddings. Add shape or ordered token-bigram features only if ablations justify them. A pooled bag cannot reliably capture all word order, negation, or action direction. Include these limitations in the model card; add hard negatives such as enable/disable and import/export. Do not claim bag features solve those cases by construction.

Word features are each lexical token's UTF-8 bytes, prefixed by `w:`. Character features are within-token 2-, 3-, and 4-grams over Unicode scalars, with boundary markers added around each token, and namespace `c:`. Boundary markers are integer sentinels outside the scalar range, serialized through a specified tagged encoding so literal user text cannot imitate them. Freeze exact serialization in `feature-spec.json` before creating data.

Use FNV-1a 32-bit with offset 2166136261 and multiplier 16777619, unsigned wrap after each byte, and bucket `hash & 1023`. TypeScript must use `Math.imul`; Python masks with `0xffffffff`. No language-native hash functions. Preserve repeated feature occurrences and their weights; do not silently deduplicate hash collisions. Pool each feature family separately, then average the two family means. Empty family means are zero. Padding is excluded from counts and gradients. Include all feature IDs and counts in golden fixtures.

Bound model work to the first 32 tokens, each token's first 64 scalars, and at most 512 character features in deterministic token-then-n-gram-length-then-position order. These encoder bounds do not change full-string exact or lexical comparisons. Report truncation frequency in data and benchmarks.

For row-vector x, tensor storage uses output-major row-major matrices:

```text
x = 0.5 * (mean(Eword[word_ids]) + mean(Echar[char_ids]))
h = tanh(x @ W1.T + b1)
z = h @ W2.T + b2
embedding = z / max(sqrt(sum(z*z)), 1e-8)
```

If the pre-normalization norm is below 1e-8, mark the embedding invalid for semantic retrieval. Accumulate f32 on GPU and the reference path. Use explicit f32 rounding where necessary in the CPU parity implementation; optimized JS arithmetic may use double intermediates only when it passes tolerance tests.

Candidate representation: encode the label, each alias, and context separately with the same encoder. Average label and alias vectors with equal weight, add context vector with weight 0.25 when present, and normalize. Skip invalid vectors. This is the initial policy, not a claim that averaging synonyms is optimal. Train and evaluate with this exact composition. Benchmark label-only and alias/context ablations. Cache one vector per candidate, including a fingerprint of content, feature version, composition version, and model hash.

### Architecture search

Compare pooled embeddings without projection, the network above, and a wider 24-dimensional model. Compare word-only and character-only variants. Start with three fixed random seeds, then run more only if uncertainty changes the decision. Measure hash bucket occupancy, collisions, and errors on unrelated brand names. Select on the quality/bytes/latency frontier, not minimum parameter count alone.

Raw storage arithmetic for the starting model is 33,576 bytes at int8 and 25,182 bytes at packed int6, excluding scales, metadata, alignment, and runtime. f32 weights occupy 134,304 bytes. At 5,000 candidates, 16-dimensional f32 candidate vectors occupy 320,000 bytes per copy. Browser memory and download size are different budgets. Brotli compression gains must be measured, not assumed.

## 7 Data specification

### Sources and limitations

Begin with a reviewed set of software navigation concepts and candidate menus from reusable public/open-source interfaces. Extract labels, headings, documented synonyms, paths, and task descriptions only where reuse is permitted. Public visibility alone is not a training-data license.

Mind2Web supplies web interaction task data [S2]; WorkArena evaluates knowledge work in ServiceNow [S3]. They are candidate sources of weak supervision, not ready-made short-query synonym datasets. An entire task description is not automatically a positive label for every clicked element. A task such as changing a billing address may involve navigation, search, and form submission. Retain a task-to-label pair only when the local action supports the association. Inspect source licenses and dependencies before ingestion. Do not assume WorkArena's code license covers every underlying asset.

Avoid relying on unavailable external corpora to bootstrap. Create a clearly labeled synthetic seed suite and a separately reviewed development set. Teacher-generated paraphrases may enlarge training data after filtering, but generated examples alone cannot validate usefulness.

Initial planning target: 15–30 product families, 200–500 concepts, 10,000–50,000 accepted query/menu records after augmentation. Quality and diversity matter more than hitting these counts. Aim for at least 1,000 evaluated queries with at least 200 semantic-only and 200 no-match examples. These are work targets; disclose actual counts.

### Record format

```json
{
  "id": "record-00001",
  "product_family": "example-suite",
  "source_id": "manifest-entry-001",
  "query": "coworkers",
  "candidates": [
    {"id": "members", "label": "Members", "context": "workspace access"},
    {"id": "billing", "label": "Billing"}
  ],
  "relevance": {"members": 3, "billing": 0},
  "judgment_status": "reviewed",
  "query_kind": "semantic",
  "augmentation_parent": null,
  "split_group": "example-suite-template-family-1"
}
```

Relevance grades: 3 directly satisfies intent; 2 useful destination; 1 related but insufficient; 0 irrelevant. Unjudged candidates are absent from the map and must not silently become negatives. Multiple positives are legal. No-match examples have all candidates judged zero. Add provenance for source URL, retrieval date, revision/hash, applicable license, extracted fields, transformation script version, teacher/model revision where used, and reviewer status. Keep sensitive user/account data out of distributed examples.

### Splits and negative mining

Split by product family and source/template lineage before augmentation. Keep aliases, paraphrases, typo children, and near-duplicate menus together. Hold out additional wording/concept families to test transfer beyond memorized vocabulary. Product-only splitting does not eliminate shared-phrase leakage; measure overlap and report it.

Use train, development, and sealed final-test sets. Mining operates on train only. Development selects architectures, thresholds, QAT choices, and stopping points. The final set is opened once for a selected release candidate. After repeated inspection it becomes a regression set; a fresh sealed set is needed for new generalization claims. Autonomous agents may not use final-test examples to generate training data.

Siblings are potential hard negatives, not automatic negatives: Members and Users can both be relevant. Mask known positives, aliases, and uncertain candidates from negative denominators. Include realistic no-answer menus, ambiguous account meanings, visually similar unrelated labels, action opposites, number-bearing labels, and plausible incorrect destinations. Oversample observed failures within train rather than accumulating easy random negatives.

## 8 MLX training and distillation

MLX is the training framework. Its official documentation provides automatic differentiation, array operations, saving/loading, and evaluation controls [S4]. Pin a tested version and Apple Silicon environment. Do not estimate training time from another project's epoch count. Teacher generation and data curation can dominate compute cost.

Implement deterministic feature fixtures in Python and TypeScript first. Cache prepared batches. Train the float reference with AdamW, recording learning rate, weight decay, batch size, temperature, clipping, seeds, and stopping rule. Initial search: learning rates 1e-3 and 3e-3, batch size 128 where feasible, temperature 0.07 and 0.15, up to 100 epochs with development patience 10. These are starting values, not mandatory exhaustive grid combinations. Materialize MLX lazy computations before stopping timers or saving results.

Use query-to-menu multi-positive contrastive loss. For each query, probability mass assigned to all judged relevant candidates is the numerator and all eligible judged candidates form the denominator. Exclude unjudged or conflicting negatives. Optionally weight positive grades. Do not mechanically use symmetric query/candidate loss: navigation relevance can be directional and many-to-many.

```text
Lcontrastive = -log(sum(exp(sim(q,p)/T), p in positives)
                    / sum(exp(sim(q,c)/T), c in eligible candidates))
```

Use numerically stable logsumexp. All-zero relevance menus do not use an empty positive numerator; use them for abstention calibration and, if justified, a separately specified margin loss. Avoid forcing an arbitrary absolute cosine target before measuring the distribution.

Train a supervised-only baseline before teacher use. If distillation is beneficial, select a pinned locally available embedding teacher or reranker and cache menu-level scores offline. Validate the teacher against reviewed UI examples first. Generic semantic similarity can confuse related but operationally different settings. Distill distributions over the same candidate menu with KL divergence and a development-selected weight; do not copy illustrative cosine values from the prior discussion as labels. Teacher scores are not ground truth or probabilities of correct navigation.

No teacher is shipped. Record model license, revision, prompts if applicable, temperatures, inference settings, and filtering rules. If access or budget blocks teacher inference, complete the supervised pipeline and document the missing experiment.

## 9 Quantization and export

Try post-training int8 first. Compare float, int8, then int6; int4 is optional. Add quantization-aware training only when it resolves a measured quality loss. Choose the last 10–20% of training steps as an initial QAT phase, with development-based selection rather than copying another project's final-epoch count.

Define one deployment-matched symmetric per-tensor scheme:

```text
Q = 2^(bits-1) - 1
scale = max(abs(W)) / Q, or 1 for an all-zero tensor
round_away(t) = sign(t) * floor(abs(t) + 0.5)
codes = clip(round_away(W / scale), -Q, Q)
Wq = codes * scale
Wfake = W + stop_gradient(Wq - W)
```

The straight-through estimator is necessary: bare round/clip is not a sufficient differentiable training recipe. Freeze scales from float master tensors for each forward evaluation, stopping gradient through the quantization path. Record scale f32 serialization. Include small bias tensors in the quantization scheme initially; retain f32 biases only as an explicitly measured format change.

Export a versioned manifest and packed binary payload. Required metadata: format version, feature version, normalization version, candidate-composition version, model architecture, ordered tensor names, shapes, element counts, quantization bits, f32 scales, byte offsets/lengths, payload SHA-256, model ID, and validated semantic cutoff. Offsets are bytes into the actual packed payload, never float element counts.

For int6, map signed code q in [-31,31] to unsigned `u=q+31` in [0,62]; 63 is invalid. Pack codes least-significant-bit first into a byte stream; zero unused trailing bits; start each tensor at a byte boundary. Declare byte order for header numbers. Reject invalid codes, nonfinite scales, overflow, overlapping ranges, unexpected shapes, and feature-version mismatches. Test pack/unpack round trips at extrema and non-byte-aligned counts.

Generate export fixtures using unpacked, quantized weights and the exact deployment computation. Save feature IDs, pooled vectors, projection outputs, norms, final vectors, and menu scores. Compare float-versus-quantized quality separately from quantized-reference-versus-browser parity.

## 10 Browser runtime

CPU implementation comes first and is the fallback. Unpack and dequantize weights once to f32 buffers, cache candidate embeddings, and compute query embedding plus dot products at search time. No ONNX, Transformers.js, or other general runtime in the production bundle. Such tools may be used as offline baselines.

Implement the same mathematical operations in WGSL, whose normative language specification is maintained by W3C [S5]. Specialized kernels cover embedding gather/reduction, projection, normalization, and candidate dot products. Begin with straightforward f32 kernels; do not require shader-f16 or attempt native six-bit arithmetic. Packed weights reduce transfer size, while runtime compute remains float.

Feature extraction, exact/fuzzy checks, and final tier ordering stay on CPU. Candidate embeddings stay resident on GPU when available. Return score buffers for deterministic final sorting. For 5,000 candidates the readback is only 20,000 bytes, but dispatch/synchronization can still dominate. Measure the complete operation.

Initialize adapter/device and pipelines once per runtime. Reuse bounded growable buffers, handle device loss, honor device limits, and dispose allocations. Parallel GPU reductions must use valid workgroup barriers and no cross-workgroup synchronization assumptions. Keep weights shareable between indexes if this does not complicate lifecycle correctness.

`auto` initially chooses CPU. Enable a GPU crossover only after end-to-end benchmarks on named devices show improvement. Candidate count alone may not capture query length, cold state, or index residency. Forced WebGPU selects it when available but falls back explicitly when unavailable. Importing the package must not request a GPU or fetch assets as a side effect.

Model assets may be bundled or explicitly loaded by the host. Document both approaches and count all resources in size reports. The default production library emits no telemetry. Render labels as text in the demo, not raw HTML.

## 11 Evaluation and verification

### Baselines

Evaluate exactly the same menus, queries, aliases, and available context with:

1. The proposed lexical engine alone.
2. Lexical engine plus a compact, curated alias map built using train/development only.
3. Character n-gram TF-IDF cosine retrieval combined with the same hard tiers.
4. A pinned general embedding model as an offline quality reference, with its actual model size reported.
5. The StarSpace-style pooled shared-embedding baseline specified in Section 6, including the controlled loss comparison.
6. Each projected tiny learned variant, with and without distillation and quantization.

An alias dictionary is a serious competing product. Include its complete compressed bytes. Keep host-supplied aliases identical across systems; a learned system must not receive metadata withheld from baselines. The general model is a reference, not a guaranteed upper bound.

### Metrics

Report nDCG@5 using gain `2^grade-1`, MRR using relevance at least 2, and Recall@1/3/5 over the same threshold. Distinguish hit rate from recall when there are multiple positives. Report conditional semantic quality both before abstention and after the cutoff, and coverage. For no-match menus report the fraction returning any semantic item and the fraction returning any item, including lexical false positives.

Slices: exact, typo, prefix, acronym, semantic-only, multiword intent, ambiguous label, action opposites, unseen products, unseen wording/concept families, no-match, and encoder truncation. Semantic-only excludes queries with a relevant eligible lexical match under the frozen matcher. Show counts and confidence intervals; tiny slices are descriptive, not definitive. Choose a global cosine cutoff on development to meet the false-positive budget; if inadequate across menu sizes, document failure before adding menu-size or margin conditioning.

### Parity and runtime tests

Feature IDs, normalization outputs, packed codes, and counts must match exactly. Starting numeric acceptance is maximum absolute embedding-component error at most 1e-4 and score error at most 2e-4 between quantized MLX, CPU, and WebGPU. Investigate deviations by stage rather than widening tolerance blindly. Rankings must agree for comparisons whose reference score gap exceeds twice the score-error tolerance. Near-tie inversions are reported separately; hard tier decisions must still agree exactly.

Include empty/zero embeddings, long inputs, repeated tokens, Unicode normalization, punctuation, digit labels, all quantizer extrema, malformed payloads, duplicate IDs, duplicate labels, abort, concurrent searches, disposal, missing GPU, lost GPU, failed asset loads, and prefix false positives. Finite outputs and deterministic tie ordering are release gates.

### Performance and size methodology

Run candidate sizes 10, 50, 250, 1,000, 5,000, and 50,000. Record cold initialization, initial index encoding, warm queries, full CPU and GPU timings, p50/p95, JS main-thread time, and memory. Include realistic labels, contexts, and aliases rather than only cached synthetic vectors. Use at least 30 untimed warmups and 200 measured queries for each warm configuration; randomize query order and record power state and browser version. Use GPU timestamps only as supplementary diagnostics, not user-visible latency.

Measure the minified production import graph and model resources as served. Pin minifier and Brotli implementation and settings; report Brotli quality 11/window 22 plus raw and gzip sizes. Compress separate network resources separately and sum them, rather than claiming the size of an unrealistically concatenated archive. Report lexical-only, CPU semantic, optional WebGPU increment, and total. Exclude source maps from deployment totals only if they are actually not served; disclose the exclusion.

## 12 Repository and reproducible commands

```text
packages/core/src/          API, normalization, lexical ranking, CPU backend
packages/webgpu/src/        GPU lifecycle and WGSL
packages/model/             generated manifest and quantized model assets
training/                  MLX model, losses, experiments, QAT, exporter
data/                      source manifests, split definitions, permitted fixtures
eval/                      metrics, baseline adapters, sealed-test runner
bench/                     real-browser timing and size reports
fixtures/                  shared feature and numerical parity fixtures
apps/demo/                 minimal command palette and benchmark view
docs/                      feature contract, model card, decisions, reproduction
```

Use one JS lockfile and one pinned Python environment. Choose concrete scripts implementing this command contract:

```sh
pnpm install --frozen-lockfile
uv sync --frozen
pnpm test:contracts
uv run python -m training.prepare --config configs/data.yaml
uv run python -m training.train --config configs/base.yaml
uv run python -m training.evaluate --split dev --checkpoint runs/selected
uv run python -m training.export --checkpoint runs/selected --bits 8
pnpm test:parity
pnpm bench:browser
pnpm bench:size
pnpm release:verify
```

These are required future entry points, not commands already run. Training requires an appropriate MLX environment. Ordinary CI must still run CPU contracts, export-format tests, and fixed-fixture parity without retraining. Hardware-dependent jobs must report skipped status accurately.

## 13 Delegation plan for the lead agent

The lead agent owns product scope, the feature/API freeze, decisions, integration, and release evidence. Delegate bounded work by owned paths. Agents may propose interface changes, but must not independently invent incompatible hashing, score meanings, tensor layouts, or export formats.

| Workstream | Ownership and output | Dependency |
| --- | --- | --- |
| Data and provenance agent | `data/`, source adapter proposals, leakage report, reviewed menus, licenses | Record schema and split policy |
| Lexical and API agent | `packages/core/` normalization, tiers, API lifecycle and tests | Frozen normalization and API |
| MLX modeling agent | `training/`, trained candidates, ablations, quantized export proposal | Shared feature fixtures and train/dev data |
| Evaluation agent | `eval/`, baselines, sealed-test process, scorecards | Frozen relevance and metric definitions |
| WebGPU agent | `packages/webgpu/`, parity and loss recovery | Selected quantized export and CPU reference |
| Packaging and demo agent | `bench/`, `apps/demo/`, size automation, browser evidence | Stable API and model loading contract |

With limited agent slots, combine packaging/demo and evaluation or defer them. The lead should continue useful integration work while agents run. A data agent and evaluation agent can work independently after the schema freezes. Do not start shader optimization while the architecture is changing.

Each assignment must include owned files, inputs, output interfaces, acceptance tests, constraints, and a completion report containing changes, commands actually run, outcomes, and blockers. Agents must not write each other's owned files without coordination. The lead resolves shared-file conflicts and runs the integrated release checks. Independent evaluation review should inspect leakage and baseline fairness before final-test execution.

## 14 Milestones and stop conditions

**M0 Contracts and evidence.** Inspect reference repositories at pinned commits, verify licenses, freeze API/features/format, prepare golden fixtures, and register evaluation targets. Confirm Apple Silicon training and real-browser test access. If unavailable, implement portable pieces and label hardware-dependent work blocked.

**M1 Baselines and data.** Deliver the lexical library, alias baseline, source manifest, train/dev/test definitions, and first evaluation report. This is a useful product even before ML.

**M2 Semantic feasibility.** Train the StarSpace-style pooled baseline and projected MLX encoders and compare against aliases. Inspect errors and run the finite architecture/feature ablations. If two materially different small models fail the semantic improvement gate, stop compression work and report whether data quality, capacity, or weak product benefit appears limiting. Do not spend indefinitely optimizing failed models.

**M3 Deployable CPU model.** Select quantization, export deterministic assets, pass CPU parity, implement abstention and lifecycle behavior, and meet the CPU transfer budget.

**M4 WebGPU parity and benefit.** Implement kernels, verify actual browser execution and fallback, and measure crossover. If GPU provides no useful speedup, retain it as an explicit experimental backend and keep auto on CPU. Do not claim acceleration from kernel-only timings.

**M5 Release candidate.** Integrate demo, run sealed evaluation once, generate model card and measured size/latency reports, and produce a local reviewable package. Publication is a separate user-authorized action.

The demo must show an editable query, candidate menu, match reasons, backend, and comparable lexical-only results. Include representative semantic and typo cases without claiming curated examples prove generalization. No unrelated dashboard or marketing work.

## 15 Final handoff checklist

- Working local repository with clean reproduction instructions and pinned dependencies.
- Exact/fuzzy/semantic behavior and typed API documented with runnable examples.
- MLX source, configuration, selected float checkpoint, and exported quantized artifact where training ran.
- Data provenance and license manifest; split leakage audit; clear synthetic versus reviewed counts.
- Baseline comparison and ablation table with actual numbers, uncertainty, and failures.
- Feature and quantized numerical fixtures; CPU/WebGPU parity results on named devices.
- Complete transfer bytes, runtime memory, and end-to-end timings.
- Model card stating intended domain, ambiguity limitations, quantization, evaluation scope, and no-confidence-probability caveat.
- Decision log explaining model choice, alias comparison, quantization, and auto-backend policy.
- Accurate list of skipped or blocked work; no fabricated training success or browser test results.

## 16 Reference evidence and corrections to the earlier discussion

The architecture and gates in this document are design decisions. The prior conversation's claims about particular model sizes, six-bit schemes, training seconds, and repository internals were not independently verified in this authoring pass because the referenced GitHub pages could not be fetched. Do not copy their numbers into a README as facts. Before borrowing code, inspect the actual repository and license at a pinned commit. There is no requirement to fork either repository; a small clean implementation may be easier to audit.

The useful proposed pattern is MLX training → deployment-matched quantization → custom weight export → CPU/WGSL implementations → numerical parity → size and retrieval gates. MLX preference does not establish that training will take seconds. Tiny weights do not establish useful semantic quality. Six-bit storage does not imply six-bit compute. Prefixes are lexical, not exact. Cosine is not calibrated confidence. Teacher outputs and UI hierarchies require relevance judgment.

| ID | Primary reference | Use and verification status |
| --- | --- | --- |
| S1 | [fastText word representations](https://fasttext.cc/docs/en/unsupervised-tutorial.html) | Accessed; subword embedding prior art, not evidence that this size target will succeed |
| S2 | [Mind2Web project](https://osu-nlp-group.github.io/Mind2Web/) | Accessed; candidate web-task source requiring extraction and license review |
| S3 | [WorkArena project](https://servicenow.github.io/WorkArena/) | Accessed; enterprise task benchmark, not a turnkey synonym dataset |
| S4 | [MLX documentation](https://ml-explore.github.io/mlx/build/html/index.html) | Accessed; training framework reference; pin the version actually used |
| S5 | [WebGPU Shading Language specification](https://www.w3.org/TR/WGSL/) | Accessed; language/operator reference for browser kernels |
| S6 | [StarSpace paper](https://arxiv.org/abs/1709.03856) | Accessed for version 1.1; shared feature-based embeddings and task-dependent retrieval similarity |
| R1 | [gpu lexer architecture](https://github.com/vercel-labs/gpu-lexer/blob/main/architecture.md) | Supplied reference; fetch failed in this pass; verify before reuse |
| R2 | [gpu cron model card](https://github.com/manuschillerdev/gpu-cron/blob/main/MODEL_CARD.md) | Supplied reference; fetch failed in this pass; verify before reuse |
| R3 | [Manu Schiller post](https://x.com/manuschiller/status/2098801726023139403) | User-supplied provenance; not independently verified here |


## 17 Revision history

Version 1.1 names the project gpu-search, credits StarSpace alongside fastText, makes the pooled retrieval baseline explicit, and requires controlled comparison before retaining projections or distillation. Core product scope, MLX deployment approach, and exact/lexical/semantic tier precedence are unchanged.
