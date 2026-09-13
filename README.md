# gpu-search

[Live demo](https://gpu-search.vercel.app) · [Verification evidence](docs/verification.md)

Small, private search for software interfaces. Normalized label matches rank first, followed by deterministic aliases, prefixes, acronyms, and typo matches. Queries stay in the browser.

**Status: improved experimental CPU model, plus a stable lexical library.** The demo runs actual pooled16 int8 weights trained with 7,961 records from licensed CLINC150, BANKING77, VS Code settings, and the original training fixtures. On the same expanded development set, semantic nDCG@5 improved from **0.028 to 0.464**; the strongest lexical baseline scored **0.385**, with new-model no-match false positives **4.83%**. These are weak-label development results, not an independent final test. Transfer to unseen software concepts remains limited. The demo shows raw model rankings without a relevance cutoff; no WebGPU backend is claimed. See [expanded evaluation](docs/expanded-evaluation.md), [data provenance](docs/data-sources.md), and [original specification](docs/specification.md).

**Latest experiment decision:** keep the current live weights. A same-size ordered-word-pair candidate reached **0.523** development nDCG, but the reserved 40-row synthetic test did not confirm improved transfer: both models abstained on every semantic-only test query, and raw overall nDCG fell from 0.545 to 0.535. The candidate remains available for research, not promoted. See the [optimization report](docs/optimization-experiments.md) for training, distillation, calibration, size and held-out evidence.

## Try locally

Requires Node.js 22.12+ and pnpm 12.4.1. The Python experiment uses the pinned environment in `uv.lock` and Apple Silicon for MLX training.

```sh
pnpm install --frozen-lockfile
pnpm dev
pnpm release:verify
```

The minimalist demo compares actual model scores with lexical results for the same editable candidate menu. Initial candidates have no aliases or context. Queries, including `coworkers`, pass through the trained encoder; no synonym lookup supplies the model results. The model panel displays raw cosine rankings without an abstention threshold, including potentially irrelevant results. Missing or invalid weights show a model error, never substitute lexical results. Auto / Light / Dark at the top follows the system or saves an explicit preference.

## Library API

The workspace package is named `gpu-search`; it has **not been published to npm**, and package-name availability is unverified. Build locally before importing it from another workspace package.

```ts
import { createIndex } from 'gpu-search/lexical'

const index = await createIndex([
  { id: 'profile', label: 'Profile' },
  { id: 'members', label: 'Members', aliases: ['coworkers'] },
  { id: 'billing', label: 'Billing' },
])

await index.search('profle')
// { results: [{ id: 'profile', label: 'Profile', match: 'lexical',
//   reason: 'typo', matchedField: 'label', lexicalScore: 6 / 7 }],
//   backend: 'lexical', degraded: false }

index.dispose()
```

The main entry also exports `createIndex`, defaulting to a request for semantics. Until a validated model exists, it returns lexical results with `degraded: true`; `{ semantic: false }` opts into the baseline explicitly. A forced `webgpu` request also falls back with explicit degradation when semantics are requested. Imports do not fetch assets or request devices.

Label equality always outranks alias equality. Within lexical results the precedence is alias equality → full prefix → ordered token prefixes → acronym → strong OSA typo. Scores rank only within a subclass; they are not confidence probabilities. Duplicate labels retain insertion order. Context is accepted and validated but has no effect on lexical ranking.

Indexes snapshot candidates and can be reused concurrently. `search(query, { limit, signal })` defaults to ten results, accepts zero, and rejects invalid limits. Empty queries return no results. `dispose()` is idempotent and invalidates future/pending results. Input errors expose `code: 'ERR_SEARCH_INPUT'` and `path`.

Build limits: 50,000 candidates; 256 Unicode scalars per label/query/context; eight aliases of up to 128 scalars each. Duplicate IDs, empty fields, malformed records, unpaired surrogates, and overflow are rejected. Applications must filter unauthorized destinations before indexing.

## Reproduce evidence

```sh
pnpm test:contracts
pnpm test:parity
pnpm build
pnpm bench:size
pnpm exec playwright install chromium webkit
pnpm bench:browser

uv sync --frozen
uv run python -m training.prepare --config configs/data.yaml
uv run python -m training.train --config configs/base.yaml
```

The commands above reproduce the original pilot. For the expanded experiment:

```sh
uv run python -m training.prepare_external
uv run python -m training.train_expanded --config configs/expanded.yaml
uv run python -m training.evaluate_expanded --old packages/model/experimental --new packages/model/candidate
uv run python -m unittest training.test_export training.test_data_sources training.test_quality
```

Training preserves existing runs and refuses to overwrite them; pass `--tag my-run` to train into new run directories when rerunning. Source extraction and pinned download commands are in [data provenance](docs/data-sources.md). The new experiment includes matched-update old-data controls, three seeds, and epoch-1/10 comparisons. Regression checks enforce source-label quality floors, no-match limits, dataset hashes, and zero-weight failure. CI runs them alongside implementation tests.

See the [model card](docs/model-card.md), [expanded evaluation](docs/expanded-evaluation.md), and [original pilot report](docs/experiment.md). No production release checkpoint or independently validated cutoff is selected. Upstream source tests remain unscored; the original reserved synthetic test was opened once after freezing the optimization candidate, and is no longer a fresh holdout. Source intent labels are inherited; their conversion to menu relevance has **zero new human-reviewed judgments**. These results do not establish generalization to arbitrary products.

[Size evidence](bench/reports/size.json) measures minified browser entry points and every demo resource separately with Brotli quality 11/window 22 and gzip level 9. [Browser evidence](bench/reports/browser.json) records the exact hardware/browser, raw samples, indexing, p50/p95, and limitations. These are local measurements, not universal latency guarantees. CI checks contracts, fixed fixtures, types, build, the 8 KiB lexical budget and the complete 50 KiB CPU-demo budget without training.

## Credits

These three projects are the main inspiration for small, focused, local learned browser tools:

- **[gpu-lexer](https://gpu-lexer.vercel.app)** by Shu Ding / Vercel Labs — [source](https://github.com/vercel-labs/gpu-lexer). Specialized learned syntax highlighting and custom WebGPU execution.
- **[gpu-time](https://gpu-time.arikko.dev)** by Arik Chakma — [source](https://github.com/arikchakma/gpu-time). Local English time parsing with CPU and WebGPU backends.
- **[gpu-cron](https://gpu-cron.vercel.app)** by Manu Schiller — [source](https://github.com/manuschillerdev/gpu-cron). A compact learned natural-language cron parser.

Also credit [fastText](https://fasttext.cc/docs/en/unsupervised-tutorial.html) for character-subword representations, [StarSpace](https://arxiv.org/abs/1709.03856) for shared-feature retrieval, [Apple MLX](https://ml-explore.github.io/mlx/build/html/index.html) for training, and [W3C WGSL](https://www.w3.org/TR/WGSL/) for the planned shader contract. New training sources: **[CLINC150](https://github.com/clinc/oos-eval)** (Larson et al., CC BY 3.0), **[BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets)** (Casanueva et al. / PolyAI, CC BY 4.0), and **[VS Code](https://github.com/microsoft/vscode)** (Microsoft Corporation, MIT). Their licenses, attribution, pinned revisions, and transformations are preserved in [data provenance](docs/data-sources.md). Third-party data retains its own license; the repository's MIT code license does not replace it.

Vite, TypeScript, pnpm, esbuild, Playwright, NumPy, and uv support development and reproducibility.

Offline teacher experiments also credit [Sentence Transformers all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (Apache-2.0), PyTorch and Hugging Face Transformers. The teacher and its dependencies are not browser assets. Its pinned revision, model-card copy, source checksums and cached scores are preserved in `runs/teacher-minilm/`; distilled students were rejected. See [optimization experiments](docs/optimization-experiments.md).

[Reference provenance](docs/credits.md) records inspected commits and license metadata. No code, weights, datasets, or performance figures from the three GPU inspiration projects were copied. The implementation is MIT licensed; third-party training data retains the licenses described above.
