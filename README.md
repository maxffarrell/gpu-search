# gpu-search

[Live demo](https://gpu-search.vercel.app) · [Verification evidence](docs/verification.md)

Small, private search for software interfaces. Normalized label matches rank first, followed by deterministic aliases, prefixes, acronyms, and typo matches. Queries stay in the browser.

**Status: real experimental CPU model demo, plus a stable lexical library.** The demo downloads the trained pooled16 int8 weights, verifies their SHA-256, and computes query/candidate embeddings and cosine scores locally. It shows raw model rankings alongside lexical results. The learned encoders did not pass the semantic release gate on the small synthetic development suite, so no validated cutoff or production semantic-quality claim is made. There is no WebGPU backend. See [experiment results](eval/feasibility.json), [decisions](docs/decisions.md), and the [original specification](docs/specification.md).

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

See [model card](docs/model-card.md) and [experiment report](docs/experiment.md) for evaluation/export commands and the stop decision. No release checkpoint is selected; commands must name an experimental run explicitly. The reserved synthetic test set is not scored. The dataset has **zero human-reviewed records**, so none of these results establish generalization to real products.

[Size evidence](bench/reports/size.json) measures minified browser entry points and every demo resource separately with Brotli quality 11/window 22 and gzip level 9. [Browser evidence](bench/reports/browser.json) records the exact hardware/browser, raw samples, indexing, p50/p95, and limitations. These are local measurements, not universal latency guarantees. CI checks contracts, fixed fixtures, types, build, and the lexical byte budget without training.

## Credits

These three projects are the main inspiration for small, focused, local learned browser tools:

- **[gpu-lexer](https://gpu-lexer.vercel.app)** by Shu Ding / Vercel Labs — [source](https://github.com/vercel-labs/gpu-lexer). Specialized learned syntax highlighting and custom WebGPU execution.
- **[gpu-time](https://gpu-time.arikko.dev)** by Arik Chakma — [source](https://github.com/arikchakma/gpu-time). Local English time parsing with CPU and WebGPU backends.
- **[gpu-cron](https://gpu-cron.vercel.app)** by Manu Schiller — [source](https://github.com/manuschillerdev/gpu-cron). A compact learned natural-language cron parser.

Also credit [fastText](https://fasttext.cc/docs/en/unsupervised-tutorial.html) for character-subword representations, [StarSpace](https://arxiv.org/abs/1709.03856) for shared-feature retrieval, [Apple MLX](https://ml-explore.github.io/mlx/build/html/index.html) for training, and [W3C WGSL](https://www.w3.org/TR/WGSL/) for the planned shader contract. Vite, TypeScript, pnpm, esbuild, Playwright, NumPy, and uv support development and reproducibility.

[Reference provenance](docs/credits.md) records inspected commits and license metadata. No reference-project code, weights, datasets, or performance figures were copied. MIT licensed.
