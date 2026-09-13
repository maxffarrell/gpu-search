# Reusing trained candidate embeddings

The experimental CPU model now supports an immutable prepared index. It validates and snapshots the menu, computes each candidate's label/alias/context vector once, and reuses those vectors as the query changes. This improves repeated inference without changing the model's 32,768 weight bytes, score ordering, or retrieval quality.

```ts
// Repository-local module; the model runtime is not published as an npm package.
import { loadModel } from '../packages/model/runtime'

const model = await loadModel(manifest, weightsArrayBuffer)
const index = model.prepare([
  { id: 'profile', label: 'Profile', aliases: ['my information'] },
  { id: 'members', label: 'Members', context: 'organization settings' },
])

const first = index.score('coworkers').slice(0, 5)
const second = index.score('personal information').slice(0, 5)

index.dispose()
```

`prepare(candidates)` is synchronous and returns `{ size, score(query), dispose() }`. Preparation follows the core candidate count/text/alias limits and rejects malformed records or duplicate IDs with `SearchInputError`. Queries are also validated. Zero candidate embeddings are omitted. Scores are raw cosines, sorted descending with input order breaking exact ties; there is no learned relevance cutoff or cross-tier comparison.

Candidate records and aliases are copied. Later caller mutation cannot change the index or its results, and changing a returned result cannot mutate stored state. Each index owns its candidate vectors; there is no unbounded global cache. `dispose()` is idempotent, releases the index's vector and label storage, and future queries throw `ModelIndexDisposedError`. Disposing one prepared index leaves the model and other prepared indexes usable.

Rebuild when the menu changes, then dispose the old index. The demo prepares once after the model loads and again when an edited menu is accepted; ordinary query typing only encodes the query and performs cached dot products and sorting. The existing `model.score(query, candidates)` remains a backward-compatible raw reference path that prepares temporary vectors for that call. Prefer the prepared API for repeated queries.

This synchronous raw-scoring API is separate from the stable lexical `createIndex`. It does not add a validated semantic backend, query cancellation, or WebGPU to that API. The model remains experimental; faster inference does not resolve its generalization and abstention limitations.

## Verification

Tests compare prepared and reference outputs exactly across multiple queries, aliases, and context; verify mutation isolation, stable ties, input limits, zero vectors, and independent disposal; and retain actual exported-weight numerical parity. The production demo browser suite verifies edited-menu inference, privacy, missing assets, and zero-weight behavior.

`bench/model-browser.ts` measures real trained CPU inference in Chromium and WebKit on the named local hardware. It reports asset fetch and validation/decode separately from menu preparation and warm queries at 100, 1,000, and 5,000 candidates. Every size uses 30 warmups and 200 measured queries; the 1,000-candidate case also times the uncached reference in the same browser. Full result sorting/materialization is included. Lexical ranking and a relevance cutoff are not included, so this is not a complete hybrid-search benchmark. Timer resolution can round very short calls to zero, and power state is uncontrolled.

On 2026-09-13, the unchanged `navigation-align0p5-seed29` model ran on an Apple M4 Max with macOS 27.0.0. Times below are milliseconds. The uncached arm follows the prepared arm in the same browser session; these are local measurements, not guarantees across devices.

| Browser | Candidates | Preparation | Prepared p50 | Prepared p95 | Uncached p50 | Uncached p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Chromium 153 | 100 | 6.9 | 0.0 | 0.1 | — | — |
| Chromium 153 | 1,000 | 42.9 | 0.2 | 0.3 | 37.1 | 39.1 |
| Chromium 153 | 5,000 | 188.1 | 1.0 | 1.1 | — | — |
| WebKit 26 | 100 | 7.0 | 0.0 | 0.0 | — | — |
| WebKit 26 | 1,000 | 36.0 | 0.0 | 1.0 | 18.0 | 19.0 |
| WebKit 26 | 5,000 | 106.0 | 1.0 | 1.0 | — | — |

Both browsers returned exactly the reference scores, made zero network requests during measured queries, and reported no errors. At 5,000 candidates the index stores 320,000 vector bytes plus candidate identifiers and labels. Preparation remains synchronous and can block the main thread for large menus; reuse removes that work from ordinary queries but does not eliminate the initial cost. Zero timings reflect browser timer resolution.

Run:

```sh
node --import tsx --test fixtures/model.test.ts
node --import tsx bench/model-browser.ts
pnpm build
pnpm bench:size
pnpm test:demo
```

Optional `CHROME_CHANNEL`, `WEBKIT_EXECUTABLE`, and `MODEL_BENCH_URL` overrides allow installed browser binaries and an already running root Vite server. Reports record overrides explicitly. See `bench/reports/model-browser.json` for current timings and model hashes, and `bench/reports/size.json` for complete separately compressed demo resources. The current prepared demo build is 36,773 bytes Brotli, below the 50 KiB CPU budget, with unchanged 32,768-byte weights.
