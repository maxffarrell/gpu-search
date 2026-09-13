# gpu-search

[Try the demo](https://gpu-search.vercel.app) · [How it works](docs/specification.md) · [Verification](docs/verification.md)

Small, private search for command palettes, settings, navigation menus, and action pickers.

Give gpu-search your destinations and let people find them without typing every character perfectly. `profle` can find **Profile**. `api k` can find **API Keys**. Add an alias and `coworkers` can find **Members**. You control the available destinations and what happens when someone selects a result.

Search runs locally. There is no backend service to operate, API key to manage, or query telemetry. The matching library needs no model download and works without a GPU.

## A list in. Useful results out.

Create an index once, then reuse it as someone types:

```ts
import { createIndex } from 'gpu-search/lexical'

const index = await createIndex([
  { id: 'profile', label: 'Profile' },
  { id: 'members', label: 'Members', aliases: ['coworkers'] },
  { id: 'api-keys', label: 'API Keys' },
])

const { results } = await index.search('profle', { limit: 5 })
// results[0].label  → 'Profile'
// results[0].reason → 'typo'

index.dispose()
```

This import refers to the repository's local workspace package. It is **not published to npm**; build and link the workspace package to use it in another project.

Behind that small API, gpu-search handles:

- **Consistent text matching:** Unicode normalization, English case handling, and extra whitespace, while keeping meaningful punctuation distinct.
- **Predictable ordering:** an exact label comes before an alias, prefix, acronym, or typo match. Ties stay consistent.
- **Explainable results:** each match says why it appeared and whether it matched the label or an alias.
- **Everyday interface details:** empty queries return nothing, result limits are supported, and cancelled searches cannot deliver stale results.
- **Index management:** candidates are copied when indexed, repeated searches can run safely, and disposing an index releases its stored entries. Invalid records, duplicate IDs, and oversized inputs produce explicit errors.

Rebuild the index when your menu changes. You do not need to assemble separate normalization, spelling, ranking, and cancellation utilities.

## Try it locally

Use Node.js 22.12+ and pnpm 12.4.1 in this repository:

```sh
pnpm install --frozen-lockfile
pnpm dev
```

The demo lets you change the query and candidate menu, compare results, and choose Auto, Light, or Dark appearance.

**The learned search is experimental.** The library above provides deterministic text matching. The demo also runs a real, small trained model that tries to connect different wording with related destinations. It can make mistakes and show unrelated results; its scores are not confidence ratings. It runs on the CPU, and no WebGPU acceleration is claimed. This experiment is separate from the library's reliable matching rules.

For implementation and research details, see the [API specification](docs/specification.md), [model card](docs/model-card.md), [experiment findings](docs/optimization-experiments.md), and [reproduction evidence](docs/verification.md).

## Credits

The three main inspirations are:

- **[gpu-lexer](https://gpu-lexer.vercel.app)** — Shu Ding / Vercel Labs.
- **[gpu-time](https://gpu-time.arikko.dev)** — Arik Chakma.
- **[gpu-cron](https://gpu-cron.vercel.app)** — Manu Schiller.

Their focused, local browser tools inspired this project. Their code, weights, and datasets were not copied.

Research builds on **fastText**, **StarSpace**, and **Apple MLX**, with **W3C WGSL** informing planned GPU work. Data comes from **CLINC150** (Larson et al., CC BY 3.0), **BANKING77** (Casanueva et al. / PolyAI, CC BY 4.0), and **VS Code** (Microsoft, MIT). Offline teacher experiments used **Sentence Transformers all-MiniLM-L6-v2** (Apache-2.0), **PyTorch**, and **Hugging Face Transformers**; these do not ship to the browser.

Thanks also to Vite, TypeScript, pnpm, esbuild, Playwright, NumPy, and uv. [Source credits](docs/credits.md), [data attribution](docs/data-sources.md), and [teacher details](docs/optimization-experiments.md) preserve provenance and license information. Project code is MIT licensed; third-party materials retain their own licenses.
