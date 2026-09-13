# Reference provenance

Inspected 2026-09-12 via primary GitHub repository metadata and READMEs. All three repository license identifiers were MIT. No source code, model weights, datasets, branding assets, or benchmark figures were copied. These are architectural inspirations, not dependencies or claims of equivalent performance.

| Project | Pinned revision | Contribution to the direction |
| --- | --- | --- |
| [gpu-lexer — Shu Ding / Vercel Labs](https://github.com/vercel-labs/gpu-lexer) · [demo](https://gpu-lexer.vercel.app) | `3bf10853186b20070096ed40d5986e0bf48742c1` | Small specialized learned browser workloads and custom WebGPU execution |
| [gpu-time — Arik Chakma](https://github.com/arikchakma/gpu-time) · [demo](https://gpu-time.arikko.dev) | `aba27e54aabe7310cba5c160fa2079045096ffb1` | Local natural-language functionality with explicit CPU/WebGPU backends |
| [gpu-cron — Manu Schiller](https://github.com/manuschillerdev/gpu-cron) · [demo](https://gpu-cron.vercel.app) | `0ddaa91d88c33838e1f76c11c0d830810f1d7914` | Compact learned natural-language parsing as a focused browser experiment |

Additional primary references: [fastText subword representations](https://fasttext.cc/docs/en/unsupervised-tutorial.html), [StarSpace shared-feature retrieval](https://arxiv.org/abs/1709.03856), [Apple MLX](https://ml-explore.github.io/mlx/build/html/index.html), and [W3C WGSL](https://www.w3.org/TR/WGSL/). Pooled encoders here are inspired baselines, not exact reproductions. Vite, TypeScript, pnpm, esbuild, Playwright, NumPy, and uv provide development and validation tooling; their licenses remain with their respective projects. Mind2Web and WorkArena were suggested research sources in the spec but their data was not ingested.


## Desktop navigation metadata

Author-provided English search keywords, destination names, and descriptions were extracted from these official repositories without executing their code:

| Provider | Source | Pinned revision |
| --- | --- | --- |
| GNOME | [GNOME Settings / gnome-control-center](https://github.com/GNOME/gnome-control-center) | `cd79a897190989ca395c6f00962f0929734103c7` |
| KDE | [Plasma Desktop](https://github.com/KDE/plasma-desktop) | `19dae94ec1d563d095b08602dd5eed60d485a5e0` |
| KDE | [Plasma Workspace](https://github.com/KDE/plasma-workspace) | `df3bba49a2c0657d4165f6511b1771138ccc75ea` |
| Xfce | [Xfce Settings](https://github.com/xfce-mirror/xfce4-settings) | `488c919d23c979c9b28abd192c05b9b2310fa732` |

Credit the GNOME, KDE, and Xfce contributors for this search metadata. [KDE's KCM documentation](https://develop.kde.org/docs/features/configuration/kcm/) explains the authored search-keyword field. [Navigation data provenance](navigation-data.md) records extraction, immutable provider/product lineage, original source paths, hashes, license evidence, and shared-keyword positive destinations. An absent keyword is not a negative relevance judgment.

Original metadata and notices remain in [data/navigation/raw](../data/navigation/raw/). GNOME/Xfce repository GPL license texts and KDE's mixed per-file GPL/LGPL and other license evidence are retained; missing file-level SPDX assignments are explicitly recorded. These materials are not relicensed under this project's MIT code license.

## Open English WordNet

Offline positive-pair experiments credit the **Open English WordNet team** and **Princeton University WordNet**. The source is [Open English WordNet, 2025 edition](https://github.com/globalwordnet/english-wordnet/releases/tag/2025-edition); the recorded repository revision is `dc343f2683279ecbb13fab4e2fd778d7b162d287`. The downloaded XML archive has SHA-256 `9ca6d1dcb75f822fdd66617f7d9da48142ace38dd544d6ad5e2feca1674ad3fe`.

Open English WordNet's additions use **CC BY 4.0**, while the underlying Princeton material retains its WordNet license and attribution. Both are preserved in [LICENSE.md](../data/lexicon/LICENSE.md) and [WNDB_License.txt](../data/lexicon/WNDB_License.txt). The [extraction manifest](../data/lexicon/manifest.json) records the source, hashes, train-only seed selection, filters, and 504 derived positive pairs. No negative judgments are inferred from missing word relationships. This is offline experimental supervision; no WordNet dictionary or lookup is shipped to the browser.
