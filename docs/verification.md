# Release evidence and limits

This is the baseline milestone allowed by the specification's semantic stop condition, not completion of its learned-search release criteria.

- TypeScript contracts cover exact/alias precedence, every lexical subclass, OSA transpositions and safeguards, Unicode and symbols, input bounds, snapshots, abort/concurrency/disposal, ties, empty queries, fallback metadata, and 200 seeded adversarial menus.
- Python/TypeScript feature IDs and normalization agree on golden fixtures. Both lexical evaluation baselines agree with the shipping engine on all development fixtures plus adversarial camelCase cases.
- Five Python export-format tests cover quantizer rounding/extrema, int6 packing, reserved codes, trailing bits and malformed byte lengths. The research exporter is not a validated browser asset loader.
- Revised Chrome production-build checks verify actual manifest/weight fetches, model scores for six queries against the real encoder, arbitrary edited candidates, invalid JSON, literal HTML-like labels, 390px containment, and Auto/Light/Dark persistence and system changes. Blocking weights shows Model unavailable; substituting valid zero weights eliminates model results while lexical search still works. Zero requests were observed during query/edit interactions after load. See `bench/reports/demo.json`.
- The custom experimental CPU runtime verifies SHA-256 and strict int8 tensor metadata. Exported NumPy fixture parity has maximum embedding-component error 8.94e-8 and cosine error 1.70e-7. Tests also check malformed assets, mutation isolation, actual weight perturbation, and alias/context composition. This verifies execution, not model usefulness.
- Full lexical search ran in Chrome 153.0.8010.36 and Playwright WebKit 26.0 on Apple M4 Max, macOS/Darwin 27.0.0. At 1,000/5,000 candidates Chrome p95 was 2.1/10.1 ms; WebKit was 2/11 ms. The 50,000 stretch workload reached 62.1/76 ms p95. See all samples and indexing timings in `bench/reports/browser.json`.
- Measurements used 30 warmups and 200 seeded randomized queries at each size. Labels have repeated synthetic families with suffixes, aliases, and contexts; this is not representative product corpus performance. Timers can round short calls to zero. Power state was not controlled. Heap reporting was available only in Chrome and is a whole-page estimate after the sweep, not per-index allocation accounting.
- The local browser benchmark used installed Chrome (`CHROME_CHANNEL=chrome`) and WebKit revision 2359 (`WEBKIT_EXECUTABLE` override), recorded in the report. Default reproduction downloads Playwright's pinned browsers; results will differ by version/device.
- The size report separately compresses all emitted resources with Node's zlib Brotli quality11/window22 and gzip level9. The lexical library meets the 8KiB target. The revised demo transfer includes the actual experimental manifest and int8 model weights; no source maps are served. Historical latency results above measure only the lexical engine.

Not claimed: reviewed held-out semantic quality, actual WebGPU kernels, device-loss recovery, semantic latency guarantees, or production model readiness. Later updates below add experimental-model parity, measured CPU transfer budgets and an offline general embedding reference; none establishes a production-ready semantic model.

## Expanded-data update (2026-09-13)

The active demo artifact is now `packages/model/candidate/` (`expanded-more-data-pooled-seed17`), with the pilot preserved for comparisons. The full pipeline passed 15 Python tests covering exporter contracts, data separation and source-label quality floors; JavaScript tests additionally cover candidate NumPy parity and the training-seen Members/Profile sanity cases. Browser verification checked actual model bytes, arbitrary candidate edits, no query transmission, themes, and missing/zero weights with the new artifact. See `docs/expanded-evaluation.md` for numerical development gates, transfer limits, and the still-unmet reviewed-final-test requirement. These results supersede the original model-quality stop finding for this expanded experiment only.

## Optimization follow-up (2026-09-13)

The same live weights were retained after 39 additional training runs and 90 calibration configurations. The strongest same-size research candidate improved development but did not improve calibrated reserved-test semantics. Its final synthetic test was opened once after model/manifest selection was frozen; the test is now consumed and remains unreviewed. See [the optimization decision](optimization-experiments.md) and `eval/optimized-evaluation.json`.

All 37 portable Python tests and 24 JavaScript tests pass, together with types, production build, complete CPU transfer-size gates, and real Chrome demo inference/failure/privacy/theme checks. The standalone research runtime also passes independent NumPy parity on real weights and Unicode/ordered-feature fixtures. Neither its runtime nor its weights enter the live demo import graph. The original CPU demo is 36,326 bytes Brotli; the isolated research build is 36,243 bytes. CI enforces both 50 KiB CPU budgets and the lexical 8 KiB limit. No training or reserved-test access occurs in CI.

## Navigation and inference reuse (2026-09-13)

The experimental demo now uses `navigation-align0p5-seed29`, retaining 32,768 raw weight bytes. Twelve navigation/data follow-ups and nine collision-capacity runs preceded a frozen one-time GNOME provider evaluation. The selected model improved raw semantic known-positive nDCG from 0.1016 to 0.1705 and passed all seven existing regression floors. It still trails the train-paraphrase TF-IDF reference on this measure and does not establish safe semantic abstention. See [the complete evaluation and release decision](navigation-evaluation.md).

Prepared candidate embeddings reuse menu vectors across queries with exact reference score parity. The browser report separates synchronous setup and warm query latency; see [the prepared API and measurements](prepared-model-index.md). Portable tests cover navigation metadata parsing, positive-only augmentation, evaluation judgments, and runtime isolation. CI does not train models or reopen the provider holdout.

## Typo-result precedence fix

Main demo results now use exact/fuzzy lexical results whenever present and otherwise show explicitly labeled real model suggestions. They do not pad successful text matches with unrelated cosine neighbors. Raw scores remain separately inspectable. Real-weight regression tests protect `profle → Profile`, exact labels, prefixes and aliases, including a fixture where the model itself ranks the wrong destination first. Browser checks cover main-result precedence and typo search with unavailable or zeroed weights. No weights were retrained or changed by this fix.
