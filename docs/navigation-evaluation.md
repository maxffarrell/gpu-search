# Same-size navigation training

The demo now uses the frozen `navigation-align0p5-seed29` experiment: **32,768 weight bytes**, exactly the previous model's size. It improves source-authored navigation retrieval and passes every existing release regression floor. It remains an experimental CPU model, not a production-approved semantic backend.

## What changed

We extracted real search keywords and destinations from pinned KDE, XFCE, and GNOME sources. KDE supplies training positives, XFCE supplies development selection, and GNOME was evaluated once after the candidate and manifest hashes were frozen. Only source-authored destinations are positive judgments; other menu items are unjudged, not proven negatives. [Data provenance and licenses](navigation-data.md) describe the sources and exclusions.

Twelve warm-start training runs tested navigation alignment, additional WordNet synonyms, and short-query augmentation. Navigation alignment won; the other additions were rejected. Nine separate experiments that traded embedding dimensions for more hash buckets also lost to matched controls. All artifacts and selection reports are retained for reproduction. No added dictionary, alias lookup, external API, or teacher model is shipped.

The selected model uses the existing feature contract and 16-dimensional int8 tables. Its SHA-256 is `fcb4be56a9a8ddb82e02de9ae31f0822e2e211b4dc65e61cc24dc285620dd6a2`. Selection chose epoch 2; additional epochs were not automatically better.

## Untouched provider evaluation

GNOME contains 246 queries, including 222 without a relevant lexical match. The following values measure those 222 queries using raw model ranking and documented positives. They do not measure confidence, exhaustive relevance, or no-match safety.

| System | nDCG@5 | Documented hit in top 1 | Documented hit in top 5 |
| --- | ---: | ---: | ---: |
| Previous demo model, 32 KiB | 0.1016 | 4.95% | 21.17% |
| New navigation model, 32 KiB | 0.1705 | 7.21% | 32.43% |
| Train-paraphrase TF-IDF reference | 0.2065 | 16.22% | 31.53% |
| Offline MiniLM reference, ~91 MB weights | 0.3825 | 27.03% | 55.86% |

The new model's nDCG gain over the previous model is 0.0688. A bootstrap grouped by documented destination sets gives a 95% interval of [0.0260, 0.1102]. This supports an improvement on this provider, not a population-wide or best-in-class claim. The split contains both seen and new normalized training queries, reported separately in the full evidence.

With the existing calibration cutoff, the new model's semantic nDCG falls to 0.0063 and its top-five documented hit rate to 1.35%. These data contain no reviewed negative queries, so they cannot validate abstention. The demo deliberately exposes raw model retrieval and explains that scores are not confidence ratings.

## Release decision and remaining work

All seven existing expanded-data and original-development transfer gates passed after selection, including coverage, no-match, and regression requirements; the raw model size also remains unchanged. The original consumed synthetic test was not reopened. We promoted the new weights only to the experimental demo; stable library behavior remains governed by its lexical contracts.

This model is **not production-ready or best-in-class**. It still loses to the training-example reference on ranking quality and has weak calibrated transfer. The next quality milestone requires independently reviewed product menus, multi-positive relevance and genuine no-match examples, with a new reserved evaluation before further tuning. The GNOME set is now consumed evaluation evidence and must not be reused as an untouched test. More epochs alone did not solve this gap.

Separately, [prepared candidate embeddings](prepared-model-index.md) remove repeated menu encoding from each query without altering scores or adding model bytes. Browser measurements distinguish one-time preparation from warm inference.

## Reproduction and evidence

- `training/prepare_navigation.py` and `training/prepare_navigation_splits.py`: pinned source extraction and provider splits.
- `training/experiment_navigation.py` and `configs/navigation-*.json`: training configurations and follow-ups.
- `eval/navigation-training-selection.json`: development-only selection across all twelve runs.
- `eval/navigation-selection.json` and `eval/navigation-holdout-access.json`: exact frozen artifact and one-time evaluation receipt.
- `eval/navigation-evaluation.json`: per-query rankings, baselines, slices, hashes, and uncertainty.
- `eval/navigation-regressions.json`: all seven release regression gates passed.
- `eval/collision-summary.md`: rejected hash-bucket/dimension tradeoffs.

The evaluator defaults to development data. Holdout access requires a matching frozen record and an exclusive receipt; rerunning it does not create another independent test. The offline teacher is pinned and optional, and is never a browser dependency.
