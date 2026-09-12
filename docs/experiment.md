# Reproducing the synthetic feasibility experiment

This experiment deliberately reports a failed feasibility gate. It does not supply a production semantic model.

```sh
uv sync --frozen
uv run python -m training.prepare --config configs/data.yaml
uv run python -m training.train --config configs/base.yaml
uv run python -m training.report
uv run python -m training.audit
uv run python -m training.evaluate --split dev --checkpoint runs/pooled-16-both-contrastive-seed17
uv run python -m unittest training.test_export
uv run python -m training.export --checkpoint runs/pooled-16-both-contrastive-seed17 --bits 8 --experimental
```

`uv.lock` pins the Python dependencies. Training requires macOS Apple Silicon with Metal access; the original sandbox could not initialize Metal, and training succeeded with authorized direct GPU access. Feature preparation, NumPy evaluation and exporter tests work without MLX. `training.evaluate` intentionally has no final-test option; a reviewed sealed-test protocol is still required. `runs/selected` does not exist because no model qualified. Export without `--experimental` rejects instead of silently blessing a checkpoint.

The source fixtures are entirely original synthetic text under this repository's MIT license. `data/sources.json` contains provenance and the preparation-script hash. `data/leakage-audit.json` records menu-family splits, literal-query overlaps and encoder truncation counts. No paid API or external dataset was used.

## Actual development results

The 80 development records contain 24 exact/typo records, 36 semantic queries and 20 no-match records. Overall nDCG below averages the 60 relevant-query records; no-match behavior is reported separately. There are no grade-1-only cases. Semantic-only means no judged relevant candidate has an eligible lexical match. Gain is `2^grade - 1`, MRR and recall use grade at least 2. Because each record has one positive, recall equals hit rate in this particular dataset.

| System | Semantic nDCG@5 after cutoff | Overall nDCG@5 | No-match any result |
| --- | ---: | ---: | ---: |
| Lexical | 0.0000 | 0.4000 | 0% |
| Lexical + training aliases | 0.0000 | 0.4000 | 0% |
| Character TF-IDF + same lexical tiers | 0.1462 | 0.4877 | 5% |
| Pooled 16d contrastive seed 17 | 0.0833 | 0.4500 | 5% |
| Pooled 16d contrastive seed 29 | 0.0000 | 0.4000 | 5% |
| Pooled 16d contrastive seed 43 | 0.0278 | 0.4167 | 5% |
| Projected 16d contrastive seed 17 | 0.0278 | 0.4167 | 5% |
| Projected 16d contrastive seed 29 | 0.0278 | 0.4167 | 5% |
| Projected 16d contrastive seed 43 | 0.0278 | 0.4167 | 5% |

All 18 individual ablations, cutoffs, coverage, recall, MRR, raw per-query ranks/scores and bootstrap intervals are in `eval/feasibility.json` and each `runs/*/dev-evaluation.json`. The pooled seed-17 semantic improvement interval is [0.0000, 0.1944], so its point estimate does not pass the interval gate. No seed passes. Metrics before abstention look much better while returning something for every no-match query; that is precisely why both cutoff behavior and coverage are reported.

Cutoffs were selected on development and are reused for these diagnostics. Bootstrap uses 10,000 paired query samples, seed 20260912, and is descriptive after development selection; it is not a final held-out confidence claim. The global cutoff allows at most 5% of development no-match queries to receive any semantic item. Results remain dominated by tiny sample size and invented judgments.

The alias map comes from training only. Host aliases/context are identically absent for every system. TF-IDF fits document frequencies on unique training labels, with smoothed unseen-gram document frequency zero, and uses the same fixed lexical tiers. It is a lexical n-gram similarity diagnostic, not a general embedding model. Neither TF-IDF nor an experimental encoder is deployed by default.

The reserved test records were never scored. The next meaningful step is human-reviewed product menus and disjoint relevant/no-match queries, with a stronger reviewed train/development alias baseline, followed by preregistered evaluation. More shader work cannot repair the current evidence gap.
