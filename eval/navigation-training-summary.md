# Navigation positive-alignment pilot

Twelve warm-start experiments were completed without accessing GNOME, original development data, or consumed source tests. The initial weights were the current v1 32 KiB int8 artifact dequantized to float; the feature contract and deployed runtime were unchanged.

The primary eight runs used navigation weights 0, 0.1, 0.5 and 1, each at seeds 17 and 29 for 15 epochs. Every step retained the expanded-data positive-menu contrastive anchor and optionally added a squared positive cosine hinge targeting 0.7. The 646 KDE training queries supplied 750 explicitly documented query/label pairs. All listed positives were included; unlisted navigation labels were never made negatives. XFCE supplied 76 development queries and 81 documented pairs; 29 normalized queries overlap KDE. No training aliases or dictionaries were used for prediction.

## Quantized primary results

| Navigation weight | Seed17 XFCE semantic raw nDCG@5 | Seed29 XFCE semantic raw nDCG@5 |
| --- | ---: | ---: |
| 0 control | 0.3884 (fails retention after quantization) | 0.3666 (initial checkpoint) |
| 0.1 | 0.5089 | 0.5206 |
| 0.5 | 0.5156 | 0.5313 |
| 1.0 | 0.5215 | 0.5113 (fails retention after quantization) |

All values are known-positive diagnostics over 73 semantic-only XFCE queries. Unlisted destinations are unjudged, so these numbers do not establish exhaustive relevance or navigation false-positive rates. Cutoffs were fitted exclusively on the existing expanded calibration set.

The frozen research candidate is `packages/model/experiments/navigation-align0p5-seed29`, selected at epoch2. Its 32,768-byte payload SHA256 is `fcb4be56a9a8ddb82e02de9ae31f0822e2e211b4dc65e61cc24dc285620dd6a2`. The calibration cutoff is `0.7053964138031006`.

It satisfies the final expanded-data gates: semantic nDCG 0.45674, overall nDCG 0.45771, relevant coverage 0.4875, and no-match rate 3.98%. Overall regression from the actual baseline 0.4653183 is 0.00761, below 0.01. These are development retention checks, not held-out proof.

XFCE semantic raw recall@1/@5 is 0.2740/0.8037, versus raw nDCG 0.5313. **At the retained calibration cutoff, semantic coverage is only 15.1%, recall@1/@5 is 0.0548/0.0548, and nDCG is 0.0548.** The raw embedding improvement is therefore not a useful cutoff guarantee. Root-owned untouched holdout evaluation remains necessary; this artifact was not deployed.

## Exploratory follow-ups

Two seed17 WordNet-positive-pair runs used the prepared 504 train-label-filtered synset pairs at weight0.1: lexicon-only and navigation0.5 plus lexicon. The combined result was 0.5072 raw semantic nDCG, below navigation0.5 alone; lexicon-only was 0.3794 and failed the quantized false-positive gate. Sense-specific synonymy was never treated as universal interface equivalence. No further lexicon search was run.

Two navigation0.5 runs then tested the fixed polite-prefix list at seeds17/29. Only positive expanded TRAIN rows were shortened, with one matching leading prefix removed and at least two remaining tokens. Negation and action words remained intact. There were 1,202 eligible derivatives; mean length fell from 8.88 to 6.85 tokens. Anchor batches mixed original and derived pools50:50 at the same 7,165 total examples and56 steps per epoch, oversampling the eligible subset. No derived query overlaps XFCE.

Both shortened-query runs retained epoch0 because trained checkpoints failed the retention gates. This augmentation was rejected. All new training stopped after these twelve runs.

## Evidence and reproduction

- `eval/navigation-training.json`: primary eight runs and independent int8 evaluation.
- `eval/navigation-training-lexiconpilot.json`: two exploratory lexical-prior runs.
- `eval/navigation-training-shortquery.json`: two fixed-prefix runs.
- `eval/navigation-training-augmentation-audit.json`: exact augmentation counts and overlap.
- `eval/navigation-training-selection.json`: frozen choice and final gate audit of all12 candidates.
- `runs/navigation-*/training.json`: full epoch histories and input hashes.

Five portable tests cover preservation of negation/action text, prefix bounds, exclusion of no-match rows, multiple documented positives, and missing-positive rejection. Training ran with pinned MLX0.31.1 on Apple Silicon Metal. Recorded loop times include development evaluation/checkpoint work and exclude feature preparation; they are not inference benchmarks.

```sh
uv run python -m unittest training.test_navigation_augmentation
uv run python -m training.experiment_navigation --tag reproduce
uv run python -m training.experiment_navigation --config configs/navigation-lexicon.json --tag reproduce-lexicon
uv run python -m training.experiment_navigation --config configs/navigation-short-query.json --tag reproduce-short
```

Tags preserve all existing artifacts. Follow-up configurations explicitly identify exploratory choices; no final-test or production-quality claim follows from repeated development selection.
