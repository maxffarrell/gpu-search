# Same-size feature ablations

All arms use two 1024×16 embedding tables (32,768 int8 weight bytes), expanded training data, MLX 0.31.1 on Metal, the same initialization seeds, AdamW .003/weight-decay .0001, contrastive temperature .15, batch 128 and clipping 1. The initial screen trains ten complete epochs. Calibration alone sets the cosine cutoff; development selects the highest semantic nDCG@5 checkpoint whose development no-match rate is at most 5%. This is development evidence, not a reviewed final test.

| Feature arm | Seed | Selected epoch | Int8 semantic nDCG@5 | Dev no-match rate |
|---|---:|---:|---:|---:|
| Original feature control | 17 | 10 | .46526 | .04830 |
| Original feature control | 29 | 1 | .27704 | .04545 |
| Original feature control | 43 | none feasible | — | >.05 throughout |
| 75% word / 25% character | 17 | none feasible | — | >.05 throughout |
| 25% ordered bigrams in word family | 17 | 8 | .50253 | .04830 |
| 25% ordered bigrams in word family | 29 | 9 | .52254 | .04830 |
| 25% ordered bigrams in word family | 43 | 8 | .52161 | .03693 |

The bigram arm improves the seed17 selected checkpoint by .0376 (conditional paired bootstrap 95% interval [.0146,.0602]). At equal ten-epoch budgets, improvements are .0625, .0050 and .0206 across seeds; the latter two intervals include zero. The large seed29 selected-checkpoint difference partly reflects the control failing the no-match guard after epoch1. This is evidence for an interaction with calibration and training dynamics, not a universal gain from ordered features.

`eval/feature-experiments-independent.json` contains independent NumPy scoring, per-source/unseen-family slices, fixed-epoch comparisons, and float/int8 comparisons. None of the three bigram exports loses semantic nDCG after int8 quantization on this development set. NumPy versus sparse MLX parity over139 representative and bounded-edge texts has maximum bigram vector/cosine difference4.18e-7, below the frozen1e-4/2e-4 tolerances (`feature-experiments-parity.json`). This is not browser parity. The 32KiB count is weights only, not complete compressed browser transfer.

## Frozen selection and one transfer check

Seed29/epoch9 was selected and its hash recorded in `feature-experiments-selection.json` before opening original pilot development. Its raw pilot nDCG@5 is .684893 versus .654799 for the original pilot artifact and .6460 for the supplied deployed reference. Both .01 regression guards pass. The paired difference versus the original pilot has interval [-.0731,.1346], so statistical improvement is not established. Calibrated combined pilot nDCG is .43333 at .45 relevant coverage and .05 no-match return rate; raw rankings intentionally return irrelevant items on no-match queries. No further feature tuning follows this transfer report.

## Proposed distinct feature contract

`training/features_bigram.py` is the portable NumPy reference, with feature version `gpu-search-features-bigram25-v1`. All base normalization, tokenization, unigram/character feature IDs and bounds remain unchanged. For adjacent pairs among the first32 tokens, each truncated to64 Unicode scalars, serialize ASCII `b:` then unsigned32-bit little-endian UTF8 byte length of token A, UTF8 A, unsigned32-bit little-endian byte length of token B, UTF8 B. Apply the existing FNV1a32 hash and bucket `&1023`. Bigrams share the word table; preserve occurrences and collisions. There are at most31 bigrams.

For two or more tokens: `word = .75*mean(unigrams) + .25*mean(bigrams)`. For a single token, `word = mean(unigrams)`. Pool `.5*(word + mean(character features))` and normalize by the existing L2 rule. Candidate aliases/context use the existing composition. The selected artifact has a standard experimental manifest with this distinct version, unchanged weight hash, and feature-aware numerical fixtures. A future browser loader must explicitly support this new version; it must never silently treat these weights as the original feature contract.

## Reproduction

```sh
uv run python -m training.experiment_features --epochs 10
uv run python -m training.experiment_features --arms control bigram25 --seeds 29 43 --epochs 10
uv run python eval/feature-experiments-reference.py
uv run python eval/feature-experiments-parity.py
uv run python eval/feature-experiments-transfer.py
```

Training refuses to overwrite checkpoint directories. Run in a fresh checkout without generated experiment directories to reproduce training. The transfer script requires an existing frozen selection record and never changes it. It is a one-time post-selection diagnostic, not a checkpoint selector.

A separately authorized bounded interaction combines these features with the existing no-match objective: `uv run python -m training.experiment_features_objective`. It uses local process-scoped cache injection, all-zero TRAIN menus only, seeds17/29/43 and at most30epochs. Outputs are separate and feature-aware. Original pilot transfer remains unopened for that combination until root selection authorizes the check.
