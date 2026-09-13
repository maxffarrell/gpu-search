# Feature experiments

The selected research candidate is `features-bigram25-seed29-e10`, epoch 9: two 1024×16 tables, 32,768 int8 weight bytes, trained with MLX 0.31.1 on Metal. It adds ordered adjacent token bigrams to the existing word table. No extra embedding table, vocabulary or inference framework is required. The byte figure covers weights, not the complete browser download.

The artifact is frozen at `packages/model/experiments/features-bigram25-seed29-e10/`, payload SHA-256 `166f105df6a3f4e0941828b1130d06671c2b5bbd759471811f607a2fa7ae30f8`. The standard experimental manifest deliberately retains `validatedSemanticCutoff: null`; its research calibration result is not a final release approval. `feature-contract.json` specifies serialization and pooling. Sixteen numerical fixtures cover multibyte text, emoji, normalization, punctuation, repeated occurrences, 32-token bounds and 64-scalar token bounds. The portable NumPy reference is `training/features_bigram.py`; browser implementations must explicitly recognize `gpu-search-features-bigram25-v1`.

## Feature comparison

The ten-epoch screen uses the same pooled 16 architecture, training rows, optimizer settings and per-seed shuffled update sequence. It compares original features, 75% word/25% character weighting, and 25% bigram weighting inside the word family. Calibration alone sets the cutoff. Development selects epochs that satisfy the 5% no-match constraint, then maximizes semantic nDCG@5 and relevant coverage. Original pilot development is excluded from selection.

| Arm | Seed | Selected epoch | Int8 semantic nDCG@5 | Dev no-match |
|---|---:|---:|---:|---:|
| Original features |17|10|.46526|4.83%|
| Original features |29|1|.27704|4.55%|
| Original features |43|None feasible|—|Above5%|
|75% word weighting|17|None feasible|—|Above5%|
| Bigram25 |17|8|.50253|4.83%|
| Bigram25 |29|9|.52254|4.83%|
| Bigram25 |43|8|.52161|3.69%|

The selected-checkpoint comparison favors bigrams, but the seed 29 control only satisfies the no-match condition at epoch 1. That large difference combines feature quality with calibration/optimization behavior. At the equal ten-epoch budget, float semantic improvements are+.0625,+.0050,+.0206 for seeds 17/29/43; paired bootstrap intervals include zero for the latter two. Do not present this as a significant equal-update gain for every seed. Independent NumPy results and per-source/held-family slices are in `eval/feature-experiments-independent.json`.

The selected seed 29 has expanded-dev semantic nDCG .522537, relevant coverage and source slices recorded in the report, and 4.83% global no-match rate. Its VSCode slice nDCG is .290658. Aggregate safety can hide weaker unseen-domain slices; the final evaluator must report those separately.

## Post-selection transfer

`eval/feature-experiments-selection.json` froze seed 29 before the one original-pilot transfer check. Raw pilot nDCG is .684893 versus .654799 for the original pilot and the supplied deployed reference .6460, passing the .01 regression guards. The paired interval versus the original pilot is [-.0731,.1346], so significant improvement is not established. Raw scores return results for no-match queries; the calibrated combined system instead has .43333 nDCG, .45 relevant coverage and .05 no-match rate on that pilot slice. No tuning followed the transfer result.

## No-match objective interaction

A separate bounded experiment combines the bigram features with the independently tested no-match training objective. It uses all-zero TRAIN judgments only, seeds 17/29/43 and at most 30 epochs. Its int8 development semantic nDCGs are .531107, .447057, .526181, with global no-match rates 4.83%, 3.98%, 4.55%. The best combination gains only .00857 over the selected simpler model; the paired interval [-.01614,.03274] includes zero. It has a worse weakest seed and lower selected VSCode nDCG(.271versus .291). The combination is therefore rejected for this candidate; its original-pilot transfer was not opened for tuning.

Those outputs remain separate in `runs/features-bigram-no-match/`, `packages/model/experiments/features-bigram-no-match-*`, and `eval/feature-experiments-bigram-no-match.json`. The training wrapper injects a feature cache only inside its own process, leaving baseline modules untouched. Its fixtures use the bigram encoder, never the original feature encoder.

## Verification and reproduction

The independent MLX-versus-NumPy pooling audit over 139 texts found maximum vector/cosine error below 4.18e-7, within the 1e-4/2e-4 tolerances. The same development set showed no semantic nDCG loss after int8 quantization. Browser parity and complete transfer-size results are separate integration evidence.

```sh
uv run python -m training.experiment_features --epochs 10
uv run python -m training.experiment_features --arms control bigram25 --seeds 29 43 --epochs 10
uv run python eval/feature-experiments-reference .py
uv run python eval/feature-experiments-parity.py
uv run python -m training.experiment_features_objective
```

Training refuses to overwrite existing run directories; use a fresh experiment checkout to reproduce it. Source dataset hashes are saved with runs. These results use weakly derived retrieval judgments and development-selected checkpoints. Neither the source datasets nor these experiments establish reviewed final-test generalization.
