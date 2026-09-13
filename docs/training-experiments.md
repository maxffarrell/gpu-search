# Training objective and capacity experiments

Twenty-one 16-dimensional runs compared seven fixed training choices over seeds 17, 29 and 43. A three-run 24-dimensional follow-up tested capacity and int6 storage. These are weak-label development experiments; none used the original development set for training or selection, and none establish reviewed final-test quality.

The existing runtime, feature contract and deployed model were preserved. The 16-dimensional experiments use two 1024×16 embedding tables, totaling 32,768 int8 weight bytes. The 24-dimensional alternative requires 49,152 int8 or 36,864 packed int6 weight bytes, before metadata and runtime code.

## Controlled setup

All arms use the frozen expanded dataset, 7,165 sampled records and 56 optimizer steps per epoch, AdamW learning rate 0.003, and three fixed seeds. The original arm uses temperature 0.15. Training stops after development patience 10, with at least 10 and at most 60 epochs; capacity is capped at 30. All corresponding selected 16-dimensional no-match epochs were below 30.

Cutoffs are fitted exclusively to calibration no-match records. Checkpoint selection first requires at most 5% development semantic false positives, then maximizes post-cutoff semantic nDCG@5 and coverage. If no epoch satisfies that gate, the retained checkpoint is explicitly **NOT SELECTED**, with a false gate flag. Such weights are failure-analysis artifacts, never automatically deployable candidates.

The no-match arm adds a weight-0.25 squared hinge above cosine 0.35 for TRAIN all-zero menus. It never creates an empty positive contrastive numerator. The hard-negative arm adds a weight-0.25 margin of 0.15 against the hardest judged negative already in that training menu; it does not infer global negative labels. Source balancing samples sources uniformly, then rows within source. These choices were fixed before their runs.

The fixed total sample budget means no-match arms see fewer positive examples per epoch, and balanced arms repeat small sources more often. Actual positive/no-match and source counts are logged. This is equal optimizer-budget experimentation, not equal unique-example exposure.

## Actual 32 KiB results

| Arm | Seeds meeting development false-positive gate | Mean semantic nDCG@5 | Mean held-family nDCG@5 | Mean unseen UI namespace nDCG@5 |
| --- | ---: | ---: | ---: | ---: |
| Control | 2/3 | 0.4162 | 0.1265 | 0.1360 |
| Temperature 0.07 | 3/3 | 0.3341 | 0.0929 | 0.1353 |
| Train no-match loss | 3/3 | 0.4598 | 0.1733 | 0.2399 |
| Hard-negative margin | 2/3 | 0.4157 | 0.1421 | 0.1206 |
| Source balanced | 3/3 | 0.4473 | 0.1610 | 0.2969 |
| Balanced plus no-match | 2/3 | 0.4441 | 0.1382 | 0.3636 |
| Balanced plus hard margin, temperature 0.07 | 3/3 | 0.3684 | 0.1089 | 0.1058 |

Means include the explicitly rejected retained checkpoint where a seed failed; the gate column is essential. No-match loss had the strongest consistent result in this objective screen: worst-seed nDCG 0.4501 and all three development false-positive rates below 5%. Source balancing helped the small UI slice, but reduced aggregate consistency. These findings do not rank separately run feature-representation experiments.

The exported no-match seed-17 candidate is at `packages/model/experiments/no-match-seed17/`. It remains explicitly experimental with no validated deployment cutoff. A later independent quantized evaluation and comparison against other workstreams must determine any promotion.

## Capacity decision

Using the same no-match objective, the 24-dimensional seed-17 float checkpoint reached 0.4975 semantic nDCG@5 and 4.83% development false positives. At its **unchanged float calibration cutoff**, int8 scored 0.4964 with loss 0.0011; int6 scored 0.4936 with loss 0.0039 and 4.26% false positives.

The other two capacity seeds failed the false-positive gate: int6 rates were 8.52% and 9.09%. All three int6 quantization losses were below 0.01, but capacity did not demonstrate consistent safe coverage. The roughly 0.025 seed-17 improvement over the smaller no-match model does not justify a larger payload and decoder/runtime change on this evidence. The capacity alternative was rejected; its artifacts remain isolated for analysis.

## Evidence and verification

`eval/training-experiments.json` contains all 21 histories' selected metrics and slice summaries. `eval/training-experiments-capacity.json` records the three capacity checkpoints and fixed-cutoff float/int8/int6 comparisons. Each `runs/experiment-*/` directory retains full histories, epoch-1/10 checkpoints, configuration, source hashes and sampled counts.

Training ran on Apple M4 Max using MLX 0.31.1 Metal. The separate feature workstream sometimes used the GPU concurrently, so timing is contention-affected and cannot compare kernels or architectures fairly. Recorded loop times include checkpointing and per-epoch development evaluation but exclude initial data/feature/cache preparation; optimizer time is recorded separately. No latency claim follows from these training times.

Eleven portable NumPy tests verify probability-mass behavior, positive/negative monotonicity, multiple positives, padding and unjudged masks, no-match and mixed batches, finite-weight rejection, and truthful saved gate flags. A separate actual MLX check covered 15 objective/mask cases: maximum objective error versus NumPy was 2.69e-08, unjudged gradients were exactly zero, and every checked gradient was finite.

```sh
uv sync --frozen
uv run python -m unittest training.test_experiment_objectives
# Requires Apple Silicon Metal; tags preserve all existing checkpoints.
uv run python -m training.experiment_training --tag reproduction
uv run python -m training.experiment_capacity --tag reproduction
uv run python -m training.verify_experiment_objectives
```

Use `--arm no-match --seed 17 --tag another-run` for a bounded objective rerun. Existing checkpoint directories are never overwritten. Development is already used for model selection; new product-quality claims require new reviewed held-out data.
