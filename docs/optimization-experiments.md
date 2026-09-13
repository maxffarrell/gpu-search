# Same-size optimization experiments

## Decision

Keep the deployed `expanded-more-data-pooled-seed17` model from commit `c5a0a11`. Preserve the strongest new candidate for research, without promoting its weights. Its development gain did not transfer to the reserved synthetic evaluation. Neither model has passed the reviewed semantic release gate.

The work comprises 39 actual MLX training runs: 21 objective ablations, three wider-capacity runs, seven feature runs, three feature/objective interactions and five teacher-distillation runs. Ninety calibration configurations were evaluated on two fixed models. All training used the existing frozen training partition; calibration fitted thresholds; development selected checkpoints. These repeated development comparisons are exploratory, not independent evidence.

## Selected research candidate

`packages/model/candidate-v2/` contains `features-bigram25-seed29-e10`, selected epoch 9, with 32,768 int8 weight bytes. It uses the same 1024-word/1024-character tables and 16 dimensions, adding ordered adjacent word-pair hashes to the existing word table. No vocabulary or teacher is shipped. Payload SHA-256: `166f105df6a3f4e0941828b1130d06671c2b5bbd759471811f607a2fa7ae30f8`.

The TypeScript research runtime is `packages/model/runtime-bigram.ts`, independently checked against the portable NumPy implementation. It is excluded from the live demo import graph. The standard loader retains its original feature contract and rejects the new feature version. The candidate's `validatedSemanticCutoff` remains null.

| Expanded development, same 1,118 semantic queries | Live baseline | Research candidate |
| --- | ---: | ---: |
| Calibrated semantic nDCG@5 | 0.46436 | 0.52254 |
| Semantic coverage | 49.55% | 55.81% |
| No-match semantic return rate, 352 menus | 4.83% | 4.83% |
| Raw semantic nDCG@5, no abstention | 0.84912 | 0.84910 |

The calibrated improvement is +0.05818, with descriptive paired query-bootstrap 95% interval [0.03250, 0.08399]. Raw ranking is essentially unchanged: the gain concerns which useful results survive calibration. Query-level intervals ignore family clustering and prior checkpoint selection. They do not establish a population-wide improvement.

Original 80-row synthetic development transfer passes the existing regression guards: raw overall nDCG rises from 0.64597 to 0.68489; calibrated overall nDCG rises from 0.41667 to 0.43333. No-match returns increase from 0/20 to 1/20. The unseen VS Code development slice still has 5/22 no-match errors (22.73%), despite passing the aggregate 5% budget. A global rate must not be advertised as a guarantee for each product family.

## Reserved test, opened after freezing selection

The model and manifest hashes were frozen in `eval/optimized-selection.json` before the one-time access recorded in `eval/sealed-test-access.json`. The same expanded-calibration thresholds were applied to the reserved 40-row synthetic test. No threshold, checkpoint or training change followed these results.

| Reserved synthetic evaluation | Live baseline | Research candidate |
| --- | ---: | ---: |
| Calibrated semantic nDCG@5, 18 queries | 0 | 0 |
| Calibrated semantic coverage | 0% | 0% |
| Calibrated overall nDCG@5 | 0.40000 | 0.40000 |
| Raw semantic nDCG@5 | 0.38866 | 0.37082 |
| Raw overall nDCG@5 | 0.54524 | 0.53454 |
| Calibrated no-match return rate | 0% | 0% |

Abstaining on every semantic query is not useful semantic search, even with zero false positives. The small raw-score decrease is not evidence of a statistically established regression, but the test supplies no positive reason to replace the live model. Development and original-transfer floors passed; **the promotion decision is still no**. This test is synthetic, unreviewed and now consumed. Future tuning requires a new independent evaluation set; do not treat this set as fresh again.

The full `eval/optimized-evaluation.json` retains hashes, cutoffs, per-query top-five indices, slice counts and bootstrap comparisons. None of this is a reviewed final release result.

## What was tried

| Experiment | Outcome |
| --- | --- |
| Ordered word pairs, unchanged 32 KiB | Selected research arm; feasible scores 0.5025 / 0.5225 / 0.5216 across three seeds. Equal-update gains are not independently significant for every seed. |
| More word weight, fewer character features | No eligible checkpoint in the bounded initial screen. |
| Explicit train no-match loss | More consistent across seeds, but no compelling gain over the selected word-pair model. |
| Hard-negative margin, lower temperature, source balancing | Mixed or weaker results; retained as ablations, not deployed changes. |
| Word pairs plus no-match loss | Best selected score 0.5311, only +0.00857 versus simpler word pairs; interval [-0.01614, 0.03274] includes zero, and worst-seed performance was weaker. Rejected. |
| 24 dimensions, int6 | 36 KiB raw weights. Quantization loss stayed below 0.01, but two of three seeds exceeded the no-match budget. Rejected. |
| Learned score/margin calibration | Ninety configurations across two fixed models; none improved over the original cosine cutoff. No runtime rule added. |
| Offline MiniLM teacher distillation | Four distilled students lost to the matched zero-distillation control. Absolute and centered cosine objectives were both tested. Rejected. |

Details: [training objectives and capacity](training-experiments.md), [features and interaction](feature-experiments.md), and [calibration](../eval/calibration-summary.md). Training timings materialize MLX work but may include contention from concurrent experiments; they are not controlled performance benchmarks.

## Teacher provenance

The offline reference is [Sentence Transformers all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, licensed Apache-2.0 according to its pinned model card. Its safetensors file is 90,868,376 bytes, versus the student's 32,768 weight bytes. The teacher was loaded without remote code, using attention-mask mean pooling and normalized embeddings. Raw teacher semantic nDCG was 0.8678; calibrated nDCG was 0.1970 with 6.25% development no-match returns. A larger model alone did not solve abstention.

PyTorch 2.10.0, Transformers 4.57.6 and NumPy 2.4.3 were installed in an isolated environment. The teacher is not a production dependency. `runs/teacher-minilm/` contains the pinned model-card copy, source/model/score hashes and cached partition scores. Only `train-scores.npy` entered student losses. Calibration, development and original-transfer teacher embeddings were precomputed separately; their values were never training targets. No sealed teacher embeddings were computed.

The four distillation students and matched control used seed 17, 30 epochs, the same row order and optimizer settings, and unchanged feature/weight shape. The centered objective was an exploratory follow-up after the absolute-cosine screen. No benefit warranted further seeds. The cached teacher scores permit reproducing student training without downloading the teacher.

## Size and verification

The complete local research demo, including UI, model metadata and weights, is **36,243 bytes Brotli (35.4 KiB)**; each resource is compressed separately. The actual live demo stays at approximately 35.5 KiB with its original weights. Both fit the 50 KiB CPU budget. Research byte counts do not imply deployment or a WebGPU implementation.

CI checks the lexical 8 KiB budget and complete CPU-demo 50 KiB budget, plus a separate isolated research build. Numerical checks cover real exported embeddings and scores, Unicode, token boundaries, ordered pairs, repeated features and truncation. Actual MLX objective checks agree with the portable reference within 2.69e-8; unjudged candidates have zero gradient in those checks. Research-quality regression tests read development only; CI does not reopen the consumed test.

```sh
pnpm release:verify
pnpm bench:size:research
uv run python -m unittest training.test_export training.test_data_sources training.test_quality training.test_calibration training.test_experiment_objectives training.test_release_candidate

# Development and original-transfer reproduction; never reads the reserved test.
uv run python -m training.evaluate_release_candidate \
  --candidate packages/model/candidate-v2 --output /tmp/gpu-search-research-evaluation.json
```

See the linked experiment documents for immutable training commands. To reproduce a distillation run, choose a new seed not already present and use `uv run python -m training.experiment_distillation --seed 29 --weights 0 1 4`. The optional teacher preparation script requires the isolated dependencies listed above and a new `--output` directory; its pinned model downloads total roughly 87 MiB. Existing run directories are preserved rather than overwritten.

The next useful investment is independently reviewed software-navigation relevance: multiple valid destinations, realistic no-answer menus, action opposites and genuinely unseen product families. More optimization against the current development set would not resolve the observed transfer limitation.
