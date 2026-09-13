# Fixed word-table scaling ablation

The five predeclared scales do not produce an eligible replacement for `navigation-align0p5-seed29`. Reducing the word-table contribution improves synthetic typo retrieval while reducing broader semantic coverage. The unchanged scale is the only arm that passes every existing floor; no new model artifact was exported.

| Word scale | Typo top-1 | Typo top-5 | Expanded semantic nDCG@5 | Expanded coverage | Expanded no-match | Expanded overall nDCG@5 | XFCE semantic raw nDCG@5 | All floors |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1.00 | .6827 | .8502 | .4567 | .4866 | .0398 | .4577 | .5313 | Pass |
| 0.75 | .7276 | .8790 | .4532 | .4785 | .0455 | .4542 | .5327 | Fail: coverage |
| 0.50 | .7804 | .9183 | .4180 | .4463 | .0455 | .4190 | .5116 | Fail |
| 0.25 | .8790 | .9639 | .2536 | .2791 | .0398 | .2550 | .5002 | Fail |
| 0.00 | .9319 | .9856 | .1063 | .1261 | .0199 | .1079 | .4694 | Fail |

The frozen floors were expanded semantic nDCG@5 ≥ .45, semantic coverage ≥ .48, no-match semantic rate ≤ .05, overall nDCG@5 ≥ .447708, and XFCE semantic raw nDCG@5 ≥ .5113. Each ratio used its own cutoff fitted only to the expanded calibration partition. Typo metrics use full raw cosine order without lexical ranking or a cutoff. The 0.75 arm misses the coverage floor by .001467; the floor was not relaxed to accept it.

The NumPy evaluator multiplies the dequantized word table by each fixed ratio and leaves the character table unchanged. Existing query normalization and candidate label/alias/context composition remain identical. This suggests the two feature branches trade typo robustness against semantic retrieval in this checkpoint, rather than establishing that character-only inference is generally better. Retraining may find a better tradeoff; this experiment provides no evidence that it will.

Only `data/typo/dev.jsonl`, `data/expanded/{calibration,dev}.jsonl`, and `data/navigation-splits/dev.jsonl` were read. There was no access to typo holdout, diagnostic, custom queries, original test, or navigation holdout. Typo judgments are synthetic one-edit origins; XFCE judgments contain documented positives with other labels unjudged. Results are development evidence, not reviewed independent quality claims.

Positive scaling can be represented through the word tensor's float scale with unchanged int8 codes and the same 32,768-byte payload. The zero arm would require zero word codes and a valid positive scale, still occupying the same payload size. No exported artifact, runtime, demo, or dataset was changed.

Reproduce with `.venv/bin/python -m eval.typo-word-scaling`. [The machine-readable report](../eval/typo-word-scaling.json) records source hashes, model hash, exact metrics, and pass/fail for every floor.

## Authorized bounded follow-up

After the initial scan, a separate follow-up froze three intermediate ratios (.8, .85, .9) plus the unchanged control. It added the requested short-label slice (at most two whitespace-separated tokens in the clean label, 300 development queries) and selected the highest mean of short-label and all-query top-1 accuracy among arms passing every floor. The initial report remains unchanged.

| Word scale | All typo top-1 | Short-label top-1 | Mean selection score | Expanded semantic nDCG@5 | Expanded coverage | XFCE semantic raw nDCG@5 | All floors |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1.00 | .6827 | .3467 | .5147 | .4567 | .4866 | .5313 | Pass |
| 0.80 | .7147 | .4133 | .5640 | .4457 | .4714 | .5392 | Fail |
| 0.85 | .7035 | .3900 | .5468 | .4654 | .4946 | .5300 | Pass |
| 0.90 | .6979 | .3900 | .5440 | .4699 | .5018 | .5324 | Pass |

The selected .85 artifact is `packages/model/experiments/navigation-align0p5-seed29-word085`. Only its word tensor's dequantization scale changes, stored as float32 and matching little-endian bytes; the 32,768 weight bytes remain byte-identical. Its manifest hash therefore distinguishes it from its parent even though the payload hash is shared. The existing TypeScript loader accepts the exported artifact.

Reloading the exact exported metadata and bytes passes every floor: semantic nDCG .465414, coverage .494633, no-match rate .039773, overall nDCG .466369, and XFCE .529963. Exact exported all-query typo top-1 is .705128; short-label top-1 is .396667. Two additional short-label queries rank correctly in the exported evaluation. The export uses a different float32 operation order (scaling before decoding versus scaling decoded values), which can change near-tied ordering; selection evidence and exported verification are reported separately. There is no further ratio tuning.

This is a development-selected candidate for comparison with the parallel training experiments, not a deployment or independent holdout result. No demo/runtime change, diagnostic access, custom-query scoring, or holdout access occurred. [Follow-up results](../eval/typo-word-scaling-followup.json) and [exact export verification](../eval/typo-word-scaling-export-verification.json) retain their own source hashes. Reproduce with `.venv/bin/python -m eval.typo-word-scaling-followup` followed by `.venv/bin/python -m eval.export-word-scaling`.
