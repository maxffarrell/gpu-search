# Expanded training evaluation

The new 16-dimensional pooled model improves the measured weakly labeled development task. Its quantized artifact is `packages/model/candidate`; the original deployed pilot artifact remains unchanged at `packages/model/experimental`. This is an experimental model, not an independently validated release.

The expanded corpus has 7,961 training rows, 1,397 calibration rows, and 1,472 development rows. Sources are licensed CLINC150 and BANKING77 intent datasets, VS Code settings descriptions, and the original 120 synthetic training records. Source classes and descriptions were converted into retrieval judgments; none of these retrieval judgments were human reviewed. See [data sources](data-sources.md) for licenses, transformations, split audits, and limitations. No sealed test was used.

The calibration partition alone selects each system's global cosine cutoff. Development selects checkpoints and compares systems. Of the three expanded runs, seed 17 at epoch 10 meets the preexisting 5% development no-match cap; the higher-nDCG seed 29 and seed 43 candidates exceed it. Selection therefore uses the previously registered constraint rather than silently relaxing it. Original pilot development data is evaluated only after selection as a transfer regression check.

An independent NumPy checkpoint audit also compares training-data controls at the same saved epoch. At epoch 10, semantic nDCG@5 for expanded versus original-data-only training is 0.4644 versus 0.0292 (seed 17), 0.5154 versus 0.0447 (seed 29), and 0.4948 versus 0.0324 (seed 43). Seed 17 improves from 0.2741 at epoch 1 to 0.4644 at epoch 10. This supports a benefit from the expanded supervision on this evaluation distribution; it does not make all expanded runs pass abstention. All selected, epoch-1, and epoch-10 checkpoints are recorded in `eval/expanded-checkpoint-audit.json`.

## Actual int8 artifact comparison

The table reports nDCG@5 on the 1,118 development queries having a relevant candidate but no relevant eligible lexical match. Coverage is the fraction returning any result. No-match rate is the fraction of all 352 no-match development menus returning a semantic item. Cosines are not confidence probabilities.

| System | Semantic nDCG@5 | No-match semantic rate |
| --- | ---: | ---: |
| Lexical | 0.0000 | 0.00% |
| Lexical plus all train-derived aliases | 0.0304 | 0.00% |
| Train-fitted word/character TF-IDF | 0.1232 | 3.98% |
| Nearest train paraphrase TF-IDF | 0.3852 | 7.10% |
| Original int8, calibrated | 0.0277 | 3.12% |
| New int8, calibrated | **0.4644** | **4.83%** |

The new artifact improves over the strongest baseline by **0.0792** nDCG@5, with paired query-bootstrap 95% interval **[0.0495, 0.1089]**. Improvement over the original artifact is **0.4366**, interval **[0.4073, 0.4655]**. These are descriptive development intervals after checkpoint selection, not independent generalization evidence; source-related queries are correlated. The strongest paraphrase baseline's calibration cutoff transfers to a 7.10% development false-positive rate, which exceeds the target and is reported rather than retuned on development.

Every system sees identical host candidates, aliases, and context. The complete train alias baseline can use more than the runtime's eight-alias limit, intentionally making this a stronger offline comparator. The nearest-paraphrase baseline scores each candidate against its complete collection of positive training utterances, not only the candidate label. Word/character vocabulary and IDF are fit exclusively on training texts. No development wording is inserted into either baseline.

Without abstention, new raw semantic nDCG@5 is 0.8491 versus 0.4537 for the old artifact. Raw scoring returns a score for every nonzero embedding, including unrelated candidates, and consequently returns something on 100% of no-match menus. It is useful for inspection, not a calibrated search policy. Calibrated semantic-query coverage is 49.55%; abstention removes many relevant results as well as unrelated ones.

## Seen intents and transfer

| Development slice | New calibrated nDCG@5 | New raw nDCG@5 | Nearest train paraphrase nDCG@5 |
| --- | ---: | ---: | ---: |
| Known intent, different wording | 0.5479 | 0.9153 | 0.4721 |
| Unseen intent family | 0.1673 | 0.5480 | 0.0698 |
| VS Code settings, unseen namespace | 0.1733 | 0.7915 | 0.1020 |

Most gains concern wording for known intents. Unseen-family calibrated coverage remains weak. In particular, direct train-fitted TF-IDF obtains 0.5174 on the unseen settings namespace slice, beating the new calibrated model's 0.1733 there. The global improvement must not be presented as superiority on every software-interface domain.

The original pilot development set was never added to training or checkpoint selection. Its combined calibrated overall nDCG@5 improves from 0.4000 to 0.4167; raw semantic-query nDCG improves from 0.5221 to 0.5256. Raw overall nDCG changes from 0.6548 to 0.6460, a 0.0088 regression within the preexisting 0.01 allowance. Both models have zero calibrated no-match false positives on that small transfer set. This small old synthetic set is a regression probe, not a generalization benchmark.

## Gates and regression checks

The expanded source-derived development comparison passes the provisional numerical gates registered in `eval/expanded-gates.json`. The reviewed-final-test gate remains **not passed**. No final test, broad human judgment study, general embedding teacher baseline, or multilingual evaluation was executed.

Actual float and int8 evaluation shows zero nDCG loss with independently calibrated cutoffs. At the frozen float cutoff, quantization loses 0.00179 semantic nDCG@5, also within the unchanged 0.01 allowance. CPU/NumPy numerical parity is tested separately from retrieval quality; WebGPU is not implemented.

`training.test_quality` checks metric math with multiple relevant answers and grade-1 judgments, hard tier precedence against adversarial semantic scores, cutoff tie behavior, train-only baseline fit, invalid/zero embedding abstention, and actual selected artifact quality against relevance judgments. The frozen regression floor is semantic nDCG 0.45, semantic coverage 0.48, and no-match rate at most 0.05. It also enforces the original transfer regression allowance and proves zeroed weights fail the semantic floor. Dataset SHA-256s are fixed in `eval/expanded-quality-floors.json`; missing data or artifacts fail, rather than silently skipping. This suite uses selection-used development data as a repeatable regression corpus, not as new independent evidence.

Reproduce from the repository root:

```sh
uv run python -m training.evaluate_expanded --old packages/model/experimental --new packages/model/candidate
uv run python -m unittest training.test_quality -v
```

The evaluation command writes `eval/expanded-evaluation.json`, `eval/expanded-transfer.json`, and `eval/expanded-quantization.json`. Reports retain per-query top-five indices and aggregate metrics, not large score matrices. Floor thresholds are intentionally not regenerated by the evaluator. Original pilot reports are preserved.
