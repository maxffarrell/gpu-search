# Weight-only typo consistency experiment

None of the eleven training runs was promoted. The best passing exported model improved the mixed typo development score from 0.522949 to 0.531955, below the separately evaluated word-table rebalancing candidate's 0.550897. Gains were small and inconsistent across seeds. All models retain the v1 pooled16 architecture and 32,768-byte int8 payload; this experiment adds no runtime dictionary, feature rule, or decoder change.

The [feature diagnosis](typo-weight-diagnosis.md) found that the existing character representation favors the intended spelling, while the word hash vector dominates the pooled result. We tested learned consistency rather than changing any individual query's ranking.

## Frozen scope and objective

Warm start: `packages/model/experiments/navigation-align0p5-seed29`, payload SHA-256 `fcb4be56a9a8ddb82e02de9ae31f0822e2e211b4dc65e61cc24dc285620dd6a2`, dequantized before optimization. Training used 7,165 positive expanded-training menus, 750 explicitly documented KDE navigation positive pairs, and 4,212 synthetic noisy/clean pairs from 351 training-derived labels. Typo label groups were split before corruption. These new augmentation groups are disjoint, but the base model may already have seen their clean labels. Synthetic corruption is not human query evidence.

The objective combines existing judged-menu contrastive cross entropy (temperature 0.15), KDE positive alignment `0.5 * mean(max(0, 0.7 - cosine)^2)`, typo consistency `weight * mean((1 - cosine(noisy, stop_gradient(clean)))^2)`, and frozen clean-teacher retention `mean((1 - cosine(current, initial))^2)`. Teacher observations include sampled anchor text, navigation positive text, and canonical typo labels. There is no explicit teacher matching for noisy variants. Missing navigation judgments never become negatives.

Every run used eight epochs, batch size 128, AdamW decay 0.0001, and gradient norm clipping at 1.0. The first seven runs used learning rate 0.001. After all seven retained the initial checkpoint, four explicitly exploratory followups changed only learning rate to 0.0001 and tested typo weights 0.5 and 2. Both stages include epoch zero for checkpoint selection, with earliest ties retained. Seeds control minibatch and positive-pair sampling; all share the same warm start.

Selection used the mean of raw typo top-1 on all 1,248 development queries and on 300 queries whose clean label has at most two tokens. The remaining 948 queries have longer labels. This prevents long setting names from hiding short-query failures. Checkpoints must also satisfy expanded overall nDCG@5 ≥ 0.447708, semantic nDCG@5 ≥ 0.45, **semantic** coverage ≥ 0.48, semantic false-positive rate on no-match menus ≤ 0.05, and XFCE raw semantic nDCG@5 ≥ 0.5113. Thresholds use expanded calibration only; expanded and XFCE development are selection gates. XFCE scores measure documented positives with incomplete judgments, not exhaustive relevance.

No typo holdout, Profile diagnostic, consumed GNOME test, or original development examples were read or scored by this trainer. Input hashes are recorded and checked after each run. The authorized initial named-query feature diagnosis is separate from selection; new checkpoints were not tested against that diagnostic here.

## Exported results

All numbers below come from independently recomputed NumPy evaluation of the exported int8 weights. A passing float checkpoint can fail after quantization and calibration; the export must pass again.

| LR | Typo weight | Seed | Selected epoch | Short top-1 | All top-1 | Long top-1 | Mixed score | Export passes gates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| .001 | 0 | 17 | 0 | .360000 | .685897 | .789030 | .522949 | yes, unchanged |
| .001 | .5 | 17 | 0 | .360000 | .685897 | .789030 | .522949 | yes, unchanged |
| .001 | .5 | 29 | 0 | .360000 | .685897 | .789030 | .522949 | yes, unchanged |
| .001 | 2 | 17 | 0 | .360000 | .685897 | .789030 | .522949 | yes, unchanged |
| .001 | 2 | 29 | 0 | .360000 | .685897 | .789030 | .522949 | yes, unchanged |
| .001 | 5 | 17 | 0 | .360000 | .685897 | .789030 | .522949 | yes, unchanged |
| .001 | 5 | 29 | 0 | .360000 | .685897 | .789030 | .522949 | yes, unchanged |
| .0001 | .5 | 17 | 1 | .366667 | .691506 | .794304 | .529087 | yes |
| .0001 | .5 | 29 | 4 | .370000 | .695513 | .798523 | .532756 | **no** |
| .0001 | 2 | 17 | 0 | .360000 | .685897 | .789030 | .522949 | yes, unchanged |
| .0001 | 2 | 29 | 2 | .370000 | .693910 | .796414 | .531955 | yes |

The failed weight-0.5, seed-29, lower-LR export has semantic nDCG@5 0.448420 and semantic coverage 0.477639, below both floors. Its float checkpoint passed; its exported weights do not. It is retained as a failed research artifact with `int8PassesGates: false`, and must not be promoted based on its higher typo score.

The best passing weight-2, seed-29, lower-LR export has expanded overall nDCG@5 0.454315, semantic nDCG@5 0.453339, semantic coverage 0.482111, no-match false-positive rate 0.036932, and XFCE raw semantic nDCG@5 0.512756. It adds only three correct short-query predictions over the starting model (111/300 versus 108/300), while reducing expanded semantic quality and approaching the XFCE floor. The other seed at weight 2 selected epoch zero. These results do not establish a robust learned improvement over the simpler rebalancing alternative.

## Evidence and reproduction

Full configuration, input hashes, selected metrics, export hashes and independent evaluations are in `eval/typo-training.json` and `eval/typo-training-lowlr.json`. Per-epoch histories and immutable selected checkpoints are in `runs/typo-consistency*`; research exports are in matching directories under `packages/model/experiments/`. No production artifact was changed by this trainer.

Using the pinned uv environment on Apple Silicon with MLX Metal:

```sh
uv sync --frozen
uv run python -m training.experiment_typo --config configs/typo-training.json
uv run python -m training.experiment_typo --config configs/typo-training-lowlr.json --tag lowlr
```

The trainer refuses to overwrite existing run or artifact directories. Reproduction therefore requires a fresh output workspace. Recorded per-run loop times were 1.47–1.54 seconds on this M4 Max, covering training and in-loop MLX development evaluation but excluding cache construction and independent post-export evaluation. These local times are not a portable throughput benchmark. Training stopped after the authorized eleven runs; further holdout evaluation and any release decision belong to the separately frozen candidate workflow.
