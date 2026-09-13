# Bucket/dimension tradeoff: no winning new feature contract

Nine real MLX runs compared the current bigram25 representation with two alternative allocations of the same or fewer weight bytes. All use existing training data, the same positive-menu contrastive loss, three seeds (17, 29, 43), and at most 30 epochs. The cutoff is fitted only on calibration; expanded development selects eligible checkpoints. No source tests, consumed old tests, or original development data were opened by this experiment.

| Allocation | Int8 weight bytes | Mean semantic nDCG@5 | Held-family mean | Unseen UI namespace mean |
| --- | ---: | ---: | ---: | ---: |
| 1024 word + 1024 char buckets, 16D control | 32,768 | 0.5290 | 0.2033 | 0.2258 |
| 2048 word + 2048 char buckets, 8D | 32,768 | 0.3860 | 0.0896 | 0.0911 |
| 2048 word + 1024 char buckets, 10D | 30,720 | 0.4520 | 0.1359 | 0.1463 |

These are independent NumPy evaluations of int8 weights, with a calibration-only cutoff fitted to each quantized model. Every exported seed meets the 5% development false-positive threshold. Both new allocations underperform their matched control at every seed and on both held-family slices in aggregate. They are rejected. This does not establish that hash collisions are harmless; it shows that reducing dimensional capacity to buy more buckets was counterproductive here.

The training corpus has 27,282 distinct word/bigram feature byte strings sharing the word table and 15,710 character feature byte strings. Doubling buckets lowers maximum word bucket occupancy from 47 to 27 and character occupancy from 27 to 18, but the retrieval loss outweighs this reduction. The counts describe collisions, not their causal contribution to error.

The control's seed-43 checkpoint reached 0.5518 quantized semantic nDCG and 4.55% development false positives at epoch 15. It is a current-contract research candidate, not proof of a production improvement; the development set has been reused for selection. No live model or runtime was changed.

Each selected checkpoint has a separate research manifest, byte payload and numeric fixtures under `packages/model/experiments/collision-*`. New feature versions never replace the baseline contract. All payloads were checked against their 32,768-byte budget. Maximum MLX/NumPy embedding-component error across all runs was 6.11e-07. Fixed-float-cutoff quantization loss was at most 0.00269, below 0.01. Metadata/runtime transfer overhead was not benchmarked for these rejected contracts.

Full histories and source hashes are under `runs/collision-*`; detailed slice metrics, fixed-cutoff quantization comparisons, vector parity and collision accounting are in `eval/collision-tradeoffs.json`. Timing includes the materialized training/evaluation loop and excludes feature preparation; concurrent research can contend for the GPU, so this is not an architecture latency comparison.

```sh
uv sync --frozen
uv run python -m training.experiment_collision --tag reproduction
```

Training requires Apple Silicon Metal. The tag preserves existing immutable checkpoints. The frozen serialization and allocation choices are in `configs/collision-contracts.json`; optimization settings are in `configs/collision-training.json`. Signed hashing, int4/QAT and new data were not mixed into this test.
