# Model card: feasibility experiment, no released semantic model

The deployed search engine is lexical with optional developer aliases. **No learned model passed the semantic release gate.** Experimental MLX weights are research artifacts and are not loaded by the demo or the default package.

## Intended scope

Short English software-navigation queries and candidate menus. Character hashing borrows the subword idea from fastText; shared pooled feature embeddings are inspired by StarSpace. This is original code, not an exact reproduction of either implementation. A bag of features cannot encode arbitrary word order or reliably distinguish action opposites. Cosine similarity is not a probability.

## Data

Original agent-authored synthetic fixtures: 120 training records, 80 development records, and 40 reserved synthetic test records. There are six menu families, six candidates per menu, 18 training concepts, 12 development concepts, and six reserved concepts. All judgments are **synthetic and unreviewed**. No external corpus or teacher was used. The actual counts are far below the proposed data targets.

The menu family is the split group, and all associated label, typo, and paraphrase examples remain together. Normalized queries do not overlap across splits. Shared ordinary English words still occur; zero literal overlap is not proof of semantic independence. The reserved synthetic test set was generated but not scored, is not genuinely sealed, and cannot replace a separately reviewed held-out set.

Every positive record has exactly one grade-3 destination; all remaining candidates are grade 0. No examples establish multi-positive relevance, grade-1 judgments, real product ambiguity, or context handling. No-match phrases are synthetic and mostly easy. These limitations prevent a release-quality conclusion even if the numeric target had been reached.

## Training

Actual runs used Python 3.12.14, MLX 0.31.1 and Metal on an Apple M4 Max, macOS as recorded in `eval/feasibility.json`. Seeds: 17, 29, 43. AdamW, learning rate 0.003, weight decay 0.0001, temperature 0.15, global gradient clipping 1.0, maximum 100 epochs and development patience 10. The complete positive training set has 90 records, so the actual full batch is 90 (configuration upper bound 128).

Eighteen checkpoints cover pooled and projected 16-dimensional encoders, pooled margin loss with fixed 0.2 margin, a 24-dimensional projected variant, word-only pooled, and character-only pooled models. The margin was a fixed pilot choice; no margin sweep was run. Every checkpoint has raw history, materialized elapsed training time, settings and weights under `runs/`. Early stopping used pre-cutoff semantic development nDCG@5. The dataset has label-only candidates, so training does not exercise aliases/context composition.

## Findings and deployment decision

At development cutoffs allowing at most one semantic false positive among 20 no-match records, learned semantic nDCG@5 ranged from 0 to 0.0833. All paired bootstrap 95% improvement intervals against lexical plus training aliases included zero. The character TF-IDF diagnostic scored 0.1462 on the same semantic slice. All 18 provisional numeric gates failed; there is also no reviewed final-test evidence.

Lexical plus the training alias map scored 0.4000 overall and 0 on the 36 semantic-only development queries. None of the train alias map's labels occur in development, so this is a weak test of a curated dictionary's usefulness. The model does not get development aliases either. This pilot suggests that the tiny synthetic data is inadequate; it does not demonstrate an architecture impossibility.

No release checkpoint, validated cosine cutoff, or trained WebGPU runtime was selected. Distillation, quantization optimization, and shader optimization stopped at the feasibility gate. The int8 artifact under `packages/model/experimental/` is an exporter smoke-test artifact from seed 17, with an explicit null validated cutoff. It is not a deployment recommendation. No int6 quality claim is made.

## Missing evidence

Human-reviewed data and final evaluation; a pinned general embedding reference; real product-source licensing/curation; teacher distillation; alias/context, action-opposite and ambiguity evaluation; MLX-to-JavaScript-to-WebGPU numeric parity for a selected model; semantic transfer bytes and browser latency. Empty and long feature inputs have shared fixtures, but that does not establish retrieval quality for them.
