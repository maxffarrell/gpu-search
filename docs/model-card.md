# Model card: expanded experimental model

Latest decision: the experimental demo now runs `navigation-align0p5-seed29`, epoch 2, using the same 16-dimensional CPU encoder and **32,768 int8 weight bytes**. Its payload SHA-256 is `fcb4be56a9a8ddb82e02de9ae31f0822e2e211b4dc65e61cc24dc285620dd6a2`. [Navigation evaluation](navigation-evaluation.md) records the frozen provider test, existing regression gates, provenance, and remaining quality gap. No production semantic release is claimed.

The previous demo artifact remains under `packages/model/candidate/`, the prior word-pair research artifact under `packages/model/candidate-v2/`, and the original pilot under `packages/model/experimental/`. Both the original synthetic test and new GNOME provider evaluation are now consumed; neither is fresh evidence for future selection. The sections below retain historical pilot and expanded-data evidence.

Training records: 7,961 total / 7,165 positive supervised, development 1,472, separate calibration 1,397. Sources are CLINC150, BANKING77, MIT VS Code descriptions and the original training fixtures. Source intent labels and documentation are adapted into weak menu relevance, not newly human-reviewed search judgments. Original dev is used only for post-selection transfer; upstream tests are unused.

On expanded development, int8 semantic nDCG@5 is 0.4644, versus old-model 0.0277 and nearest-training-paraphrase TF-IDF 0.3852. The +0.0792 baseline delta has paired bootstrap 95% interval [0.0495, 0.1089], descriptive after checkpoint selection. No-match false positives are 4.83%; semantic relevant-query coverage is 49.55%. Measured int8 nDCG loss is zero with separately calibrated cutoffs and 0.00179 at the frozen float cutoff, within the 0.01 allowance. Provisional numerical development gates pass; independent reviewed final evaluation remains absent.

The model is much stronger on new wording of seen intents than on unseen domains. Original pilot transfer semantic nDCG@5 remains only 0.0278 with the expanded calibration cutoff. Raw demo rankings deliberately apply no cutoff and can return irrelevant answers. The stable lexical engine remains responsible for guaranteed exact/typo ordering; there is no WebGPU implementation or generalization guarantee.

Only seed17 met the development no-match limit among the three selected new checkpoints. Other seeds with higher retrieval scores were rejected for excessive false positives. Matched-update old-data controls and fixed epoch comparisons separate extra examples from extra optimization. See [expanded evaluation](expanded-evaluation.md) and [source/license provenance](data-sources.md). More training did help this larger dataset through epoch10, while later selected checkpoints/other seeds did not establish better acceptable coverage.

## Original pilot history (superseded findings, retained for reproducibility)

The library remains lexical with optional developer aliases. **No learned model passed the semantic release gate.** The first model demo explicitly ran the pilot MLX-trained pooled16 int8 weights through a custom TypeScript CPU encoder, exposing raw cosine rankings alongside the lexical engine. The default package does not load those weights. The model panel has no validated cutoff and may rank unrelated candidates.

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

Human-reviewed data and final evaluation; a pinned general embedding reference; real product-source licensing/curation; teacher distillation; alias/context, action-opposite and ambiguity evaluation; selected-model MLX-to-JavaScript-to-WebGPU numeric parity (the experimental CPU encoder is checked against exported NumPy fixtures); semantic transfer bytes and browser latency. Empty and long feature inputs have shared fixtures, but that does not establish retrieval quality for them.
