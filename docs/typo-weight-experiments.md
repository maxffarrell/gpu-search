# Investigating the model's typo failure

We found a **partial weight-level improvement**, not a full fix. The demo now uses `navigation-align0p5-seed29-word085`: the word-table dequantization scale is 15% smaller, while the character table is unchanged. The int8 payload remains exactly **32,768 bytes**. This changes the effective embedding weights; it adds no dictionary, query correction, ranking exception, or inference code.

The raw model still ranks `profle` against **Invoices** above **Profile**. The main search correctly returns Profile through its separate text-matching tier. These are different claims: safer application behavior does not establish reliable learned spelling.

## Cause

[Feature diagnosis](typo-weight-diagnosis.md) showed that the character representation already favors Profile. The unrelated word-hash vector selected by `profle` dominates the pooled representation: its norm is about 3.13 times the mean character vector's norm. Equal family coefficients do not mean equal influence. Hash buckets share rows across unrelated words, limiting what a single weight update can repair without affecting other inputs.

## Experiments and selection

We generated single-edit spelling variants from existing training labels, splitting clean labels before augmentation. Training had 4,212 pairs from 351 labels, development 1,248 cases from 104 labels, and the reserved evaluation 600 cases from 50 labels. Profile was excluded from augmentation training and development selection. The baseline may have seen the clean labels during its original training: this is a reserved test of new spelling robustness, not fresh evidence of semantic generalization.

Eight fixed word-table scales and eleven consistency-training runs were compared. Training aligned noisy and clean embeddings, retained frozen clean representations, and preserved existing semantic objectives. Seven initial runs and four smaller-learning-rate follow-ups failed to beat the eligible scale adjustment. Stronger spelling improvements commonly reduced semantic quality or coverage. One selected floating-point checkpoint failed its gates after quantization and was rejected.

Selection used the mean of short-label and overall development top-one accuracy, subject to fixed semantic, coverage, no-match, and XFCE regression floors. The 0.85 scale also passed all seven existing release regression gates before the reserved spelling test was opened. No gate was relaxed to promote it.

## Reserved evaluation

| Measure | Previous weights | Rebalanced weights |
| --- | ---: | ---: |
| Top-one accuracy, 600 synthetic typo cases | 64.33% | 67.33% |
| Top-one accuracy, 144 short-label cases | 42.36% | 45.83% |
| Raw int8 payload | 32 KiB | 32 KiB |

The rebalanced model fixes 20 previously wrong cases and regresses 2 previously correct cases. The overall gain is **3.0 percentage points**. A bootstrap clustered by clean label gives a 95% interval of **[1.5, 4.67] percentage points**. NumPy and the actual TypeScript runtime agree on these aggregate held-out values. They were evaluated as the same frozen event, not independent replications. Small floating-point differences affect a few near-tied development rankings; both implementations are recorded.

The known demo checks still return Members for `coworkers` and Profile for `my information`. However, `profle` still returns Invoices in raw inference, with cosine 0.7021. The scale adjustment is therefore a measured broad improvement, not resolution of the reported model bug.

## What would constitute a fuller fix?

These results support further investigation of how the model allocates capacity between word and character features. A future experiment could constrain word-vector dominance while teaching the character branch semantic retrieval, potentially changing pooling while keeping the same weight budget. That proposal is untested. Simply increasing epochs, strengthening consistency loss, or reducing word influence further did not satisfy the current gates.

No production-ready or best-in-class claim follows from this synthetic spelling benchmark. A stronger model needs retained semantic accuracy, reliable spelling on short unseen labels, reviewed ambiguous/no-match queries, and a new reserved evaluation. This spelling holdout is now consumed and cannot be represented as untouched in later selection.

## Reproduction and evidence

- `training/prepare_typo.py`, `data/typo/manifest.json`: deterministic perturbations, label separation and source hashes. Ambiguous shared corruptions and spellings equal to another clean label are excluded. Judgments are synthetic, not human-reviewed.
- `configs/typo-training*.json`, `training/experiment_typo.py`, `eval/typo-training*.json`: bounded training runs and exact-export gates. [Training findings](typo-training.md).
- `eval/typo-word-scaling*.json`: initial sweep, bounded follow-up and exact exported-model verification. [Scaling findings](typo-word-scaling.md).
- `eval/typo-selection.json`, `eval/typo-holdout-access.json`: frozen payload and manifest hashes, candidate selection and one-time evaluation receipt.
- `eval/typo-evaluation.json`, `eval/typo-runtime-holdout.json`: reserved NumPy and actual serving-runtime evidence. `eval/typo-known-diagnostic.json` separately records the already-known failures.
- `eval/typo-release-regressions.json`: all seven existing release regression gates passed. Original consumed semantic tests were not reopened.

The training-only consistency approach was informed by [CharacterBERT and Self-Teaching for Improving the Robustness of Dense Retrievers on Queries with Typos](https://arxiv.org/abs/2204.00716). Our small hashed encoder and synthetic label task differ substantially from that work; its results are not evidence that this implementation will match them.
