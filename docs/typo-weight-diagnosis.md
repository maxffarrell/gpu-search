# Why the learned encoder confuses `profle`

The current navigation checkpoint scores `profle` against `Profile` at **0.2390**, but against `Invoices` at **0.7065**. This is a weight/representation problem even though the library's separate lexical tier can recognize the typo. No runtime or ranking changes were made for this diagnosis.

The character features already provide the right signal: character-only cosine is **0.8080** for `profle`/`profile` and **−0.1876** for `profle`/`invoices`. The typo retains 12 shared character buckets with `profile`.

The word and character families have equal *coefficients*, but unequal vector magnitudes. For `profle`, the single word embedding has norm **0.8285**, while the average of 18 character features has norm **0.2646**—a 3.13× ratio. Averaging many character rows reduces their magnitude; equal family coefficients do not imply equal influence after normalization.

| Query/candidate | Full cosine | Query word contribution | Query character contribution |
| --- | ---: | ---: | ---: |
| `profle` → `Profile` | 0.2390 | 0.0444 | 0.1946 |
| `profle` → `Invoices` | 0.7065 | 0.6642 | 0.0423 |

The contributions sum to full cosine and include the candidate's complete embedding. They are not the same as independently normalized word-only or character-only similarity.

The word hash makes an unseen spelling select an unrelated learned row:

- `profle` hashes to word bucket 208, shared by TRAIN tokens including `shift`, `trim`, `350` and `transferred.`.
- `profile` hashes to bucket 17, shared with tokens including `dictation` and `fields`.
- `invoices` hashes to bucket 84, shared with `accessibility`, `tree` and others. `accessibility` occurs 19 times across the inventoried unique TRAIN texts.

These are actual feature aliases; they do not prove which particular training example caused the final vectors. The typo and `Invoices` do not directly share their word bucket. Rather, the typo receives an unrelated word vector whose direction dominates its useful character signal.

A weight-level experiment should therefore add **train-only clean/noisy embedding consistency**, alongside the existing positive-menu and navigation objectives, while preserving clean embeddings with a frozen clean teacher. This asks the model to compensate for noisy word hashes using retained subword information. It adds no inference bytes. Broad typo-label training is preferable to a special case for `profle`; the named Profile example must remain outside training and checkpoint selection.

Do not infer that stronger consistency is always better: shared word rows also represent legitimate unrelated terms, so excessive alignment can damage clean retrieval. Keep clean-quality, coverage and no-match gates, and evaluate corruptions grouped by previously unused clean labels. Family renormalization or a character-heavy inference mix would change runtime behavior and is outside this weight-only experiment.

The exact vectors, feature IDs, norms, collision terms and source hashes are in `eval/typo-weight-diagnosis.json`. The artifact payload hash is `fcb4be56a9a8ddb82e02de9ae31f0822e2e211b4dc65e61cc24dc285620dd6a2`; only existing TRAIN corpora were scanned. No consumed source test or typo holdout was read.
