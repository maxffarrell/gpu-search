# Fixed-weight calibration experiment

No learned rejection rule improved the existing global cosine cutoff while retaining the development no-match constraint and overall regression allowance. Keep the existing cutoff; do not add runtime coefficients for this result.

The finite grid was recorded before scoring in `calibration-settings.json`: cosine alone; cosine with best-other margin; cosine with mean-other margin; both margins; and mean margin with score dispersion. Each uses three ridge strengths and three calibration false-positive budgets (1%, 2.5%, 5%), for 45 variants per fixed model. Coefficients fit only calibration candidate relevance labels; thresholds use only calibration no-match maxima. Development selects among the frozen variants. A candidate's margin is an optional feature, not a mandatory positive gap: tied relevant candidates can both pass, and hard lexical tiers and cosine ordering remain unchanged.

| Fixed int8 model | Original semantic nDCG@5 | Selected rule nDCG@5 | No-match rate | Paired improvement interval |
| --- | ---: | ---: | ---: | --- |
| Current expanded seed 17 | 0.46436 | 0.46436 | 4.83% | [0, 0] |
| New no-match-loss seed 17 | 0.46774 | 0.46774 | 4.55% | [0, 0] |

Both selections reduce to a monotonic score-only transformation, returning exactly the existing ranking. On the current model, representative ridge-1 results at a 5% calibration budget are 0.35242 nDCG / 5.40% no-match for best-other margin, 0.41441 / 4.55% for mean-other margin, 0.35063 / 4.83% for both, and 0.40500 / 5.11% for mean-margin plus dispersion. Lower false-positive budgets sacrifice further coverage. Full settings, coefficients, slice metrics, and unsuccessful variants remain in the JSON reports.

All expanded menus have 10 candidates, so a menu-size coefficient is not identifiable. Candidate artifacts explicitly restrict support to 10 candidates and retain the existing scalar cutoff as the proposed fallback. No new runtime implementation or deployment is justified. Original pilot transfer was unchanged for the already-selected current model. The new no-match-loss candidate was screened with `--skip-transfer`, and its original pilot transfer rows were neither read nor scored here while training selection was still in progress.

These are weak source-derived labels, predominantly with one relevant candidate per menu. Multi-positive behavior is covered mechanically by a tied-positive regression test; this is not a measured multi-positive retrieval-quality claim. Calibration fitting and cutoff selection reuse calibration labels, and development selected 45 settings per model. The query bootstrap is descriptive and does not address source-family clustering or selection bias. No sealed test or human-reviewed final evaluation was used.

```sh
uv run python -m training.experiment_calibration
uv run python -m training.experiment_calibration --model packages/model/experiments/no-match-seed17 --prefix eval/calibration-new --skip-transfer
uv run python -m unittest training.test_calibration -v
```
