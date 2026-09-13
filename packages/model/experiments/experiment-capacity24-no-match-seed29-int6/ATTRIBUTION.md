# Training-source attribution

Model: `experiment-capacity24-no-match-seed29-int6`, pooled24, int6, experimental.

Training used adapted examples from:

- CLINC150, Stefan Larson and collaborators, “An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction” (2019), https://github.com/clinc/oos-eval — CC BY 3.0, https://creativecommons.org/licenses/by/3.0/.
- BANKING77, Iñigo Casanueva, Tadas Temčinas, Daniela Gerz, Matthew Henderson, Ivan Vulić / PolyAI, “Efficient Intent Detection with Dual Sentence Encoders” (2020), https://github.com/PolyAI-LDN/task-specific-datasets — CC BY 4.0, https://creativecommons.org/licenses/by/4.0/.
- Visual Studio Code documentation/source, Copyright Microsoft Corporation, https://github.com/microsoft/vscode — MIT. The complete license is preserved at `data/ui-settings/LICENSE-vscode.txt`.
- Original gpu-search synthetic training fixtures.

Changes included selected source subsets, intent-ID humanization, documentation cleaning, derived menu distractors and removed-positive negatives. The examples were used to train new weights; source authors do not endorse this model. See `docs/data-sources.md` and `data/expanded/manifest.json` for pinned revisions, retained licenses, exact transformations and split hashes. Preserve this attribution when redistributing the model.

Source intent classes are weak retrieval supervision. This artifact is not independently validated for production software search and has no validated semantic cutoff.
