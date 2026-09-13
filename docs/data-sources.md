# Expanded training data provenance

This dataset adapts licensed intent-classification examples and software-setting documentation into local retrieval menus. The labels and menu relevance are **mechanically derived, not human-reviewed retrieval judgments**. No source establishes reliable generalization to arbitrary software interfaces.

## Sources and attribution

- **CLINC150**, Stefan Larson and collaborators, [An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction](https://aclanthology.org/D19-1131/) (EMNLP-IJCNLP 2019). [Primary repository](https://github.com/clinc/oos-eval/tree/828f8093932c8fe6ca7936c3d2e52903b1c523de), revision `828f8093932c8fe6ca7936c3d2e52903b1c523de`. Licensed **CC BY 3.0**, preserved in `data/external/clinc150/LICENSE`; original README and attribution preserved alongside. Changes: selected interface-adjacent intent families, humanized class IDs, generated distractor menus, and removed-positive negatives.
- **BANKING77**, Iñigo Casanueva, Tadas Temčinas, Daniela Gerz, Matthew Henderson, Ivan Vulić / PolyAI, [Efficient Intent Detection with Dual Sentence Encoders](https://aclanthology.org/2020.nlp4convai-1.5/) (2020). [Primary repository](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b), revision `57ec275d8078af65b7731c2a98be812d844a6d6b`. Licensed **CC BY 4.0**, preserved in `data/external/banking77/LICENSE`; original README preserved alongside. Changes: capped source-train records, grouped intents, humanized class IDs, and generated menus/negatives.
- **Visual Studio Code**, Microsoft Corporation. [Primary repository](https://github.com/microsoft/vscode/tree/8e35945bae3f2b0b3d0276963281180f1ce10cb0), revision `8e35945bae3f2b0b3d0276963281180f1ce10cb0`. **MIT**, preserved in `data/ui-settings/LICENSE-vscode.txt`. Root extraction records 460 description/setting-ID pairs and source locations in `data/ui-settings/pairs.json` and `source.json`. Documentation descriptions are weak query supervision, not naturally collected search queries. Changes: cleaned original descriptions and mechanically humanized setting IDs, whole-namespace splits, deduplication, menus, negatives. Each emitted row includes the pinned source URL and line.
- Original `data/train.jsonl` synthetic examples only. They retain their `synthetic-unreviewed` status and source identity. Original `data/dev.jsonl` is separately reserved for regression evaluation; preparation never opens original dev or test.

Third-party dataset material retains its source license and attribution, independently of this repository's code license. Neither source authors nor Microsoft endorse this derived dataset or model. Source README files retain the full recommended citations.

## Reproduction

```sh
python -m training.prepare_external --download
python -m training.prepare_external
```

The first command fetches only pinned public source URLs and saves complete license texts. CLINC distributes train/val/test in one transport file: the JSON container is decoded, but `test` and `oos_test` members are never accessed, exported, sampled, or scored. BANKING77 `test.csv` is not downloaded. Extracted CLINC train, val, and OOS train/val members are preserved; OOS examples are currently unused. Full upstream transport hashes are recorded by `--download`. Source asset hashes, output counts and SHA-256 values appear in `data/expanded/manifest.json` and `data/external/download-manifest.json`.

The expanded adapter consumes the checked-in VSCode pairs; `--download` refreshes CLINC/BANKING assets only. To independently re-extract the VSCode pairs after installing the pinned JavaScript dependencies:

```sh
mkdir -p /tmp/gpu-search-vscode-source
curl --fail --location https://codeload.github.com/microsoft/vscode/tar.gz/8e35945bae3f2b0b3d0276963281180f1ce10cb0 --output /tmp/gpu-search-vscode-source/source.tar.gz
tar -xzf /tmp/gpu-search-vscode-source/source.tar.gz -C /tmp/gpu-search-vscode-source
node --import tsx training/prepare_settings.ts /tmp/gpu-search-vscode-source/vscode-8e35945bae3f2b0b3d0276963281180f1ce10cb0
python -m training.prepare_external
```

This inspects TypeScript ASTs without executing upstream application code. Source locations, the upstream MIT license, and the extraction script hash are preserved alongside the pairs. Retain the dataset attribution and license files with redistributed training material and checkpoints; the repository's MIT code license does not replace the CLINC/BANKING source licenses.

## Splits and derivation

Intent-family groups are declared in `training/prepare_external.py` before partition assignment. CLINC music/location and BANKING identity families are unseen development targets; CLINC employment and BANKING virtual-card families are unseen calibration targets. None of those intent labels appears in any training menu. Held CLINC families use source validation only, reserving their source-train wording unused. Known CLINC families keep source train exclusively in training and divide official validation wording between dev/calibration. BANKING77 has no official validation partition: source train is deterministically split and capped at 40/10/10 examples per known intent for train/dev/calibration, with 20 source-train examples per held intent. Its official test remains reserved.

VSCode setting namespaces are assigned wholly to train/dev/calibration using a fixed SHA-256 namespace rule. No setting ID or namespace occurs in two splits, and candidate menus only draw from the same split's settings. Exact normalized duplicate queries are globally removed with training taking precedence, then dev, then calibration. VSCode cross-split near duplicates use a frozen token-set Jaccard threshold of 0.85 (minimum four tokens), checked for target labels and queries; queries are also checked against other-source examples. This conservative lexical audit does not prove absence of paraphrase leakage.

Menus contain ten label-only candidates for new source rows. Distractors favor shared intent/label tokens and fill deterministically. Source class inequality is treated as a weak negative, though neighboring intents can be jointly relevant. Approximately one tenth of training and one quarter of evaluation examples remove the source-positive candidate entirely and are marked `removed-positive-uncertain`. They model unavailable destinations but are not reviewed proof that every remaining option is irrelevant. Existing original synthetic menus retain their six-item form.

No alias or context text is added. CLINC food/travel/chitchat and unrelated intents are excluded, BANKING training is capped, and real VSCode descriptions add a software-domain transfer slice. Banking and voice-assistant queries still dominate the count: macro source/slice reporting and the separately reserved original UI regression set are required. Do not report aggregate auxiliary improvements as a passed software-search release gate.

`data/expanded/split-audit.json` records duplicate drops, zero normalized query overlap, and the held-family candidate exclusion check. `manifest.json` records exact final counts, source counts, no-match counts, slice counts, intent-family definitions and hashes. Development and calibration have distinct queries and separate unseen-intent families, but known-intent labels are intentionally shared. These are development resources; no final test has been opened.
