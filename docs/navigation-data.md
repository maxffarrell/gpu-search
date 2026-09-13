# Author-provided settings search metadata

This extraction preserves real software search metadata from three desktop providers. It contains **85 entries, 1,209 keyword-to-destination pairs, and 1,047 within-product query groups**. There are **128 groups with multiple inherited positive destinations**. It does not create training splits, candidate menus, relevance grades for other entries, or no-match judgments.

| Provider | Repository | Pinned revision | Entries | Keyword pairs |
|---|---|---|---:|---:|
| GNOME | [gnome-control-center](https://github.com/GNOME/gnome-control-center) | `cd79a897190989ca395c6f00962f0929734103c7` | 27 | 315 |
| KDE | [plasma-desktop](https://github.com/KDE/plasma-desktop), [plasma-workspace](https://github.com/KDE/plasma-workspace) | `19dae94ec1d563d095b08602dd5eed60d485a5e0`, `df3bba49a2c0657d4165f6511b1771138ccc75ea` | 40 | 754 |
| Xfce | [xfce4-settings](https://github.com/xfce-mirror/xfce4-settings) | `488c919d23c979c9b28abd192c05b9b2310fa732` | 18 | 140 |

The [freedesktop Desktop Entry specification](https://specifications.freedesktop.org/desktop-entry/latest/recognized-keys.html) describes `Keywords` as metadata useful for entry search. The [official KDE KCM documentation](https://develop.kde.org/docs/features/configuration/kcm/) identifies `X-KDE-Keywords` as terms used to search settings modules. These fields were authored for search, unlike documentation descriptions previously repurposed as queries. They remain inherited author choices, not new independently reviewed judgments or collected end-user queries.

## Contents

- `data/navigation/raw/`: exact selected upstream metadata files, license texts, copyright notices and available SPDX/REUSE evidence. No source code is executed.
- `downloads.json`: pinned archive URL, archive hash, retained file paths, byte sizes and hashes.
- `panels.json`: original English label, description, keyword list, source path/line references, entry role, desktop visibility flags, immutable provider/product lineage and license evidence.
- `keyword-pairs.json`: exact authored query spelling and one inherited positive destination per row. It explicitly leaves other destinations unjudged.
- `query-groups.json`: groups case-insensitively within a product and unions all matching authored destinations. It never converts absent keywords into negative relevance. Original spelling remains in pairs.
- `manifest.json`: counts, pinned revisions, parser skips and explicit unresolved split/relevance policy.

Only unlocalized English fields are selected. Translation keys such as `Name[fr]` are ignored. Desktop keywords use semicolons; KDE keywords use commas. Desktop string escapes are decoded, but no paraphrases, spelling corrections, synonym guesses or descriptions-as-queries are generated. Exact duplicate pairs within one entry are removed.

There are 75 direct settings panels/modules, eight preferred-application launchers, one settings catalog and one background service. These roles are exposed so menu construction can exclude helper/service entries explicitly. Visibility flags alone cannot identify a settings panel: integrated GNOME entries may intentionally be hidden from application launchers.

## Licensing and lineage

Raw metadata remains under its upstream terms. GNOME and Xfce preserve their repository `COPYING` GPL version 2 texts. KDE retains its mixed GPL/LGPL and other `LICENSES` texts plus available per-file notices and REUSE records. One extracted file has an explicit `GPL-3.0-or-later` SPDX identifier; the other 84 metadata files lack a direct file-level SPDX declaration. Their records retain the actual evidence and flag that absence. A repository license collection does not prove every metadata file has the same license, and this extraction does not invent a GPL version, an “or later” grant, or an MIT assignment.

The repository's MIT code license does not relicense these upstream files or derived metadata. Preserve original paths, full license notices and attribution when redistributing them. File-license resolution and any downstream model distribution decision must use the retained evidence rather than treating these records as MIT training assets.

Lineage IDs are fixed as `gnome:gnome-control-center`, `kde:plasma-desktop`, `kde:plasma-workspace`, and `xfce:xfce4-settings`. KDE products share a provider and may share concepts or wording; splitting them does not establish independent-provider generalization. Root coordination must freeze train/calibration/test lineage policy before consuming these extractions. Unlisted keywords are not evidence of irrelevance, especially for broad shared terms such as “screen” or “keyboard.”

## Reproduction

```sh
python -m training.prepare_navigation --download
python -m training.prepare_navigation
```

The first command downloads exact pinned official repository archives and retains selected metadata/license files without extracting or executing application code. The second deterministically reproduces the extraction from checked-in raw files. Existing training datasets, models and evaluation splits are not modified.

Parser contracts run with `python -m unittest training.test_navigation -v`. They use temporary synthetic metadata, covering localized-field exclusion, shared-keyword positive unions, no invented negative grades, entry roles, preserved visibility/license evidence, desktop escapes and byte-identical reproduction. These tests do not train or score any provider, including GNOME.
