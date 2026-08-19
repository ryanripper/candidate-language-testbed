# Repository restructure notes

The research in this repository was produced in a series of analysis sessions
inside a sandboxed working directory, not in a git repository. Turning it into a
public repo required a folder reorganization and a set of path fixes. This file
records exactly what changed, so that any difference between the code here and
the code that produced the committed results is auditable.

**Nothing about the analyses changed** — no seeds, no hyperparameters, no metrics,
no decision rules. The edits are confined to where files live and how scripts
find each other.

## Folder mapping

| Original folder | In this repository |
|---|---|
| `ws0/` | `ws0-harness/` |
| `embeddings-pca-analysis/` | `analyses/00-embeddings-pca/` |
| `sentence-transformer-analysis/` | `analyses/ws1-sentence-transformers/` |
| `topic-modeling-bakeoff/` | `analyses/ws2-topic-bakeoff/` |
| `llm-scaling-analysis/` | `analyses/ws3-llm-scaling/` |
| `ws4-preanalysis/` | `analyses/ws4-preanalysis/` |
| `synthesis/` | `analyses/synthesis/` |
| `synthetic-candidate-tweets/` | `data/synthetic-candidate-tweets/` |
| plan / notes / survey `.md` files at top level | `docs/`, `docs/plans/`, `docs/skills/` |
| `write-ups-for-review/` | removed — byte-identical duplicates; see [writeups.md](writeups.md) |

## Code edits (65 substitutions across 26 files)

**1. Cross-workstream path resolution.** Scripts located the harness and each
other by hardcoded sibling directory names (`HERE.parent / "ws0"`,
`ROOT / "topic-modeling-bakeoff"`, …). These now resolve to the new locations:

- `HERE.parent / "ws0"` → `HERE.parents[1] / "ws0-harness"`
- `ROOT / "ws0..."` → `ROOT.parent / "ws0-harness..."`
- `ROOT / "sentence-transformer-analysis"` → `ROOT / "ws1-sentence-transformers"`
- `ROOT / "topic-modeling-bakeoff"` → `ROOT / "ws2-topic-bakeoff"`
- `ROOT / "llm-scaling-analysis"` → `ROOT / "ws3-llm-scaling"`
- `ROOT / "synthetic-candidate-tweets"` → `ROOT.parent / "data" / "synthetic-candidate-tweets"`

**2. Hardcoded sandbox absolute paths.** Several scripts pointed at
`/home/claude/work/...` and `/mnt/user-data/uploads/...`, which exist only in the
session container the analyses were run in — these were broken for any other
user, in any layout. Each affected script now derives its own location:

```python
_HERE = _Path(__file__).resolve().parent.parent   # this analysis folder
_ROOT = _HERE.parents[1]                          # repository root
```

and `DATA` / `OUT` / `FIG` are computed from it. In `ws0-harness/`, `DATA` and
`FROZEN_VALIDATION` are computed from the existing `HERE` constant.

One special case: `analyses/ws4-preanalysis/scripts/02_embed.py` needs a compiled
GloVe binary, previously at a fixed sandbox path. It now reads the `GLOVE_BIN`
environment variable, falling back to `third_party/glove/build`. GloVe is not
vendored here — build it from the Stanford source if you want to reproduce the
GloVe column of the WS4 preanalysis table.

**3. Navigational READMEs** (`ws0-harness/README.md`, `analyses/*/README.md`)
had their folder references and their links to the execution plan updated to the
new locations.

**4. Link targets in historical documents.** One narrow exception to the
"don't edit historical documents" rule: `NOTES-readout.md`, the plan documents,
and two write-ups contained `computer://` links pointing at absolute paths on the
author's own machine. Those links are meaningless to anyone else and leak a local
filesystem path, so the **link targets** were rewritten to repository-relative
paths. No prose, number, claim, or date was altered.

**5. One added directory.** `analyses/ws1-sentence-transformers/intermediate/`
holds a `.gitkeep`, because `scripts/01_embed_corpus.py` writes into that folder
without creating it first — on a fresh clone the script would otherwise fail on
its first write.

## What was deliberately NOT edited

`preregistration.md`, `*-writeup.md`, `article_draft.md`, `NOTES.md`,
`NOTES-readout.md`, and the plan documents in `docs/plans/` are **historical
records** — several are pre-registrations that were written and dated *before*
the corresponding unseal, and their evidential value depends on them not being
rewritten after the fact. They are preserved verbatim and therefore refer to
files by their original folder names. Use the mapping table above when following
a path mentioned in one of those documents.

## Files excluded from version control

See `.gitignore` and the `REGENERATE.md` file in each affected folder. Roughly
106 MB of derived binary artifacts (distance matrices, candidate representation
matrices, sealed-corpus parquets) are excluded and rebuildable from committed
scripts. Also omitted: a technical writing sample PDF and its annotated
companion, which are job-application artifacts rather than research outputs.
