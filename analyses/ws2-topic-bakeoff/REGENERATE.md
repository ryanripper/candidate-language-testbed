# Regenerating the excluded WS2 artifacts

Two files are excluded from version control (~28 MB total):
`outputs/stagec_llm.npz` and `outputs/stagec_llm_refined.npz` — the per-topic
candidate-distance tensors used in Stage C. Every scoreboard, ladder, validity
table, topic assignment array, and figure is committed.

Rebuild from inside this folder:

```bash
python scripts/07_stagec.py llm
python scripts/07_stagec.py llm_refined
```

Stage C needs the WS1 tier-B (Model2Vec) tweet embeddings, so run
`../ws1-sentence-transformers/scripts/01_embed_corpus.py` first if
`../ws1-sentence-transformers/intermediate/` is empty.

**Also never committed** (regenerable, and not needed to read the results):

- `intermediate/` token and TF-IDF matrices — `python scripts/01_prepare_tokens.py`
- Stage C tensors for the `lda`, `lsa`, `nmf`, and `bertopic` entrants — same
  command as above with the entrant name; they were computed, scored, and
  discarded because those entrants lost Stage A.
- LLM label chunk files — `scripts/03_llm_sample.py` and `04_llm_propagate.py`.
  **These call an external model API and are not free to re-run.** Their results
  are committed as `outputs/llm_taxonomy.json`, `outputs/llm_sample_labels.csv`,
  and the `assignments_llm*.npy` arrays, so nothing downstream needs the API.

`outputs/assignments_llm_refined.npy` (K = 13, ARI .890) is the assignment file
WS3 and the synthesis stage consume. It is committed.
