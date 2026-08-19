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

Stage C needs the WS1 tier-A (Model2Vec) tweet embeddings, and
`09_stageb_augment.py` needs tier B (MiniLM), so run
`python ../ws1-sentence-transformers/scripts/01_embed_corpus.py A` (and `B`)
first if `../ws1-sentence-transformers/intermediate/` is empty.

**Also never committed** (regenerable, and not needed to read the results):

- `intermediate/` token and TF-IDF matrices — `python scripts/01_prepare_tokens.py`
- Stage C tensors for the `lda`, `lsa`, `nmf`, and `bertopic` entrants — same
  command as above with the entrant name; they were computed, scored, and
  discarded because those entrants lost Stage A.
- LLM label chunk files — `scripts/03_llm_sample.py` builds the sample and
  `04_llm_propagate.py` propagates labels, but **the labeling itself was
  performed by in-session LLM agents at pilot scale (per preregistration D1),
  not by API-calling code in this repo — there is no committed code that can
  re-run it.** The results are committed as `outputs/llm_taxonomy.json`,
  `outputs/llm_sample_labels.csv`, and the `assignments_llm*.npy` arrays, so
  nothing downstream needs re-labeling.

`outputs/assignments_llm_refined.npy` (K = 13, ARI .890) is the assignment file
WS3 and the synthesis stage consume. It is committed.

**Post-unseal reconstructions (2026-08 audit):**
`outputs/exploratory_rt_routing.csv` and `outputs/stagec_validity_refined.csv`
were originally ad-hoc session computations; `scripts/11_reconstruct_exploratory.py`
now regenerates both (the routing table reproduces the committed file exactly;
the refined validity table needs `stagec_llm_refined.npz` rebuilt first, per
above). `figures/fig5b_pair_topic_heatmap_blindwinner.png` was originally a
manual rename; `10b_fig5_refined.py` now preserves it in code.
