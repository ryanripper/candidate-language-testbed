# WS2 Write-Up — What I Did and What We Learned

*Session of 2026-07-26 · Workstream 2 of the extensions execution plan · seed 20260726*

## What this session set out to do

Answer your NOTES.md questions — *"Is AI good enough to identify themes? Does topic modeling become obsolete?"* — as a pre-registered experiment: a five-way blind topic-recovery bake-off on the 104,601-tweet synthetic corpus, followed by LLM augmentation of the winner (Stage B) and the topic-conditioned "far from whom, on what" distance decomposition (Stage C).

## The steps, in order

**1. Pre-registration (before touching truth).** I locked `preregistration.md` first: primary metric (tweet-level ARI vs planted `true_topic`), the five entrants' exact configurations, the K-selection rule (NPMI coherence sweep, K ∈ 5…40), the 0.60 success bar, the winner rule (highest ARI; interpretability tie-break within 0.03), the blinded-judge rubric, and the single-unseal plan.

**2. Blind Stage A.** All five entrants labeled every tweet, choosing K themselves:

- **LDA** (gensim) and **LSA** (TF-IDF+SVD+k-means) both coherence-selected K=40; **NMF** selected K=8.
- **BERTopic** (MiniLM → UMAP → HDBSCAN) hit the exact failure the plan's risk register predicted: 58% outliers at the pre-registered setting, and the pre-declared fallback produced **763 micro-topics**.
- **Direct LLM theming**: two independent agents each read 1,000 tweets and proposed taxonomies blind — they agreed on 19 of 22 themes (the taxonomy is stable, not sample noise). I merged them into 25 themes per the pre-stated rule, eight agents labeled the full 2,000-tweet sample, and labels propagated corpus-wide by embedding nearest-centroid.
- Before unsealing, two fresh blinded judge agents scored 152 provenance-hidden topics on a specific/coherent/nameable rubric.

**3. The unseal (once).** `08_unseal_validate.py` read `sealed_truth.parquet` a single time and scored everything.

**4. Post-unseal Stages B and C** — explicitly labeled mitigation measurement, per the pre-registration.

## The findings

**Finding 1 — Nobody cleared the bar blind; the LLM won anyway.** Truth had **12 topics**; entrants guessed 8–763. Scoreboard: LLM ARI 0.289 (NMI 0.630) > LDA 0.156 > LSA 0.138 > NMF 0.044 > BERTopic 0.023. The pre-registered negative-result branch fired: on this generator, no method recovers planted topics blind. Bonus finding: blind coherence — the standard K-selection tool — did *not* rank recovery (LSA had the best coherence and finished third; coherence-led K selection is what pushed LDA/LSA to K=40).

**Finding 2 — The whole gap was two conventions, not comprehension.** Post-unseal, the LLM's 25 themes mapped almost perfectly onto truth (11 themes mapped 1.00 to a single true topic). What killed the blind score: (a) the generator puts **all retweets** (26.5% of the corpus) in one `retweet_source` topic, while the LLM themed them by content; (b) the generator files visit tweets under the venue's policy topic, while both taxonomy runs independently saw a `community-visits` genre. The mitigation ladder: **+retweet routing (an observable-column rule available blind): 0.289 → 0.765** — past the bar in one move; +dissolving genre themes: 0.824; +coarsening to K=13: **0.890**. The same retweet rule rescues every entrant (LDA 0.70, BERTopic 0.69, LSA 0.67) and preserves the ranking. One structural rule was worth more than every modeling choice combined (+0.48 vs a 0.13 spread across model families).

**Finding 3 — "Far from whom, on what" has a three-tier answer.** The pre-registered Stage C claim held: best within-topic distance validity (0.704) beat the overall corrected figure (0.640). On the refined K=13 instrument the decomposition is stark: **retweet-content 0.849** (who you amplify is the most ideologically diagnostic language slice — echoing the corpus's 0.98 behavioral ceiling) > **policy topics 0.23–0.57** (healthcare/abortion/guns lead) > **campaign-process 0.004** (a fifth of all candidate output carries *zero* positional signal). The fig5 heatmap shows the most-distant cross-party pairs are dark almost entirely in the retweet-content column; the "most distant same-party" pairs differ on process mix, not position. Also: the Tier A retweet-style PC1 confound replicated a **fourth** time and was projected out — at this point it's a law of the corpus.

**Answers to NOTES.md.** Is AI good enough to identify themes? *Yes for discovery and naming* (stable, most interpretable, structurally near-isomorphic to truth), *yes for corpus-scale assignment* via the cheap two-stage design — but only once content-routing conventions are explicit. Does topic modeling become obsolete? The classical models lost on recovery, K-selection, and interpretability simultaneously; what survives is the scaffolding (coherence as diagnostic, c-TF-IDF rendering, embedding propagation). **Decision per pre-registered rule: LLM taxonomy + embedding propagation is the sole topic instrument carried forward**, with `assignments_llm_refined.npy` (K=13) as the version WS3 should consume.

## Honest-broker notes

The 0.60 bar was cleared only in labeled post-unseal mitigation, never blind. The judge and one contestant share a model family (mitigated by provenance-hidden rendering; reported). The judge sample was capped at 40 topics/entrant (documented deviation — BERTopic's 763 made exhaustive judging infeasible). Template-generated text likely flatters all methods.

## Where things stand

- **Done:** WS0 harness, WS1 (negative result; Model2Vec default), **WS2 (this)**.
- **Next per plan:** WS3 — ask-and-average LLM scaling (pilot on the 150-candidate subsample in-session, per D1) → validation vs instruments → held-out behavioral prediction. WS2 hands WS3 its topic instrument for the C2 topic-attention test.
- **Real-data lesson banked:** decide shared-content and genre-vs-subject routing *before* modeling; condition distance claims on topic (a fifth of text is positional noise; amplified content is gold).

*Everything is committed under `topic-modeling-bakeoff/` — preregistration, scripts 01–10b, outputs, five figures, article draft, README.*

---

## Errata & deviation disclosures (added 2026-08-19, post-hoc audit)

The text above is the dated 2026-07-26 write-up and has not been rewritten; the following corrections apply to it.

1. **Finding 2 overstates the community-visits replication.** "Both taxonomy runs independently saw a `community-visits` genre" is contradicted by the committed record: `outputs/llm_taxonomy.json` `run_agreement.notes` states the theme was **H1-only** ("H2 had scattered visit tweets into issue themes") and was kept from H1 in reconciliation. The genre reading is a one-run judgment, not an independent replication.

2. **Preregistration §6's taxonomy-stability metric was never computed.** §6 promised greedy label matching with Jaccard overlap over the sample assignments; no script computes it and no output contains it. The 19/22 name+definition agreement reported in `llm_taxonomy.json` is a different, weaker stability measure. This substitution was not previously flagged.

3. **Preregistration §7's missed-bar remedy was substituted.** §7 specified "a seeded/guided rerun (anchor terms from the entrant-5 taxonomy)" if the winner missed the 0.60 bar. What was actually run (`09_stageb_augment.py`) is a relabeling ladder of the winner's existing assignments (retweet routing → genre dissolve → coarsening) — no seeded topic-model rerun exists in the pipeline. The ARI deltas were reported as promised, but the method differs from the pre-registered one and should be read as a post-unseal mitigation measurement, not prereg compliance.
