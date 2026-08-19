# WS1 Write-up — Sentence-Transformer Embeddings on the Synthetic Testbed

*Session of 2026-07-25 (seed 20260725). Executes §2 of `extensions-execution-plan.md` (E1.1–E1.5) on the WS0 shared harness. Companion documents: `preregistration.md` (locked before unseal), `article_draft.md` (public-facing narrative), `outputs/validation_results.csv` and `outputs/decision.json` (the scorecard and the formal outcome).*

---

## 1. What this session set out to test

The 2026-07-20 analysis left a standing bar: TF-IDF + SVD recovers the planted `true_ideology` of the 910 synthetic candidates at r = 0.9738 — essentially the generator's ceiling — and the best distance instrument was TF-IDF cosine at validity 0.6238 (corrected word2vec: 0.5921). WS1 asked whether modern sentence-level embeddings beat those numbers, split deliberately into two questions, because the earlier round proved a method can win one and lose the other: does a context-aware embedding produce a better ideological *axis*, and does it produce more valid *distances* — in particular via the genuinely new capability sentence embeddings unlock, comparing candidates as distributions of tweet vectors (energy distance, MMD) instead of collapsed centroids.

Two models ran as pre-registered tiers: Model2Vec `potion-base-8M` (Tier A, a static model distilled from a transformer; 256-d) and `all-MiniLM-L6-v2` (Tier B, the standard compact sentence transformer; 384-d). A stronger third model (`bge-small-en-v1.5`, Tier C) sat behind a blind gate — Ryan's D2 decision this session — described below.

## 2. Protocol decisions made before any scoring

The preregistration was written and dated before anything was scored, naming the primary metric (Pearson r of the Tier B corrected-space partisan axis vs `true_ideology`), the full comparison grid, the frozen baselines, the decision rule, and the confound-gate mechanics. Three choices deserve note.

First, the **primary instrument was fixed as MiniLM's corrected axis** — not "the best axis found," which would have been post-hoc selection dressed up as a result. Everything else (Tier A, centered-space variants) is reported as secondary.

Second, the **D2 Tier-C question was resolved with a blind gate**. The execution plan had suggested running the stronger model "only if MiniLM materially beats TF-IDF," but that comparison uses sealed truth. The workaround, pre-registered in §2a: Tier C runs only if MiniLM's *blind-safe* diagnostics beat TF-IDF's — either its best between/within party ratio reaching TF-IDF's frozen 1.3255, or its blind partisan-axis D/R correlation exceeding TF-IDF's 0.9449. This keeps the tier decision inside the blind protocol instead of leaking validation information into a design choice.

Third, the **style-axis criterion got an explicit edge-case rule**: a PC correlating ≥ 0.6 with a behavioral covariate is projected out as style — *unless* it is the blind-identified partisan PC and its D/R correlation exceeds its covariate correlation. That clause turned out to matter (below).

## 3. What was run

Script 01 embedded all 104,601 tweets once per tier (retweets embedded as their text, consistent with the retweets-as-speech stance): Model2Vec in 12 seconds, MiniLM in 6.3 minutes on the container's 2 CPU cores. Script 02 built the anisotropy variants (raw / centered / whitened), per-candidate centroids, and the centroid PCA with blind axis identification. Script 03 ran the mandatory confound gate: top-10 PCs regressed on retweet share, log tweet volume, and a blind topic-mix-entropy proxy (k-means, k = 15, over a TF-IDF+SVD tweet representation built with the WS0 recipe), then projected out style axes and rebuilt corrected spaces. Script 04 computed the full pre-registered distance grid — centroid cosine for all four variants, plus energy distance and RBF-MMD over full tweet clouds (no subsampling) for the centered and corrected spaces — and evaluated the Tier-C gate. Script 05 performed the single unseal and scored everything.

Two implementation notes for the record. The energy/MMD computation over all 413,595 candidate pairs was done with a blocked matrix-multiplication engine that accumulates per-pair cross-distance and kernel sums in one pass over the corpus; it was verified against brute-force computation on synthetic clouds (uneven sizes, agreement to 1e-4) before its outputs were used. An initial out-of-memory crash in the bandwidth heuristic (a naive 2000×2000×d broadcast) was fixed to a gemm-based form; no results were produced by the faulty version.

## 4. Blind-phase findings

**The retweet-style confound replicated — a third time, in both new spaces.** Model2Vec's PC1 correlates with retweet share at r = −0.97, MiniLM's at −0.84. Word2vec (07-20), a distilled static transformer, and a full sentence transformer all organize candidate language first by *how much of it is retweets* and only second by politics. This is now better read as a structural property of averaged social-media embeddings than as a quirk of any model family.

**MiniLM's contamination is not confined to one component.** Its *partisan* PC2 also carries r = 0.48 with retweet share — style and substance are entangled in the political direction itself, not separable by projecting out a single PC. Model2Vec's partisan PC2, by contrast, is nearly style-clean (r = 0.14), though it correlates −0.68 with the topic-entropy proxy — which is where the pre-registered exemption clause fired: PC2's D/R correlation (0.876) exceeded its covariate correlation, so it was correctly retained as signal rather than removed as style.

**The Tier-C gate failed, and Tier C never ran.** MiniLM's best blind ratio came in at 1.3243 against the 1.3255 reference (a miss by 0.001 — the gate was almost generous), and its blind axis D/R correlation at 0.700 against TF-IDF's 0.945 (not close). Scaling up the transformer while the small one was losing on blind diagnostics would have been hope, not method; the pre-registered rule made that call automatic.

## 5. Results at unseal

| Instrument | axis r | best distance validity | between/within |
|---|---|---|---|
| TF-IDF + SVD (frozen) | **0.974** | 0.624 | 1.326 |
| word2vec (frozen) | 0.886 | 0.592 (corrected) | 1.352 |
| Model2Vec potion-8M (Tier A) | 0.900 | **0.640** (corrected centroid) | 1.456 (corrected energy) |
| MiniLM-L6-v2 (Tier B, **primary**) | 0.721 | 0.455 (centered centroid) | 1.324 |

The full grid (20 measures) is in `outputs/validation_results.csv`; the highlights:

- **The primary metric lost decisively.** MiniLM's corrected axis reached r = 0.721 — below TF-IDF's 0.974 and below word2vec's 0.886. The most sophisticated model this corpus has met finished last, and the ordering of the four instruments inverts model sophistication almost exactly.
- **Model2Vec beat word2vec but not TF-IDF** (0.900 vs 0.886 vs 0.974), at 12 seconds of compute.
- **The distributional hypothesis failed.** Energy and MMD distances peaked at validity 0.48 — well under the corrected centroid's 0.640 — despite energy distance posting the best *blind* ratio of the batch (1.456). Per-tweet variation is real but is mostly noise with respect to ideology; averaging it away is a feature.
- **Whitening actively hurt** (validity ≈ 0.31 across the board): it re-inflates exactly the nuisance directions the correction suppresses.
- **The confound correction was worth +0.16 validity where style was separable** (Model2Vec: 0.476 → 0.640) **and was mildly harmful where it wasn't** (MiniLM: 0.455 → 0.397) — the entanglement diagnosed blind in §4 predicted the unseal outcome.

## 6. The formal outcome, and one exploratory finding

Under the pre-registered decision rule the result is the third branch: **a negative result**. The ST axis did not beat 0.974, and distributional distances did not beat the centroid/TF-IDF benchmarks. Consequences, as pre-registered: TF-IDF + SVD remains the axis instrument, and **Model2Vec becomes the default embedding for the real-data sweep on cost grounds**.

One finding sits outside the decision rule and is reported with the *exploratory* label it must keep: **Model2Vec's corrected centroid cosine distances are the most valid distance instrument this testbed has produced** (0.640 vs TF-IDF's 0.624). The rule named distributional distances as the challenger, so this does not formally crown a new distance instrument — but it is a concrete, cheap hypothesis for WS2's topic-conditioned distance work to test properly.

The honest caveat carried from the corpus itself: this testbed is template-generated with a small vocabulary, which is maximally friendly to lexical methods. The right conclusion is not "transformers are bad at political text" but "when the signal is lexical, lexical instruments win" — real campaign tweets may sit in the other regime, and the real-data follow-on measures exactly that.

## 7. What this changes downstream

WS2's BERTopic entrant reuses the MiniLM tweet embeddings from `scripts/01_embed_corpus.py B` (regenerable in ~6 minutes; the arrays themselves were deliberately not committed). The blind topic-entropy proxy built here (`outputs/topic_entropy_proxy.csv`) and the verified energy/MMD engine in `scripts/04_distances.py` are both reusable. All 16 distance matrices are frozen in `outputs/` for the synthesis stage's Mantel/Procrustes agreement analysis.

Decision points now live for Ryan: **D1** (LLM annotator and budget — blocks WS2's Stage-B judge and WS3's full run), **D3** (three articles vs one combined paper — the plan deferred this until after WS1; the negative result arguably strengthens the combined-paper option, since WS1 alone is a supporting chapter rather than a headline), and **D4** (whether to extend the job-application writing sample with these results or keep it frozen).

---

*Verification performed this session: metrics.py self-tests passed; the distance engine was checked against brute-force computation on synthetic clouds; headline numbers (axis r for both tiers, Tier A corrected centroid validity, TF-IDF baseline validity) were independently recomputed from committed arrays via a separate code path and matched to 4 decimals; all 41 files were committed to `sentence-transformer-analysis/` with zero rejections.*
