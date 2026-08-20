# Synthesis write-up — five instruments, one planted axis

*Run 2026-07-27, session seed 20260727. Completes §5 of the extensions
execution plan: instrument-agreement matrix, Mantel/Procrustes geometry,
divergence case studies, consolidated validation table. All analyses are
post-unseal by design; the workstreams' preregistered claims were
already settled in WS1–WS3.*

**Scope note (D1).** The WS3 ask-and-average pilot cleared its
pre-registered bar (r = .970 ≥ .90) and the recorded recommendation was
a full 910 × 5 API run. Ryan **deferred that scale-up on 2026-07-27**
(pinned; not funded for now). Every comparison involving the LLM
instrument therefore runs on the n = 150 stratified pilot subsample —
the support where all five instrument families exist on identical
candidates. Non-LLM comparisons are reported on the full n = 910 as
well. If the scale-up is ever revisited, scripts 01–04 take a
full-corpus score file without structural change.

## 1. The agreement matrix (fig 1)

On the pilot support, four scores are nearly interchangeable rankings of
the same axis: true ideology, behavioral (split-A mean retweet-source
ideology), TF-IDF+SVD PC1, and the LLM ask-and-average score — all
pairwise Pearson r ≥ .970 (behavioral–LLM .979 is the tightest estimated
pair). The embedding family trails in the order WS1 established:
Model2Vec ≈ .90 band, word2vec ≈ .88, MiniLM ≈ .72–.74. The full-corpus
matrix (n = 910, LLM absent) shows the same structure, with the
behavioral score at its generator ceiling (r = .980 vs truth,
reproduced exactly from the plan's reference value).

The more interesting object is the **error-correlation matrix**: partial
the oracle out of every estimated instrument and correlate what
remains. Two families appear.

* **Content family:** LLM–behavioral residual r = .58, LLM–TF-IDF .55,
  TF-IDF–behavioral .38. The LLM's misses are not idiosyncratic — they
  lean the same way the behavioral and lexical instruments lean on the
  same candidates.
* **Style family:** word2vec–MiniLM residual r = .61, MiniLM–Model2Vec
  .47. The two spaces WS1 diagnosed as style-contaminated share their
  blind spots with each other, not with the content family
  (cross-family residuals run .15–.38).

Practical reading: a real-data ensemble gains little from adding a
second instrument *within* a family. The diversity worth buying is
across families — one content instrument, one style-corrected embedding
instrument — because their errors are closest to independent.

## 2. Geometry: Mantel and Procrustes (fig 2)

Mantel tests (999 permutations, seed 20260727, all p ≤ .001) on the
pilot support: the three score-derived distance matrices (oracle,
behavioral, LLM |gap|) agree at r = .93–.95; every text-space distance
matrix sits far lower against the oracle geometry — Model2Vec corrected
.66, w2v corrected .60, TF-IDF .58, MiniLM .44. Procrustes similarity
of 2-D MDS maps tells the same story (.95–.96 within the score trio;
.72 TF-IDF; .43–.64 the rest). Full-corpus Mantel values match the
frozen distance-validity ladder (truth × TF-IDF .624, truth × Model2Vec
.640, truth × w2v .592).

Two honest caveats. First, score-derived |gap| matrices are **rank-1 by
construction** — a 1-D score can only produce distances that agree with
another 1-D score's distances if the scores correlate, so the .93–.95
trio is partly bookkeeping. It is still informative that no
high-dimensional text geometry gets above .66 against the oracle: on
this corpus, *direction is much easier than distance* — a result now
replicated across four instrument families. Second, the WS1 distance
figure carried forward here (corrected Model2Vec centroid, dv = .640)
keeps its EXPLORATORY label from WS1's decision rule.

## 3. The WS2 × WS3 tie: signal tiers without truth (fig 2, right)

WS2's refined instrument decomposed distance signal into three tiers on
the full corpus: retweet-content (dv = .849) ≫ policy topics (.23–.57)
≫ campaign-process (.004). Replacing the oracle with the LLM pilot
score — no truth anywhere in the pipeline — reproduces that ladder
almost exactly on the topic supports (retweet-content .87, healthcare
.64, …, campaign-process .00; tier-order Spearman ρ = .97 against the
WS2 full-corpus ordering, and the per-topic correlations track the
oracle's own within ±.06 on all 13 topics). This is the synthesis
stage's most transferable finding: **the "far from whom, on what"
decomposition can be estimated end-to-end truth-free** — LLM taxonomy
topics (WS2) + LLM positional scores (WS3) + per-topic centroid
distances — which is the exact configuration available on real data.

## 4. Divergence case studies (fig 3, packet in outputs/)

Disagreement index = SD of six instrument z-scores per candidate
(n = 150). No single covariate explains it (|r| ≤ .18 for retweet
share, volume, LLM rep SD, extremity) — divergence lives in
conjunctions, which is precisely why the plan pairs the index with
reading. The top eight cases (7 D, 1 R, against a near-balanced 73R/71D/6I
subsample — itself a lead worth keeping) sort cleanly into three mechanisms, all embedding-family
departures; the content trio never breaks apart by more than 0.7 z,
while embedding scores land up to 2.1 z from truth.

**Style overshoot (C0501 Quimby, C0742 Xiong).** Retweet shares of
.70/.68 drag w2v and MiniLM to −2.2…−2.5 z against truth ≈ −1.2. This
is the retweet-style confound — replicated in WS0/WS1/WS2 and now
visible at the level of a named candidate — surviving correction when
the diet is extreme enough. The LLM compresses slightly on the same
candidates (−.92/−1.01) for a different, visible reason: with 70% of
the bundle being org-voiced retweet text, few original sentences carry
candidate-authored position (Quimby also has the pilot's highest rep
SD, .152).

**Process-diet flips (C0256 Ostrander, C0692 Northcutt, C0474
Kowalczyk).** Small bundles (17–41 tweets) dominated by
campaign-process and local-visit content ("Met with local business
owners…", "Rode along with officers last night") — the topic tier WS2
measured at dv = .004 — leave the embedding centroids anchored in
positionless text: w2v/MiniLM/Model2Vec land near zero or on the wrong
side of it (MiniLM puts Northcutt, truth −.91, at **+1.14**).
Northcutt is also genuinely cross-pressured on content: three copies of
"both parties share the blame" on immigration plus a balanced-budget
line; every text instrument compresses her toward center (LLM −.51,
TF-IDF −.55), while the behavioral score (−.30) is itself thinned by a
7% retweet share. The instruments aren't wrong about her *presentation*
— the generator's latent is more liberal than the language she was
dealt.

**Mixed-frame bundles (C0591 Oglesby, C0089 Quintanilla, C0853
Pemberton).** Strong-to-moderate liberals whose bundles contain planted
counter-frame lines (Oglesby, truth −1.52: "Criminals don't follow gun
laws — disarming law-abiding OH citizens makes us less safe";
Quintanilla: "China is eating our lunch on trade"). The content trio
absorbs these (Oglesby: behav −1.30, llm −1.30, tfidf −1.46); the
embedding centroids do not (Model2Vec −.22, MiniLM **+.61**). MiniLM
ends on the wrong side of center in five of the eight cases, and
exactly at zero on a sixth — the
per-candidate face of its style-contaminated partisan axis from WS1.

The quant-qual loop closes the way the 07/20/26 NOTES.md entry hoped:
the index only flags *where* to look; attributing each case to
style-overshoot vs process-diet vs mixed-frame required reading twelve
tweets per candidate, and took minutes once the packet existed. (The
packet also makes the template-text caveat concrete — identical planted
sentences recur verbatim across candidates.)

## 5. Consolidated validation table (fig 4, CSV in outputs/)

`consolidated_validation.csv` assembles every headline number from
WS0–WS3 plus synthesis, each read programmatically from its source
file. The testbed's final instrument ranking, on identical support
where it matters:

* **Axis:** behavioral .977 > TF-IDF .974 > LLM .970 (pilot) >
  Model2Vec .900 > w2v .878 > MiniLM .721. Bar of .90: three
  instruments clear it; the LLM is the only one that does so with zero
  corpus training.
* **Distance:** Model2Vec corrected .640 (exploratory) ≈ TF-IDF .624 >
  w2v .592 ≫ MiniLM .397; within-topic retweet-content slice .849
  tops everything.
* **Behavior (held-out C1):** oracle 2.317 < TF-IDF 2.348 < LLM 2.368 <
  Model2Vec 2.430 ≪ nulls ≈ 2.98–3.00.
* **Topics:** blind bake-off — nobody near the .60 bar (LLM best,
  .289); with the observable retweet-routing convention, LLM refined
  K = 13 reaches ARI .890.

Per D4, `technical_writing_sample.pdf` remains frozen; this table is
the writing sample's "second results section" in waiting, not in place.

## Caveats (inherited, restated)

Template-generated text flatters every recovery number — these are
transfer results, not real-tweet forecasts. WS3 annotator agents and
the orchestrator share a model family (blinded, reported). Frozen
baselines saw split-B text (leakage favors baselines). Identifier
stripping was largely vacuous here but is load-bearing on real data;
contamination-immunity is a testbed property. 21/150 small-bundle
candidates have understated rep SDs. Score-derived distance matrices
are rank-1 (see §2). LLM coverage is the n = 150 pilot only (D1
deferred 2026-07-27).

## What carries to real data

One instrument per family, chosen by the workstream decision rules:
TF-IDF+SVD as the axis instrument; Model2Vec corrected centroids as the
(exploratory) distance instrument; LLM taxonomy + embedding propagation
as the topic instrument, with routing conventions decided *before*
modeling; LLM ask-and-average as the contamination-robust second
opinion on the axis — pending the deferred scale-up for full-corpus
coverage. Validation target changes from planted truth to roll-call
behavior (DW-NOMINATE), where the behavioral-instrument ceiling
observed here suggests behavior-derived references are the right
gold standard.

---

## Errata (added 2026-08-19, post-hoc audit)

The text above is the dated synthesis write-up and has not been rewritten;
the following corrections apply to it.

1. **The cross-family residual band ".15–.38" (§ error families) omits one
   pair.** `outputs/error_correlation_150.csv` puts TF-IDF × word2vec at
   **.449**; the other eight cross-family pairs are .149–.376. The
   within-vs-across-family contrast stands, but the band as stated
   understates the leakiest cross-family pairing.

2. **§4 "Quimby also has the pilot's highest rep SD, .152" is wrong in
   scope.** .152 is the maximum among the eight divergence cases only; the
   pilot-wide maximum is **.254** (C0450), with fifteen candidates above .152
   (`outputs/divergence_index.csv` / ws3 `scores_main.csv`).

3. **§4's style-overshoot range "−2.2…−2.5 z" is too narrow.**
   `outputs/divergence_cases.csv` runs to **−2.66** (Quimby, MiniLM). The
   article draft's "−2.2 to −2.7" is the correct range.
