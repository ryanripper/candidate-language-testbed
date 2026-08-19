# Five instruments, one axis: what a measurement zoo agrees on, where it errs together, and when you just have to read the tweets

*Synthesis stage of the extensions execution plan — the piece that runs after the workstreams stop. Run 2026-07-27, seed 20260727, on the shared harness's frozen artifacts: the 2026-07-20 embeddings/PCA baselines (WS0), sentence-transformer axes and distances (WS1), the refined LLM topic instrument (WS2), and the ask-and-average pilot scores (WS3). No new blind claims — every preregistered gate was already settled upstream; this stage asks what the instruments say about* each other.

## The setup

Four analyses on one synthetic testbed (910 fictional 2022-cycle congressional candidates, 104,601 tweets, planted ideology/topic/framing) produced five families of ideology instrument: a **behavioral** score (mean ideology of the org accounts a candidate retweets), a **lexical** axis (TF-IDF+SVD PC1, r = .974, the incumbent), two **corpus-trained** embedding axes (word2vec .878), two **pretrained** embedding axes (Model2Vec .900, MiniLM .721), and an **LLM ask-and-average** score (.970 on the 150-candidate pilot). Each was validated against planted truth in its own workstream. The synthesis questions are different: do they agree with *each other*, do they fail on the *same candidates*, do their *geometries* match, and what happens on the candidates where they fall out?

One scope note up front: the WS3 pilot cleared its preregistered gate (r ≥ .90), and the recorded recommendation was a full 910-candidate API run. That scale-up is **deferred as of 2026-07-27** — pinned by decision of the author, not funded for now. So every all-instrument comparison here lives on the 150-candidate stratified pilot support; the gate result stands on the record if the run is revisited.

## Result 1 — Agreement is a two-club town

The correlation matrix (fig 1) has a crisp block structure. Truth, behavioral, TF-IDF, and LLM are pairwise r ≥ .970 — near-interchangeable rankings, with the tightest *estimated* pair being behavioral–LLM at .979. The embedding axes trail exactly as WS1 left them (Model2Vec ~.90, w2v ~.88, MiniLM ~.72).

But agreement among instruments that all track a strong common signal is cheap. The sharper object partials the oracle out of every instrument and correlates the residuals — pure shared *error*. Two families emerge:

| | residual r |
|---|---|
| LLM × behavioral | **.58** |
| LLM × TF-IDF | .55 |
| TF-IDF × behavioral | .38 |
| word2vec × MiniLM | **.61** |
| MiniLM × Model2Vec | .47 |
| cross-family pairs | .15–.38 (8 of 9; TF-IDF × word2vec .45) |

A **content family** (LLM, lexical, behavioral) misses on the same candidates in the same direction — the LLM annotator, it turns out, errs most like the *retweet-source* signal, which is a satisfying mechanical hint: when the text is ambiguous, whoever a candidate amplifies is also what tips a bundle's read. And a **style family** (word2vec, MiniLM — the two spaces WS1 caught carrying a retweet-style axis) shares its own, different blind spots. The ensemble lesson is direct: a second instrument from the same family buys correlated errors; diversity worth paying for crosses the family line.

## Result 2 — Direction is easy; distance still is not

Mantel tests over pairwise-distance matrices (999 permutations each) repeat a lesson this project keeps re-learning, now in its most general form. Score-derived geometries (oracle, behavioral, LLM |gap|) agree at r = .93–.95 — though as rank-1 objects they partly must. The real content is the ceiling for text-space geometry against the oracle's: Model2Vec corrected centroids .66, corrected word2vec .60, TF-IDF .58, MiniLM .44 (pilot support; the full-corpus values land on the frozen distance-validity ladder, .640/.592/.624/.397). Procrustes on 2-D MDS maps says the same. Every instrument family recovers *who is left of whom* far better than *how far apart two candidates are* — across lexical, corpus-trained, pretrained, and LLM instruments alike. On this testbed, distance claims need topic conditioning (below) or they are mostly noise dressed as cartography.

## Result 3 — "Far from whom, on what," now truth-free

WS2's best result was a decomposition: within-topic distance validity splits into three tiers — retweet-content (.849) ≫ policy topics (.23–.57) ≫ campaign-process (.004). The synthesis test replaces the oracle with the WS3 pilot score and re-runs the whole decomposition with **no ground truth anywhere**: WS2's topics come from an LLM taxonomy, the distances from embedding centroids, the reference gaps from LLM ask-and-average scores. The tier ladder reproduces at Spearman ρ = .97, and each topic's truth-free correlation lands within ±.06 of its oracle counterpart (retweet-content .87 vs .85; campaign-process .00 vs .00). That is the most portable sentence this testbed has produced: the *entire* "far from whom, on what" pipeline — taxonomy, routing, per-topic distances, positional reference — can be estimated on real data, where no `true_ideology` column will ever exist.

## Result 4 — The divergence cases: three mechanisms, all visible by reading

Rank the 150 pilot candidates by the SD of their six instrument z-scores and no single covariate explains the top of the list (all |r| ≤ .18) — disagreement lives in conjunctions. So, per the plan (and the NOTES.md quant-qual entry that inspired it): pull the eight most-contested candidates, print each one's scores next to twelve of their split-A tweets, and read. Every case resolves, and they sort into three mechanisms:

**Style overshoot.** Junia Quimby (D, truth z −1.2) tweets 70% retweets; word2vec and MiniLM score her −2.2 to −2.7 — the retweet-style confound, diagnosed statistically in three prior analyses, here visible as a named candidate flung past the pole. The LLM compresses her slightly instead (−0.9) and posts the highest repetition SD among the eight divergence cases (.152; the pilot-wide max is .254): with so few original sentences per bundle, the annotator has little candidate-authored position to read. Same pattern, same diagnosis: Tamika Xiong (68% retweets).

**Process-diet flips.** Colton Ostrander and Maeve Northcutt (both D, small bundles) spend their feeds on yard signs, ride-alongs, and "met with local business owners" — the topic tier WS2 measured at *zero* positional signal. Embedding centroids anchored in positionless text drift to the middle or across it: MiniLM puts Northcutt (truth −0.9) at **+1.1**. Northcutt is also the packet's one genuinely cross-pressured candidate — three copies of "both parties share the blame" on immigration, a balanced-budget line — and every text instrument duly compresses her toward center. The instruments read her *presentation* correctly; the generator's latent is simply more liberal than the language it dealt her. That distinction — presentation vs latent — is exactly what no correlation table can surface and one minute of reading can.

**Mixed-frame bundles.** Petra Oglesby (D, truth z −1.5) is a strong liberal whose bundle includes planted counter-frame lines ("Criminals don't follow gun laws — disarming law-abiding OH citizens makes us less safe"). The content family shrugs (behav −1.3, LLM −1.3, TF-IDF −1.5); embedding centroids average the frames and land at −0.2 (Model2Vec) and **+0.6** (MiniLM). Across the eight cases MiniLM ends on the wrong side of center five times (and exactly at zero once) — WS1's style-contaminated axis, one candidate at a time.

Note what did *not* happen: the content trio never split by more than 0.7 z on any case, while embedding scores landed as far as 2.1 z from truth. When behavioral, lexical, and LLM instruments genuinely disagree with each other is apparently not where this corpus's action is — the action is embedding spaces reading style and diet as position.

## The consolidated scoreboard

One table now spans the testbed (`synthesis/outputs/consolidated_validation.csv`, every value read programmatically from its source file): axis recovery behavioral .977 > TF-IDF .974 > LLM .970 > Model2Vec .900 > w2v .878 > MiniLM .721; distance validity topping out at .640 overall but .849 within the retweet-content slice; held-out behavioral prediction oracle 2.317 < TF-IDF 2.348 < LLM 2.368 < Model2Vec 2.430 ≪ null ~3.0; blind topic recovery defeated (best ARI .289) until one observable routing convention rescued it (.765, then .890 refined). Per decision D4 the frozen job-application writing sample stays frozen; this table is its second results section in waiting.

## What carries to real data

One instrument per family, each chosen by a preregistered decision rule, with error-independence now measured rather than hoped for: TF-IDF+SVD for the axis; Model2Vec corrected centroids (exploratory label intact) for distances, conditioned on topic; the LLM taxonomy for topics, with content-routing conventions set *before* modeling; ask-and-average as the contamination-robust cross-check — on real candidates the only family whose validity argument does not depend on corpus-specific training. The deferred scale-up is the missing piece of coverage, not of validity; the synthesis pipeline takes a full-corpus score file whenever it exists.

## Limitations

Everything inherited: template text flatters every number here (transfer results, not real-tweet forecasts); WS3's annotators and this orchestrator share a model family; frozen baselines saw split-B text; identifier stripping was vacuous on this corpus and will not be on real data; small-bundle repetition SDs are understated; rank-1 score geometries inflate their own Mantel agreement; and the all-instrument support is n = 150 by the D1 deferral. The divergence reading is one analyst reading eight packets — computational grounded theory's "machine proposes, human reads" loop, run once, not a coded reliability study.

*Scripts 01–05, outputs, and figures in `synthesis/`; agreement matrices, Mantel/Procrustes tables, the divergence packet, and the consolidated table in `synthesis/outputs/`.*
