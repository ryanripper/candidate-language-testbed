# Can you just ask the model where a politician stands? An ask-and-average pilot on 150 synthetic candidates — and what the scores predict

*Workstream 3 of the extensions execution plan. Run 2026-07-26, seed 20260726, under the project's blind-then-validate protocol: every design choice was locked in `preregistration.md` before the planted `true_ideology` was unsealed, and truth was read exactly once. Companion pieces: the 2026-07-20 embeddings/PCA study, WS1 (sentence transformers), and WS2 (topic bake-off).*

## The question

The Political Analysis ask-and-average literature reports that LLMs place party manifestos on a left–right scale at r ≈ 0.90 against expert benchmarks. Manifestos are long, canonical, and — the standard objection — almost certainly in training data. This study asks whether the method survives two removals at once: from manifestos to **25-tweet bundles**, and from real parties to **910 fictional candidates**, where training-data contamination is impossible by construction. Then it asks the question that matters more than recovery: do the scores **predict behavior the annotator never saw**?

The design, per the pre-registration: for each of the 150 candidates in the frozen stratified subsample, five independent bundles of 25 tweets drawn only from split A of the frozen tweet split, identifiers stripped (all cues in this corpus live in `RT @OrgHandle:` prefixes — the stripping rules for party labels and mentions are stated and applied but turn out to be vacuous here), each bundle scored −1…+1 by a fresh blinded in-session annotator that never sees candidate identity, another bundle, or any truth; the candidate's score is the mean of five, the SD an instrument-stability measure. 1,730 scorings in total including ablations; zero API cost (decision D1's pilot-then-scale rule).

## Result 1 — The method transfers: r = 0.970

Pre-registered bar: r ≥ 0.90, the manifesto-literature figure. The pilot cleared it with room:

| instrument (same 150 candidates) | Pearson r | Spearman ρ |
|---|---|---|
| Behavioral ceiling (mean retweet-source ideology, split A) | 0.977 | 0.938 |
| TF-IDF+SVD PC1 (frozen incumbent) | 0.974 | 0.940 |
| **LLM ask-and-average (pilot)** | **0.970** | **0.918** |
| WS1 Tier A Model2Vec axis | 0.899 | 0.852 |
| word2vec partisan axis (frozen) | 0.878 | 0.841 |

Three readings. First, the ask-and-average method survives both removals — short noisy bundles, guaranteed-uncontaminated targets — essentially undegraded. Second, it does **not beat the dumb baseline**: TF-IDF+SVD, trained on this corpus for pennies, is 0.004 ahead. On this testbed the LLM's value is not accuracy; it is that it hit the lexical ceiling **with zero corpus-specific training, no vocabulary, and no fitted axis** — the score arrives on an interpretable −1…+1 scale from a cold start. Third, the ordering TF-IDF ≈ LLM ≫ Model2Vec ≫ w2v reproduces WS1's conclusion from a completely different direction: on short planted political text, lexical signal is the ceiling, and instruments differ mainly in how much of it they reach.

Instrument behavior under the hood (fig2, fig5): median across-repetition SD 0.089 — about 9% of the scale — but stability does *not* flag error (r = −0.11): a candidate scored consistently is not scored correctly, so repetition SD cannot substitute for validation. Misses concentrate slightly at the **poles**, not the middle (|error| vs |truth| r = +0.13): the annotator compresses extremes, rarely venturing past ±0.85 — visible as Spearman (0.918) trailing Pearson (0.970). Moderates, the usual worry, were not the problem. Both confound screens — blind (score vs retweet share / volume / topic entropy) and post-unseal (error vs same) — came back clean, max |r| ≈ 0.14: the 07-20 retweet-style artifact that haunts every embedding space in this project does **not** touch the declared-scale instrument.

## Result 2 — The ablations: cues were redundant, bundles weren't needed

**Cue bias.** The party-cue literature made identifier stripping mandatory in the main condition; the ablation re-scored the *same* rep-1/2 bundles with the 20 org handles left in. Effect on accuracy: Δr = **+0.004** (0.968 vs 0.964) — nothing. Effect on scores: mean |paired shift| 0.055, with the only directional pattern a mild outward drift for Democrats (−0.038 further left; Republicans +0.010). When the text itself separates the parties at point-biserial 0.955, cues have nothing left to add — they nudge scores toward the poles without changing rank. The measurement is the finding: on *this* corpus stripping cost nothing, but the polarizing direction of the shift is exactly the failure mode the literature warns about, arriving here in miniature.

**Bundle vs tweet-level.** Scoring 25 tweets as one bundle vs scoring each tweet alone and averaging: r = 0.953 vs 0.951 on the 30-candidate subset — equivalent. The bundle's supposed advantage (context accumulation) and its risk (one vivid tweet anchoring the read) apparently cancel on template-generated text. Practically this matters for scale-up: tweet-level calls are smaller, parallelize better, and price identically.

## Result 3 — The behavioral horse race: 93% of the oracle's edge, from text alone

Stage C is the direction Ryan asked for: scores estimated on split A predicting behavior measured on split B, against the oracle ceiling (`true_ideology` itself run through the identical pipeline).

**C1 — Who do you amplify? (pre-registered primary).** Each candidate's split-B retweets modeled as a choice among the 20 org accounts, P(org) ∝ softmax(−β·|score − org ideology|), calibration and β fit on split A only. Held-out log-loss per retweet:

| model | log-loss | top-1 acc |
|---|---|---|
| Oracle (true_ideology) | 2.317 | 0.151 |
| TF-IDF | 2.348 | 0.149 |
| **LLM** | **2.368** | **0.141** |
| WS1 Model2Vec | 2.430 | 0.122 |
| Null: base rates | 2.980 | 0.057 |
| Null: uniform | 2.996 | 0.050 |

Every text instrument recovers the large majority of the oracle's 0.68-nat edge over the null — the LLM 93% of it, leaving 0.05 nats "on the table." The pre-registered headline was the instrument *ranking*, and it is stable across log-loss and top-1: **oracle > TF-IDF > LLM > WS1 ≫ null**, i.e. behavioral inference quality tracks Stage B recovery almost exactly. No instrument's scores unlock behavior the others' don't; whatever ideology signal an instrument captures is what buys behavioral prediction. (Fit is limited by the corpus itself: candidates retweet probabilistically across several nearby orgs, so even the oracle tops out at 15% top-1 against a 20-way choice.)

**C2 — What do you talk about?** The designed negative of the study: predicting split-B topic shares (WS2's refined K=13 instrument) from any single ideology score barely beats the grand-mean null — **including from the oracle** (JS 0.151 vs null 0.158; all instruments within 0.002 of each other). The generator tilts topic mixes ideologically, but per-candidate Dirichlet noise dominates that tilt. The lesson generalizes: a scalar position is the wrong tool for *attention*; topic mix needs its own instrument (WS2's), not a projection from ideology.

**C3 — How do you frame it?** Split-B lexical framing intensity is predicted well by everything (LLM 0.933), with an instructive anomaly: TF-IDF (0.955) beats the *oracle* (0.945). That is not magic — it is shared method variance. The target is lexical framing; TF-IDF is a lexical instrument; they co-vary through word choice above and beyond latent ideology. The within-topic decomposition (healthcare, abortion, guns — WS2's three strongest policy topics) preserves the ordering at smaller n. A useful caution for real data: when the validation target is itself text-derived, lexical instruments get flattered.

## What the pilot buys, and what it doesn't

The pre-registered D1 gate is cleared, so the standing recommendation is the full 910 × 5 run via API (≈4,550 bundles, ≈2.7M input tokens — single-digit dollars on a small model; the tweet-vs-bundle equivalence says test the cheap configuration first). What the full run adds is not a better r — the pilot's n=150 estimate is tight — but coverage: per-candidate scores usable in the synthesis stage's instrument-agreement matrix and divergence case studies across all 910. *As of 2026-07-27 the scale-up is deferred by decision of the author: the gate result stands on the record, but the run is unfunded for now, and the synthesis stage proceeds on the 150-candidate pilot coverage.*

What the pilot does *not* license: on real candidates, contamination protection vanishes (every real politician is in training data), identifier stripping becomes load-bearing rather than vacuous, and the cue-bias ablation — which cost nothing here — becomes the experiment that matters. The design ports; the null results do not.

## Limitations

Annotator and orchestrating analyst share a model family (the annotator agents were blinded to identity, mapping, and truth; the design was preregistered before any scoring — but this is the WS2 honest-broker caveat again, reported not solved). Template-generated text likely flatters recovery for all instruments; the r = 0.97 says the *method* transfers to short bundles, not that real tweets are this easy. The frozen TF-IDF and WS1 axes saw split-B text at training time (a leakage that favors the *baselines* in Stage C, making the LLM's showing conservative — direction pre-registered). Small-bundle candidates (21 of 150 with < 25 split-A tweets) have understated repetition SDs; they were excluded from the stability diagnostic. C2/C3 per-topic figures rest on few split-B tweets per candidate per topic (n's in fig4).

*Scripts 01–08, outputs (including the full per-bundle audit trail), and figures in `llm-scaling-analysis/`; preregistration locked before unseal; single truth-read in `06_unseal_validate.py`.*
