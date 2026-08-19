# NOTES Read-Out — Mapping Ryan's Notes to Project Findings & Analyses

*This is the running read-out that [NOTES.md](NOTES.md) calls for: Claude parses each dated entry and maps it to potential findings, analyses, and content changes in the project. Newest entries first. Started 2026-07-20.*

---

## 07/27/26 — Update: the synthesis stage closes loops opened by both 07/20/26 entries

*No new NOTES.md entry; this is a read-out update recording that the [synthesis stage](../analyses/synthesis/synthesis-writeup.md) (run 2026-07-27) answered questions both existing entries carried forward.*

1. **The quant-qual bridge was executed, not just theorized.** The divergence case studies are the "numbers flag *where* to look; reading the tweets says *why*" loop in practice: an instrument-disagreement index over six measurements picked 8 of 150 candidates, and no single covariate explained the ranking (all |r| ≤ .18) — but reading twelve tweets per case resolved every one into three nameable mechanisms (retweet-style overshoot, positionless process diets, mixed-frame bundles). Packet + interpretation: [divergence_case_tweets.md](../analyses/synthesis/outputs/divergence_case_tweets.md), writeup §4. This is skills-doc §1.9's computational grounded theory pattern (machine proposes → human deep-reads) run end-to-end on this project's own data.

2. **"Is AI good enough to identify themes?" now has a stronger answer than WS2's.** The WS2 topic instrument and WS3 LLM scores jointly reproduce the "far from whom, on what" signal ladder (retweet-content ≫ policy ≫ campaign-process) with **no ground truth anywhere in the pipeline** — tier-order ρ = .97 vs the oracle version. The AI-vs-topic-modeling question ends, on this testbed, as coexistence with division of labor: LLM taxonomy + embedding propagation for themes, and the whole decomposition estimable on real data where no truth column exists.

3. **New finding neither entry anticipated: instruments err in families.** Partial the oracle out and LLM/lexical/behavioral instruments share their remaining errors (residual r ≈ .4–.6), as do the two style-contaminated embedding spaces (.61) — while cross-family errors are near-independent. Measurement diversity is a property of *families*, not of instrument count.

**Changes triggered:** `synthesis/` folder created (scripts 01–05, figures 1–4, consolidated validation table); D1 scale-up deferral (Ryan, 2026-07-27) recorded across plan §6, WS3 outputs, and the synthesis docs. Writing sample unchanged per D4.

---

## 07/20/26 — Topic Modeling in Research

**What the note says.** Interest in discovering hidden themes (LDA, NMF, LSA); potential use on focus-group/qualitative data and as a replacement for subjective human coding; a fear that discovered topics come out too broad, too inclusive, or hard to interpret; and open questions about whether topic modeling survives the AI era — can the two coexist, or does AI make it obsolete?

**How it maps to the project.**

1. **You've already run LSA — and it won.** LSA is SVD on the (weighted) document-term matrix, which is exactly the TF-IDF+SVD baseline in the 2026-07-20 embeddings analysis ([workflow](../analyses/00-embeddings-pca/00_workflow_outline.md)). It recovered the planted ideology at r = .974 — essentially the generator's ceiling — beating corpus-trained word2vec. Your instinct toward the classical family has empirical support *inside this project*.

2. **"Is AI good enough to identify themes?" is now a designed experiment, not a rhetorical question.** The synthetic corpus carries a planted `true_topic` column that the embeddings analysis sealed and never touched. That enables a blind topic-recovery bake-off — LDA vs. NMF vs. BERTopic vs. direct LLM theming, scored against the planted topics (ARI/NMI) plus coherence and label-quality metrics — reusing the seal-then-validate protocol that already worked. This is now the opening move of Phase 2 in the [extension plan](plans/candidate-language-research-extension-plan.md) (§4 and Phase 2, revised 2026-07-20).

3. **The broad/uninterpretable fear becomes a measurement.** Topic coherence metrics quantify "too broad / hard to interpret," and seeded/guided topic modeling (extension plan §4) is the mitigation: pin the taxonomy to a fixed issue list so topics can't drift into mush.

4. **Coexistence, not obsolescence, is where the 2025–26 literature landed.** Topic models find structure cheaply, deterministically, and transparently; LLMs label topics, describe them, and apply codebooks — each covers the other's weakness (skills doc §1.6, and the new §1.9). The bake-off in (2) will say whether that consensus holds on this project's data.

5. **Focus groups / avoiding subjective coding** connects to the LLM-annotation strand already in the extension plan (Törnberg 2025 — LLMs outperforming expert coders on political social media) and to the new skills-doc section on LLM-assisted qualitative coding, whose consistent finding is: machines rival humans on explicit themes, humans stay in the loop for latent ones.

**Changes triggered:** extension plan §4 + Phase 2 revised (classical baselines, topic-recovery study); literature scan updated with LDA/NMF/BERTopic comparison studies.

---

## 07/20/26 — Bridging Quantitative and Qualitative Data

**What the note says.** How do quantitative and qualitative data speak to each other — numerically, thematically, heuristically? What role does each play, and does that differ by domain? What do patterns, theme identification, reporting, and sharing look like on each side?

**How it maps to the project.**

1. **This project *is* a bridge, run in one direction.** The candidate research takes qualitative material (campaign language) and produces quantitative measurements (positions, distances, correlations). The three linkage modes in your note have concrete counterparts: *numerically* — quantitizing (an ideology score from word choice); *thematically* — the planned topic-conditioned distances ("far from whom, on what"); *heuristically* — the figures and joint displays that put numeric maps next to interpretable examples (the PCA maps + most-alike/unalike pair tables in the [article draft](../analyses/00-embeddings-pca/article_medium_draft.md) are already this).

2. **The reverse direction is the project's current gap — now covered.** The skills collection had no entry on computational qualitative analysis, so a new section was added: [skills doc §1.9](skills/cutting-edge-data-science-skills-2026.md) — computational grounded theory (machine proposes patterns → human deep-reads → machine confirms), LLM-assisted qualitative coding, human–machine reliability metrics (κ/α across the bridge), topic models as qualitative instruments, and mixed-methods joint displays.

3. **"Does the relationship differ by domain?" — treat as an open research question.** In measurement-oriented domains (political text) the quantitative strand dominates and qualitative reading validates; in meaning-oriented domains (focus groups, health research) the qualitative strand dominates and computation scales it. Worth revisiting after the Phase 2 bake-off, which will generate evidence from the measurement side.

4. **Reporting and sharing** map onto the project's existing deliverable pattern: numeric validity evidence (correlation tables) side-by-side with interpretable artifacts (labeled maps, ranked pair lists, topic labels). The Medium draft's structure — figure, table, plain-language reading of each — is the house style for bridging in reports.

**Changes triggered:** new skills doc §1.9 with 2025–26 sources.

**Open questions carried forward:** domain-dependence of the quant/qual relationship (note 3); whether LLM theming beats classical topic models on *this* corpus (bake-off pending); what a qualitative-first deliverable (e.g., focus-group-style analysis of candidate language) would look like using §1.9 methods.
