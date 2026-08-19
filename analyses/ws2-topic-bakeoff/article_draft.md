# Is AI good enough to identify themes? A five-way blind topic bake-off on 104,601 synthetic campaign tweets

*Workstream 2 of the extensions execution plan. Run 2026-07-26, seed 20260726, under the project's blind-then-validate protocol: every design choice below was locked in `preregistration.md` before the planted `true_topic` labels were unsealed, and truth was read exactly once. Companion pieces: the 2026-07-20 embeddings/PCA study and WS1's sentence-transformer study.*

## The question

The project notes (07/20/26) asked two questions this study answers as an experiment rather than an opinion: **"Is AI good enough to identify themes?"** and **"Does topic modeling become obsolete?"** — plus the standing worry that discovered topics end up "too broad, too inclusive, difficult to interpret."

The testbed is the synthetic 2022-cycle corpus: 910 fictional candidates, 104,601 tweets, with a planted `true_topic` per tweet. Because every candidate is fictional, LLM training-data contamination is impossible by construction — the standard objection to LLM-based text analysis cannot apply here.

## The contest

Five entrants, each assigning all 104,601 blind tweets to topics of its own choosing, including its own number of topics K (that ambiguity is part of what was tested). Classical entrants chose K by NPMI coherence sweep over K ∈ {5…40}; all shared identical lexical preprocessing.

| Entrant | Configuration | K chosen |
|---|---|---|
| LDA | gensim, coherence-swept | 40 (39 used) |
| NMF | sklearn on TF-IDF, coherence-swept | 8 |
| LSA | TF-IDF+SVD+k-means (the incumbent family) | 40 |
| BERTopic | MiniLM → UMAP → HDBSCAN → c-TF-IDF | 763 (!) |
| Direct LLM theming | Claude reads a stratified 2,000-tweet sample → taxonomy → labels → nearest-centroid propagation in MiniLM space | 25 |

The LLM entrant's taxonomy step was run twice on disjoint 1,000-tweet halves: 19 of 22 themes matched one-to-one across runs — the taxonomy is stable, not sample noise. Interpretability was scored before unsealing by a blinded judge (topics rendered identically as c-TF-IDF terms + example tweets, provenance hidden, shuffled; one caveat reported openly: contestant 5 and the judge share a model family, mitigated by the blind rendering).

## Result 1 — Everyone failed the bar; the LLM failed it least

Pre-registered success bar: ARI ≥ 0.60 vs the planted topics. **Nobody cleared it blind.**

| entrant | ARI | NMI | K | judge interpretability (1–5) |
|---|---|---|---|---|
| **direct LLM** | **0.289** | **0.630** | 25 | 4.40 |
| LDA | 0.156 | 0.513 | 39 | 4.06 |
| LSA | 0.138 | 0.535 | 40 | 4.39 |
| NMF | 0.044 | 0.237 | 8 | 3.75 |
| BERTopic | 0.023 | 0.515 | 763 | 4.52 |

The generator had planted **12** topics: ten policy issues, one `campaign_logistics`, and one `retweet_source`. Nobody guessed near 12 — and that K mismatch, not thematic confusion, turns out to be almost the whole story (Result 2).

Two side findings worth keeping. First, **blind coherence did not rank recovery** (fig2): LSA had the best NPMI (0.33) and finished third; the LLM won ARI with middling coherence. Optimizing K by coherence — the standard practice — pointed LDA and LSA to K=40, more than triple the truth. Second, **BERTopic's failure mode was exactly the one the plan's risk register predicted**: at the pre-registered min_cluster_size=200 it left 58% of tweets as outliers; the pre-declared fallback (min_cluster_size=60) got outliers to 17% at the cost of shattering the corpus into 763 micro-clusters — highest judge interpretability per topic (each micro-cluster is a crisp template), lowest ARI (ARI punishes shattering), and near-zero topic diversity (0.03: the same words appear in hundreds of topics).

## Result 2 — The gap was two conventions, not comprehension

Post-unseal diagnosis (everything from here on is labeled mitigation measurement, not blind discovery). Mapping each LLM theme to its majority true topic showed the taxonomy itself was nearly clean — abortion, foreign-policy, veterans, trade, democracy-reform, and all six process themes mapped 1.00 to a single true topic. The blind score was destroyed by exactly two structural conventions where the LLM's (defensible) reading and the generator's differed:

1. **Retweets.** The generator assigns every retweet — 26.5% of the corpus — to a single `retweet_source` topic. The LLM themed retweets by their *content* (a voter-purge retweet → election-integrity).
2. **Visit tweets.** The generator files "toured the plant / stopped by the school" under the venue's *policy* topic; the LLM's first taxonomy run saw a coherent `community-visits` genre — 12,523 tweets spread across every policy topic. (The second run scattered visit tweets into issue themes; the genre theme was kept from H1 in reconciliation, so this is a one-run reading, not an independent replication.)

The refinement ladder quantifies this (fig3): routing retweets to their own topic — a rule using only the **observable** `is_retweet` column, available blind — jumps ARI from 0.289 to **0.765**, past the bar in one move. Dissolving the two genre themes adds 0.06; coarsening the taxonomy to K=13 lands at **ARI 0.890 / NMI 0.877**. Applied to the other entrants (exploratory), the same retweet rule rescues nearly everyone (LDA 0.70, BERTopic 0.69, LSA 0.67) while preserving the blind ranking — the LLM stays first.

The honest headline is therefore *not* "LLM theming recovers planted topics at 0.89." It is: **one observable structural rule was worth more than every modeling choice combined** (+0.48 ARI vs a 0.13 spread across four model families), and the bake-off's blind loss was a taxonomy-*convention* mismatch, not a comprehension failure. On real data, where there is no generator whose conventions must be matched, the practical lesson survives translation: decide explicitly how shared/quoted content and genre-vs-subject tweets should be treated *before* modeling — the choice dominates the model.

## Result 3 — Far from whom, on what

Stage C built per-topic candidate distances in the corrected Model2Vec space (WS1's best distance instrument; the retweet-style PC1 confound reappeared for the fourth consecutive time and was projected out — it is now a law of this corpus, not an anecdote). The pre-registered claim — that within-topic distances should beat the overall figure because the generator plants framing conditionally on topic — **held**: best within-topic validity 0.704 (blind winner's election-integrity topic) vs 0.640 overall.

On the refined K=13 instrument the decomposition is cleaner still (fig4, fig5):

- **retweet-content: 0.849** — by far the most ideologically diagnostic slice of candidate language, echoing the corpus's behavioral ceiling (mean retweet-source ideology correlates 0.98 with truth). Who you amplify beats what you say.
- **Policy topics: 0.23–0.57** (healthcare 0.57, abortion 0.55, guns 0.52 lead) — real but weaker signal; interestingly the *fine* blind themes often carried more signal than their merged versions, because merging pooled distinct framings.
- **campaign-process: 0.004** — a fifth of candidate output (GOTV, rallies, fundraising, holidays) carries literally zero positional information. Distance claims computed over all text dilute signal with this noise.

The pair × topic heatmap (fig5) makes the "far from whom, on what" artifact concrete: the most-distant cross-party pairs are dark almost entirely in the retweet-content column and in one or two policy columns where both members are active — while same-party "most distant" pairs turn out to differ on *process mix*, not position.

## Answers to the notes' questions

**Is AI good enough to identify themes?** For theme *discovery and naming*: yes — the LLM taxonomy was stable across independent runs, judged most interpretable among real contenders, and structurally almost isomorphic to the planted truth. For *corpus-scale assignment*: yes with the two-stage design (2,000 labels propagated by embeddings cost ~zero), but only after the structural conventions are fixed; blind, it scored 0.29 like everything else.

**Does topic modeling become obsolete?** The classical models lost on every axis at once — worse recovery, worse or degenerate K selection, and no better interpretability. What survives is not LDA but the *scaffolding*: coherence metrics (as diagnostics, not K-selectors), c-TF-IDF rendering, and embedding propagation. On this evidence the topic instrument carried to real data is **LLM taxonomy + embedding propagation, with content-routing rules made explicit** — and topic-conditioning is not optional for distance work: a fifth of the corpus is positional noise and one slice (amplified content) is worth more than everything else combined.

## Limitations

Template-generated text flatters every method's coherence and may understate real-tweet difficulty; the judge and one contestant share a model family (blind-rendered, but reported); Stage B/C refinements are post-unseal mitigation measurements and labeled as such throughout; the 0.60 bar was cleared only under those labeled conditions, not blind. The per-entrant judge sample was capped at 40 topics (BERTopic's 763 made exhaustive judging infeasible) — a documented deviation from the preregistration.

*Scripts 01–10b, outputs, and figures in `topic-modeling-bakeoff/`; preregistration locked before unseal; single truth-read in `08_unseal_validate.py`.*
