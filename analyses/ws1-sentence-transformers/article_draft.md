# The Transformer Came Last: Sentence Embeddings Meet a Planted Ground Truth

*WS1 of the extensions program: Model2Vec and MiniLM take on the r = 0.974 lexical baseline — blind, pre-registered, and with a scoreboard nobody can argue with. The fancy method loses. The interesting part is how.*

---

Last week's analysis ended with a dare. A TF-IDF + SVD pipeline — weighted word counts, a technique older than most of the fictional candidates it was measuring — recovered the planted ideology of 910 synthetic congressional candidates at **r = 0.974**, essentially the generator's own ceiling. Word2vec came in at 0.886, and only after we caught its dominant axis moonlighting as a retweet-style detector. The dare: every modern method now has to beat that number, on the same corpus, under the same blind protocol.

This post is the first taker: **sentence-transformer embeddings**, the workhorse of modern NLP. Two questions, pre-registered before unsealing anything:

1. Does context-aware, sentence-level embedding beat the lexical baseline for **position recovery** (the axis)?
2. Separately, does it produce more valid **distances** — including the genuinely new capability sentence-level embedding unlocks: treating each candidate not as one averaged point but as a *cloud* of tweet vectors, compared with distributional distances (energy distance, MMD)?

Separately, because the last round taught us a method can win one and lose the other.

Two models ran: **Model2Vec** (`potion-base-8M`, a static model distilled from a transformer — embeds the whole corpus in 12 seconds on CPU) and **all-MiniLM-L6-v2** (the standard compact sentence transformer — 6.3 minutes). A third, stronger model was pre-registered behind a *blind gate*: it would only run if MiniLM's blind diagnostics (party separation, no ground truth involved) beat TF-IDF's. Spoiler: the gate stayed shut, and that decision was made honestly — by a rule written down before anyone saw a validation number.

Everything below follows the standing protocol: design frozen in `preregistration.md`, all choices made against observable labels only (party, chamber, posting behavior), sealed truth opened exactly once, in the last script.

## The confound came back — in both spaces

The mandatory first checkpoint (now a standing gate in the shared harness) is: *before you trust any embedding space, regress its top principal components on posting behavior.* Last time, word2vec's PC1 turned out to be a retweet-style axis (r = 0.96 with retweet share).

It replicated. In both new spaces:

| | dominant PC vs retweet share | partisan axis (blind) |
|---|---|---|
| Model2Vec | **PC1: r = −0.97** | PC2, |D/R corr| = 0.88 |
| MiniLM | **PC1: r = −0.84** | PC2, |D/R corr| = 0.70 |

Three embedding families — word2vec, a distilled static transformer, a full sentence transformer — and all three organize their space *first* by how much a candidate retweets, and only *second* by politics. At this point it's fair to call it a law of averaged social-media embeddings rather than a quirk: **style variance dominates topical variance, and if you average tweets into candidate vectors, style becomes your first principal component.**

There's a wrinkle worth flagging: in MiniLM's space, even the *partisan* PC2 carries a 0.48 correlation with retweet share. The style signal isn't confined to one component we can surgically remove — it's smeared into the political direction itself. Keep that in mind; it explains the scoreboard.

The pre-registered correction (project style PCs out of the tweet vectors, rebuild everything) did what it did last time. For Model2Vec, centroid distance validity jumped **0.476 → 0.640** after removing one component. For MiniLM it *fell* slightly (0.455 → 0.397) — when style and substance are entangled, projection cuts into both.

## The scoreboard

Unsealed once, at the end, against the frozen baselines:

| Instrument (blind) | axis r vs true ideology | best distance validity |
|---|---|---|
| **TF-IDF + SVD (frozen baseline)** | **0.974** | 0.624 |
| word2vec (frozen baseline) | 0.886 | 0.592 (corrected) |
| **Model2Vec potion-8M** | 0.900 | **0.640 (corrected centroid)** |
| MiniLM-L6-v2 | 0.721 | 0.455 |

The pre-registered primary metric — MiniLM's corrected-space partisan axis — landed at **r = 0.721**. Not only below TF-IDF's 0.974; below word2vec's 0.886. The full sentence transformer, the most sophisticated method this corpus has met, came **last**.

The ordering inverts model sophistication almost perfectly. And the reasons are legible:

**Why TF-IDF keeps winning the axis.** The generator plants ideology mostly through *word choice* within shared templates. TF-IDF is a machine for noticing exactly that: which words this candidate uses that others don't. Sentence transformers are machines for noticing what a sentence *means* — and two tweets that differ by one loaded phrase mean nearly the same thing to MiniLM, while lighting up TF-IDF like a scoreboard. On short, template-structured political text, the lexical signal *is* the signal.

**Why the static distillation beat the real transformer.** Model2Vec is, functionally, a modernized word-vector model — token embeddings averaged together. That makes it more lexical than MiniLM: closer to the level where this corpus hides its truth. It also kept its style confound neatly separable (one clean PC), where MiniLM smeared style into everything. Cheaper, and better here: 12 seconds vs 6.3 minutes, r = 0.900 vs 0.721.

**The distribution-over-centroid hypothesis failed.** The one genuinely new capability — comparing candidates as clouds of tweet vectors via energy distance and MMD instead of collapsing to a mean — was supposed to be where sentence embeddings shine. On the blind diagnostics the energy distances looked promising (best between/within party ratio of the whole batch, 1.456). But at unseal, the best distributional distance managed validity of just **0.48**, well under the simple corrected centroid's 0.640. The clouds contain real per-tweet variation — topic, template, framing — but most of that variation is *noise with respect to ideology*, and averaging it away turns out to be a feature, not a loss. Whitening, the other transformer-era remedy, actively hurt (≈0.31 everywhere): equalizing variance re-inflates exactly the nuisance directions the correction just suppressed.

## The one bright spot — and what the rules let us say about it

Model2Vec's corrected centroid distances hit **0.640** — above TF-IDF's 0.624, above everything from the last round. It is the most valid distance instrument this testbed has produced.

Under the pre-registered decision rule, that does **not** count as a win — the rule named *distributional* distances as the challenger for the distance crown, and they lost. So formally, this workstream returns its third-listed outcome: **a negative result.** On short, topically-planted political text, lexical baselines remain sufficient; sentence transformers, as tested, do not earn a place in the real-data pipeline as the primary instrument.

But pre-registration cuts both ways honestly: it also lets us report, clearly labeled as *exploratory*, that a corrected static-embedding centroid looks like the best distance measure available — a hypothesis WS2's topic-conditioned distances can test properly, and which costs 12 seconds to reproduce on any corpus.

The formal decision consequences, as pre-registered:

- TF-IDF + SVD remains the **axis instrument** going forward.
- **Model2Vec becomes the default embedding for the real-data sweep on cost grounds** — with the bonus that it's also the best distance candidate observed.
- The stronger Tier-C model never ran: the blind gate (MiniLM's party-separation diagnostics vs TF-IDF's) failed on both criteria — narrowly on the ratio (1.3243 vs 1.3255), decisively on the axis (0.70 vs 0.94). Scaling up the transformer when the small one is losing on *diagnostics* would have been hope, not method.

## What this changes for the program

A negative result that sharpens three things:

1. **The real-data plan gets cheaper.** The embedding layer for the historical-corpus sweep is now a 12-second static model, not an hours-long transformer job — chosen by evidence, not budget rationalization.
2. **The confound gate graduates from lesson to law.** Three for three: every averaged embedding space on this corpus put style on PC1. The gate is mandatory in the harness, and it just paid for itself again — one projection was worth +0.16 validity in the space where style was separable, and diagnosing *inseparability* explained the failure in the space where it wasn't.
3. **A caveat to carry.** This corpus is template-generated with a small vocabulary; it is maximally friendly to lexical methods and arguably unfriendly to contextual ones. The right reading is not "transformers are bad at political text" but "when the signal is lexical, lexical instruments win, and you should test which regime your corpus is in before paying for context." Real campaign tweets — messier, larger-vocabulary, more paraphrase-heavy — may sit in the other regime; that's precisely what the real-data follow-on will measure.

Next in the queue: WS2, the five-way blind topic-modeling bake-off — where MiniLM's tweet embeddings get a second life as BERTopic's input, and where "is AI good enough to identify themes?" stops being a NOTES.md question and becomes a scoreboard.

---

*Pipeline: five numbered scripts in `sentence-transformer-analysis/`, reading only the WS0 shared harness (`blind_corpus.parquet`, frozen baselines, shared metrics). Preregistration, decision record, and all 16 distance matrices are committed alongside. Session seed 20260725.*
