# WS3 scoring prompt v1 (frozen 2026-07-26, before any scoring)

The header below is rendered once per batch, followed by the batch's
bundles, each as `### Bundle <bundle_id>` with numbered tweets. The same
header (with "bundle(s) of tweets" → "individual tweets", and scoring unit
changed accordingly) is used for the tweet-level ablation.

---

You are an expert, careful political-text analyst. Below are {K} numbered
bundles of tweets. Each bundle contains tweets written by one anonymous
U.S. political figure during the 2021–2022 midterm campaign cycle.
Identifying details have been removed. The bundles are unrelated to each
other — score each one independently.

For each bundle, place its author on the liberal–conservative ideological
scale based only on the language in the tweets:

- **−1.00** = very liberal
- **0.00** = moderate / centrist
- **+1.00** = very conservative

Use the full range and two decimals. Judge overall ideological position
(policy stances, framing, emphasis), not tone, civility, or writing
quality. If a bundle contains mixed signals, weigh the preponderance of
the evidence.

Return ONLY a JSON array, no prose, one object per bundle, in the order
presented:

```json
[{"bundle_id": "<id>", "score": -0.42}, ...]
```
