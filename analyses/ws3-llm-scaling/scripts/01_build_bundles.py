"""
01_build_bundles.py — WS3 Stage A prep (BLIND: reads blind_corpus only)
-----------------------------------------------------------------------
Builds, per preregistration.md §2:
  * main condition: 150 candidates x 5 reps, n=25 split-A tweets, cues stripped
  * cue condition:  same tweet sets as main reps 1-2, text left intact
  * tweet-level ablation: 30-candidate subset, rep-1 bundle tweets scored solo

Outputs (ws3-llm-scaling/outputs/):
  bundle_map.csv          bundle_id -> candidate_id, rep, condition, tweet_ids
  batches/batch_*.md      agent-ready batch files (blind: ids + text only)
  batch_manifest.json     batch -> bundle_ids, expected counts
Seeds: sampling 20260726*10+rep; shuffling/batching 20260726 (preregistered).
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WS = ROOT / "ws3-llm-scaling"
OUT = WS / "outputs"
BATCH = OUT / "batches"
BATCH.mkdir(parents=True, exist_ok=True)

SEED = 20260726
N_TWEETS = 25
M_REPS = 5
BATCH_MAIN = 25     # bundles per agent batch (main)
BATCH_CUE = 50      # bundles per agent batch (cue ablation)
BATCH_TWEET = 125   # tweets per agent batch (tweet-level ablation)

RT_PREFIX = re.compile(r"^RT @\w+:\s*")
MENTION = re.compile(r"@\w+")
PARTY = re.compile(r"\b(democrats?|democratic|dems?|republicans?|gop)\b", re.I)

def clean(text: str, handle: str, name: str) -> str:
    t = RT_PREFIX.sub("", text)
    t = MENTION.sub("@user", t)
    t = PARTY.sub("[party]", t)
    if name:
        t = re.sub(re.escape(name), "[name]", t, flags=re.I)
    if handle:
        t = re.sub(re.escape(handle), "@user", t, flags=re.I)
    return t.strip()

def main() -> None:
    bc = pd.read_parquet(ROOT.parent / "ws0-harness" / "blind_corpus.parquet")
    ab = pd.read_parquet(ROOT.parent / "ws0-harness" / "tweet_split_ab.parquet")
    sub = json.load(open(ROOT.parent / "ws0-harness" / "splits.json"))["subsample_150"]["candidate_ids"]
    df = bc.merge(ab[["tweet_id", "split"]], on="tweet_id", validate="1:1")
    A = df[(df.split == "A") & df.candidate_id.isin(sub)].copy()
    cands = sorted(sub)
    assert len(cands) == 150

    rows, bundles = [], {}
    # ---- main condition (stripped) + record tweet sets ----------------------
    for rep in range(1, M_REPS + 1):
        rng = np.random.default_rng(SEED * 10 + rep)
        for cid in cands:
            g = A[A.candidate_id == cid]
            k = min(N_TWEETS, len(g))
            pick = g.sample(n=k, random_state=int(rng.integers(0, 2**31 - 1)))
            order = rng.permutation(len(pick))
            pick = pick.iloc[order]
            texts = [clean(r.text, r.handle, r.candidate_name) for r in pick.itertuples()]
            bid = f"M{rep}{cid}"  # opaque enough; agents never see the map anyway
            bundles[bid] = texts
            rows.append(dict(bundle_id=bid, candidate_id=cid, rep=rep,
                             condition="main", n_tweets=k,
                             small_bundle=len(g) < N_TWEETS,
                             tweet_ids="|".join(pick.tweet_id)))
    # ---- cue condition: same tweet sets as main reps 1-2, raw text ----------
    txt = dict(zip(df.tweet_id, df.text))
    for r in [r for r in rows if r["condition"] == "main" and r["rep"] <= 2]:
        bid = "C" + r["bundle_id"][1:]
        bundles[bid] = [txt[t] for t in r["tweet_ids"].split("|")]
        rows.append(dict(bundle_id=bid, candidate_id=r["candidate_id"], rep=r["rep"],
                         condition="cue", n_tweets=r["n_tweets"],
                         small_bundle=r["small_bundle"], tweet_ids=r["tweet_ids"]))
    # ---- tweet-level ablation: 30-candidate subset, rep-1 tweets ------------
    rng = np.random.default_rng(SEED)
    subset30 = sorted(rng.choice(cands, size=30, replace=False))
    json.dump(subset30, open(OUT / "tweetlevel_subset30.json", "w"), indent=1)
    titems = []
    for r in [r for r in rows if r["condition"] == "main" and r["rep"] == 1
              and r["candidate_id"] in subset30]:
        for i, (tid, tx) in enumerate(zip(r["tweet_ids"].split("|"),
                                          bundles[r["bundle_id"]])):
            titems.append(dict(item_id=f"T{r['candidate_id']}{i:02d}",
                               candidate_id=r["candidate_id"], tweet_id=tid, text=tx))
    pd.DataFrame(titems).to_csv(OUT / "tweetlevel_map.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "bundle_map.csv", index=False)

    # ---- batch files --------------------------------------------------------
    header = (WS / "prompts" / "scoring_prompt_v1.md").read_text().split("---\n", 1)[1]
    manifest = {}
    rng = np.random.default_rng(SEED)

    def emit_bundle_batches(prefix, ids, per):
        ids = list(ids); rng.shuffle(ids)
        for b in range(0, len(ids), per):
            chunk = ids[b:b + per]
            name = f"{prefix}_{b // per:02d}"
            lines = [header.replace("{K}", str(len(chunk))), ""]
            for bid in chunk:
                lines.append(f"### Bundle {bid}\n")
                lines += [f"{i+1}. {t}" for i, t in enumerate(bundles[bid])]
                lines.append("")
            (BATCH / f"{name}.md").write_text("\n".join(lines))
            manifest[name] = chunk

    emit_bundle_batches("main", [r["bundle_id"] for r in rows if r["condition"] == "main"], BATCH_MAIN)
    emit_bundle_batches("cue", [r["bundle_id"] for r in rows if r["condition"] == "cue"], BATCH_CUE)

    theader = (header.replace("numbered\nbundles of tweets", "individual tweets")
               .replace("Each bundle contains tweets written by one anonymous\nU.S. political figure",
                        "Each tweet was written by an anonymous U.S. political figure")
               .replace("The bundles are unrelated to each\nother — score each one independently.",
                        "The tweets are unrelated to each other — score each one independently.")
               .replace("For each bundle, place its author", "For each tweet, place its author")
               .replace("If a bundle contains mixed signals", "If a tweet contains mixed signals")
               .replace("one object per bundle", "one object per tweet")
               .replace('"bundle_id"', '"item_id"'))
    ids = [t["item_id"] for t in titems]; rng.shuffle(ids)
    tmap = {t["item_id"]: t["text"] for t in titems}
    for b in range(0, len(ids), BATCH_TWEET):
        chunk = ids[b:b + BATCH_TWEET]
        name = f"tweet_{b // BATCH_TWEET:02d}"
        lines = [theader.replace("{K}", str(len(chunk))), ""]
        lines += [f"### Tweet {iid}\n{tmap[iid]}\n" for iid in chunk]
        (BATCH / f"{name}.md").write_text("\n".join(lines))
        manifest[name] = chunk
    json.dump(manifest, open(OUT / "batch_manifest.json", "w"))

    n_main = sum(r["condition"] == "main" for r in rows)
    n_cue = sum(r["condition"] == "cue" for r in rows)
    print(f"bundles: main={n_main} cue={n_cue} tweet_items={len(titems)}")
    print(f"batches: {len(manifest)} "
          f"(main {sum(k.startswith('main') for k in manifest)}, "
          f"cue {sum(k.startswith('cue') for k in manifest)}, "
          f"tweet {sum(k.startswith('tweet') for k in manifest)})")
    print("small_bundle candidates:",
          len({r['candidate_id'] for r in rows if r['condition']=='main' and r['small_bundle']}))

if __name__ == "__main__":
    sys.exit(main())
