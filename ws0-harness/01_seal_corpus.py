"""
01_seal_corpus.py — WS0.1
-------------------------
Shared-harness seal script (extensions-execution-plan.md, step 0.1).

Splits the synthetic corpus into:
  blind_corpus.parquet  — every observable column (what workstreams may read)
  sealed_truth.parquet  — tweet_id/candidate_id + true_topic, true_framing,
                          true_ideology (read ONLY at pre-registered
                          validation steps)

Both files are hash-stamped into seal_manifest.json. Any workstream can
verify it is reading the canonical blind corpus by checking the sha256.

Blind protocol (established 2026-07-20, embeddings-pca-analysis):
design decisions are made against blind_corpus only; sealed_truth is opened
once per workstream, at the validation step named in its preregistration.md.

Session seed convention: 20260725 (run date). The seal itself is
deterministic — no randomness here.
"""

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = (
    HERE.parent
    / "data"
    / "synthetic-candidate-tweets"
    / "synthetic_candidate_tweets_2022.csv.gz"
)
TRUE_COLS = ["true_topic", "true_framing", "true_ideology"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    # Guard the committed seal record: rebuilding the parquet views is fine,
    # but silently rewriting seal_manifest.json would let a modified corpus
    # pass verification against its own fresh manifest. Rebuilds must either
    # keep the manifest byte-identical or be explicitly flagged as a reseal.
    manifest_path = HERE / "seal_manifest.json"
    reseal = "--reseal" in sys.argv
    prior = json.load(open(manifest_path)) if manifest_path.exists() else None
    if prior is not None and not reseal:
        if sha256(DATA) != prior.get("source_sha256"):
            sys.exit(
                "ERROR: source corpus hash does not match the committed "
                "seal_manifest.json. Refusing to overwrite the seal record. "
                "If a reseal is genuinely intended, rerun with --reseal "
                "(and say so loudly in the commit message)."
            )

    df = pd.read_csv(DATA, compression="gzip")
    n_cand = df["candidate_id"].nunique()
    print(f"Loaded {len(df):,} rows, {n_cand} candidates")
    assert all(c in df.columns for c in TRUE_COLS), "true_* columns missing"

    # Stable row order: chronological as shipped (firehose order preserved).
    sealed = df[["tweet_id", "candidate_id"] + TRUE_COLS].copy()
    blind = df.drop(columns=TRUE_COLS)

    blind_path = HERE / "blind_corpus.parquet"
    sealed_path = HERE / "sealed_truth.parquet"
    blind.to_parquet(blind_path, index=False, compression="zstd")
    sealed.to_parquet(sealed_path, index=False, compression="zstd")

    # On a plain rebuild (same source, no --reseal) the parquet hashes may
    # legitimately change with pandas/pyarrow versions; refresh them but
    # preserve the original seal date so the seal's provenance is not
    # silently re-stamped.
    manifest = {
        "sealed_on": (prior["sealed_on"] if prior is not None and not reseal
                      else date.today().isoformat()),
        "rebuilt_on": date.today().isoformat(),
        "source_file": DATA.name,
        "source_sha256": sha256(DATA),
        "blind_corpus": {
            "file": blind_path.name,
            "sha256": sha256(blind_path),
            "rows": int(len(blind)),
            "candidates": int(n_cand),
            "columns": list(blind.columns),
        },
        "sealed_truth": {
            "file": sealed_path.name,
            "sha256": sha256(sealed_path),
            "rows": int(len(sealed)),
            "columns": list(sealed.columns),
        },
        "protocol": (
            "Workstreams read blind_corpus.parquet only. sealed_truth.parquet "
            "is opened once per workstream at the validation step declared in "
            "its preregistration.md, written BEFORE unsealing."
        ),
    }
    with open(HERE / "seal_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"blind_corpus.parquet : {len(blind):,} rows, "
          f"{len(blind.columns)} cols, sha256 {manifest['blind_corpus']['sha256'][:12]}…")
    print(f"sealed_truth.parquet : {len(sealed):,} rows, "
          f"sha256 {manifest['sealed_truth']['sha256'][:12]}…")
    print("Seal manifest written.")


if __name__ == "__main__":
    main()
