# Regenerating the WS3 artifacts

Nothing in WS3 is excluded from version control except two intermediate
directories that were never committed:

- `outputs/batches/` — deterministic bundle packets; rebuild with
  `python scripts/01_build_bundles.py`.
- `outputs/raw_scores/` — the per-agent score JSONs, the **non-regenerable
  primary data** of this workstream. They are committed as the audit archive
  `outputs/raw_scores.tar.gz`; before running `02_collect_scores.py`,
  extract it in place:

  ```bash
  cd outputs && tar -xzf raw_scores.tar.gz
  ```

The scoring itself was performed by in-session LLM agents against the frozen
prompt (`prompts/scoring_prompt_v1.md`), not by API-calling code in this
repo — there is no committed code that can re-score. Everything downstream
of `02_collect_scores.py` reproduces from the committed raw scores.

Script numbering jumps from 03 to 06 because the unseal script's name was
fixed by preregistration §7 — hence the 04–05 gap (see README).

Re-running `06_unseal_validate.py` preserves the hand-recorded D1 deferral
fields (`d1_status`, `d1_status_note`) in `outputs/decision.json`, which
`synthesis/scripts/05_consolidated_table.py` reads.
