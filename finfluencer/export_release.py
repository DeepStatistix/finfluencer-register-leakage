"""Build the public data release (data/released/) from the local, private data.

    python export_release.py --out <repository>/data/released

Released: one row per analysed account with a salted-hash identifier, registration status and the rule that produced it,
numeric platform-behaviour features, the out-of-fold ranking score, the validation group and the two coders' judgements;
plus every results table used in the manuscript.
NOT released: channel names, URLs, titles, descriptions, comments, register names or numbers, and the hashing salt.
The salt is read from data/finfluencer/.salt (or the FIN_SALT environment variable) and must stay private; without it the
hashed identifiers cannot be linked back to channels.
"""
import argparse
import hashlib
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import eval_pu as E
from common import ANN, DATA, PROC

RESULT_FILES = ["pu_results.csv", "balance_check.csv", "flag_pilot_results.csv", "active_learning_summary.csv", "active_learning_curves.csv",
                "leakage_gap_ci.csv", "nlp_oof_summary.csv", "nlp_ablation.csv", "scrub_levels.csv", "corpus_stats.csv", "top_terms_scrubbed.csv", "nested_tuning.csv", "nested_tuning_gap.csv"]


def salt():
    p = DATA / ".salt"
    return os.environ.get("FIN_SALT") or p.read_text().strip()


def main(a):
    out = Path(a.out)
    (out / "results").mkdir(parents=True, exist_ok=True)
    s = salt()
    h = lambda cid: hashlib.sha256((s + str(cid)).encode()).hexdigest()[:12]

    acc = pd.read_csv(PROC / "accounts.csv")
    acc = acc[acc.n_videos > 0]
    frame = E.build_frame()
    beh = frame[["channel_id"] + E.BEHAV].set_index("channel_id")
    pool = set(frame.channel_id)
    tbl = acc[["channel_id", "reg_status", "reg_resolution_rule", "n_videos"]].copy()
    tbl["in_analysis_pool"] = tbl.channel_id.isin(pool)
    tbl["subscribers_decile"] = pd.qcut(acc.set_index("channel_id").subscribers.reindex(tbl.channel_id).rank(method="first"), 10, labels=False).to_numpy() + 1
    tbl = tbl.join(beh, on="channel_id")

    sc = pd.read_csv(PROC / "pu_scores.csv")[["channel_id", "score"]].rename(columns={"score": "pu_out_of_fold_score"})
    tbl = tbl.merge(sc, on="channel_id", how="left")

    key = pd.read_csv(ANN / "topk" / "key_hidden.csv")[["channel_id", "group"]].rename(columns={"group": "validation_group"})
    tbl = tbl.merge(key, on="channel_id", how="left")
    for who in "AB":
        c = pd.read_csv(ANN / "topk" / f"coder_{who}.csv", dtype=str).dropna(subset=["channel_id"])
        c = c[["channel_id", "gives_specific_recommendations", "sells_tips_or_advisory", "entity_type"]]
        c.columns = ["channel_id"] + [f"coder{who}_{x}" for x in ("gives_specific_recommendations", "sells_tips_or_advisory", "entity_type")]
        tbl = tbl.merge(c, on="channel_id", how="left")
    bounds = pd.read_csv(PROC / "topk_validation_bounds.csv")[["channel_id", "strict", "lenient"]].rename(columns={"strict": "flag_strict", "lenient": "flag_lenient"})
    tbl = tbl.merge(bounds, on="channel_id", how="left")
    tbl.insert(0, "account_id", tbl.channel_id.map(h))
    tbl = tbl.drop(columns="channel_id").sort_values("account_id")
    tbl.to_csv(out / "accounts.csv", index=False)
    for f in RESULT_FILES:
        if (PROC / f).exists():
            shutil.copy(PROC / f, out / "results" / f)
    print(f"wrote {len(tbl)} accounts and {len(list((out / 'results').iterdir()))} result tables to {out}")
    print("columns:", list(tbl.columns))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
