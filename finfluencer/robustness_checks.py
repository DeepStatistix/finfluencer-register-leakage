"""Two robustness checks of the leakage-gap result.

    python robustness_checks.py groups    # cross-validation that keeps accounts of the same registered entity together
    python robustness_checks.py deletion  # control: delete the same NUMBER of random sentences per account instead of the regulatory ones

groups    Several registered advisers run more than one channel (the same registration number is quoted by more than one account).
          With ordinary stratified folds such related accounts can fall into training and test parts of the same fold. Here the
          accounts are grouped by registration number and StratifiedGroupKFold is used, with the same models and metrics as the main analysis.
deletion  Scrubbing removes about 13% of the characters. If the AUC fell merely because the text is shorter, deleting the same number of
          randomly chosen sentences from every account would lower the AUC as well. The control keeps the length effect and drops the content effect.
Outputs   processed/robustness_groups.csv, processed/robustness_deletion.csv
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import common
import eval_pu as E
from common import PROC
from leakage_miner import sentences
from nlp_experiments import boot_auc, boot_pair, collect, featurise, splits


def lr_scores(texts, sp, y):
    mats = featurise(texts, sp, "word")
    s = [LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5).fit(Xtr, y[tr]).predict_proba(Xte)[:, 1] for (Xtr, Xte), (_, tr, te) in zip(mats, sp)]
    return collect(sp, len(y), s)


def groups(a):
    df = E.build_frame()
    y = df.y.to_numpy()
    acc = pd.read_csv(PROC / "accounts.csv").set_index("channel_id")
    nums = acc.verified_reg_nos.reindex(df.channel_id).fillna("")
    gid, seen = [], {}
    for i, (cid, n) in enumerate(zip(df.channel_id, nums)):
        key = n.split(";")[0] if n else f"single_{i}"
        gid.append(seen.setdefault(key, len(seen)))
    gid = np.array(gid)
    sizes = pd.Series(gid).value_counts()
    print(f"{len(df)} accounts in {len(sizes)} groups; groups with more than one account: {(sizes > 1).sum()} (accounts involved: {int(sizes[sizes > 1].sum())})")
    sp = []
    for r in range(3):
        for k, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=r).split(np.zeros(len(y)), y, gid)):
            sp.append((r, tr, te))
    raw = lr_scores(df.text_leaky.tolist(), sp, y)
    scr = lr_scores(df.text_clean.tolist(), sp, y)
    pr, ps = np.mean(raw, axis=0), np.mean(scr, axis=0)
    ar, as_ = np.mean([roc_auc_score(y, s) for s in raw]), np.mean([roc_auc_score(y, s) for s in scr])
    ci = boot_pair(y, {"raw": pr, "scr": ps}, {"gap": lambda yy, s: roc_auc_score(yy, s["raw"]) - roc_auc_score(yy, s["scr"])})["gap"]
    out = pd.DataFrame([{"cv": "grouped by registration number", "groups_with_several_accounts": int((sizes > 1).sum()), "AUC_raw": round(ar, 4), "AUC_scrubbed": round(as_, 4),
                         "gap": round(ar - as_, 4), "gap_ci_lo": round(ci[0], 4), "gap_ci_hi": round(ci[1], 4)}])
    out.to_csv(PROC / "robustness_groups.csv", index=False)
    print(out.T.to_string())


def deletion(a):
    df = E.build_frame()
    y = df.y.to_numpy()
    rx = common._BOILER
    rng = np.random.default_rng(0)
    raw_t, scr_t, rnd_t, removed_counts = [], [], [], []
    for t in df.text_leaky.tolist():
        ss = sentences(t)
        flags = np.array([bool(rx.search(s)) for s in ss])
        k = int(flags.sum())
        removed_counts.append(k)
        raw_t.append(" ".join(ss))
        scr_t.append(" ".join(s for s, f in zip(ss, flags) if not f))
        keep = np.ones(len(ss), dtype=bool)
        if k:
            keep[rng.choice(len(ss), size=min(k, len(ss)), replace=False)] = False
        rnd_t.append(" ".join(s for s, kp in zip(ss, keep) if kp))
    sp = splits(y)
    res = {name: lr_scores(txt, sp, y) for name, txt in (("raw", raw_t), ("scrubbed", scr_t), ("random_matched_deletion", rnd_t))}
    pooled = {k: np.mean(v, axis=0) for k, v in res.items()}
    rows = []
    for k, v in res.items():
        lo, hi = boot_auc(y, pooled[k])
        rows.append({"text": k, "AUC": round(float(np.mean([roc_auc_score(y, s) for s in v])), 4), "CI_lo": round(lo, 3), "CI_hi": round(hi, 3)})
    fns = {"raw_minus_scrubbed": lambda yy, s: roc_auc_score(yy, s["raw"]) - roc_auc_score(yy, s["scrubbed"]),
           "raw_minus_random_deletion": lambda yy, s: roc_auc_score(yy, s["raw"]) - roc_auc_score(yy, s["random_matched_deletion"])}
    ci = boot_pair(y, pooled, fns)
    for k, fn in fns.items():
        est = fn(y, pooled)
        rows.append({"text": k, "AUC": round(est, 4), "CI_lo": round(ci[k][0], 3), "CI_hi": round(ci[k][1], 3)})
    out = pd.DataFrame(rows)
    out.to_csv(PROC / "robustness_deletion.csv", index=False)
    print(f"sentences removed per account: median {np.median(removed_counts):.0f}, mean {np.mean(removed_counts):.1f}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["groups", "deletion"])
    a = ap.parse_args()
    {"groups": groups, "deletion": deletion}[a.cmd](a)
