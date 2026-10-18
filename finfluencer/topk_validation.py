"""Stage 6 - human validation of the PU ranking: do high-scoring UNREGISTERED accounts really operate like advisers?

    python topk_validation.py make --n-top 100 --n-control 50     # blind sheets for two coders + hidden key
    (two coders fill coder_A.csv / coder_B.csv independently, ~2 min per account, read-only: no need to watch videos)
    python topk_validation.py kappa                                # agreement + adjudication file
    python topk_validation.py score                                # precision@k with CIs, lift over control

Design
  * top group    = the n_top highest out-of-fold scores among UNREGISTERED accounts (pu_scores.csv).
  * control group = n_control accounts drawn at random from the LOWER-scoring half of unregistered accounts.
  * rows are shuffled and the score/group are hidden, so coders cannot be biased by the ranking.
  * outcome 'flag' (per account, after adjudication) = gives_specific_recommendations = 1 OR sells_tips_or_advisory = 1,
    AND entity_type is not 'other_regulated_entity' (brokers, banks, AMCs are regulated elsewhere).
  * precision@k = share of the top group with flag = 1; lift = precision(top) / precision(control).
Sample size: with 100 top accounts, a precision near 0.5 has a 95% interval of about +/-0.10; 150 double-coded accounts give a
kappa interval of about +/-0.12 (see sample_size.py).
"""
import argparse
import math

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from common import ANN, PROC, RAW, read_jsonl

FIELDS = ["gives_specific_recommendations", "sells_tips_or_advisory"]
ENTITY = "entity_type"
ENTITY_VALUES = ["educator", "news_media", "other_regulated_entity", "individual_adviser_like", "crypto_forex_trader", "other"]
DIR = ANN / "topk"


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def make(a):
    DIR.mkdir(exist_ok=True)
    s = pd.read_csv(PROC / "pu_scores.csv")
    u = s[s.reg_status == "unregistered"].sort_values("score", ascending=False).reset_index(drop=True)
    top = u.head(a.n_top).assign(group="top")
    lower = u.iloc[len(u) // 2:]
    ctrl = lower.sample(min(a.n_control, len(lower)), random_state=a.seed).assign(group="control")
    pick = pd.concat([top, ctrl]).sample(frac=1, random_state=a.seed)

    chans = {c["channel_id"]: c for c in read_jsonl(RAW / "channels.jsonl")}
    vids = {}
    for v in read_jsonl(RAW / "videos.jsonl"):
        vids.setdefault(v["channel_id"], []).append(v)
    rows = []
    for _, r in pick.iterrows():
        c = chans[r.channel_id]
        vs = sorted(vids.get(r.channel_id, []), key=lambda v: v["published_at"] or "", reverse=True)
        row = {"channel_id": r.channel_id, "channel_url": "https://www.youtube.com/channel/" + r.channel_id,
               "title": c["title"], "subscribers": c["subscribers"], "channel_description": (c["description"] or "")[:700]}
        for i, v in enumerate(vs[:6], 1):
            row[f"video{i}_title"] = v["title"]
            row[f"video{i}_description"] = (v["description"] or "")[:300]
        rows.append(row)
    sheet = pd.DataFrame(rows)
    for f in FIELDS:
        sheet[f] = ""
    sheet[ENTITY] = ""
    sheet["notes"] = ""
    sheet.to_csv(DIR / "coder_A.csv", index=False, encoding="utf-8-sig")
    sheet.to_csv(DIR / "coder_B.csv", index=False, encoding="utf-8-sig")
    pick[["channel_id", "group", "score"]].to_csv(DIR / "key_hidden.csv", index=False)
    print(f"{len(sheet)} accounts ({a.n_top} top + {len(ctrl)} control) -> {DIR}")


def load(which):
    return pd.read_csv(DIR / f"coder_{which}.csv", dtype=str).set_index("channel_id")


def kappa(a):
    A, B = load("A"), load("B")
    ids = A.index.intersection(B.index)
    out = []
    for f in FIELDS + [ENTITY]:
        m = A.loc[ids, f].notna() & B.loc[ids, f].notna()
        x, y = A.loc[ids[m], f].to_numpy(), B.loc[ids[m], f].to_numpy()
        boots = []
        rng = np.random.default_rng(0)
        for _ in range(1000):
            i = rng.integers(0, len(x), len(x))
            if len(set(x[i])) > 1 or len(set(y[i])) > 1:
                boots.append(cohen_kappa_score(x[i], y[i]))
        lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
        out.append({"field": f, "n": int(m.sum()), "raw_agreement": round(float((x == y).mean()), 3),
                    "kappa": round(cohen_kappa_score(x, y), 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
    print(pd.DataFrame(out).to_string(index=False))
    d = pd.DataFrame({"channel_id": ids})
    for f in FIELDS + [ENTITY]:
        d[f + "_A"], d[f + "_B"] = A.loc[ids, f].values, B.loc[ids, f].values
    cols = [f + s for f in FIELDS + [ENTITY] for s in ("_A", "_B")]
    d = d[(d[[c for c in cols if c.endswith("_A")]].values != d[[c for c in cols if c.endswith("_B")]].values).any(axis=1)]
    for f in FIELDS + [ENTITY]:
        d[f] = ""
    d.to_csv(DIR / "adjudicated.csv", index=False, encoding="utf-8-sig")
    print(f"{len(d)} accounts need adjudication -> adjudicated.csv (fill the plain columns: {FIELDS + [ENTITY]})")


def score(a):
    A, B = load("A"), load("B")
    adj = pd.read_csv(DIR / "adjudicated.csv", dtype=str).set_index("channel_id")
    key = pd.read_csv(DIR / "key_hidden.csv").set_index("channel_id")
    rows = []
    for cid in A.index.intersection(B.index):
        r = {"channel_id": cid}
        for f in FIELDS + [ENTITY]:
            r[f] = A.at[cid, f] if A.at[cid, f] == B.at[cid, f] else adj.at[cid, f]
        rows.append(r)
    d = pd.DataFrame(rows).set_index("channel_id").join(key)
    d["flag"] = (((d[FIELDS[0]] == "1") | (d[FIELDS[1]] == "1")) & (d[ENTITY] != "other_regulated_entity")).astype(int)
    res = []
    for g, x in d.groupby("group"):
        k, n = int(x.flag.sum()), len(x)
        lo, hi = wilson(k, n)
        res.append({"group": g, "n": n, "flagged": k, "precision": round(k / n, 3), "ci95": f"[{lo:.2f}, {hi:.2f}]"})
    print(pd.DataFrame(res).to_string(index=False))
    t, c = d[d.group == "top"].flag, d[d.group == "control"].flag
    if len(c) and c.mean() > 0:
        print(f"\nlift (top / control) = {t.mean() / c.mean():.2f}")
    print("\nentity type among TOP group (shows e.g. brokers/media being ranked high):")
    print(d[d.group == "top"][ENTITY].value_counts().to_string())
    for k in (25, 50, 100):
        x = d[d.group == "top"].sort_values("score", ascending=False).head(k)
        if len(x) == k:
            print(f"precision@{k}: {x.flag.mean():.2f}")
    d.reset_index().to_csv(PROC / "topk_validation_final.csv", index=False)


def bounds(a):
    """Precision without a third adjudicator: strict = BOTH coders flag, lenient = EITHER coder flags."""
    A, B = load("A").dropna(subset=["entity_type"]), load("B").dropna(subset=["entity_type"])
    ids = A.index.intersection(B.index).dropna()
    key = pd.read_csv(DIR / "key_hidden.csv").set_index("channel_id")

    def flag(X):
        return (((X.loc[ids, FIELDS[0]] == "1") | (X.loc[ids, FIELDS[1]] == "1")) & (X.loc[ids, ENTITY] != "other_regulated_entity")).astype(int)
    d = key.loc[ids].assign(flag_A=flag(A), flag_B=flag(B))
    d["strict"], d["lenient"] = d.flag_A & d.flag_B, d.flag_A | d.flag_B
    d["regulated_A"], d["regulated_B"] = A.loc[ids, ENTITY] == "other_regulated_entity", B.loc[ids, ENTITY] == "other_regulated_entity"
    print(f"flag agreement between coders: {(d.flag_A == d.flag_B).mean():.3f}   kappa = {cohen_kappa_score(d.flag_A, d.flag_B):.3f}")
    rows = []
    for g, x in d.groupby("group"):
        for rule in ("strict", "lenient"):
            k, n = int(x[rule].sum()), len(x)
            lo, hi = wilson(k, n)
            rows.append({"group": g, "rule": rule, "n": n, "flagged": k, "precision": round(k / n, 3), "ci95": f"[{lo:.2f}, {hi:.2f}]"})
    print(pd.DataFrame(rows).to_string(index=False))
    for rule in ("strict", "lenient"):
        t, c = d[d.group == "top"][rule], d[d.group == "control"][rule]
        print(f"lift ({rule}) = {t.mean() / c.mean():.2f}" if c.mean() > 0 else f"lift ({rule}): control rate 0")
    top = d[d.group == "top"].sort_values("score", ascending=False)
    for k in (10, 25, 50, 100):
        print(f"precision@{k}: strict {top.head(k).strict.mean():.2f}  lenient {top.head(k).lenient.mean():.2f}")
    print("\nentity type (coder A / B) in TOP group:")
    print(pd.concat([A.loc[top.index, ENTITY].value_counts().rename("A"), B.loc[top.index, ENTITY].value_counts().rename("B")], axis=1).fillna(0).astype(int).to_string())
    print("regulated-elsewhere share in TOP group (either coder):", round(float((top.regulated_A | top.regulated_B).mean()), 3))
    d.reset_index().to_csv(PROC / "topk_validation_bounds.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--n-top", type=int, default=100)
    m.add_argument("--n-control", type=int, default=50)
    m.add_argument("--seed", type=int, default=11)
    sub.add_parser("kappa")
    sub.add_parser("score")
    sub.add_parser("bounds")
    args = ap.parse_args()
    {"make": make, "kappa": kappa, "score": score, "bounds": bounds}[args.cmd](args)
