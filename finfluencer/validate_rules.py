"""Stage 3b - validate the content rules on real videos (precision per rule, recall of the advisory rules).

    python validate_rules.py make     # writes rule_hits_to_check.csv and rule_misses_to_check.csv
    (a human fills the blank 'correct' / 'is_advisory_call' columns with 1 or 0)
    python validate_rules.py score    # precision with Wilson 95% CI per rule; estimated recall of trade_call|guarantee

Unit = one video (title + description). Hits are sampled with at most 3 per channel so a few prolific
channels cannot dominate. Channel labels are NOT shown, so the check is blind to the weak label.
Sample sizes: 100 hits/rule gives a precision interval of about +/-0.10 at p=0.5 and +/-0.06 at p=0.9;
150 misses detect a miss rate of 10% to about +/-0.05 (see sample_size.py for the arithmetic).
"""
import argparse
import math
import re

import pandas as pd

from common import ANN, PROC, RAW, RULES, read_jsonl, rule_flags

RULE_FILE, MISS_FILE = ANN / "rule_hits_to_check.csv", ANN / "rule_misses_to_check.csv"
ADVISORY = ("trade_call", "guarantee")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def context(text, pattern, width=140):
    m = pattern.search(text)
    if not m:
        return ""
    a, b = max(m.start() - width, 0), min(m.end() + width, len(text))
    return ("..." if a else "") + text[a:m.start()] + "[[" + m.group(0) + "]]" + text[m.end():b] + ("..." if b < len(text) else "")


def make(args):
    pool = set(pd.read_csv(PROC / "analysis_pool.csv").channel_id)
    chans = {c["channel_id"]: c["title"] for c in read_jsonl(RAW / "channels.jsonl")}
    vids = [v for v in read_jsonl(RAW / "videos.jsonl") if v["channel_id"] in pool]
    for v in vids:
        v["text"] = f"{v['title']}\n{v['description'] or ''}"
        v["flags"] = rule_flags(v["text"])
    print(len(vids), "videos in the analysis pool")

    rows = []
    for rule, pat in RULES.items():
        hits = pd.DataFrame([v for v in vids if v["flags"][rule] > 0])
        n_hits = len(hits)
        if n_hits:
            hits = hits.sample(frac=1, random_state=args.seed).groupby("channel_id").head(3).head(args.n_hits)
        for _, v in hits.iterrows():
            rows.append({"rule": rule, "population_hits": n_hits, "video_id": v.video_id, "channel": chans.get(v.channel_id),
                         "video_title": v.title, "matched_in_context": context(v.text, pat), "correct": ""})
        print(f"{rule:20s} population hits: {n_hits:6d}   sampled: {len(hits)}")
    pd.DataFrame(rows).to_csv(RULE_FILE, index=False, encoding="utf-8-sig")

    nonhits = pd.DataFrame([v for v in vids if not any(v["flags"][r] for r in ADVISORY)])
    miss = nonhits.sample(min(args.n_misses, len(nonhits)), random_state=args.seed)
    pd.DataFrame({"population_nonhits": len(nonhits), "video_id": miss.video_id, "channel": miss.channel_id.map(chans),
                  "video_title": miss.title, "description_start": miss.description.fillna("").str[:500],
                  "is_advisory_call": ""}).to_csv(MISS_FILE, index=False, encoding="utf-8-sig")
    print(f"non-hit videos in pool: {len(nonhits)}; sampled {len(miss)}\nwrote {RULE_FILE.name}, {MISS_FILE.name} in {ANN}")


def score(args):
    h = pd.read_csv(RULE_FILE, dtype=str)
    h = h[h.correct.isin(["0", "1"])]
    out = []
    for rule, g in h.groupby("rule"):
        k, n = int((g.correct == "1").sum()), len(g)
        lo, hi = wilson(k, n)
        out.append({"rule": rule, "checked": n, "correct": k, "precision": round(k / n, 3), "ci95": f"[{lo:.2f}, {hi:.2f}]",
                    "population_hits": int(g.population_hits.iloc[0])})
    res = pd.DataFrame(out)
    print(res.to_string(index=False))
    res.to_csv(ANN / "rule_precision.csv", index=False)

    m = pd.read_csv(MISS_FILE, dtype=str)
    m = m[m.is_advisory_call.isin(["0", "1"])]
    if len(m):
        rate = (m.is_advisory_call == "1").mean()
        lo, hi = wilson(int((m.is_advisory_call == "1").sum()), len(m))
        n_non = int(m.population_nonhits.iloc[0])
        # precision of the union rule: approximate with the lower of the two (conservative) when both were checked
        prec = res[res.rule.isin(ADVISORY)].precision.min() if res.rule.isin(ADVISORY).any() else float("nan")
        n_hit = int(res[res.rule.isin(ADVISORY)].population_hits.max()) if res.rule.isin(ADVISORY).any() else 0
        tp, fn = prec * n_hit, rate * n_non
        print(f"\nMissed advisory calls among non-hit videos: {rate:.3f}  (95% CI [{lo:.3f}, {hi:.3f}], n={len(m)})")
        print(f"Estimated recall of trade_call|guarantee at video level ~ {tp / (tp + fn):.2f}"
              f"   (uses the conservative precision {prec:.2f} and {n_hit} hits vs {n_non} non-hits)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--n-hits", type=int, default=100)
    m.add_argument("--n-misses", type=int, default=150)
    m.add_argument("--seed", type=int, default=7)
    sub.add_parser("score")
    a = ap.parse_args()
    {"make": make, "score": score}[a.cmd](a)
