"""Sample-size plan: collect what the analysis needs, no more.   python sample_size.py

Every figure here goes into the paper's methods section. Assumptions are explicit and can be changed.
Unit of inference is the ACCOUNT; videos and comments are repeated measures inside an account and only
need to be large enough to estimate that account's profile precisely, not the population.
"""
import math

import pandas as pd

from common import PROC, RAW, read_jsonl

Z = 1.96  # 95% confidence


def n_proportion(p, d):
    """Accounts needed to estimate a proportion p (recall, precision, accuracy) to +/- d."""
    return math.ceil(Z ** 2 * p * (1 - p) / d ** 2)


def wilson_half_width(p, n):
    centre_adj = Z * math.sqrt(p * (1 - p) / n + Z ** 2 / (4 * n ** 2)) / (1 + Z ** 2 / n)
    return centre_adj


def n_kappa(po, pe, d):
    """Doubly-coded accounts for Cohen's kappa +/- d (large-sample SE; Fleiss-style approximation)."""
    return math.ceil(Z ** 2 * po * (1 - po) / (d ** 2 * (1 - pe) ** 2))


def comments_per_account(m_videos, per_video, icc=0.05, d_target=0.10):
    """Effective sample size of comments when they cluster in videos (design effect 1+(m-1)ICC)."""
    n = m_videos * per_video
    deff = 1 + (per_video - 1) * icc
    n_eff = n / deff
    return n, n_eff, Z * math.sqrt(0.25 / n_eff)  # worst-case p = .5


def main():
    print("== 1. Accounts per class (precision of per-class recall/precision) ==")
    for p, d in ((0.80, 0.07), (0.80, 0.05), (0.90, 0.05), (0.50, 0.05)):
        print(f"  expected p={p:.2f}, half-width +/-{d:.2f}: n = {n_proportion(p, d)} accounts in that class")
    target = 200
    print(f"  -> TARGET: >= {target} gold accounts per class (recall 0.80 +/- {Z * math.sqrt(.8 * .2 / target):.3f})."
          " Going from 200 to 300/class buys only ~0.01 more precision: stop there.")

    print("\n== 2. Human double-coding for inter-coder agreement ==")
    for po, pe, d in ((0.85, 0.5, 0.10), (0.85, 0.5, 0.12), (0.90, 0.5, 0.10)):
        print(f"  observed agreement {po}, chance {pe}, kappa +/- {d}: n = {n_kappa(po, pe, d)} doubly-coded accounts")
    print("  -> PLAN: second coder on 150 accounts (kappa CI about +/-0.12); one coder on the rest, disagreements adjudicated.")

    print("\n== 3. Comments per account (clustered in videos; ICC assumed 0.05 - verify on pilot data) ==")
    for m, k in ((10, 100), (8, 40), (6, 30), (5, 20)):
        n, ne, hw = comments_per_account(m, k)
        print(f"  {m} videos x {k} comments = {n:>4} collected, effective n = {ne:>5.0f}, worst-case half-width +/-{hw:.3f}")
    print("  -> PLAN: 8 videos x 40 comments (<= 320/account). 10 x 100 costs 3x the quota for only ~0.02 better precision.")

    print("\n== 4. Videos per account ==")
    print("  Share of advisory videos estimated from 30 videos: worst-case SE ="
          f" {math.sqrt(.25 / 30):.3f}; flag thresholds (0.25 vs 0.05) are far apart relative to that. Keep 30.")

    print("\n== 5. Models: events-per-variable guard ==")
    print("  Account-level features after TF-IDF cap (20k, penalised LR) and 25 behavioural features: use penalised/regularised"
          " models; with ~200/class the minority class supports regularised LR/GBM but not unpenalised high-dim fits.")

    print("\n== Current inventory vs target ==")
    acc = pd.read_csv(PROC / "accounts.csv") if (PROC / "accounts.csv").exists() else None
    if acc is not None:
        print("  weak labels (accounts with videos):", acc[acc.weak_label != "no_videos"].weak_label.value_counts().to_dict())
        n_auth = int(((acc.weak_label == "authorized")).sum())
        print(f"  authorized so far: {n_auth} (target {target}) -> {'short by ' + str(target - n_auth) if n_auth < target else 'enough'}")
    com = RAW / "comments.jsonl"
    if com.exists():
        c = pd.DataFrame(read_jsonl(com))
        print(f"  comments collected: {len(c)} from {c.channel_id.nunique()} channels (cap per account in plan: 320)")


if __name__ == "__main__":
    main()
