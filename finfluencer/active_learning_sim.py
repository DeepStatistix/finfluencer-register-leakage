"""Stage 9 - annotation-efficient screening: how many human codes are needed?

    python active_learning_sim.py --repeats 100

Question  A regulator can afford few expert codes. Can choosing WHICH accounts to code beat coding a random sample?
Data      the 150 doubly-coded unregistered accounts (flag = adviser-like behaviour; eval_flag.frame). NOT a random
          sample of the population (100 top-scored + 50 control), so this shows relative efficiency of query strategies
          on this pool, not absolute accuracy for YouTube at large.
Protocol  Each repeat: stratified split into a pool (70%) and a held-out test set (30%). Start with 10 random labels,
          then query batches of 5 until 100 labels are used. After every batch a logistic regression on TF-IDF of the
          regulatory-scrubbed text is refit on the revealed labels and scored on the test set (AUC). All strategies share
          the same splits and seeds (paired). TF-IDF is fitted on pool text only (no test information).
Strategies  random | uncertainty (closest to 0.5) | diversity (greedy k-center in TF-IDF cosine space) |
            uncertainty+diversity (5 k-center points among the 25 most uncertain).
Outputs   processed/active_learning_curves.csv, active_learning_summary.csv, active_learning_curve.png
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

import eval_flag as F
from common import PROC

INIT, BATCH, BUDGET = 10, 5, 100
STRATS = ["random", "uncertainty", "diversity", "uncertainty+diversity"]
COLORS = {"random": "#2a78d6", "uncertainty": "#eb6834", "diversity": "#1baf7a", "uncertainty+diversity": "#eda100"}  # palette slots 1-4, fixed order


def fit(X, y, idx):
    if len(set(y[idx])) < 2:
        return None
    return LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5).fit(X[idx], y[idx])


def kcenter(X, cand, labelled, k):
    """Greedy farthest-point selection among cand, distance = 1 - cosine to the nearest chosen/labelled point."""
    chosen, ref = [], list(labelled)
    d = 1 - (X[cand] @ X[ref].T).toarray().max(axis=1) if ref else np.ones(len(cand))
    for _ in range(k):
        j = int(np.argmax(d))
        chosen.append(cand[j])
        d = np.minimum(d, 1 - (X[cand] @ X[cand[j]].T).toarray().ravel())
        d[j] = -1
    return chosen


def pick(strategy, X, y, labelled, unl, model, rng):
    if strategy == "random" or model is None:
        return list(rng.choice(unl, size=min(BATCH, len(unl)), replace=False))
    p = model.predict_proba(X[unl])[:, 1]
    unc = np.abs(p - 0.5)
    if strategy == "uncertainty":
        return [unl[i] for i in np.argsort(unc)[:BATCH]]
    if strategy == "diversity":
        return kcenter(X, np.array(unl), labelled, BATCH)
    top = np.array(unl)[np.argsort(unc)[:25]]
    return kcenter(X, top, labelled, BATCH)


def run_repeat(df, seed):
    y = df.y.to_numpy()
    tr, te = next(StratifiedShuffleSplit(1, test_size=0.3, random_state=seed).split(df, y))
    vec = TfidfVectorizer(min_df=3, ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode").fit(df.text_clean.iloc[tr])
    X = vec.transform(df.text_clean)
    full = fit(X, y, tr)
    ref_auc = roc_auc_score(y[te], full.predict_proba(X[te])[:, 1])
    rng0 = np.random.default_rng(seed)
    init = list(rng0.choice(tr, size=INIT, replace=False))
    while len(set(y[init])) < 2:
        init = list(rng0.choice(tr, size=INIT, replace=False))
    rows = []
    for st in STRATS:
        rng = np.random.default_rng(seed + 1000)
        lab = list(init)
        while True:
            m = fit(X, y, lab)
            auc = roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]) if m is not None else 0.5
            rows.append({"seed": seed, "strategy": st, "labels": len(lab), "auc": auc, "ref_auc": ref_auc, "positives_found": int(y[lab].sum())})
            if len(lab) >= BUDGET:
                break
            unl = [i for i in tr if i not in set(lab)]
            lab += pick(st, X, y, lab, unl, m, rng)
    return rows


def labels_to_reach(g, target):
    ok = g[g.auc >= target]
    return int(ok.labels.min()) if len(ok) else np.nan


def main(a):
    df = F.frame("lenient")
    print(f"pool of coded accounts: n={len(df)}, flagged={int(df.y.sum())} ({df.y.mean():.0%})")
    res = pd.DataFrame([r for s in range(a.repeats) for r in run_repeat(df, s)])
    res.to_csv(PROC / "active_learning_curves.csv", index=False)

    curve = res.groupby(["strategy", "labels"]).auc.agg(["mean", lambda s: np.percentile(s, 2.5), lambda s: np.percentile(s, 97.5)])
    curve.columns = ["mean_auc", "lo", "hi"]
    ref = res.groupby("seed").ref_auc.first().mean()
    out = []
    piv = res.pivot_table(index=["seed", "labels"], columns="strategy", values="auc")
    for st in STRATS:
        r = {"strategy": st}
        for b in (20, 30, 40, 60, 100):
            r[f"AUC@{b}"] = curve.loc[(st, b), "mean_auc"]
        # labels needed to reach 95% of the all-pool-labels AUC, per repeat then averaged
        need = [labels_to_reach(g, 0.95 * g.ref_auc.iloc[0]) for _, g in res[res.strategy == st].groupby("seed")]
        r["labels_to_95%_of_full"] = np.nanmean(need)
        r["share_reaching_95%"] = 1 - np.mean(np.isnan(need))
        pos = res[(res.strategy == st) & (res.labels == 40)].positives_found
        r["positives_in_first_40"] = pos.mean()
        if st != "random":   # paired difference in mean AUC over the whole curve vs random
            d = (piv[st] - piv["random"]).groupby("seed").mean()
            b = np.random.default_rng(0).choice(d.to_numpy(), (2000, len(d))).mean(axis=1)
            r["dAUC_vs_random(mean over curve)"] = f"{d.mean():+.3f} [{np.percentile(b, 2.5):+.3f}, {np.percentile(b, 97.5):+.3f}]"
        out.append(r)
    summ = pd.DataFrame(out).round(3)
    summ.to_csv(PROC / "active_learning_summary.csv", index=False)
    print(f"\nreference: AUC with ALL pool labels (~105) = {ref:.3f}\n")
    print(summ.to_string(index=False))
    plot(curve, ref, a.repeats)


def plot(curve, ref, repeats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bg, ink, muted = "#fcfcfb", "#0b0b0b", "#52514e"
    fig, ax = plt.subplots(figsize=(6.6, 4.0), dpi=200, facecolor=bg)
    ax.set_facecolor(bg)
    ends = {}
    for st in STRATS:
        c = curve.loc[st]
        ax.plot(c.index, c.mean_auc, color=COLORS[st], lw=2, label=st, solid_capstyle="round")
        ends[st] = c.mean_auc.iloc[-1]
    ax.axhline(ref, color=muted, lw=1, ls=(0, (4, 3)))
    ax.text(INIT + 1, ref + 0.006, "AUC with all pool labels", ha="left", va="bottom", fontsize=7.5, color=muted)
    ax.axhline(0.5, color=muted, lw=0.8)
    ax.text(INIT + 1, 0.505, "chance", ha="left", va="bottom", fontsize=7.5, color=muted)
    for st, y in ends.items():   # direct labels, in text ink; coloured line carries identity
        pass
    ax.set_xlim(INIT, BUDGET)
    ax.set_ylim(0.48, 0.85)  # axis from chance upward: differences between strategies are ~0.01 and must look that small
    ax.set_xlabel("Human-coded accounts used for training", color=ink, fontsize=9)
    ax.set_ylabel("AUC on held-out coded accounts", color=ink, fontsize=9)
    ax.grid(axis="y", color="#e4e3df", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bdbcb6")
    ax.tick_params(colors=muted, labelsize=8)
    leg = ax.legend(loc="center right", frameon=False, fontsize=8, labelcolor=ink, title=f"Query strategy (mean of {repeats} paired repeats)", title_fontsize=7.5)
    leg.get_title().set_color(muted)
    fig.tight_layout()
    fig.savefig(PROC / "active_learning_curve.png", facecolor=bg)
    print("wrote active_learning_curve.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=100)
    main(ap.parse_args())
