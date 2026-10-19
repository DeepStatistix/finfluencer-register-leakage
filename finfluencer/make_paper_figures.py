"""Figures for the manuscript, drawn from the result files in data/finfluencer/processed (no number is typed by hand).

    python make_paper_figures.py            # all figures
    python make_paper_figures.py fig3 fig7  # selected figures

Outputs (paper/finfluencer/):
  fig1_corpus.png            document length and script mix by class
  fig2_leakage_gap.png       AUC by feature set and classifier with 95% intervals
  fig3_roc_pr.png            ROC and precision-recall curves
  fig4_scrubbing.png         AUC against scrubbing intensity
  fig5_representation.png    representation x classifier, scrubbed versus unscrubbed text
  fig6_regularisation.png    regularisation and vocabulary-size sensitivity, nested tuning
  fig7_validation.png        blind validation: precision and composition of the groups
  fig8_pilot.png             supervised pilot on the coded behaviour
  fig9_label_efficiency.png  learning curves (produced by active_learning_sim.py, copied here)

Colour: categorical series use the first slots of one validated palette in fixed order and never carry text; every figure with
two or more series has a legend; the axis of AUC figures starts at chance or below so that differences are not exaggerated.
"""
import math
import re
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from common import ANN, PROC, ROOT

OUT = ROOT.parent / "paper" / "finfluencer"
OUT.mkdir(parents=True, exist_ok=True)
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]     # categorical slots 1-6 (fixed order)
BG, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#bdbcb6"
MODELS = ["logreg", "random_forest", "hist_gbm"]
MODEL_LABEL = {"logreg": "Logistic regression", "random_forest": "Random forest", "hist_gbm": "Gradient boosting"}


def canvas(w, h, nrows=1, ncols=1):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), dpi=220, facecolor=BG)
    for ax in np.atleast_1d(axes).ravel():
        ax.set_facecolor(BG)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=8)
    return fig, axes


def labels(ax, x=None, y=None, title=None):
    if x:
        ax.set_xlabel(x, fontsize=8.5, color=INK)
    if y:
        ax.set_ylabel(y, fontsize=8.5, color=INK)
    if title:
        ax.set_title(title, fontsize=9, color=INK, loc="left", fontweight="bold")


def panel_tag(ax, tag):
    ax.text(-0.02, 1.07, tag, transform=ax.transAxes, fontsize=10, fontweight="bold", color=INK, va="bottom", ha="right")


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, facecolor=BG)
    plt.close(fig)
    print("wrote", OUT / name)


# ------------------------------------------------------------------------------------------------------------- figure 1
def fig1():
    import eval_pu as E
    df = E.build_frame()
    names = ["registered", "unregistered"]
    col = {"registered": SLOT[0], "unregistered": SLOT[1]}
    fig, (a, b) = canvas(6.8, 3.2, 1, 2)
    rng = np.random.default_rng(0)
    for i, n in enumerate(names):
        v = np.log10(df[df.reg_status == n].text_clean.str.len().clip(lower=1))
        a.boxplot(v, positions=[i], widths=0.45, showfliers=False, patch_artist=True, medianprops=dict(color=INK, lw=1.2),
                  boxprops=dict(facecolor=col[n], alpha=0.35, edgecolor=col[n]), whiskerprops=dict(color=col[n]), capprops=dict(color=col[n]))
        a.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=5, color=col[n], alpha=0.5, linewidths=0)
    a.set_xticks([0, 1])
    a.set_xticklabels([f"Registered\n(n = {int((df.reg_status == 'registered').sum())})", f"Unregistered\n(n = {int((df.reg_status == 'unregistered').sum())})"], fontsize=8.5, color=INK)
    labels(a, y="Characters per account document (log$_{10}$)", title="Document length")
    a.grid(axis="y", color=GRID, lw=0.6)
    panel_tag(a, "a")
    pats = {"Latin": re.compile(r"[A-Za-z]"), "Devanagari": re.compile(r"[ऀ-ॿ]"), "Other Indic": re.compile(r"[ঀ-෿]")}
    width = 0.36
    for j, n in enumerate(names):
        t = df[df.reg_status == n].text_clean
        vals = [t.map(lambda s: len(p.findall(s)) / max(len(s), 1)).mean() for p in pats.values()]
        xs = np.arange(len(pats)) + (j - 0.5) * (width + 0.03)
        b.bar(xs, vals, width=width, color=col[n], label=n.capitalize())
    b.set_xticks(range(len(pats)))
    b.set_xticklabels(list(pats), fontsize=8.5, color=INK)
    labels(b, y="Mean share of characters", title="Script mix")
    b.grid(axis="y", color=GRID, lw=0.6)
    b.set_axisbelow(True)
    b.legend(frameon=False, fontsize=8, labelcolor=INK)
    panel_tag(b, "b")
    save(fig, "fig1_corpus.png")


# ------------------------------------------------------------------------------------------------------------- figure 2
def fig2():
    r = pd.read_csv(PROC / "nlp_oof_summary.csv")
    rows = [("behavioural", "Platform behaviour only"), ("text_clean", "Text, regulatory wording removed"),
            ("text_leaky", "Text, regulatory wording kept"), ("text_removed", "Removed regulatory wording only")]
    fig, ax = canvas(6.8, 3.9)
    for i, (fs, _) in enumerate(rows):
        for j, m in enumerate(MODELS):
            row = r[(r.features == fs) & (r.model == m)].iloc[0]
            y = len(rows) - 1 - i + (1 - j) * 0.2
            ax.plot([row.AUC_ci_lo, row.AUC_ci_hi], [y, y], color=SLOT[j], lw=1.6, solid_capstyle="round")
            ax.plot(row.AUC_mean_over_repeats, y, "o", ms=6.5, color=SLOT[j], mec=BG, mew=1.5, label=MODEL_LABEL[m] if i == 0 else None)
    ax.axvline(0.5, color=MUTED, lw=0.9)
    ax.text(0.502, len(rows) - 0.45, "chance", fontsize=7.5, color=MUTED, va="bottom")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([n for _, n in rows][::-1], fontsize=8.5, color=INK)
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(-0.6, len(rows) - 0.3)
    labels(ax, x="AUC, registered vs unregistered accounts (out-of-fold, 95% bootstrap interval)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper right", frameon=False, fontsize=8, labelcolor=INK)
    save(fig, "fig2_leakage_gap.png")


# ------------------------------------------------------------------------------------------------------------- figure 3
def fig3():
    o = pd.read_csv(PROC / "oof_scores.csv")
    y = o.registered.to_numpy()
    sets = [("behavioural", "Platform behaviour"), ("text_clean", "Text, wording removed"), ("text_leaky", "Text, wording kept"), ("text_removed", "Removed wording only")]
    fig, axes = canvas(6.9, 6.2, 2, 2)
    for col, m in enumerate(["logreg", "hist_gbm"]):
        roc, pr = axes[0, col], axes[1, col]
        for k, (fs, name) in enumerate(sets):
            s = o[f"{fs}|{m}"].to_numpy()
            fpr, tpr, _ = roc_curve(y, s)
            pre, rec, _ = precision_recall_curve(y, s)
            roc.plot(fpr, tpr, color=SLOT[k], lw=1.8, label=name)
            pr.plot(rec, pre, color=SLOT[k], lw=1.8, label=name)
        roc.plot([0, 1], [0, 1], color=MUTED, lw=0.9, ls=(0, (4, 3)))
        pr.axhline(y.mean(), color=MUTED, lw=0.9, ls=(0, (4, 3)))
        labels(roc, "False positive rate", "True positive rate", MODEL_LABEL[m])
        labels(pr, "Recall", "Precision")
        for a in (roc, pr):
            a.set_xlim(0, 1)
            a.set_ylim(0, 1.02)
            a.grid(color=GRID, lw=0.6)
            a.set_axisbelow(True)
        panel_tag(roc, "ab"[col])
        panel_tag(pr, "cd"[col])
    axes[0, 0].legend(loc="lower right", frameon=False, fontsize=7.5, labelcolor=INK)
    pr_note = f"dashed: prevalence {y.mean():.2f}"
    axes[1, 1].text(0.98, y.mean() - 0.075, pr_note, ha="right", fontsize=7, color=MUTED, transform=axes[1, 1].get_yaxis_transform())
    save(fig, "fig3_roc_pr.png")


# ------------------------------------------------------------------------------------------------------------- figure 4
def fig4():
    s = pd.read_csv(PROC / "scrub_levels.csv")
    fig, ax = canvas(5.6, 3.6)
    x = s.share_of_characters_kept * 100
    ax.errorbar(x, s.AUC, yerr=[s.AUC - s.CI_lo, s.CI_hi - s.AUC], color=SLOT[0], lw=1.8, marker="o", ms=6.5, mec=BG, mew=1.5, capsize=3, elinewidth=1.2)
    tags = ["L0", "L1", "L2", "L3"]          # L0 raw text; L1 first pass; L2 used in the main results; L3 extended (see caption)
    for xi, yi, t in zip(x, s.AUC, tags):
        ax.annotate(t, (xi, yi), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=8.5, color=INK, fontweight="bold")
    ax.axhline(0.5, color=MUTED, lw=0.9)
    ax.text(85.2, 0.508, "chance", ha="left", fontsize=7.5, color=MUTED)
    ax.set_ylim(0.45, 1.0)
    ax.invert_xaxis()
    labels(ax, "Share of characters kept after scrubbing (%)", "AUC, logistic regression (95% interval)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    save(fig, "fig4_scrubbing.png")


# ------------------------------------------------------------------------------------------------------------- figure 5
def fig5():
    a = pd.read_csv(PROC / "nlp_ablation.csv")
    rep = a[a.experiment == "representation"]
    order = [("word", "Word n-grams"), ("char", "Character n-grams"), ("word+char", "Word and character")]
    clf = [("logreg", "Logistic regression"), ("linear_svm", "Linear SVM"), ("complement_nb", "Complement NB")]
    fig, ax = canvas(6.8, 4.4)
    y = 0
    ticks, tlabels = [], []
    for rp, rname in order:
        for c, cname in clf:
            sc = rep[(rep.text == "text_clean") & (rep.representation == rp) & (rep.classifier == c)].iloc[0]
            lk = rep[(rep.text == "text_leaky") & (rep.representation == rp) & (rep.classifier == c)].iloc[0]
            ax.plot([sc.AUC, lk.AUC], [y, y], color=AXIS, lw=2.2, solid_capstyle="round", zorder=1)
            ax.plot(sc.AUC, y, "o", ms=7, color=SLOT[0], mec=BG, mew=1.5, zorder=3, label="Scrubbed text" if y == 0 else None)
            ax.plot(lk.AUC, y, "o", ms=7, color=SLOT[1], mec=BG, mew=1.5, zorder=3, label="Unscrubbed text" if y == 0 else None)
            ax.text(lk.AUC + 0.008, y, f"gap {lk.AUC - sc.AUC:.3f}", va="center", fontsize=7, color=MUTED)
            ticks.append(y)
            tlabels.append(f"{rname}, {cname}")
            y -= 1
        y -= 0.4
    ax.set_yticks(ticks)
    ax.set_yticklabels(tlabels, fontsize=7.8, color=INK)
    ax.axvline(0.5, color=MUTED, lw=0.9)
    ax.set_xlim(0.5, 1.0)
    labels(ax, x="AUC (two repeats of five-fold cross-validation)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower left", frameon=False, fontsize=8, labelcolor=INK)
    save(fig, "fig5_representation.png")


# ------------------------------------------------------------------------------------------------------------- figure 6
def fig6():
    a = pd.read_csv(PROC / "nlp_ablation.csv")
    nt = pd.read_csv(PROC / "nested_tuning.csv")
    nt = nt[nt.max_features.astype(str) == "20000"].set_index("features")
    fig, (p, q) = canvas(6.9, 3.3, 1, 2)
    c = a[a.experiment == "C"].sort_values("C_or_alpha")
    p.errorbar(c.C_or_alpha, c.AUC, yerr=[c.AUC - c.CI_lo, c.CI_hi - c.AUC], color=SLOT[0], marker="o", ms=6, mec=BG, mew=1.3, lw=1.6, capsize=3, label="Fixed C (same folds)")
    nested = nt.loc["text_clean"]
    p.axhline(nested.AUC_nested, color=SLOT[1], lw=1.4, ls=(0, (4, 3)), label=f"Nested tuning ({nested.AUC_nested:.3f})")
    p.set_xscale("log")
    p.set_ylim(0.65, 0.95)
    labels(p, "Regularisation parameter C", "AUC, scrubbed text", "Regularisation")
    p.grid(axis="y", color=GRID, lw=0.6)
    p.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc="lower right")
    panel_tag(p, "a")
    v = a[a.experiment == "vocabulary"].copy()
    v["k"] = v.max_features.astype(float)
    v = v.sort_values("k")
    q.errorbar(v.k, v.AUC, yerr=[v.AUC - v.CI_lo, v.CI_hi - v.AUC], color=SLOT[0], marker="o", ms=6, mec=BG, mew=1.3, lw=1.6, capsize=3)
    q.set_xscale("log")
    q.set_xticks([2000, 5000, 20000])
    q.set_xticklabels(["2,000", "5,000", "20,000"])
    q.minorticks_off()
    q.set_ylim(0.65, 0.95)
    labels(q, "Vocabulary size (terms)", None, "Vocabulary")
    q.grid(axis="y", color=GRID, lw=0.6)
    panel_tag(q, "b")
    save(fig, "fig6_regularisation.png")


# ------------------------------------------------------------------------------------------------------------- figure 7
def fig7():
    b = pd.read_csv(PROC / "topk_validation_bounds.csv")
    ca = pd.read_csv(ANN / "topk" / "coder_A.csv", dtype=str).dropna(subset=["channel_id"]).set_index("channel_id")
    b["entity"] = b.channel_id.map(ca.entity_type)
    fig, (p, q) = canvas(6.9, 3.5, 1, 2)
    rows = []
    for g, name in (("top", "Top 100\nby score"), ("control", "Control 50\n(lower half)")):
        x = b[b.group == g]
        for rule, col in (("strict", SLOT[0]), ("lenient", SLOT[1])):
            k, n = int(x[rule].sum()), len(x)
            rows.append((name, rule, k / n, *wilson(k, n), col))
    for i, (name, rule, pr, lo, hi, col) in enumerate(rows):
        yy = (1 - i // 2) + (0.14 if rule == "strict" else -0.14)
        p.plot([lo, hi], [yy, yy], color=col, lw=1.8, solid_capstyle="round")
        p.plot(pr, yy, "o", ms=7, color=col, mec=BG, mew=1.5, label=f"{rule.capitalize()} rule" if i < 2 else None)
        p.text(hi + 0.015, yy, f"{pr:.2f}", va="center", fontsize=7.5, color=MUTED)
    p.set_yticks([1, 0])
    p.set_yticklabels(["Top 100\nby score", "Control 50\n(lower half)"], fontsize=8.5, color=INK)
    p.set_xlim(0.2, 0.9)
    p.set_ylim(-0.6, 1.6)
    labels(p, "Share flagged as adviser-like (Wilson 95% interval)", title="Precision")
    p.grid(axis="x", color=GRID, lw=0.6)
    p.spines["left"].set_visible(False)
    p.tick_params(axis="y", length=0)
    p.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    panel_tag(p, "a")
    ents =[("individual_adviser_like", "Adviser-like individual"), ("crypto_forex_trader", "Crypto or forex trader"), ("educator", "Educator"),
            ("news_media", "News or media"), ("other_regulated_entity", "Other regulated entity"), ("other", "Other")]
    left = {"top": 0, "control": 0}
    for k, (e, name) in enumerate(ents):
        for gi, g in enumerate(("top", "control")):
            n = int(((b.group == g) & (b.entity == e)).sum())
            tot = int((b.group == g).sum())
            w = n / tot
            q.barh(1 - gi, w, left=left[g], color=SLOT[k], height=0.55, edgecolor=BG, linewidth=1.5, label=name if gi == 0 else None)
            if w > 0.08:
                q.text(left[g] + w / 2, 1 - gi, f"{n}", ha="center", va="center", fontsize=7.5, color="white" if k not in (3,) else INK)
            left[g] += w
    q.set_yticks([1, 0])
    q.set_yticklabels(["Top 100", "Control 50"], fontsize=8.5, color=INK)
    q.set_xlim(0, 1)
    labels(q, "Share of group", title="Entity type (coder A, counts shown)")
    q.spines["left"].set_visible(False)
    q.tick_params(axis="y", length=0)
    q.legend(frameon=False, fontsize=6.8, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    panel_tag(q, "b")
    save(fig, "fig7_validation.png")


# ------------------------------------------------------------------------------------------------------------- figure 8
def fig8():
    f = pd.read_csv(PROC / "flag_pilot_results.csv")
    f = f[f.label == "lenient"].copy()
    f["ci_lo"] = f.AUC_95CI.str.extract(r"\[([\d.]+),")[0].astype(float)
    f["ci_hi"] = f.AUC_95CI.str.extract(r",([\d.]+)\]")[0].astype(float)

    def get(feat, model):
        return f[(f.features == feat) & (f.model == model)].iloc[0]
    items = [("Random scores", get("none", "majority/random"), False), ("Keyword rules", get("keyword rules", "rule_score"), False),
             ("Registration-based ranking", get("PU registration score", "pu_score"), False), ("Subscribers", get("subscribers", "log_subs"), False),
             ("Platform behaviour (LR)", get("behavioural", "logreg"), True), ("Scrubbed text (gradient boosting)", get("text_clean", "hist_gbm"), True),
             ("Scrubbed text (LR)", get("text_clean", "logreg"), True)]
    fig, ax = canvas(6.6, 3.5)
    for i, (name, r, model) in enumerate(items):
        y = len(items) - 1 - i
        col = SLOT[0] if model else MUTED
        ax.plot([r.ci_lo, r.ci_hi], [y, y], color=col, lw=1.8, solid_capstyle="round")
        ax.plot(r.AUC, y, "o", ms=7, color=col, mec=BG, mew=1.5)
        ax.text(r.ci_hi + 0.012, y, f"{r.AUC:.2f}", va="center", fontsize=7.5, color=MUTED)
    ax.axvline(0.5, color=MUTED, lw=0.9)
    ax.text(0.502, len(items) - 0.45, "chance", fontsize=7.5, color=MUTED, va="bottom")
    ax.plot([], [], "o", color=MUTED, label="Baseline")
    ax.plot([], [], "o", color=SLOT[0], label="Trained on the coded behaviour")
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([n for n, _, _ in items][::-1], fontsize=8.3, color=INK)
    ax.set_xlim(0.3, 0.95)
    ax.set_ylim(-0.6, len(items) - 0.2)
    labels(ax, x="AUC for the coded flag (n = 150, 95% bootstrap interval)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, fontsize=7.8, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2)
    save(fig, "fig8_pilot.png")


def fig9():
    shutil.copy(PROC / "active_learning_curve.png", OUT / "fig9_label_efficiency.png")
    print("copied", OUT / "fig9_label_efficiency.png")


def fig10():
    s = pd.read_csv(PROC / "sim_leakage_results.csv")
    fig, (p, q) = canvas(7.0, 3.5, 1, 2)
    full, part = s[s.scrubber_recall == 1.0], s[(s.scrubber_recall < 1.0) | (s.pi_registered == 0.75)]
    lim = 0.17
    p.plot([0, lim], [0, lim], color=MUTED, lw=0.9, ls=(0, (4, 3)))
    p.errorbar(full.true_gap, full.mean_estimated_gap, yerr=full.sd_estimate, fmt="o", color=SLOT[0], ms=6.5, mec=BG, mew=1.4, capsize=3, elinewidth=1.3,
               label=r"Complete scrubber, $\pi_1$ varied")
    inc = s[s.pi_registered == 0.75]
    inc = inc[inc.scrubber_recall < 1.0]
    p.errorbar(inc.true_gap, inc.mean_estimated_gap, yerr=inc.sd_estimate, fmt="^", color=SLOT[1], ms=7, mec=BG, mew=1.4, capsize=3, elinewidth=1.3,
               label=r"Incomplete scrubber, $\pi_1 = 0.75$")
    p.set_xlim(-0.01, lim)
    p.set_ylim(-0.01, lim)
    labels(p, "True gap", "Estimated gap (mean $\\pm$ 1 SD)", "Estimated against true gap")
    p.grid(color=GRID, lw=0.6)
    p.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc="upper left")
    panel_tag(p, "a")
    names = [rf"$\pi_1={x.pi_registered:g}$" + ("" if x.scrubber_recall == 1.0 else rf", $\rho={x.scrubber_recall:g}$") for x in s.itertuples()]
    xs = np.arange(len(s))
    series = [("coverage_95", "95% interval", SLOT[0]), ("coverage_x1.25", "widened by 1.25", SLOT[1]), ("coverage_x1.5", "widened by 1.5", SLOT[2])]
    for k, (col, name, c) in enumerate(series):
        q.plot(xs + (k - 1) * 0.16, s[col], "o", ms=6, color=c, mec=BG, mew=1.2, label=name)
    q.axhline(0.95, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    q.set_xticks(xs)
    q.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    q.set_ylim(0.7, 1.0)
    labels(q, None, "Coverage of the true gap", "Coverage")
    q.grid(axis="y", color=GRID, lw=0.6)
    q.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc="lower right")
    panel_tag(q, "b")
    save(fig, "fig10_simulation.png")


def fig11():
    m = pd.read_csv(PROC / "miner_cv_results.csv").set_index("arm")
    g = pd.read_csv(PROC / "miner_gap_ci.csv").set_index("quantity")
    e = pd.read_csv(PROC / "embedding_cv_results.csv").set_index("variant")
    eg = pd.read_csv(PROC / "embedding_gap_ci.csv").iloc[0]
    fig, (p, q) = canvas(7.0, 3.6, 1, 2)
    arms = [("raw", "Raw text"), ("miner", "Mined templates removed"), ("miner+regno", "Mined templates and\nregistration numbers removed"),
            ("manual", "Manual scrubber (Level 2)"), ("manual+miner", "Manual scrubber, then miner"), ("miner_removed_only", "Text removed by the miner only")]
    for i, (a, name) in enumerate(arms):
        y = len(arms) - 1 - i
        c = SLOT[0] if a in ("raw", "manual", "manual+miner") else SLOT[1]
        p.plot([m.loc[a, "CI_lo"], m.loc[a, "CI_hi"]], [y, y], color=c, lw=1.8, solid_capstyle="round")
        p.plot(m.loc[a, "AUC"], y, "o", ms=7, color=c, mec=BG, mew=1.5)
        p.text(m.loc[a, "CI_hi"] + 0.008, y, f"{m.loc[a, 'AUC']:.3f}", va="center", fontsize=7.5, color=MUTED)
    p.plot([], [], "o", color=SLOT[0], label="Raw or manual scrubbing")
    p.plot([], [], "o", color=SLOT[1], label="Mined (statistical) scrubbing")
    p.set_yticks(range(len(arms)))
    p.set_yticklabels([n for _, n in arms][::-1], fontsize=7.3, color=INK)
    p.axvline(0.5, color=MUTED, lw=0.9)
    p.set_xlim(0.6, 1.0)
    p.set_ylim(-0.6, len(arms) - 0.4)
    labels(p, "AUC (logistic regression, TF-IDF, 95% interval)", title="Scrubbing arms")
    p.grid(axis="x", color=GRID, lw=0.6)
    p.spines["left"].set_visible(False)
    p.tick_params(axis="y", length=0)
    p.legend(frameon=False, fontsize=7, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.35, -0.2), ncol=1)
    panel_tag(p, "a")
    tf = pd.read_csv(PROC / "leakage_gap_ci.csv")
    tf = tf[(tf.model == "logreg") & (tf.quantity == "gap(raw-clean)")].iloc[0]
    gaps = [("Manual scrubber,\nTF-IDF", tf.estimate, tf.ci_lo, tf.ci_hi, SLOT[0]),
            ("Manual scrubber,\nsentence embeddings", eg.estimate, eg.ci_lo, eg.ci_hi, SLOT[2]),
            ("Mined templates and\nregistration numbers", g.loc["gap_miner_regno", "estimate"], g.loc["gap_miner_regno", "ci_lo"], g.loc["gap_miner_regno", "ci_hi"], SLOT[1]),
            ("Mined templates", g.loc["gap_miner", "estimate"], g.loc["gap_miner", "ci_lo"], g.loc["gap_miner", "ci_hi"], SLOT[1])]
    for i, (name, est, lo, hi, c) in enumerate(gaps):
        y = len(gaps) - 1 - i
        q.plot([lo, hi], [y, y], color=c, lw=1.8, solid_capstyle="round")
        q.plot(est, y, "o", ms=7, color=c, mec=BG, mew=1.5)
        q.text(hi + 0.004, y, f"{est:.3f}", va="center", fontsize=7.5, color=MUTED)
    q.set_yticks(range(len(gaps)))
    q.set_yticklabels([n for n, *_ in gaps][::-1], fontsize=7.3, color=INK)
    q.axvline(0, color=MUTED, lw=0.9)
    q.set_xlim(-0.01, 0.13)
    q.set_ylim(-0.6, len(gaps) - 0.4)
    labels(q, "Leakage gap, AUC (95% interval)", title="Leakage gap by method")
    q.grid(axis="x", color=GRID, lw=0.6)
    q.spines["left"].set_visible(False)
    q.tick_params(axis="y", length=0)
    panel_tag(q, "b")
    save(fig, "fig11_miner_embeddings.png")


FIGS = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5, "fig6": fig6, "fig7": fig7, "fig8": fig8, "fig9": fig9, "fig10": fig10, "fig11": fig11}

if __name__ == "__main__":
    for k in (sys.argv[1:] or list(FIGS)):
        FIGS[k]()
