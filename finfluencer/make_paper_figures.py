"""Figures for the manuscript, drawn from processed/pu_results.csv (no numbers typed by hand).
    python make_paper_figures.py      -> paper/finfluencer/fig1_leakage_gap.png
"""
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import PROC, ROOT

OUT = ROOT.parent / "paper" / "finfluencer"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ["logreg", "random_forest", "hist_gbm"]
LABEL = {"logreg": "Logistic regression", "random_forest": "Random forest", "hist_gbm": "Gradient boosting"}
COLOR = {"logreg": "#2a78d6", "random_forest": "#eb6834", "hist_gbm": "#1baf7a"}   # palette slots 1-3, fixed order
ROWS = [("behavioural", "Platform behaviour only"), ("text_clean", "Text, regulatory wording removed"),
        ("combined_clean", "Behaviour + text, wording removed"), ("text_leaky", "Text, regulatory wording kept")]

r = pd.read_csv(PROC / "nlp_oof_summary.csv")      # leakage-gap run: one protocol for all four feature sets
ROWS = [("behavioural", "Platform behaviour only"), ("text_clean", "Text, regulatory wording removed"),
        ("text_leaky", "Text, regulatory wording kept"), ("text_removed", "Removed regulatory wording only")]
bg, ink, muted = "#fcfcfb", "#0b0b0b", "#52514e"
fig, ax = plt.subplots(figsize=(6.8, 3.9), dpi=220, facecolor=bg)
ax.set_facecolor(bg)
for i, (fs, name) in enumerate(ROWS):
    for j, m in enumerate(MODELS):
        row = r[(r.features == fs) & (r.model == m)].iloc[0]
        lo, hi = row.AUC_ci_lo, row.AUC_ci_hi
        y = len(ROWS) - 1 - i + (1 - j) * 0.2
        ax.plot([lo, hi], [y, y], color=COLOR[m], lw=1.6, solid_capstyle="round")
        ax.plot(row.AUC_mean_over_repeats, y, "o", ms=6.5, color=COLOR[m], mec=bg, mew=1.5, label=LABEL[m] if i == 0 else None)
ax.axvline(0.5, color=muted, lw=0.9)
ax.text(0.502, len(ROWS) - 0.45, "chance", fontsize=7.5, color=muted, va="bottom")
ax.set_yticks(range(len(ROWS)))
ax.set_yticklabels([n for _, n in ROWS][::-1], fontsize=8.5, color=ink)
ax.set_xlim(0.45, 1.0)
ax.set_ylim(-0.6, len(ROWS) - 0.3)
ax.set_xlabel("AUC, registered vs unregistered accounts (out-of-fold, 95% bootstrap interval)", fontsize=8.5, color=ink)
ax.grid(axis="x", color="#e4e3df", lw=0.6)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#bdbcb6")
ax.tick_params(axis="x", colors=muted, labelsize=8)
ax.tick_params(axis="y", length=0)
ax.legend(loc="upper right", frameon=False, fontsize=8, labelcolor=ink)
fig.tight_layout()
fig.savefig(OUT / "fig1_leakage_gap.png", facecolor=bg)
print("wrote", OUT / "fig1_leakage_gap.png")
