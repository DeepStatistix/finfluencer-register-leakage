"""Stage 5 - positive-unlabelled (PU) screening: which UNREGISTERED accounts look like registered advisers?

    python eval_pu.py --repeats 5 --folds 5

Setup
  P = accounts on the SEBI register (known positives).  U = random sample of unregistered accounts
  (UNLABELLED: mostly educators/news, some unregistered advisers).  We train P-vs-U, score every account
  out-of-fold, and read the ranking of U: high-scoring unregistered accounts resemble registered advisers.
Why this is honest
  * every U account is scored by models that never saw it (out-of-fold), otherwise the model just memorises U;
  * AUC(P vs U) is a LOWER bound on separability because U contains hidden positives (reported as such);
  * leakage control: registration claims/numbers and regulatory boilerplate (common.scrub_regulatory) are removed
    from text, and an UNSCRUBBED run is reported as a leakage diagnostic;
  * baselines: subscribers only, a keyword-rule score, random.
Outputs: processed/pu_results.csv (metrics), processed/pu_scores.csv (out-of-fold score per account).
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from common import PROC, RAW, extract_regulatory, read_jsonl, rule_flags, scrub_regulatory

BEHAV = ["log_subs", "log_views_per_sub", "log_video_count", "channel_age_days", "uploads_per_month",
         "like_rate", "comment_rate", "median_log_views", "mean_title_len", "share_caps_titles"]
COMMENT_CAP, FLAG_RATE = 320, 0.20


def build_frame(seed=42):
    pool = pd.read_csv(PROC / "analysis_pool.csv")
    ch = pd.DataFrame(read_jsonl(RAW / "channels.jsonl")).set_index("channel_id")
    vids = pd.DataFrame(read_jsonl(RAW / "videos.jsonl"))
    vids = vids[vids.channel_id.isin(pool.channel_id)].copy()
    vids["pub"] = pd.to_datetime(vids.published_at, utc=True, format="ISO8601")
    vids["caps"] = vids.title.fillna("").map(lambda t: sum(c.isupper() for c in t) / max(len(t), 1) > 0.5)
    com = pd.DataFrame(read_jsonl(RAW / "comments.jsonl"))
    com = com[com.channel_id.isin(pool.channel_id) & ~com.by_channel_owner.fillna(False)]
    com = com.sample(frac=1, random_state=seed)  # shuffle, then keep at most COMMENT_CAP per account
    com = com[com.groupby("channel_id").cumcount() < COMMENT_CAP]

    g = vids.groupby("channel_id")
    span = (g.pub.max() - g.pub.min()).dt.days.clip(lower=30) / 30
    f = pd.DataFrame({
        "uploads_per_month": g.size() / span,
        "like_rate": g.likes.sum() / g.views.sum().clip(lower=1),
        "comment_rate": g.comments.sum() / g.views.sum().clip(lower=1),
        "median_log_views": g.views.median().map(np.log1p),
        "mean_title_len": g.title.apply(lambda s: s.fillna("").str.len().mean()),
        "share_caps_titles": g.caps.mean(),
        "text_raw": g.apply(lambda d: " ".join((d.title.fillna("") + " " + d.description.fillna("")).tolist()), include_groups=False),
    })
    f["comments_raw"] = com.groupby("channel_id").text.apply(lambda s: " ".join(s.dropna())).reindex(f.index).fillna("")
    f["chan_raw"] = (ch.title.fillna("") + " " + ch.description.fillna("")).reindex(f.index)
    f["log_subs"] = ch.subscribers.reindex(f.index).fillna(0).map(np.log1p)
    f["log_views_per_sub"] = (ch.views / ch.subscribers.clip(lower=1)).reindex(f.index).map(np.log1p)
    f["log_video_count"] = ch.video_count.reindex(f.index).map(np.log1p)
    f["channel_age_days"] = (pd.Timestamp.now(tz="UTC") - pd.to_datetime(ch.published_at, utc=True, format="ISO8601")).dt.days.reindex(f.index)
    # leakage-safe text: boilerplate sentences removed; leaky text: everything
    f["text_clean"] = (f.chan_raw + " " + f.text_raw).map(scrub_regulatory) + " " + f.comments_raw.map(scrub_regulatory)
    f["text_leaky"] = f.chan_raw + " " + f.text_raw + " " + f.comments_raw
    # the wording the scrubber removes, analysed on its own (leakage-channel decomposition)
    f["text_removed"] = (f.chan_raw + " " + f.text_raw).map(extract_regulatory) + " " + f.comments_raw.map(extract_regulatory)
    # keyword-rule baseline score: no. of videos with guarantee / trade call / paid / performance flags
    flags = vids.assign(t=vids.title.fillna("") + " " + vids.description.fillna("")).t.map(rule_flags)
    f["rule_score"] = pd.DataFrame(list(flags), index=vids.index).drop(columns=["disclaimer", "claims_registration"]).gt(0).sum(axis=1).groupby(vids.channel_id).mean()
    df = pool.merge(f, left_on="channel_id", right_index=True, how="inner").reset_index(drop=True)
    df["y"] = (df.reg_status == "registered").astype(int)
    return df


def make_models(feature_set, behav=None):
    behav = behav or BEHAV
    text_col = {"text_clean": "text_clean", "text_leaky": "text_leaky", "combined_clean": "text_clean", "text_removed": "text_removed"}.get(feature_set)

    def prep(max_feat):
        parts = []
        if feature_set in ("behavioural", "combined_clean"):
            parts.append(("num", StandardScaler(), behav))
        if text_col:
            parts.append(("txt", TfidfVectorizer(min_df=5, max_features=max_feat, ngram_range=(1, 2), sublinear_tf=True,
                                                 strip_accents="unicode"), text_col))
        return ColumnTransformer(parts)

    def dense():
        return Pipeline([("p", prep(600)), ("d", FunctionTransformer(lambda X: X.toarray() if hasattr(X, "toarray") else X))])

    return {
        "logreg": Pipeline([("p", prep(20000)), ("m", LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5))]),
        "random_forest": Pipeline([("p", dense()), ("m", RandomForestClassifier(
            300, class_weight="balanced_subsample", min_samples_leaf=2, n_jobs=-1, random_state=0))]),
        "hist_gbm": Pipeline([("p", dense()), ("m", HistGradientBoostingClassifier(
            class_weight="balanced", max_depth=4, learning_rate=0.05, max_iter=200, random_state=0))]),
    }


def recall_at_flag_rate(y, s, rate=FLAG_RATE):
    cut = np.quantile(s, 1 - rate)
    return float((s[y == 1] >= cut).mean())


def boot(fn, y, s, n=1000, seed=0):
    rng, v = np.random.default_rng(seed), []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            v.append(fn(y[i], s[i]))
    return np.percentile(v, [2.5, 97.5])


def oof_scores(model, df, X_cols, folds, repeats):
    """Mean out-of-fold P(positive) per account across repeats, plus per-repeat score vectors."""
    y = df.y.to_numpy()
    rkf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=0)
    per = [np.full(len(df), np.nan) for _ in range(repeats)]
    for k, (tr, te) in enumerate(rkf.split(df, y)):
        m = clone(model).fit(df.iloc[tr][X_cols], y[tr])
        per[k // folds][te] = m.predict_proba(df.iloc[te][X_cols])[:, 1]
    return per


def main(a):
    df = build_frame()
    y = df.y.to_numpy()
    print(f"accounts: {len(df)}  registered (P): {y.sum()}  unregistered sample (U): {(1 - y).sum()}")
    cols = ["text_clean", "text_leaky", "rule_score"] + BEHAV
    rows, best = [], {}

    def add(name, fs, per):
        aucs = [roc_auc_score(y, s) for s in per]
        pooled = np.mean(per, axis=0)
        lo, hi = boot(roc_auc_score, y, pooled)
        rows.append({"features": fs, "model": name, "AUC_P_vs_U": np.mean(aucs), "AUC_95CI(bootstrap)": f"[{lo:.2f},{hi:.2f}]",
                     "avg_precision": np.mean([average_precision_score(y, s) for s in per]),
                     "positive_recall@top20%": np.mean([recall_at_flag_rate(y, s) for s in per])})
        best[(fs, name)] = pooled

    # baselines (no fitting): random, subscribers only, keyword-rule score
    rng = np.random.default_rng(0)
    add("random", "none", [rng.random(len(y)) for _ in range(a.repeats)])
    add("subscribers_only", "log_subs", [df.log_subs.to_numpy()] * a.repeats)
    add("keyword_rule_score", "rules", [df.rule_score.to_numpy()] * a.repeats)

    for fs in ("behavioural", "text_clean", "combined_clean", "text_leaky"):
        for name, model in make_models(fs).items():
            add(name, fs, oof_scores(model, df, cols, a.folds, a.repeats))
            print(f"  done {fs:15s} {name}")
    # behavioural without audience-size features: is it only 'big channel' vs 'small channel'?
    no_size = [c for c in BEHAV if c not in ("log_subs", "log_views_per_sub", "median_log_views")]
    for name, model in make_models("behavioural", no_size).items():
        add(name, "behavioural_no_size", oof_scores(model, df, cols, a.folds, a.repeats))

    res = pd.DataFrame(rows).round(3)
    res.to_csv(PROC / "pu_results.csv", index=False)
    print("\n", res.to_string(index=False))

    # final ranking = best leakage-safe model by AUC; Elkan-Noto style estimate of hidden positives in U
    safe = res[res.features.isin(["behavioural", "text_clean", "combined_clean"])].sort_values("AUC_P_vs_U", ascending=False)
    top = safe.iloc[0]
    s = best[(top.features, top.model)]
    c = float(np.mean(s[y == 1]))  # P(labelled | positive) under SCAR, estimated from held-out positives
    hidden = float(np.minimum(1, s[y == 0] / c).sum())
    print(f"\nbest leakage-safe model: {top.model} on {top.features} (AUC {top.AUC_P_vs_U})")
    print(f"Elkan-Noto estimate of positive-like accounts inside U: {hidden:.0f} of {(y == 0).sum()} "
          f"({hidden / (y == 0).sum():.0%}); assumes labelled positives are a random draw of all positives (SCAR) - likely violated, report as indicative only")
    out = df[["channel_id", "reg_status"]].assign(score=s, model=f"{top.model}/{top.features}")
    out["rank_in_unregistered"] = out.loc[out.reg_status == "unregistered", "score"].rank(ascending=False)
    out.to_csv(PROC / "pu_scores.csv", index=False)
    print("wrote pu_results.csv, pu_scores.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    main(ap.parse_args())
