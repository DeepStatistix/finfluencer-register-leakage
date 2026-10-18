"""Stage 7 (PILOT) - supervised detection of adviser-like behaviour among UNREGISTERED accounts, trained on the human flag.

    python eval_flag.py --repeats 10 --folds 5

Target  flag = (gives_specific_recommendations OR sells_tips_or_advisory) AND entity_type != other_regulated_entity,
        coded blind by two coders (topk_validation.py). Primary label = lenient (either coder), sensitivity = strict (both).
Sample  150 coded accounts: 100 top-scored + 50 control from the unregistered pool. NOT a random sample of unregistered
        accounts, so prevalence and absolute metrics describe this sample only. Treat as a PILOT with wide intervals.
Design  account-level repeated stratified CV; TF-IDF/scaling fitted inside folds; regulatory boilerplate scrubbed from
        text; class weights for balance; baselines = majority, keyword rules, the PU registration score (already shown
        to carry no signal); bootstrap CIs over accounts. The keyword-rule AUC here is the proper validation of the rules.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

import eval_pu as E
from common import PROC


def frame(label):
    df = E.build_frame()
    lab = pd.read_csv(PROC / "topk_validation_bounds.csv")[["channel_id", "group", "score", "strict", "lenient"]]
    df = df.merge(lab, on="channel_id", how="inner")
    df["y"] = df[label].astype(int)
    return df


def models(feature_set):
    def prep(max_feat):
        parts = []
        if feature_set in ("behavioural", "combined_clean"):
            parts.append(("num", StandardScaler(), E.BEHAV))
        if feature_set in ("text_clean", "combined_clean"):
            parts.append(("txt", TfidfVectorizer(min_df=3, max_features=max_feat, ngram_range=(1, 2), sublinear_tf=True,
                                                 strip_accents="unicode"), "text_clean"))
        return ColumnTransformer(parts)
    dense = Pipeline([("p", prep(500)), ("d", FunctionTransformer(lambda X: X.toarray() if hasattr(X, "toarray") else X))])
    return {"logreg": Pipeline([("p", prep(5000)), ("m", LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5))]),
            "hist_gbm": Pipeline([("p", dense), ("m", HistGradientBoostingClassifier(
                class_weight="balanced", max_depth=3, learning_rate=0.05, max_iter=150, random_state=0))])}


def ci(y, s, fn, n=2000, seed=0):
    rng, v = np.random.default_rng(seed), []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            v.append(fn(y[i], s[i]))
    return np.percentile(v, [2.5, 97.5])


def run(label, a):
    df = frame(label)
    y = df.y.to_numpy()
    print(f"\n=== label = {label}: n={len(df)}, flagged={y.sum()} ({y.mean():.0%}) ===")
    cols = ["text_clean"] + E.BEHAV
    rows = []

    def add(feat, name, per):
        pooled = np.mean(per, axis=0)
        lo, hi = ci(y, pooled, roc_auc_score)
        rows.append({"label": label, "features": feat, "model": name, "AUC": np.mean([roc_auc_score(y, s) for s in per]),
                     "AUC_95CI": f"[{lo:.2f},{hi:.2f}]", "avg_precision": np.mean([average_precision_score(y, s) for s in per]),
                     "prevalence": y.mean(), "precision@top25%": np.mean([y[np.argsort(-s)[:len(y) // 4]].mean() for s in per])})

    add("none", "majority/random", [np.random.default_rng(i).random(len(y)) for i in range(a.repeats)])
    add("keyword rules", "rule_score", [df.rule_score.to_numpy()] * a.repeats)
    add("PU registration score", "pu_score", [df.score.to_numpy()] * a.repeats)
    add("subscribers", "log_subs", [df.log_subs.to_numpy()] * a.repeats)
    for fs in ("behavioural", "text_clean", "combined_clean"):
        for name, model in models(fs).items():
            rkf = RepeatedStratifiedKFold(n_splits=a.folds, n_repeats=a.repeats, random_state=0)
            per = [np.full(len(df), np.nan) for _ in range(a.repeats)]
            for k, (tr, te) in enumerate(rkf.split(df, y)):
                m = clone(model).fit(df.iloc[tr][cols], y[tr])
                per[k // a.folds][te] = m.predict_proba(df.iloc[te][cols])[:, 1]
            add(fs, name, per)
            print("  done", fs, name, flush=True)
    res = pd.DataFrame(rows).round(3)
    print(res.to_string(index=False))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=10)
    a = ap.parse_args()
    out = pd.concat([run("lenient", a), run("strict", a)])
    out.to_csv(PROC / "flag_pilot_results.csv", index=False)
    print("wrote flag_pilot_results.csv")
