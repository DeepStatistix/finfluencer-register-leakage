"""Stage 10 - technical NLP/ML experiments reported in the manuscript.

    python nlp_experiments.py stats       # corpus statistics by class (script mix, length, vocabulary)
    python nlp_experiments.py gap         # leakage gap with paired bootstrap CI, leakage-channel decomposition, Brier scores
    python nlp_experiments.py scrub       # AUC at four scrubbing intensities
    python nlp_experiments.py ablation    # representation (word / character / both), classifier, C and vocabulary sensitivity
    python nlp_experiments.py tune        # nested cross-validation of C and vocabulary size (tuned leakage gap)
    python nlp_experiments.py terms       # highest-weight terms of the scrubbed logistic regression

All experiments use the 591-account pool (141 registered, 450 unregistered) and repeated stratified 5-fold cross-validation.
Vectorisers are fitted inside each training fold. TF-IDF matrices are computed once per fold and shared by the classifiers
compared on them; folds are processed in parallel (joblib). Outputs are written to data/finfluencer/processed/.
"""
import argparse
import re

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import eval_pu as E
from common import PROC

REPEATS, FOLDS, N_JOBS = 3, 5, 6
DEVANAGARI, LATIN = re.compile(r"[ऀ-ॿ]"), re.compile(r"[A-Za-z]")
OTHER_INDIC = re.compile(r"[ঀ-෿]")   # Bengali, Gurmukhi, Gujarati, Odia, Tamil, Telugu, Kannada, Malayalam, Sinhala
WORD = re.compile(r"\w+", re.UNICODE)


# ----------------------------------------------------------------------------------------------- helpers
def splits(y, repeats=REPEATS, folds=FOLDS):
    rkf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=0)
    return [(k // folds, tr, te) for k, (tr, te) in enumerate(rkf.split(np.zeros(len(y)), y))]


def vec_word(max_feat=20000):
    return TfidfVectorizer(min_df=5, ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode", max_features=max_feat)


def vec_char():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=5, sublinear_tf=True, max_features=100000)


def _vectorise(kind, tr_texts, te_texts, max_feat=20000):
    v = vec_word(max_feat) if kind == "word" else vec_char()
    return v.fit_transform(tr_texts), v.transform(te_texts)


def featurise(texts, sp, kind, max_feat=20000):
    """TF-IDF matrices for every fold, computed in parallel; texts is a list of strings aligned with the data frame."""
    texts = np.asarray(texts, dtype=object)
    return Parallel(n_jobs=N_JOBS)(delayed(_vectorise)(kind, texts[tr].tolist(), texts[te].tolist(), max_feat) for _, tr, te in sp)


def top_columns_by_df(Xtr, k):
    return np.argsort(-np.asarray((Xtr > 0).sum(axis=0)).ravel(), kind="stable")[:k]


def collect(sp, n, fold_scores):
    """fold_scores: list of score vectors aligned with sp; returns one out-of-fold score vector per repeat."""
    per = {}
    for (r, tr, te), s in zip(sp, fold_scores):
        per.setdefault(r, np.full(n, np.nan))[te] = s
    return [per[r] for r in sorted(per)]


def boot_auc(y, s, n=2000, seed=0):
    rng, v = np.random.default_rng(seed), []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            v.append(roc_auc_score(y[i], s[i]))
    return np.percentile(v, [2.5, 97.5])


def rank_pool(per):
    return np.mean([pd.Series(s).rank().to_numpy() for s in per], axis=0)   # margins and probabilities are not comparable


def score_of(m, X):
    return m.decision_function(X) if hasattr(m, "decision_function") else m.predict_proba(X)[:, 1]


# ----------------------------------------------------------------------------------------------- stats
def stats(_):
    df = E.build_frame()
    rows = []
    for name, g in df.groupby("reg_status"):
        t = g.text_clean
        chars = t.str.len()
        dev = t.map(lambda s: len(DEVANAGARI.findall(s)) / max(len(s), 1))
        lat = t.map(lambda s: len(LATIN.findall(s)) / max(len(s), 1))
        oth = t.map(lambda s: len(OTHER_INDIC.findall(s)) / max(len(s), 1))
        toks = t.map(lambda s: len(WORD.findall(s)))
        vocab = set(w.lower() for s in t for w in WORD.findall(s))
        rows.append({"class": name, "accounts": len(g), "median_chars": chars.median(), "IQR_chars": f"{chars.quantile(.25):.0f}-{chars.quantile(.75):.0f}",
                     "median_tokens": toks.median(), "vocabulary_types": len(vocab), "mean_share_Latin": lat.mean(),
                     "mean_share_Devanagari": dev.mean(), "mean_share_other_Indic": oth.mean(),
                     "accounts_>10%_Devanagari": float((dev > .10).mean()), "accounts_>10%_other_Indic": float((oth > .10).mean()),
                     "median_comments_chars": g.comments_raw.str.len().median()})
    out = pd.DataFrame(rows).round(3)
    out.to_csv(PROC / "corpus_stats.csv", index=False)
    print(out.T.to_string())


# ----------------------------------------------------------------------------------------------- leakage gap
def boot_pair(y, sc, fns, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    out = {k: [] for k in fns}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            for k, fn in fns.items():
                out[k].append(fn(y[i], {m: s[i] for m, s in sc.items()}))
    return {k: np.percentile(v, [2.5, 97.5]) for k, v in out.items()}


def gap(a):
    df = E.build_frame()
    y = df.y.to_numpy()
    n = len(y)
    sp = splits(y)
    names = ["logreg", "random_forest", "hist_gbm"]
    pooled, rows, per_all = {}, [], {}

    def fit_models(Xtr, ytr, Xte, dense_tr, dense_te):
        out = {}
        lr = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5).fit(Xtr, ytr)
        out["logreg"] = lr.predict_proba(Xte)[:, 1]
        rf = RandomForestClassifier(300, class_weight="balanced_subsample", min_samples_leaf=2, n_jobs=-1, random_state=0).fit(dense_tr, ytr)
        out["random_forest"] = rf.predict_proba(dense_te)[:, 1]
        gb = HistGradientBoostingClassifier(class_weight="balanced", max_depth=4, learning_rate=0.05, max_iter=200, random_state=0).fit(dense_tr, ytr)
        out["hist_gbm"] = gb.predict_proba(dense_te)[:, 1]
        return out

    for fs in ("text_clean", "text_leaky", "text_removed", "behavioural"):
        if fs == "behavioural":
            mats = None
        else:
            mats = featurise(df[fs].tolist(), sp, "word")
        fold_scores = {m: [] for m in names}
        for i, (r, tr, te) in enumerate(sp):
            if mats is None:
                sc = StandardScaler().fit(df.iloc[tr][E.BEHAV])
                Xtr, Xte = sc.transform(df.iloc[tr][E.BEHAV]), sc.transform(df.iloc[te][E.BEHAV])
                res = fit_models(Xtr, y[tr], Xte, Xtr, Xte)
            else:
                Xtr, Xte = mats[i]
                cols = top_columns_by_df(Xtr, 600)           # trees use the 600 most frequent terms (dense matrix)
                res = fit_models(Xtr, y[tr], Xte, Xtr[:, cols].toarray(), Xte[:, cols].toarray())
            for m in names:
                fold_scores[m].append(res[m])
        for m in names:
            per = collect(sp, n, fold_scores[m])
            per_all[(fs, m)] = per
            pooled[(fs, m)] = np.mean(per, axis=0)
            lo, hi = boot_auc(y, pooled[(fs, m)])
            rows.append({"features": fs, "model": m, "AUC_ci_lo": lo, "AUC_ci_hi": hi,
                         "AUC_mean_over_repeats": np.mean([roc_auc_score(y, s) for s in per]),
                         "AUC_sd_over_repeats": np.std([roc_auc_score(y, s) for s in per]),
                         "AUC_pooled": roc_auc_score(y, pooled[(fs, m)]), "Brier_pooled": brier_score_loss(y, np.clip(pooled[(fs, m)], 0, 1))})
            print("done", fs, m, flush=True)
    res = pd.DataFrame(rows).round(4)
    res.to_csv(PROC / "nlp_oof_summary.csv", index=False)
    oof = pd.DataFrame({"channel_id": df.channel_id, "registered": y})
    for (fs, m), s in pooled.items():
        oof[f"{fs}|{m}"] = s
    oof.to_csv(PROC / "oof_scores.csv", index=False)      # repeat-averaged out-of-fold scores, for ROC / PR curves
    out = []
    for m in names:
        sc = {"raw": pooled[("text_leaky", m)], "clean": pooled[("text_clean", m)], "removed": pooled[("text_removed", m)], "beh": pooled[("behavioural", m)]}
        fns = {"gap(raw-clean)": lambda yy, s: roc_auc_score(yy, s["raw"]) - roc_auc_score(yy, s["clean"]),
               "clean-behaviour": lambda yy, s: roc_auc_score(yy, s["clean"]) - roc_auc_score(yy, s["beh"])}
        ci = boot_pair(y, sc, fns)
        point = {"gap(raw-clean)": roc_auc_score(y, sc["raw"]) - roc_auc_score(y, sc["clean"]),
                 "clean-behaviour": roc_auc_score(y, sc["clean"]) - roc_auc_score(y, sc["beh"])}
        for k in fns:
            out.append({"model": m, "quantity": k, "estimate": point[k], "ci_lo": ci[k][0], "ci_hi": ci[k][1]})
    o = pd.DataFrame(out).round(4)
    o.to_csv(PROC / "leakage_gap_ci.csv", index=False)
    print("\n", res.to_string(index=False), "\n\n", o.to_string(index=False))


# ----------------------------------------------------------------------------------------------- scrubbing intensity
def scrub_levels():
    """Four scrubbing intensities. L1 = registration and mandatory-disclosure cues; L2 = the version used in all main results
    (adds 'for illustration / not a recommendation / liability' wording); L3 = L2 plus warranty/accuracy/responsibility wording."""
    import common
    l1 = re.compile(common.BOILER_L1, re.I)
    l2 = re.compile(common.BOILER_L1 + "|" + common.BOILER_L2_ADD, re.I)
    l3 = re.compile(common.BOILER_L1 + "|" + common.BOILER_L2_ADD + "|" + common.BOILER_L3_ADD, re.I)

    def make(rx):
        def f(text):
            parts = re.split(r"(?<=[.!?|])\s+|\n+", text or "")
            return " ".join(p for p in parts if p and not rx.search(p))
        return f
    return {"L0 raw": lambda t: t, "L1 first pass": make(l1), "L2 used in main results": make(l2), "L3 extended": make(l3)}


def scrub(_):
    df = E.build_frame()
    y = df.y.to_numpy()
    sp = splits(y)
    rows = []
    for name, f in scrub_levels().items():
        text = ((df.chan_raw + " " + df.text_raw).map(f) + " " + df.comments_raw.map(f)).tolist()
        mats = featurise(text, sp, "word")
        scores = [LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5).fit(Xtr, y[tr]).predict_proba(Xte)[:, 1] for (Xtr, Xte), (_, tr, te) in zip(mats, sp)]
        per = collect(sp, len(y), scores)
        lo, hi = boot_auc(y, np.mean(per, axis=0))
        kept = sum(len(t) for t in text) / (df.chan_raw + " " + df.text_raw + " " + df.comments_raw).str.len().sum()
        rows.append({"scrubbing_level": name, "model": "logreg", "AUC": round(float(np.mean([roc_auc_score(y, s) for s in per])), 4),
                     "CI_lo": round(lo, 3), "CI_hi": round(hi, 3), "share_of_characters_kept": round(float(kept), 3)})
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(PROC / "scrub_levels.csv", index=False)
    print(out.to_string(index=False))


# ----------------------------------------------------------------------------------------------- ablation
def ablation(_):
    df = E.build_frame()
    y = df.y.to_numpy()
    sp = splits(y, repeats=2)
    n = len(y)
    clfs = {"logreg": lambda: LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5),
            "linear_svm": lambda: LinearSVC(C=0.1, class_weight="balanced"),
            "complement_nb": lambda: ComplementNB(alpha=0.3)}
    rows = []

    def evaluate(mats, make, sel=None):
        scores = []
        for (Xtr, Xte), (_, tr, te) in zip(mats, sp):
            if sel is not None:
                cols = top_columns_by_df(Xtr, sel)
                Xtr, Xte = Xtr[:, cols], Xte[:, cols]
            scores.append(score_of(make().fit(Xtr, y[tr]), Xte))
        per = collect(sp, n, scores)
        lo, hi = boot_auc(y, rank_pool(per))
        return float(np.mean([roc_auc_score(y, s) for s in per])), lo, hi

    def add(group, text, rep, clf, c, mf, res):
        rows.append({"experiment": group, "text": text, "representation": rep, "classifier": clf, "C_or_alpha": c, "max_features": mf,
                     "AUC": round(res[0], 4), "CI_lo": round(res[1], 3), "CI_hi": round(res[2], 3)})
        print(f"  {group:14s} {text:11s} {rep:9s} {clf:13s} C={c} mf={mf}: {res[0]:.3f}", flush=True)

    for text in ("text_clean", "text_leaky"):
        word = featurise(df[text].tolist(), sp, "word")
        char = featurise(df[text].str.slice(0, 20000).tolist(), sp, "char")      # character n-grams on the first 20,000 characters
        both = [(sparse.hstack([a[0], b[0]]).tocsr(), sparse.hstack([a[1], b[1]]).tocsr()) for a, b in zip(word, char)]
        for rep, mats in (("word", word), ("char", char), ("word+char", both)):
            for name, mk in clfs.items():
                add("representation", text, rep, name, "", 20000 if rep == "word" else "100000", evaluate(mats, mk))
        if text == "text_clean":
            for C in (0.05, 0.5, 5, 50):
                add("C", text, "word", "logreg", C, 20000, evaluate(word, lambda C=C: LogisticRegression(max_iter=3000, class_weight="balanced", C=C)))
            for k in (2000, 5000, 20000):
                add("vocabulary", text, "word", "logreg", 0.5, k, evaluate(word, clfs["logreg"], sel=k))
    out = pd.DataFrame(rows)
    out.to_csv(PROC / "nlp_ablation.csv", index=False)
    print(out.to_string(index=False))


# ----------------------------------------------------------------------------------------------- nested tuning
def tune(_):
    """Nested cross-validation for the logistic-regression regularisation parameter C (and the vocabulary cap).
    Within every outer training fold, C is chosen by 3-fold inner cross-validation on the AUC; the chosen model is refitted on
    the whole outer training fold and scored on the outer test fold. The TF-IDF vocabulary/idf of the outer training fold is
    reused in the inner folds (unsupervised, no label information from the outer test fold)."""
    from sklearn.model_selection import StratifiedKFold
    df = E.build_frame()
    y = df.y.to_numpy()
    n = len(y)
    sp = splits(y)
    grid = [0.5, 5, 50, 500]
    rows, pooled, chosen = [], {}, []
    for fs in ("text_clean", "text_leaky", "text_removed"):
        for mf in (20000, 50000, None):
            if mf != 20000 and fs == "text_removed":
                continue
            mats = featurise(df[fs].tolist(), sp, "word", mf)
            scores, picked = [], []
            for (Xtr, Xte), (r, tr, te) in zip(mats, sp):
                inner = StratifiedKFold(3, shuffle=True, random_state=r)
                best_c, best = None, -1
                for C in grid:
                    aucs = []
                    for itr, iva in inner.split(Xtr, y[tr]):
                        m = LogisticRegression(max_iter=3000, class_weight="balanced", C=C).fit(Xtr[itr], y[tr][itr])
                        aucs.append(roc_auc_score(y[tr][iva], m.predict_proba(Xtr[iva])[:, 1]))
                    if np.mean(aucs) > best:
                        best, best_c = np.mean(aucs), C
                picked.append(best_c)
                scores.append(LogisticRegression(max_iter=3000, class_weight="balanced", C=best_c).fit(Xtr, y[tr]).predict_proba(Xte)[:, 1])
            per = collect(sp, n, scores)
            pooled[(fs, mf)] = np.mean(per, axis=0)
            lo, hi = boot_auc(y, pooled[(fs, mf)])
            rows.append({"features": fs, "max_features": mf if mf else "all", "AUC_nested": round(float(np.mean([roc_auc_score(y, s) for s in per])), 4),
                         "CI_lo": round(lo, 3), "CI_hi": round(hi, 3), "median_selected_C": float(np.median(picked))})
            print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(PROC / "nested_tuning.csv", index=False)
    gaps = []
    for mf in (20000,):
        sc = {"raw": pooled[("text_leaky", mf)], "clean": pooled[("text_clean", mf)]}
        fn = {"gap": lambda yy, s: roc_auc_score(yy, s["raw"]) - roc_auc_score(yy, s["clean"])}
        ci = boot_pair(y, sc, fn)["gap"]
        gaps.append({"vocabulary": mf, "tuned_leakage_gap": round(roc_auc_score(y, sc["raw"]) - roc_auc_score(y, sc["clean"]), 4),
                     "CI_lo": round(ci[0], 4), "CI_hi": round(ci[1], 4)})
    pd.DataFrame(gaps).to_csv(PROC / "nested_tuning_gap.csv", index=False)
    print(out.to_string(index=False), "\n", pd.DataFrame(gaps).to_string(index=False))


# ----------------------------------------------------------------------------------------------- terms
def terms(_):
    df = E.build_frame()
    v = vec_word()
    X = v.fit_transform(df.text_clean)
    m = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5).fit(X, df.y)
    t = np.array(v.get_feature_names_out())
    o = np.argsort(m.coef_[0])
    out = pd.DataFrame({"toward_registered": t[o[-25:]][::-1], "weight_reg": m.coef_[0][o[-25:]][::-1].round(3),
                        "toward_unregistered": t[o[:25]], "weight_unreg": m.coef_[0][o[:25]].round(3)})
    out.to_csv(PROC / "top_terms_scrubbed.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stats", "gap", "ablation", "terms", "scrub", "tune"])
    a = ap.parse_args()
    {"stats": stats, "gap": gap, "ablation": ablation, "terms": terms, "scrub": scrub, "tune": tune}[a.cmd](a)
