"""Sensitivity of the PU results to the class-imbalance strategy (141 registered : 450 unregistered).

    python balance_check.py

Compares, on leakage-safe text features, three treatments done INSIDE each training fold:
  none          no correction
  class_weight  'balanced' weights (the setting used in eval_pu.py)
  undersample   bag of models, each trained on all positives + an equal-sized random subset of U
SMOTE is deliberately not used: it interpolates between sparse TF-IDF vectors, producing non-text points, and
done before the split it leaks. Class weights / undersampling are the standard safe choices here.
Metrics are imbalance-robust (AUC, average precision, recall at a fixed 20% flag rate), never plain accuracy.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

import eval_pu as E
from common import PROC


class UnderBag(BaseEstimator, ClassifierMixin):
    """Average of n models, each fit on all positives and an equal number of random negatives."""

    def __init__(self, base=None, n_bags=10, seed=0):
        self.base, self.n_bags, self.seed = base, n_bags, seed

    def fit(self, X, y):
        y = np.asarray(y)
        rng = np.random.default_rng(self.seed)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        self.models_ = []
        for _ in range(self.n_bags):
            idx = np.concatenate([pos, rng.choice(neg, size=min(len(pos), len(neg)), replace=False)])
            self.models_.append(clone(self.base).fit(X.iloc[idx], y[idx]))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.mean([m.predict_proba(X)[:, 1] for m in self.models_], axis=0)
        return np.column_stack([1 - p, p])


def models():
    prep = E.make_models("text_clean")
    lr = prep["logreg"].named_steps["p"]
    gb = prep["hist_gbm"].named_steps["p"]
    out = {}
    for name, pre, mk in (("logreg", lr, lambda cw: LogisticRegression(max_iter=3000, C=0.5, class_weight=cw)),
                          ("hist_gbm", gb, lambda cw: HistGradientBoostingClassifier(
                              max_depth=4, learning_rate=0.05, max_iter=200, class_weight=cw, random_state=0))):
        out[(name, "none")] = Pipeline([("p", clone(pre)), ("m", mk(None))])
        out[(name, "class_weight")] = Pipeline([("p", clone(pre)), ("m", mk("balanced"))])
        out[(name, "undersample")] = UnderBag(Pipeline([("p", clone(pre)), ("m", mk(None))]), n_bags=8)
    return out


def main(a):
    df = E.build_frame()
    y = df.y.to_numpy()
    cols = ["text_clean"] + E.BEHAV
    rows = []
    for (name, strat), model in models().items():
        rkf = RepeatedStratifiedKFold(n_splits=a.folds, n_repeats=a.repeats, random_state=0)
        per = [np.full(len(df), np.nan) for _ in range(a.repeats)]
        for k, (tr, te) in enumerate(rkf.split(df, y)):
            m = clone(model).fit(df.iloc[tr][cols], y[tr])
            per[k // a.folds][te] = m.predict_proba(df.iloc[te][cols])[:, 1]
        rows.append({"model": name, "balancing": strat, "AUC": np.mean([roc_auc_score(y, s) for s in per]),
                     "avg_precision": np.mean([average_precision_score(y, s) for s in per]),
                     "positive_recall@top20%": np.mean([E.recall_at_flag_rate(y, s) for s in per])})
        print("done", name, strat, flush=True)
    res = pd.DataFrame(rows).round(3)
    res.to_csv(PROC / "balance_check.csv", index=False)
    print("\n", res.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    main(ap.parse_args())
