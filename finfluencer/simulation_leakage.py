"""Monte Carlo study of the leakage-gap estimator (manuscript Section 4.11 / 5.11).

    python simulation_leakage.py --reps 200

Question  Does the cross-validated leakage gap  Delta = AUC(raw text) - AUC(scrubbed text)  recover the true expected gap, and does
          the paired bootstrap interval of the paper have nominal coverage? What happens when the scrubber is incomplete?
Design    Synthetic bag-of-words accounts. Content words follow a Zipf law; registered accounts (class 1) over-use a random subset of
          words by the factor (1 + delta), which is the legitimate signal. In addition every account can contain a block of 10 'regulatory'
          tokens drawn from a pool of 40; the block appears in a registered account with probability pi1 and in an unregistered account
          with probability pi0 = 0.02 (the leakage channel). A scrubber removes a present block with probability r (its recall).
Truth     For a scenario, the true gap is the mean over 200 independent training sets of the same size as a cross-validation training
          fold (0.8 n) of AUC(raw) - AUC(scrubbed) measured on one large fresh test sample (4,000 accounts, same class ratio).
Estimate  Per replicate: one dataset of n = 591 accounts (141 positives), one repetition of 5-fold cross-validation with TF-IDF fitted
          inside each fold and logistic regression (C = 0.5, balanced weights), gap_hat = AUC_raw - AUC_scrubbed on the out-of-fold scores,
          paired percentile bootstrap interval (B = 500 resamples of accounts, same resample for both AUCs).
Output    processed/sim_leakage_results.csv
"""
import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

from common import PROC

V, B, BLOCK, LENGTH = 1500, 40, 10, 150     # content vocabulary, boilerplate pool, tokens per block, content tokens per account
N, N_POS, PI0 = 591, 141, 0.02
DELTA, N_SIGNAL = 0.95, 150                # legitimate signal: 150 words over-used by (1 + DELTA) in class 1 (calibrated to a scrubbed AUC of about 0.83, as in the real data)


def world(seed):
    rng = np.random.default_rng(seed)
    base = 1 / np.arange(1, V + 1)
    base /= base.sum()
    boost = np.ones(V)
    boost[rng.choice(V, N_SIGNAL, replace=False)] = 1 + DELTA
    p1 = base * boost
    return base, p1 / p1.sum()


def draw(rng, p0, p1, n, n_pos, pi1, recall):
    """Counts matrices (raw, scrubbed) and labels for n accounts."""
    y = np.zeros(n, dtype=int)
    y[:n_pos] = 1
    X = np.zeros((n, V + B), dtype=np.float32)
    for i in range(n):
        X[i, :V] = rng.multinomial(LENGTH, p1 if y[i] else p0)
    has = rng.random(n) < np.where(y == 1, pi1, PI0)
    for i in np.where(has)[0]:
        X[i, V + rng.choice(B, BLOCK, replace=False)] += 1
    removed = has & (rng.random(n) < recall)
    S = X.copy()
    S[removed, V:] = 0
    return sparse.csr_matrix(X), sparse.csr_matrix(S), y


def model():
    return make_pipeline(TfidfTransformer(sublinear_tf=True), LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5))


def truth(p0, p1, pi1, recall, seed, n_train_sets=200, n_test=4000):
    rng = np.random.default_rng(seed)
    n_tr = int(0.8 * N)
    Xt, St, yt = draw(rng, p0, p1, n_test, int(round(n_test * N_POS / N)), pi1, recall)
    gaps = []
    for _ in range(n_train_sets):
        X, S, y = draw(rng, p0, p1, n_tr, int(round(n_tr * N_POS / N)), pi1, recall)
        a = roc_auc_score(yt, model().fit(X, y).predict_proba(Xt)[:, 1])
        b = roc_auc_score(yt, model().fit(S, y).predict_proba(St)[:, 1])
        gaps.append((a, b, a - b))
    g = np.array(gaps)
    return g.mean(axis=0)


def replicate(seed, p0, p1, pi1, recall, truth_gap, boots=500):
    rng = np.random.default_rng(seed)
    X, S, y = draw(rng, p0, p1, N, N_POS, pi1, recall)
    oof_x, oof_s = np.zeros(N), np.zeros(N)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        oof_x[te] = model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        oof_s[te] = model().fit(S[tr], y[tr]).predict_proba(S[te])[:, 1]
    est = roc_auc_score(y, oof_x) - roc_auc_score(y, oof_s)
    bs = []
    for _ in range(boots):
        i = rng.integers(0, N, N)
        if 0 < y[i].sum() < N:
            bs.append(roc_auc_score(y[i], oof_x[i]) - roc_auc_score(y[i], oof_s[i]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, lo, hi, bool(lo <= truth_gap <= hi)


def main(a):
    p0, p1 = world(2026)
    scenarios = [(pi, 1.0) for pi in (0.0, 0.25, 0.5, 0.75, 1.0)] + [(0.75, r) for r in (0.9, 0.7, 0.5)]
    rows, reps_rows = [], []
    for k, (pi1, r) in enumerate(scenarios):
        auc_raw, auc_scr, tg = truth(p0, p1, pi1, r, seed=1000 + k)
        res = Parallel(n_jobs=6)(delayed(replicate)(5000 + 97 * k + j, p0, p1, pi1, r, tg) for j in range(a.reps))
        est = np.array([x[0] for x in res])
        lo = np.array([x[1] for x in res])
        hi = np.array([x[2] for x in res])
        cov = np.mean([x[3] for x in res])
        reps_rows.extend({"pi_registered": pi1, "scrubber_recall": r, "true_gap": tg, "estimate": e, "lo": l, "hi": h} for e, l, h in zip(est, lo, hi))
        infl = {f"coverage_x{k}": round(float(np.mean((est - k * (est - lo) <= tg) & (tg <= est + k * (hi - est)))), 3) for k in (1.25, 1.5)}
        rows.append({"pi_registered": pi1, "scrubber_recall": r, "true_AUC_raw": round(auc_raw, 3), "true_AUC_scrubbed": round(auc_scr, 3),
                     "true_gap": round(tg, 4), "mean_estimated_gap": round(est.mean(), 4), "bias": round(est.mean() - tg, 4), "sd_estimate": round(est.std(ddof=1), 4),
                     "coverage_95": round(cov, 3), **infl, "mean_interval_width": round(float((hi - lo).mean()), 4),
                     "interval_sd_ratio": round(float((hi - lo).mean() / 3.92 / est.std(ddof=1)), 3), "replicates": a.reps})
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    pd.DataFrame(reps_rows).to_csv(PROC / "sim_leakage_replicates.csv", index=False)
    out.to_csv(PROC / "sim_leakage_results.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    main(ap.parse_args())
