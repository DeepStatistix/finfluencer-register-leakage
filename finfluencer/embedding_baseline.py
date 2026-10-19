"""Neural baseline: multilingual sentence embeddings and the leakage gap.

    python embedding_baseline.py embed     # encode every account (raw, scrubbed, removed-wording texts) and cache the vectors
    python embedding_baseline.py cv        # repeated 5-fold cross-validation, leakage gap with paired bootstrap interval

Model     sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384 dimensions, covers Hindi and other Indian languages),
          used frozen. An account is represented by the mean of the embeddings of up to 40 chunks of about 90 words, spread evenly
          over its document, so that long documents are covered at bounded cost.
Variants  raw text; text with the regulatory wording removed (Level 2 scrubber, as in the main analysis); the removed wording alone.
Classifier standardised embeddings + logistic regression with the regularisation parameter chosen by inner three-fold cross-validation
          (grid 0.001 ... 1) inside every outer training fold.
"""
import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import eval_pu as E
from common import PROC

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VARIANTS = {"raw": "text_leaky", "scrubbed": "text_clean", "removed_only": "text_removed"}
WORDS, MAX_CHUNKS = 90, 40


def chunks(text):
    w = text.split()
    wins = [" ".join(w[i:i + WORDS]) for i in range(0, max(len(w), 1), WORDS)]
    if len(wins) > MAX_CHUNKS:
        wins = [wins[i] for i in np.linspace(0, len(wins) - 1, MAX_CHUNKS).round().astype(int)]
    return wins or [""]


def embed(a):
    from sentence_transformers import SentenceTransformer
    df = E.build_frame()
    m = SentenceTransformer(MODEL, device="cpu")
    m.max_seq_length = 128
    for name, col in VARIANTS.items():
        vecs = []
        for k, t in enumerate(df[col].tolist()):
            e = m.encode(chunks(t), batch_size=64, show_progress_bar=False)
            vecs.append(e.mean(axis=0))
            if k % 100 == 0:
                print(f"  {name}: {k}/{len(df)}", flush=True)
        np.save(PROC / f"emb_{name}.npy", np.vstack(vecs))
    df[["channel_id", "y"]].to_csv(PROC / "emb_index.csv", index=False)
    print("saved embeddings for", len(df), "accounts")


def fold(X, y, tr, te):
    m = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=[0.001, 0.01, 0.1, 1], cv=3, scoring="roc_auc", class_weight="balanced", max_iter=3000))
    m.fit(X[tr], y[tr])
    return m.predict_proba(X[te])[:, 1], float(m[-1].C_[0])


def cv(a):
    from nlp_experiments import boot_auc, boot_pair, collect, splits
    idx = pd.read_csv(PROC / "emb_index.csv")
    y = idx.y.to_numpy()
    sp = splits(y)
    pooled, rows = {}, []
    for name in VARIANTS:
        X = np.load(PROC / f"emb_{name}.npy")
        res = Parallel(n_jobs=6)(delayed(fold)(X, y, tr, te) for _, tr, te in sp)
        per = collect(sp, len(y), [r[0] for r in res])
        pooled[name] = np.mean(per, axis=0)
        lo, hi = boot_auc(y, pooled[name])
        rows.append({"variant": name, "AUC": round(float(np.mean([roc_auc_score(y, s) for s in per])), 4), "CI_lo": round(lo, 3), "CI_hi": round(hi, 3),
                     "median_selected_C": float(np.median([r[1] for r in res]))})
        print(rows[-1], flush=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(PROC / "embedding_cv_results.csv", index=False)
    fn = {"leakage_gap": lambda yy, s: roc_auc_score(yy, s["raw"]) - roc_auc_score(yy, s["scrubbed"])}
    ci = boot_pair(y, pooled, fn)["leakage_gap"]
    gap = roc_auc_score(y, pooled["raw"]) - roc_auc_score(y, pooled["scrubbed"])
    g = pd.DataFrame([{"quantity": "leakage_gap_embeddings", "estimate": round(gap, 4), "ci_lo": round(ci[0], 4), "ci_hi": round(ci[1], 4)}])
    g.to_csv(PROC / "embedding_gap_ci.csv", index=False)
    print(tab.to_string(index=False), "\n", g.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["embed", "cv"])
    a = ap.parse_args()
    {"embed": embed, "cv": cv}[a.cmd](a)
