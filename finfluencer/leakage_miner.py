"""Automatic mining of leakage cues: a template-sentence scrubber with a statistical test.

    python leakage_miner.py real        # nested cross-validated comparison with the manual scrubber on the YouTube data
    python leakage_miner.py overlap     # overlap of the mined and manual sentence sets, example templates (descriptive)
    python leakage_miner.py sensitivity # minimum document frequency x FDR level
    python leakage_miner.py sim         # known-truth evaluation on synthetic documents with templates

Idea  Mandatory disclosures are TEMPLATED: (near-)identical sentences repeated across many accounts, and concentrated in one class.
      Legitimate content is rarely repeated verbatim across accounts. The miner therefore
        1. splits every account document into sentences and normalises them (lower case, URLs and digits masked, punctuation removed);
        2. keeps as candidates the sentences (and 6-token sentence prefixes, which catch templates with variable endings) that occur in
           at least m_min training accounts;
        3. tests each candidate for class skew with Fisher's exact test (registered vs unregistered accounts containing it), controlling the
           false discovery rate with Benjamini-Hochberg at level q;
        4. removes every sentence matching a selected candidate from training AND test documents.
      Labels are used only on the training accounts of each fold, so the scrubber cannot leak the target into the evaluation.
Outputs  processed/miner_*.csv
"""
import argparse
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import fisher_exact
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import common
import eval_pu as E
from common import PROC

SENT_SPLIT = re.compile(r"(?<=[.!?|])\s+|\n+")
URL = re.compile(r"https?://\S+|www\.\S+")
MIN_TOKENS, MIN_CHARS, PREFIX = 5, 25, 6
N_JOBS = 3


# --------------------------------------------------------------------------------------------- units
def sentences(text):
    return [p for p in (s.strip() for s in SENT_SPLIT.split(text or "")) if p]


def normalise(s):
    s = URL.sub(" url ", s.lower())
    s = re.sub(r"\d+", "0", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def keys_of(sentence):
    """Matching keys of a sentence: its full normalised form and (for longer sentences) its first PREFIX tokens."""
    n = normalise(sentence)
    toks = n.split()
    if len(toks) < MIN_TOKENS or len(n) < MIN_CHARS:
        return ()
    k = [("f", n)]
    if len(toks) > PREFIX:
        k.append(("p", " ".join(toks[:PREFIX])))
    return tuple(k)


def prepare(texts):
    """Sentences and their keys for every document (computed once; labels are not involved)."""
    sents = [sentences(t) for t in texts]
    ids, inv = {}, []

    def code(k):
        if k not in ids:
            ids[k] = len(inv)
            inv.append(k)
        return ids[k]
    keys = [[tuple(code(k) for k in keys_of(s)) for s in ss] for ss in sents]   # integer codes keep memory low (572,906 sentences)
    regno = [[bool(common.REG_NO.search(s)) for s in ss] for ss in sents]       # structural cue only: a SEBI registration-number pattern
    return sents, keys, regno, inv


def bh(p, q):
    """Benjamini-Hochberg: boolean mask of rejected hypotheses at FDR level q."""
    p = np.asarray(p)
    n = len(p)
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    thr = q * (np.arange(1, n + 1) / n)
    ok = p[order] <= thr
    k = np.max(np.where(ok)[0]) + 1 if ok.any() else 0
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask


# --------------------------------------------------------------------------------------------- the miner
class TemplateMiner:
    def __init__(self, min_df=5, q=0.05):
        self.min_df, self.q = min_df, q
        self.selected = {}

    def fit(self, keys, y):
        y = np.asarray(y)
        n_pos, n_neg = int(y.sum()), int((1 - y).sum())
        df, pos = defaultdict(int), defaultdict(int)
        for ks, yi in zip(keys, y):
            present = {k for sk in ks for k in sk}
            for k in present:
                df[k] += 1
                pos[k] += int(yi)
        cand = [k for k, c in df.items() if c >= self.min_df]
        p = []
        for k in cand:
            a, b = pos[k], df[k] - pos[k]
            p.append(fisher_exact([[a, n_pos - a], [b, n_neg - b]])[1])
        keep = bh(p, self.q)
        self.selected = {k: (df[k], pos[k], pv) for k, pv, m in zip(cand, p, keep) if m}
        return self

    def split(self, sents, keys, regno=None, use_regno=False):
        """Return (kept_text, removed_text) of one document."""
        kept, removed = [], []
        for j, (s, ks) in enumerate(zip(sents, keys)):
            drop = any(k in self.selected for k in ks) or (use_regno and regno is not None and regno[j])
            (removed if drop else kept).append(s)
        return " ".join(kept), " ".join(removed)


# --------------------------------------------------------------------------------------------- real data
def manual_text(texts):
    rx = common._BOILER
    out = []
    for t in texts:
        out.append(" ".join(s for s in sentences(t) if not rx.search(s)))
    return out


def fold_job(arm, tr, te, y, texts, sk, min_df, q):
    """One outer fold: scrub (arm-specific), vectorise on the training fold, fit logistic regression, return test scores."""
    if arm in ("raw", "manual"):
        trt, tet = [texts[i] for i in tr], [texts[i] for i in te]
        removed = ""
    else:
        sents, keys, regno, _ = sk
        m = TemplateMiner(min_df, q).fit([keys[i] for i in tr], y[tr])
        parts = [m.split(sents[i], keys[i], regno[i], use_regno=(arm == "miner+regno")) for i in range(len(texts))]
        if arm == "miner_removed_only":
            trt, tet = [parts[i][1] for i in tr], [parts[i][1] for i in te]
        else:
            trt, tet = [parts[i][0] for i in tr], [parts[i][0] for i in te]
    v = TfidfVectorizer(min_df=5, ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode", max_features=20000)
    Xtr = v.fit_transform(trt)
    Xte = v.transform(tet)
    s = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5).fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]
    n_sel = len(m.selected) if arm.startswith("miner") else 0
    return s, n_sel


def run_arms(df, arms, repeats, min_df=5, q=0.05):
    from nlp_experiments import splits, collect
    y = df.y.to_numpy()
    raw = df.text_leaky.tolist()
    man = manual_text(raw)
    prep = {"raw": prepare(raw), "manual": None}
    prep_man = prepare(man)
    sp = splits(y, repeats=repeats)
    out = {}
    for arm in arms:
        texts = man if arm in ("manual", "manual+miner") else raw
        sk = prep_man if arm == "manual+miner" else prep["raw"] if arm.startswith("miner") else None
        res = Parallel(n_jobs=N_JOBS)(delayed(fold_job)("miner" if arm == "manual+miner" else arm, tr, te, y, texts, sk, min_df, q) for _, tr, te in sp)
        per = collect(sp, len(y), [r[0] for r in res])
        out[arm] = (per, float(np.mean([r[1] for r in res])))
        print(f"  {arm:20s} AUC {np.mean([roc_auc_score(y, s) for s in per]):.4f}  templates selected per fold: {out[arm][1]:.0f}", flush=True)
    return out, y


def real(a):
    from nlp_experiments import boot_auc, boot_pair
    df = E.build_frame()
    arms = ["raw", "manual", "miner", "miner+regno", "manual+miner", "miner_removed_only"]
    res, y = run_arms(df, arms, repeats=3)
    pooled = {k: np.mean(v[0], axis=0) for k, v in res.items()}
    rows = []
    for k in arms:
        lo, hi = boot_auc(y, pooled[k])
        rows.append({"arm": k, "AUC": round(float(np.mean([roc_auc_score(y, s) for s in res[k][0]])), 4), "CI_lo": round(lo, 3), "CI_hi": round(hi, 3),
                     "templates_selected_per_fold": round(res[k][1], 1)})
    tab = pd.DataFrame(rows)
    tab.to_csv(PROC / "miner_cv_results.csv", index=False)
    fns = {"gap_manual": lambda yy, s: roc_auc_score(yy, s["raw"]) - roc_auc_score(yy, s["manual"]),
           "gap_miner": lambda yy, s: roc_auc_score(yy, s["raw"]) - roc_auc_score(yy, s["miner"]),
           "gap_miner_regno": lambda yy, s: roc_auc_score(yy, s["raw"]) - roc_auc_score(yy, s["miner+regno"]),
           "miner_regno_minus_manual_AUC": lambda yy, s: roc_auc_score(yy, s["miner+regno"]) - roc_auc_score(yy, s["manual"]),
           "miner_minus_manual_AUC": lambda yy, s: roc_auc_score(yy, s["miner"]) - roc_auc_score(yy, s["manual"]),
           "manual+miner_minus_manual_AUC": lambda yy, s: roc_auc_score(yy, s["manual+miner"]) - roc_auc_score(yy, s["manual"])}
    ci = boot_pair(y, pooled, fns)
    point = {"gap_manual": roc_auc_score(y, pooled["raw"]) - roc_auc_score(y, pooled["manual"]),
             "gap_miner": roc_auc_score(y, pooled["raw"]) - roc_auc_score(y, pooled["miner"]),
             "gap_miner_regno": roc_auc_score(y, pooled["raw"]) - roc_auc_score(y, pooled["miner+regno"]),
             "miner_regno_minus_manual_AUC": roc_auc_score(y, pooled["miner+regno"]) - roc_auc_score(y, pooled["manual"]),
             "miner_minus_manual_AUC": roc_auc_score(y, pooled["miner"]) - roc_auc_score(y, pooled["manual"]),
             "manual+miner_minus_manual_AUC": roc_auc_score(y, pooled["manual+miner"]) - roc_auc_score(y, pooled["manual"])}
    g = pd.DataFrame([{"quantity": k, "estimate": round(point[k], 4), "ci_lo": round(ci[k][0], 4), "ci_hi": round(ci[k][1], 4)} for k in fns])
    g.to_csv(PROC / "miner_gap_ci.csv", index=False)
    print(tab.to_string(index=False), "\n", g.to_string(index=False))


def overlap(a):
    df = E.build_frame()
    y = df.y.to_numpy()
    raw = df.text_leaky.tolist()
    sents, keys, _, inv = prepare(raw)
    m = TemplateMiner(5, 0.05).fit(keys, y)
    sel_str = {inv[k]: v for k, v in m.selected.items()}          # readable keys for the descriptive output
    rx = common._BOILER
    inst = []
    for ss, ks in zip(sents, keys):
        for s, k in zip(ss, ks):
            inst.append((bool(rx.search(s)), any(kk in m.selected for kk in k), len(s)))
    inst = pd.DataFrame(inst, columns=["manual", "miner", "chars"])
    both = (inst.manual & inst.miner).sum()
    tab = pd.DataFrame([{"sentences": len(inst), "manual_removed": int(inst.manual.sum()), "miner_removed": int(inst.miner.sum()), "both": int(both),
                         "recall_of_manual": round(both / inst.manual.sum(), 3), "precision_wrt_manual": round(both / inst.miner.sum(), 3),
                         "chars_removed_manual": int(inst[inst.manual].chars.sum()), "chars_removed_miner": int(inst[inst.miner].chars.sum()),
                         "selected_templates": len(m.selected)}])
    tab.to_csv(PROC / "miner_overlap.csv", index=False)
    print(tab.T.to_string())
    top = sorted(sel_str.items(), key=lambda kv: -kv[1][0])[:40]
    pd.DataFrame([{"kind": k[0], "template": k[1][:160], "accounts": v[0], "registered_accounts": v[1], "p": v[2]} for k, v in top]).to_csv(PROC / "miner_top_templates.csv", index=False)
    # share of selected templates that the manual regex does not catch (candidate residual leakage)
    extra = [(k, v) for k, v in sel_str.items() if not rx.search(k[1])]
    print("selected keys:", len(sel_str), "| not matched by the manual regex:", len(extra))
    for k, v in sorted(extra, key=lambda kv: -kv[1][0])[:15]:
        print("  ", k[0], v[0], v[1], k[1][:110])


def sensitivity(a):
    df = E.build_frame()
    rows = []
    for m in (3, 5, 10, 20):
        for q in (0.01, 0.05, 0.2):
            res, y = run_arms(df, ["miner"], repeats=1, min_df=m, q=q)
            per, nsel = res["miner"]
            rows.append({"min_df": m, "q": q, "AUC": round(float(np.mean([roc_auc_score(y, s) for s in per])), 4), "templates_per_fold": round(nsel, 1)})
    out = pd.DataFrame(rows)
    out.to_csv(PROC / "miner_sensitivity.csv", index=False)
    print(out.to_string(index=False))


# --------------------------------------------------------------------------------------------- simulation with known truth
def sim_world(rng, n=591, n_pos=141, p_leak=0.5, p_benign=0.25, n_leak=20, n_benign=20, variable_suffix=0.5, n_content=25, vocab=3000, delta=0.95, n_signal=150):
    base = 1 / np.arange(1, vocab + 1)
    base /= base.sum()
    boost = np.ones(vocab)
    boost[rng.choice(vocab, n_signal, replace=False)] = 1 + delta
    p1 = base * boost
    p1 /= p1.sum()
    def letters(i):                      # letters only: the normaliser masks digits, which would merge distinct synthetic tokens
        out = ""
        i += 1
        while i:
            i, r = divmod(i - 1, 26)
            out = chr(97 + r) + out
        return out
    vocab_words = ["w" + letters(i) for i in range(vocab)]
    tpl = lambda kind, i: " ".join([f"{kind}{letters(i)}x{letters(j)}" for j in range(10)])
    y = np.zeros(n, dtype=int)
    y[:n_pos] = 1
    docs, truth = [], {}
    for i in range(n_leak):
        truth[tpl("leak", i)] = "leak"
    for i in range(n_benign):
        truth[tpl("ben", i)] = "benign"
    for yi in y:
        words = rng.choice(vocab, size=n_content * 12, p=p1 if yi else base)
        sents = [" ".join(vocab_words[w] for w in words[k * 12:(k + 1) * 12]) + "." for k in range(n_content)]
        for i in range(n_leak):
            if rng.random() < (p_leak if yi else 0.02):
                s = tpl("leak", i) + (" " + " ".join(vocab_words[rng.integers(vocab)] for _ in range(3)) if rng.random() < variable_suffix else "")
                sents.append(s + ".")
        for i in range(n_benign):
            if rng.random() < p_benign:
                s = tpl("ben", i) + (" " + " ".join(vocab_words[rng.integers(vocab)] for _ in range(3)) if rng.random() < variable_suffix else "")
                sents.append(s + ".")
        rng.shuffle(sents)
        docs.append(" ".join(sents))
    return docs, y, truth


def sim_replicate(seed, p_leak):
    rng = np.random.default_rng(seed)
    docs, y, truth = sim_world(rng, p_leak=p_leak)
    n = len(y)
    idx = rng.permutation(n)
    tr, te = idx[: int(0.7 * n)], idx[int(0.7 * n):]
    if y[te].sum() == 0:
        return None
    sents, keys, _, inv = prepare(docs)
    m = TemplateMiner(5, 0.05).fit([keys[i] for i in tr], y[tr])
    # which true templates were selected (by the full or the prefix key)
    sel_prefix = {inv[k][1] for k in m.selected if inv[k][0] == "p"}
    sel_full = {inv[k][1] for k in m.selected if inv[k][0] == "f"}
    hit = {}
    for t, kind in truth.items():
        toks = normalise(t).split()
        hit[t] = (" ".join(toks[:PREFIX]) in sel_prefix) or any(f.startswith(" ".join(toks)) for f in sel_full)
    leak = [t for t, k in truth.items() if k == "leak"]
    ben = [t for t, k in truth.items() if k == "benign"]
    recall = float(np.mean([hit[t] for t in leak]))
    fpr = float(np.mean([hit[t] for t in ben]))
    out = {"recall_leak_templates": recall, "false_positive_rate_benign": fpr, "selected_total": len(m.selected)}
    def auc(trt, tet):
        v = TfidfVectorizer(min_df=3, sublinear_tf=True, token_pattern=r"\S+")
        Xtr = v.fit_transform(trt)
        return roc_auc_score(y[te], LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5).fit(Xtr, y[tr]).predict_proba(v.transform(tet))[:, 1])
    out["AUC_raw"] = auc([docs[i] for i in tr], [docs[i] for i in te])
    parts = [m.split(sents[i], keys[i])[0] for i in range(n)]
    out["AUC_miner"] = auc([parts[i] for i in tr], [parts[i] for i in te])
    oracle = [" ".join(s for s in sents[i] if not any(s.startswith(t) for t in leak)) for i in range(n)]
    out["AUC_oracle"] = auc([oracle[i] for i in tr], [oracle[i] for i in te])
    return out


def sim(a):
    rows = []
    for p_leak in (0.2, 0.5, 0.8):
        res = Parallel(n_jobs=N_JOBS)(delayed(sim_replicate)(9000 + j, p_leak) for j in range(a.reps))
        res = [r for r in res if r]
        d = pd.DataFrame(res)
        rows.append({"p_leak_registered": p_leak, **{c: round(float(d[c].mean()), 4) for c in d.columns}, "replicates": len(d)})
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(PROC / "miner_sim_results.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["real", "overlap", "sensitivity", "sim"])
    ap.add_argument("--reps", type=int, default=100)
    a = ap.parse_args()
    {"real": real, "overlap": overlap, "sensitivity": sensitivity, "sim": sim}[a.cmd](a)
