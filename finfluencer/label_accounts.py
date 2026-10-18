"""Stage 3 - link accounts to the register, compute content flags, assign WEAK labels.

    python label_accounts.py

Outputs data/finfluencer/processed/accounts.csv, one row per channel:
  match_status  reg_no_verified | reg_no_claimed_not_in_register | name_match_high | name_match_review | none
  rule shares   share of recent videos matching each content rule (see common.RULES)
  weak_label    authorized | unauthorized_candidate | neutral_candidate | review

The label hierarchy that avoids the circularity of the original paper:
  * authorized  comes from the REGISTER (external, legal, independent of text features).
  * unauthorized_candidate / neutral_candidate come from rules + are only candidates. The gold label is
    assigned by two human coders on a sample (annotation.py). The rule columns are therefore excluded
    from the model features in eval_models.py, otherwise the classifier just re-learns the rules.
"""
import re
from difflib import SequenceMatcher

import pandas as pd

from common import PROC, RAW, REG, norm_name, read_jsonl, registration_numbers, rule_flags

HIGH, REVIEW = 0.97, 0.85  # personal names collide easily (Anurag Singh / Anurag Singhal = 0.92), so be strict


def _key(norm):
    return " ".join(sorted(norm.split()))


def build_index(reg):
    idx = {}
    for i, n in enumerate(reg["norm"]):
        for t in set(n.split()):
            idx.setdefault(t, []).append(i)
    return idx


def match_name(title, reg, idx):
    """Best fuzzy match of a channel title against register names -> (score, reg_no, name)."""
    n = norm_name(title)
    if not n:
        return 0.0, None, None
    cands = {i for t in n.split() for i in idx.get(t, [])}
    best = (0.0, None, None)
    kn = _key(n)
    for i in cands:
        s = SequenceMatcher(None, kn, _key(reg.norm.iat[i])).ratio()
        if s > best[0]:
            best = (s, reg.reg_no.iat[i], reg.name.iat[i])
    # single short token names are too ambiguous to auto-accept
    if len(n.split()) < 2 and len(n) < 6:
        best = (min(best[0], REVIEW), best[1], best[2])
    return best


def holder_consistency(reg_nos, text, holders):
    """Share of the register holder's name tokens that appear in the channel's own text (0..1, best over numbers)."""
    best = 0.0
    toks = set(norm_name(text).split())
    for no in reg_nos:
        h = set(norm_name(holders.get(no, "")).split())
        if h:
            best = max(best, len(h & toks) / len(h))
    return best


def reg_status(r):
    """PRIMARY label: legal registration status, from the SEBI register only (no text judgement).
    registered   number on the register AND channel consistent with its holder, or a near-exact name match
    review       a person must decide (number found but holder does not look like the channel; fuzzy name match)
    unregistered everything else (includes channels quoting a number that is not on the register)"""
    if r["match_status"] == "name_match_high":
        return "registered"
    if r["match_status"] == "reg_no_verified":
        return "registered" if r.get("holder_consistency", 1.0) >= 0.5 else "review"
    if r["match_status"] == "name_match_review":
        return "review"
    return "unregistered"


_DENIES = re.compile(r"\bnot\s+(?:a\s+)?(?:sebi[\s-]*regist\w*|(?:financial|investment)\s+advis\w*|research\s+analyst)|"
                     r"नहीं\s*हूं|नहीं\s*हूँ|nahi\s*hoon", re.I)


def _squash(s):
    return re.sub(r"[^a-z0-9]", "", norm_name(s, drop_generic=False))


def auto_resolve(ch, verified, holder_names, name_score):
    """Resolve a 'review' account by explicit rules; returns (status, rule) so every decision is auditable.
    Precedence: denial > exact register-name > self-claim of a verified number in channel-level text
    > distinctive holder-name part in channel title/description > no evidence."""
    text = f"{ch['title']}\n{ch['description'] or ''}"
    if _DENIES.search(text):
        return "unregistered", "channel_denies_registration"
    if name_score >= HIGH:
        return "registered", "exact_register_name"
    if any(n in registration_numbers(text) for n in verified):
        return "registered", "channel_quotes_register_number"
    sq = _squash(text)
    for h in holder_names:
        for tok in norm_name(h).split():
            if len(tok) >= 4 and tok in sq:
                return "registered", "holder_name_in_channel_text"
    if verified:
        return "ambiguous", "number_only_in_videos_and_no_name_link"  # likely a brand channel; excluded, not forced to a class
    return "unregistered", "fuzzy_name_only"


def weak_label(r):
    """Pure function of an account row; see module docstring for the logic."""
    if r.get("n_videos", 1) == 0:
        return "no_videos"
    if r["match_status"] == "reg_no_verified" and r.get("holder_consistency", 1.0) < 0.5:
        return "review"  # number exists but the channel does not look like its holder: impersonation or brand name
    if r["match_status"] in ("reg_no_verified", "name_match_high"):
        return "authorized"
    advisory = max(r["share_trade_call"], r["share_guarantee"])
    if (r["match_status"] == "reg_no_claimed_not_in_register" or r["share_guarantee"] >= 0.10 or advisory >= 0.25
            or (r["share_paid_solicitation"] >= 0.25 and advisory > 0)):
        return "unauthorized_candidate"
    if advisory < 0.05 and r["share_paid_solicitation"] < 0.10 and r["match_status"] != "name_match_review":
        return "neutral_candidate"
    return "review"


def main():
    reg = pd.read_csv(REG / "register.csv", dtype=str).fillna("")
    reg = reg[reg.active.astype(str) == "True"].reset_index(drop=True)
    known = set(reg.reg_no)
    holders = dict(zip(reg.reg_no, reg.name))
    idx = build_index(reg)

    vids = {}
    for v in read_jsonl(RAW / "videos.jsonl"):
        vids.setdefault(v["channel_id"], []).append(v)

    rows = []
    for ch in read_jsonl(RAW / "channels.jsonl"):
        vs = vids.get(ch["channel_id"], [])
        texts = [f"{v['title']}\n{v['description']}" for v in vs]
        chan_text = f"{ch['title']}\n{ch['description']}"
        claimed = registration_numbers(" ".join([chan_text] + texts))
        verified = [x for x in claimed if x in known]
        score, rno, rname = match_name(ch["title"], reg, idx)
        if verified:
            status = "reg_no_verified"
        elif claimed:
            status = "reg_no_claimed_not_in_register"
        elif score >= HIGH:
            status = "name_match_high"
        elif score >= REVIEW:
            status = "name_match_review"
        else:
            status = "none"
        flags = [rule_flags(t) for t in texts]
        n = max(len(vs), 1)
        row = {"channel_id": ch["channel_id"], "title": ch["title"], "subscribers": ch["subscribers"],
               "n_videos": len(vs), "claimed_reg_nos": ";".join(claimed), "verified_reg_nos": ";".join(verified),
               "name_match_score": round(score, 3), "name_match_reg_no": rno, "name_match_name": rname,
               "match_status": status}
        row["holder_consistency"] = round(holder_consistency(verified, chan_text, holders), 2) if verified else None
        row["holder_names"] = ";".join(holders[x] for x in verified)
        for rule in flags[0] if flags else rule_flags(""):
            row[f"share_{rule}"] = sum(1 for f in flags if f[rule] > 0) / n if vs else 0.0
        row["chan_desc_flags"] = ";".join(k for k, v in rule_flags(chan_text).items() if v)
        row["reg_status"] = reg_status({**row, "holder_consistency": row["holder_consistency"] if verified else 1.0})
        row["reg_resolution_rule"] = ""
        if row["reg_status"] == "review":
            row["reg_status"], row["reg_resolution_rule"] = auto_resolve(
                ch, verified, [holders[x] for x in verified], score)
        row["false_registration_claim"] = status == "reg_no_claimed_not_in_register"
        row["weak_label"] = weak_label({**row, "holder_consistency": row["holder_consistency"] if verified else 1.0})
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(PROC / "accounts.csv", index=False, encoding="utf-8")
    print(out.weak_label.value_counts(), "\n", out.match_status.value_counts())
    v = out[out.n_videos > 0]
    print("\nPRIMARY label, accounts with videos:\n", v.reg_status.value_counts().to_string())
    # review queue: only registration identity needs a human (no advice judgement)
    q = v[v.reg_status == "review"].copy()
    q["channel_url"] = "https://www.youtube.com/channel/" + q.channel_id
    q["decision (registered/unregistered)"] = ""
    q["notes"] = ""
    q[["channel_id", "channel_url", "title", "match_status", "claimed_reg_nos", "holder_names", "holder_consistency",
       "name_match_name", "name_match_reg_no", "name_match_score", "decision (registered/unregistered)", "notes"]].to_csv(
        PROC.parent / "annotation" / "registration_review_queue.csv", index=False, encoding="utf-8-sig")
    print(f"\nreview queue: {len(q)} accounts -> annotation/registration_review_queue.csv")


if __name__ == "__main__":
    main()
