"""Shared paths, I/O helpers and the content rules used across the pipeline."""
import hashlib
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

for _s in (sys.stdout, sys.stderr):  # channel titles contain emoji/Devanagari; Windows consoles default to cp1252
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("FIN_DATA", ROOT.parent / "data" / "finfluencer"))
RAW, PROC, REG, ANN = DATA / "raw", DATA / "processed", DATA / "register", DATA / "annotation"
for _d in (RAW, PROC, REG, ANN):
    _d.mkdir(parents=True, exist_ok=True)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, rows):
    with Path(path).open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def pseudonym(value):
    """Salted hash for commenters. FIN_SALT must stay private and constant for a study."""
    salt = os.environ.get("FIN_SALT")
    if not salt:
        raise SystemExit("Set FIN_SALT (any long random string) before collecting user-level data.")
    return hashlib.sha256((salt + str(value)).encode()).hexdigest()[:16]


# ---------------------------------------------------------------- name handling
_STOP = {"pvt", "private", "ltd", "limited", "llp", "official", "channel", "india", "the", "and",
         "research", "advisory", "advisors", "advisers", "analyst", "analytics", "capital",
         "investments", "investment", "services", "solutions", "financial", "finance", "tv",
         "shri", "mr", "dr", "ca", "cs", "sir"}


def norm_name(s, drop_generic=True):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    toks = [t for t in s.split() if not (drop_generic and t in _STOP)]
    return " ".join(toks)


# ---------------------------------------------------------------- content rules
# Patterns cover English, romanised Hindi (Hinglish) and Devanagari. They encode
# SEBI-relevant behaviours and are used (i) to prioritise accounts for human coding and
# (ii) as weak labels. They are NOT model features in the main evaluation (see eval_models.py).
_R = lambda p: re.compile(p, re.I)
RULES = {
    # positive promise of returns; negated disclaimers ("does not guarantee returns") are removed first (_scrub)
    "guarantee": _R(r"guarante|100\s*%\s*(sure|accura|profit|return)|sure[\s-]?shot|pakka\s*profit|"
                    r"confirm(ed)?\s*(profit|return|call)|no\s*loss|risk[\s-]?free|double\s*your\s*money|"
                    r"गारंटी|पक्का\s*प्रॉफिट|100\s*%\s*पक्का"),
    # an actionable call: buy/sell with target/SL, explicit target/SL/entry levels, CMP, named call products.
    # Plain mentions of "stop loss"/"target" in educational titles do not count.
    "trade_call": _R(r"\b(buy|sell)\b.{0,40}\b(target|tgt|sl|stop\s*loss|stoploss)\b|"
                     r"\b(target|tgt)\s*[:\-@]\s*₹?\s*\d{2,}|\b(target|tgt)\s*₹\s*\d{2,}|\btgt\s*\d{2,}|"
                     r"\b(sl|stop\s*loss|stoploss)\s*[:\-@]\s*₹?\s*\d{2,}|\bCMP\b\s*[:\-@]?\s*₹?\s*\d{2,}|"
                     r"\bentry\s*[:\-@]\s*₹?\s*\d{2,}|\b(btst|stbt)\s+(call|stock|tip)s?\b|"
                     r"\b(intraday|option|jackpot|bank\s*nifty|nifty)\s*(tip|call)s?\b"),
    # SELLING access: paid/premium/VIP groups, fees, subscriptions. Bare telegram/whatsapp links are NOT counted
    # (nearly every channel has them, including free educational ones).
    "paid_solicitation": _R(r"(premium|vip|paid)\s*(group|channel|plan|community|member|membership|telegram|course|tips|calls|trades)|"
                            r"dm\s*(me\s*)?for\s*(price|fees?|plan|premium|details)|subscription\s*(plan|fee)|"
                            r"\bfees?\s*[:\-]?\s*(₹|rs\.?)|(₹|rs\.?)\s*\d{3,}\s*(/|per)\s*(month|mo|year|yr)"),
    "performance_claim": _R(r"\b\d{2,}\s*%\s*(return|profit|gain|accuracy|strike\s*rate)|"
                            r"(made|earned|booked)\s*(₹|rs\.?)\s*[\d,]{4,}|profit\s*proof|screenshot\s*of\s*profit"),
    "disclaimer": _R(r"not\s*sebi\s*registered|no[t]?\s*(a\s*)?(sebi\s*)?(registered|recommendation)|"
                     r"for\s*educational\s*purpose|educational\s*purposes?\s*only|not\s*investment\s*advice|"
                     r"disclaimer|do\s*your\s*own\s*research|\bDYOR\b|सेबी\s*पंजीकृत\s*नहीं"),
    # a POSITIVE claim of SEBI registration; "I am not SEBI registered" is removed first (_scrub)
    "claims_registration": _R(r"sebi[\s-]*(registered|reg\.?|regn?\.?)|registered\s*(research\s*analyst|investment\s*advis[eo]r)|"
                              r"\bIN[AH](?:\d{9}|IFSC\d{5})\b"),
}
# negations removed before matching the rules that they would otherwise invert
_NEG_GUARANTEE = _R(r"(?:\b(?:no|not|never|nor|cannot|can't|cant|without|do not|does not)\b|n't)[^.\n]{0,60}?guarante\w*"
                    r"|(?:do not|does not|don't|never)\s+provide[^.\n]{0,120}?guarante\w*"
                    r"|guarante\w*[^.\n]{0,30}?\b(?:not|nahi|no)\b|गारंटी[^.\n]{0,30}?(?:नहीं|नही)")
_NEG_PAID = _R(r"(?:\b(?:no|not|never|nor|without|free of)\b|n't)[^.\n]{0,60}?(?:premium|vip|paid)\s*\w+")
_NEG_REGISTRATION = _R(r"(?:\b(?:not|never|nor|no)\b|n't)[\s\w,&-]{0,25}?sebi[\s-]*regist\w*(?:\s+\w+){0,3}"
                       r"|sebi[\s-]*regist\w*[^.\n]{0,50}?\b(?:nahi|not)\b")
REG_NO = re.compile(r"\bIN[AH](?:\d{9}|IFSC\d{5})\b", re.I)  # IA = INA, RA = INH; IFSC entities: INAIFSC#####


def _scrub(rule, text):
    if rule == "guarantee":
        return _NEG_GUARANTEE.sub(" ", text)
    if rule == "claims_registration":
        return _NEG_REGISTRATION.sub(" ", text)
    if rule == "paid_solicitation":
        return _NEG_PAID.sub(" ", text)
    return text


def rule_flags(text):
    """Return {rule: number_of_matches} for a text string (negated disclaimers removed for guarantee/registration)."""
    text = text or ""
    return {k: len(p.findall(_scrub(k, text))) for k, p in RULES.items()}


def registration_numbers(text):
    return sorted({m.upper() for m in REG_NO.findall(text or "")})


# ---------------------------------------------------------------- regulatory-boilerplate scrubber
# Registered advisers MUST print certain wording (registration, enlistment, NISM, grievance, risk warnings).
# If a text model sees it, it identifies registration without learning anything behavioural, i.e. label leakage.
# Any sentence/line containing one of these cues is dropped before text features are built.
# Scrubber, in the three increments in which it was built (see the manuscript): L1 registration and mandatory-disclosure
# cues; L2 adds the standard "for illustration / not a recommendation / liability" wording found by inspecting the
# highest-weight terms (the version used for every main result); L3 adds warranty/accuracy/responsibility wording and is
# used only in the scrubbing-intensity experiment (nlp_experiments.py scrub).
BOILER_L1 = (r"sebi|nism|enlist|registr|regist\w+|research\s*analy|investment\s*advis|\bria\b|\braasb?\b|\bIN[AHZPB]\w{9}\b|"
             r"compliance|grievance|\bscores?\b|investor\s*charter|market\s*risks?|related\s*documents|disclaimer|disclosure|"
             r"\bbse\b|\bnse\b\s*(?:reg|member)|\bcin\b|\bgst|licen[cs]e|authori[sz]ed|रजिस्टर|पंजीकृत|सेबी")
BOILER_L2_ADD = r"illustrat|recommendatory|\bliab\w*|\bquoted\b|acting\s+on|not\s+(?:a\s+)?recommend\w*|solely\s+for|educational\s+purpose"
BOILER_L3_ADD = (r"warrant\w*|completeness|\baccuracy\b|reliab\w*|responsib\w*|obligation|past\s+performance|"
                 r"information\s+(?:is|are|provided)|for\s+information")
_BOILER = re.compile(BOILER_L1 + "|" + BOILER_L2_ADD, re.I)


def scrub_regulatory(text):
    """Drop every sentence/line that carries registration or mandatory-disclaimer wording."""
    parts = re.split(r"(?<=[.!?|])\s+|\n+", text or "")
    return " ".join(p for p in parts if p and not _BOILER.search(p))


def extract_regulatory(text):
    """The complement of scrub_regulatory: only the sentences/lines that carry registration or disclosure wording."""
    parts = re.split(r"(?<=[.!?|])\s+|\n+", text or "")
    return " ".join(p for p in parts if p and _BOILER.search(p))
