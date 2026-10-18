"""Stage 8b - download the PDFs of candidate SEBI orders, extract text, flag social-media / unregistered-advice cases.

    python fetch_order_text.py

Reads processed/sebi_order_candidates.csv (scrape_sebi_orders.py), keeps orders whose title or keywords suggest
advice/recommendation/social-media cases, downloads each order PDF once (polite, 1.5 s apart), extracts text with pypdf
into data/finfluencer/raw/sebi_orders/<id>.txt (git-ignored) and writes processed/sebi_order_flags.csv.
Flags are regex-based triage only; relevance is confirmed by reading the orders. Names of persons/channels are never
published - only aggregate counts and the matching status.
"""
import os
import io
import re
import time

import pandas as pd
import requests
from pypdf import PdfReader

from common import PROC, RAW

UA = "Mozilla/5.0 (academic research; contact " + os.environ.get("CONTACT_EMAIL", "set CONTACT_EMAIL") + ")"
RELEVANT = re.compile(r"recommend|youtube|telegram|social media|investment advis|unregistered|research analyst|advisory|tips|"
                      r"academy|institute|education|finfluenc|influencer", re.I)
FLAGS = {
    "unregistered": r"not\s+(?:a\s+)?registered|unregistered|without\s+(?:obtaining\s+)?(?:a\s+)?(?:valid\s+)?(?:certificate\s+of\s+)?registration",
    "youtube": r"you\s*tube", "telegram": r"telegram", "instagram": r"instagram", "whatsapp": r"whatsapp",
    "twitter_x": r"\btwitter\b|\bX\.com\b|\bX\s*\(formerly", "social_media": r"social\s+media",
    "advice": r"investment\s+advice|investment\s+advis|research\s+analyst|recommendations?",
    "pump": r"pump|manipulat|FUTP|price\s+rigging|artificial\s+(?:rise|price)",
}


def main():
    cand = pd.read_csv(PROC / "sebi_order_candidates.csv")
    cand = cand[cand.title.str.contains(RELEVANT) | cand.keywords.str.contains("advis|unregistered|telegram|youtube|research analyst|tips|academy|education", regex=True)]
    cand = cand.drop_duplicates("url").reset_index(drop=True)
    print(len(cand), "relevant candidates")
    out_dir = RAW / "sebi_orders"
    out_dir.mkdir(exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = UA
    rows = []
    for i, r in cand.iterrows():
        mo = re.search(r"_(\d+)\.html", r.url)
        oid = mo.group(1) if mo else "u" + str(abs(hash(r.url)) % 10**9)
        f = out_dir / f"{oid}.txt"
        if not f.exists():
            try:
                page = s.get(r.url, timeout=60).text
                m = re.search(r"file=(https?://[^'\"&]+\.pdf)", page) or re.search(r"(https?://[^'\"]+\.pdf)", page)
                if not m:
                    f.write_text("", encoding="utf-8")
                else:
                    time.sleep(1.5)
                    pdf = s.get(m.group(1), timeout=120)
                    text = "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(pdf.content)).pages)
                    f.write_text(text, encoding="utf-8")
            except Exception as e:
                print("  fail", oid, type(e).__name__)
                f.write_text("", encoding="utf-8")
            time.sleep(1.5)
            if i % 10 == 0:
                print(f"  {i}/{len(cand)}", flush=True)
        text = f.read_text(encoding="utf-8")
        row = {"order_id": oid, "date": r.date, "title": r.title, "url": r.url, "n_chars": len(text)}
        for k, p in FLAGS.items():
            row["n_" + k] = len(re.findall(p, text, re.I))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(PROC / "sebi_order_flags.csv", index=False, encoding="utf-8")
    print(df.shape, "-> sebi_order_flags.csv")


if __name__ == "__main__":
    main()
