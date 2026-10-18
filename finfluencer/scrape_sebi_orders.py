"""Stage 8a - find SEBI enforcement orders about unregistered advisory / finfluencers (candidate list).

    python scrape_sebi_orders.py --since 2021

The SEBI orders listing (enforcement/orders) has a title-keyword search per order category. We run a list of keywords
across all categories, keep orders dated >= --since, and write data/finfluencer/processed/sebi_order_candidates.csv.
Titles alone rarely say who is a finfluencer ("In the matter of <name>"), so a second step (fetch_order_text.py) reads
the order pages and flags social-media / unregistered-advice language.
Polite: one request per ~1.5 s, same session/headers a browser sends, stops on any non-200 instead of retrying.
Only public orders; names of persons are NOT to be published in the paper (aggregate results only).
"""
import os
import argparse
import html
import re
import time
from datetime import datetime

import pandas as pd
import requests

from common import PROC

B = "https://www.sebi.gov.in"
UA = "Mozilla/5.0 (academic research; contact " + os.environ.get("CONTACT_EMAIL", "set CONTACT_EMAIL") + ")"
KEYWORDS = ["investment adviser", "investment advisory", "unregistered", "research analyst", "advisory", "finfluenc",
            "academy", "institute", "education", "classes", "telegram", "youtube", "tips", "trading", "stock", "chart",
            "market guru", "learning", "wealth", "research"]
ROW = re.compile(r"<td[^>]*>\s*([A-Za-z]{3} \d{1,2}, \d{4})\s*</td>\s*<td[^>]*>\s*<a[^>]*href=['\"]([^'\"]+)['\"][^>]*title=['\"]([^'\"]*)['\"]", re.S)


def category_ids(sess):
    t = sess.get(B + "/enforcement/orders.html", timeout=60).text
    return {int(m.group(1)): html.unescape(m.group(2)) for m in
            re.finditer(r"smid=(\d+)'[^>]*><span>([^<]+)</span>", t)}


def search(sess, smid, smname, kw, page_cache):
    """Return all (date, url, title) rows for a keyword in one category (all pages)."""
    listing = B + f"/sebiweb/home/HomeAction.do?doListing=yes&sid=2&ssid=9&smid={smid}"
    f = sess.get(listing, timeout=60)
    f.raise_for_status()
    headers = {"Referer": f.url, "X-Requested-With": "XMLHttpRequest", "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"}
    out, page = [], -1
    while True:
        d = {"nextValue": max(page + 1, 1), "next": "s" if page < 0 else "n", "search": kw, "fromDate": "", "toDate": "", "fromYear": "",
             "toYear": "", "deptId": "", "sid": 2, "ssid": 9, "smid": smid, "ssidhidden": 9, "intmid": -1, "sText": "Enforcement",
             "ssText": "Orders", "smText": smname, "doDirect": page}
        r = sess.post(B + "/sebiweb/ajax/home/getnewslistinfo.jsp", data=d, headers=headers, timeout=60)
        if r.status_code != 200:
            print("  stop: HTTP", r.status_code)
            break
        t = r.text.split("#@#")[0]
        rows = ROW.findall(t)
        out += rows
        m = re.search(r"(\d+) to (\d+) of (\d+) records", t)
        total = int(m.group(3)) if m else len(rows)
        time.sleep(1.5)
        if not rows or len(out) >= total:
            break
        page = page + 1 if page >= 0 else 1
    return out


def main(a):
    sess = requests.Session()
    sess.headers["User-Agent"] = UA
    cats = {k: v for k, v in category_ids(sess).items() if k in a.categories}
    print("categories:", cats)
    seen = {}
    for smid, name in cats.items():
        for kw in KEYWORDS:
            rows = search(sess, smid, name, kw, None)
            new = 0
            for dt_, url, title in rows:
                dt = datetime.strptime(dt_, "%b %d, %Y")
                if dt.year < a.since:
                    continue
                if url not in seen:
                    seen[url] = {"date": dt.date().isoformat(), "category": name, "title": html.unescape(title).strip(),
                                 "url": url if url.startswith("http") else B + url, "keywords": kw}
                    new += 1
                else:
                    seen[url]["keywords"] += "|" + kw
            print(f"  [{name[:28]}] {kw!r}: {len(rows)} rows, {new} new (total {len(seen)})", flush=True)
    df = pd.DataFrame(seen.values()).sort_values("date", ascending=False)
    df.to_csv(PROC / "sebi_order_candidates.csv", index=False, encoding="utf-8")
    print(len(df), "candidate orders ->", PROC / "sebi_order_candidates.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=int, default=2021)
    ap.add_argument("--categories", type=int, nargs="+", default=[2])  # 2 = Orders of Chairperson/Members (final/interim orders)
    main(ap.parse_args())
