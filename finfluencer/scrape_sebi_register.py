"""Stage 2a - download SEBI's public register of intermediaries (no export button exists on the site).

    python scrape_sebi_register.py --intm-id 13 --name investment_advisers      # IA, INA...
    python scrape_sebi_register.py --intm-id <id> --name research_analysts      # RA, INH... (id: see README)

NOTE: the site's AJAX pagination endpoint answered "530 BLOCKED" to scripted requests, and we do not
try to circumvent that. Fallback that always works: open the page in your browser, click "Show All
Records" (or step through the pages), save each page as HTML (Ctrl+S, "Webpage, HTML only"), then
    python scrape_sebi_register.py --from-html page1.html page2.html ... --name investment_advisers

Source page: https://www.sebi.gov.in/sebiweb/other/OtherAction.do?doRecognisedFpi=yes&intmId=13
The page paginates through a POST to /sebiweb/ajax/other/getintmfpiinfo.jsp (25 records per page). We
send one request per page with a pause, and keep ONLY name, registration number and validity. E-mail,
phone, address and contact-person fields are deliberately dropped (personal data, not needed).
Writes data/finfluencer/register/<name>.csv in the layout build_register.py expects, and prints the
"as on" date of the register, which is the snapshot date to pass to build_register.py.
"""
import os
import argparse
import html
from pathlib import Path
import re
import time

import pandas as pd
import requests

from common import REG

BASE = "https://www.sebi.gov.in"
UA = "Mozilla/5.0 (academic research; contact " + os.environ.get("CONTACT_EMAIL", "set CONTACT_EMAIL") + ")"
PAIR = re.compile(r"""<div class=["']title["']><span>(.*?)</span></div><div class=["']value[^"']*["']><span>(.*?)</span>""", re.S)


def parse(page_html):
    rows, cur = [], None
    for k, v in PAIR.findall(page_html):
        k, v = html.unescape(k).strip(), re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", v))).strip()
        if k == "Name":
            cur = {"Name": v}
            rows.append(cur)
        elif cur is not None and k in ("Registration No.", "Validity"):
            cur[k] = v
    return rows


def main(a):
    if a.from_html:
        rows = []
        for f in a.from_html:
            got = parse(open(f, encoding="utf-8", errors="replace").read())
            print(f, len(got))
            rows += got
        df = pd.DataFrame(rows).drop_duplicates("Registration No.")
        out = REG / f"{a.name}.csv"
        df.to_csv(out, index=False, encoding="utf-8")
        print("wrote", out, len(df))
        return
    s = requests.Session()
    s.headers["User-Agent"] = UA
    first = s.get(f"{BASE}/sebiweb/other/OtherAction.do", params={"doRecognisedFpi": "yes", "intmId": a.intm_id}, timeout=60)
    first.raise_for_status()
    # send what the page's own script sends (same session cookie, Referer, XHR header); nothing more
    s.headers.update({"Referer": first.url, "X-Requested-With": "XMLHttpRequest",
                      "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"})
    m = re.search(r"as on date\s*([A-Za-z]{3} \d{1,2}, \d{4})", first.text)
    total = re.search(r"of\s+(\d+)\s+records", first.text)
    last = re.search(r"searchFormFpi\('n','(\d+)'\);\"\s+title=\"Last\"", first.text)
    n_pages = int(last.group(1)) + 1 if last else 1  # 'Last' link holds the zero-based index
    print(f"intmId={a.intm_id}: register as on {m.group(1) if m else '?'}, {total.group(1) if total else '?'} records, {n_pages} pages")
    cache = Path(a.cache_dir) if a.cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
        (cache / f"intm{a.intm_id}_p0.html").write_text(first.text, encoding="utf-8")
    rows = parse(first.text)
    for p in range(1, n_pages):  # zero-based page index; page 0 came with the first GET
        time.sleep(a.pause)
        r = s.post(f"{BASE}/sebiweb/ajax/other/getintmfpiinfo.jsp", timeout=60, data={
            "nextValue": p, "next": "n", "intmId": a.intm_id, "contPer": "", "name": "", "regNo": "", "email": "",
            "location": "", "exchange": "", "affiliate": "", "alp": "", "doDirect": p, "intmIds": ""})
        r.raise_for_status()
        if cache:
            (cache / f"intm{a.intm_id}_p{p}.html").write_text(r.text, encoding="utf-8")
        got = parse(r.text)
        print(f"page {p + 1}/{n_pages}: {len(got)}")
        if not got:
            raise SystemExit("empty page - stopping rather than hammering the server")
        rows += got
    print("rows before de-duplication:", len(rows))
    df = pd.DataFrame(rows).drop_duplicates("Registration No.")
    if total and len(df) != int(total.group(1)):
        print(f"WARNING: parsed {len(df)} unique records but the page says {total.group(1)}; check pagination")
    out = REG / f"{a.name}.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print("wrote", out, len(df))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--intm-id", type=int, default=13)
    ap.add_argument("--from-html", nargs="+")
    ap.add_argument("--name", required=True)
    ap.add_argument("--cache-dir", help="also save raw pages here (for debugging; do not commit)")
    ap.add_argument("--pause", type=float, default=1.5)
    main(ap.parse_args())
