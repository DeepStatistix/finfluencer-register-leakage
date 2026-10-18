"""Stage 2 - build the ground-truth register of SEBI-authorized advisers.

Download the lists from SEBI (Intermediaries -> "Investment Advisers" and "Research Analysts",
or the SEBI Check portal, https://check.sebi.gov.in) and save each as CSV in data/finfluencer/register/.
I could not verify the exact export layout, so column names are auto-detected:
  a column containing 'name' and a column containing 'regist' (the INA/INH number).
Optional: a column containing 'status'/'valid' (to drop expired or suspended registrations).

    python build_register.py --snapshot-date 2026-10-16

The snapshot date matters: label = "registered at observation time". Re-download the register at
the end of data collection and report both snapshots; registrations lapse and are cancelled.
"""
import argparse
import re

import pandas as pd

from common import REG, norm_name


def load_one(path):
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
    cols = {c.lower(): c for c in df.columns}
    name = next((cols[c] for c in cols if "name" in c), None)
    regno = next((cols[c] for c in cols if "regist" in c and "date" not in c), None)
    status = next((cols[c] for c in cols if "status" in c or "valid" in c), None)
    if not name or not regno:
        raise SystemExit(f"{path.name}: cannot find name/registration columns in {list(df.columns)}")
    out = pd.DataFrame({"name": df[name].str.strip(), "reg_no": df[regno].str.upper().str.extract(r"(IN[AH](?:\d{9}|IFSC\d{5}))")[0]})
    out["status"] = df[status].str.strip().str.lower() if status else "unknown"
    out["type"] = out.reg_no.str[:3].map({"INA": "investment_adviser", "INH": "research_analyst"})
    out["source_file"] = path.name
    return out.dropna(subset=["reg_no"])


def main(a):
    files = [p for p in REG.glob("*.csv") if p.name != "register.csv"]
    if not files:
        raise SystemExit(f"Put SEBI register CSV exports in {REG}")
    reg = pd.concat([load_one(p) for p in files], ignore_index=True).drop_duplicates("reg_no")
    reg["norm"] = reg.name.map(norm_name)
    reg["snapshot_date"] = a.snapshot_date
    reg["active"] = ~reg.status.str.contains("cancel|expire|surrender|suspend|lapse|inactive", regex=True)
    reg.to_csv(REG / "register.csv", index=False, encoding="utf-8")
    # search terms for finding the channels of registered advisers (positive class); 100 quota units each
    top = reg[reg.active & (reg.norm.str.split().str.len() >= 2)].name
    (REG / "register_queries.txt").write_text("\n".join(top.head(a.n_queries)), encoding="utf-8")
    print(reg.groupby(["type", "active"]).size())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot-date", required=True)
    ap.add_argument("--n-queries", type=int, default=300)
    main(ap.parse_args())
