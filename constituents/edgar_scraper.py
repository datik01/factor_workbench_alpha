"""
edgar_scraper.py
Fetch historical IWM (Russell 2000 ETF) holdings from SEC EDGAR N-PORT filings.

The iShares Russell 2000 ETF (IWM) is filed under the iShares Trust umbrella:
  - CIK: 0001100663 (iShares Trust)
  - Series ID: S000004344 (IWM — corrected from originally published S000004212)
  - Series Name: "iShares Russell 2000 ETF"

Each iShares Trust NPORT-P filing covers ONE specific ETF.
We scan the primary_doc.xml for seriesId == S000004344 or
seriesName containing "Russell 2000 ETF" (excluding BuyWrite/Growth/Value).
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import re


# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

ETF_TARGETS = {
    "R2K": {
        "cik": "0001100663",
        "series_id": "S000004344",
        "name_contains": "Russell 2000 ETF",
        "short_name": "IWM"
    },
    "SP500": {
        "cik": "0001100663",
        "series_id": "S000004310",
        "name_contains": "S&P 500 ETF",
        "short_name": "IVV"
    },
    "NDX": {
        "cik": "0001067839",
        "series_id": "S000101292",
        "name_contains": "QQQ",
        "short_name": "QQQ"
    }
}

# SEC requires a legitimate User-Agent with contact info
SEC_HEADERS = {
    "User-Agent": "Danny Atik da494@cornell.edu",
    "Accept-Encoding": "gzip, deflate",
}


# ═══════════════════════════════════════════════════════════════
# Step 1: Discover IWM filing accession numbers
# ═══════════════════════════════════════════════════════════════

def _load_all_nport_accessions(cik: str) -> list:
    """
    Load ALL NPORT-P accession numbers for a given SEC CIK.
    The submissions JSON has a 'recent' block plus additional paginated files.

    Returns list of (filing_date_str, accession_number)
    """
    base_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(base_url, headers=SEC_HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    all_nport = []

    # Recent filings
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    for i in range(len(forms)):
        if forms[i] == "NPORT-P":
            all_nport.append((
                recent["filingDate"][i],
                recent["accessionNumber"][i],
            ))

    # Additional paginated files
    extra_files = data.get("filings", {}).get("files", [])
    for ef in extra_files:
        fname = ef.get("name", "")
        if not fname:
            continue
        url2 = f"https://data.sec.gov/submissions/{fname}"
        try:
            resp2 = requests.get(url2, headers=SEC_HEADERS, timeout=30)
            d2 = resp2.json()
            forms2 = d2.get("form", [])
            for i in range(len(forms2)):
                if forms2[i] == "NPORT-P":
                    all_nport.append((d2["filingDate"][i], d2["accessionNumber"][i]))
        except Exception:
            pass
        time.sleep(0.15)

    return all_nport


def discover_etf_filings(
    etf_key: str = "R2K",
    max_filings: int = 20,
    progress_callback=None,
) -> list:
    """
    Scan EDGAR NPORT-P filings to find those belonging to a specific ETF series.

    Parameters
    ----------
    max_filings : int
        Maximum number of IWM filings to discover
    progress_callback : callable(current, total, msg)

    Returns
    -------
    list of dict
        [{"accession": str, "filing_date": str, "reporting_date": str, "n_holdings": int}]
    """
    target = ETF_TARGETS.get(etf_key)
    if not target:
        raise ValueError(f"Unknown ETF key: {etf_key}")

    if progress_callback:
        progress_callback(0, 0, f"Loading Trust filing index from SEC EDGAR for CIK {target['cik']}...")

    all_nport = _load_all_nport_accessions(target["cik"])

    if progress_callback:
        progress_callback(0, len(all_nport), f"Scanning {len(all_nport)} filings for {target['short_name']} series...")

    found = []
    checked = 0

    for filing_date, acc in all_nport:
        clean = acc.replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{target['cik']}/{clean}/primary_doc.xml"

        try:
            resp = requests.get(url, headers=SEC_HEADERS, timeout=15)
            soup = BeautifulSoup(resp.text, "lxml-xml")

            sid = soup.find("seriesId")
            sname = soup.find("seriesName")

            is_match = False
            if sid and sid.text.strip() == target["series_id"]:
                is_match = True
            elif sname and target["name_contains"].lower() in sname.text.lower() and "buywrite" not in sname.text.lower():
                is_match = True

            if is_match:
                rep_date = soup.find("repPdDate")
                n_holdings = len(soup.find_all("invstOrSec"))
                entry = {
                    "accession": acc,
                    "filing_date": filing_date,
                    "reporting_date": rep_date.text.strip() if rep_date else filing_date,
                    "n_holdings": n_holdings,
                }
                found.append(entry)

                if progress_callback:
                    progress_callback(
                        len(found), max_filings,
                        f"Found {target['short_name']} filing: {entry['reporting_date']} ({n_holdings} holdings)"
                    )

                if len(found) >= max_filings:
                    break

        except Exception:
            pass

        checked += 1
        time.sleep(0.12)  # SEC rate limit (10 req/s)

        if progress_callback and checked % 100 == 0:
            progress_callback(len(found), max_filings, f"Scanned {checked} filings, found {len(found)} {target['short_name']}...")

    if progress_callback:
        progress_callback(len(found), len(found), f"Found {len(found)} {target['short_name']} N-PORT filings")

    return found


# ═══════════════════════════════════════════════════════════════
# Step 2: Extract holdings from a single IWM filing
# ═══════════════════════════════════════════════════════════════

def extract_etf_holdings(accession: str, etf_key: str = "R2K") -> tuple:
    """
    Download and parse a single ETF N-PORT filing.
    Extract all holdings with CUSIPs.

    Parameters
    ----------
    accession : str
        SEC accession number

    Returns
    -------
    tuple of (pd.DataFrame, str)
        Holdings DataFrame and reporting date string
    """
    target = ETF_TARGETS.get(etf_key)
    clean = accession.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{target['cik']}/{clean}/primary_doc.xml"

    resp = requests.get(url, headers=SEC_HEADERS, timeout=120)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml-xml")

    rep_date_tag = soup.find("repPdDate")
    reporting_date = rep_date_tag.text.strip() if rep_date_tag else None

    holdings = []
    for inv in soup.find_all("invstOrSec"):
        name = inv.find("name")
        cusip = inv.find("cusip")
        val = inv.find("valUSD")
        balance = inv.find("balance")
        pct = inv.find("pctVal")

        if name:
            record = {
                "issuer_name": name.text.strip(),
                "cusip": cusip.text.strip() if cusip else None,
                "value_usd": float(val.text) if val else None,
                "shares": float(balance.text) if balance else None,
                "pct_net_assets": float(pct.text) if pct else None,
            }
            holdings.append(record)

    df = pd.DataFrame(holdings)

    # Filter out invalid CUSIPs
    if "cusip" in df.columns:
        df = df[df["cusip"].notna() & (df["cusip"] != "000000000") & (df["cusip"].str.len() >= 6)]

    df["reporting_date"] = reporting_date

    return df, reporting_date
