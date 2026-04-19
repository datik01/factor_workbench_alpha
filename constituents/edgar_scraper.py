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

def _load_all_holdings_accessions(cik: str) -> list:
    """
    Load ALL Holdings accession numbers for a given SEC CIK (NPORT-P, N-Q, N-CSR, N-CSRS).
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
        if forms[i] in ["NPORT-P", "N-Q", "N-CSR", "N-CSRS"]:
            all_nport.append((
                recent["filingDate"][i],
                recent["accessionNumber"][i],
                forms[i]
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
                if forms2[i] in ["NPORT-P", "N-Q", "N-CSR", "N-CSRS"]:
                    all_nport.append((d2["filingDate"][i], d2["accessionNumber"][i], forms2[i]))
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

    all_nport = _load_all_holdings_accessions(target["cik"])

    if progress_callback:
        progress_callback(0, len(all_nport), f"Scanning {len(all_nport)} filings for {target['short_name']} series...")

    # Fast-forward optimization for R2K: We already mapped the modern 20 Quarters (5 Years) locally!
    found = []
    
    if target["short_name"] == "IWM":
        known_accs = [
            ("2025-12-31", "0002071691-26-004226"), ("2025-09-30", "0002071691-25-007652"),
            ("2025-06-30", "0001752724-25-210405"), ("2025-03-31", "0001752724-25-119784"),
            ("2024-12-31", "0001752724-25-043851"), ("2024-09-30", "0001752724-24-269957"),
            ("2024-06-30", "0001752724-24-194120"), ("2024-03-31", "0001752724-24-123298"),
            ("2023-12-31", "0001752724-24-043096"), ("2023-09-30", "0001752724-23-264256"),
            ("2023-06-30", "0001752724-23-191317"), ("2023-03-31", "0001752724-23-123227"),
            ("2022-12-31", "0001752724-23-039260"), ("2022-09-30", "0001752724-22-268676"),
            ("2022-06-30", "0001752724-22-193728"), ("2022-03-31", "0001752724-22-122853"),
            ("2021-12-31", "0001752724-22-046410"), ("2021-09-30", "0001752724-21-255836"),
            ("2021-06-30", "0001752724-21-186233"), ("2021-03-31", "0001752724-21-116355")
        ]
        # Only inject if we are requesting historical limits
        for rd, ka in known_accs[:max_filings]:
            found.append({"accession": ka, "reporting_date": rd, "form_type": "NPORT-P"})
            
        if len(found) >= max_filings:
            return found

    checked = 0
    
    target_bounds = {
        "IWM": "2021-01-01",
        "QQQ": "2021-01-01",
        "IVV": "2023-09-01"
    }
    skip_bound = target_bounds.get(target["short_name"], "9999-99-99")

    for filing_date, acc, form_type in all_nport:
        if filing_date > skip_bound:
            # Bypass discovery for periods that exist in local parity shards!
            continue

        clean = acc.replace("-", "")
        
        try:
            if form_type == "NPORT-P":
                url = f"https://www.sec.gov/Archives/edgar/data/{target['cik']}/{clean}/primary_doc.xml"
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
                        "form_type": form_type,
                        "doc_url": url
                    }
                    found.append(entry)
            
            else:
                # N-Q, N-CSR, N-CSRS Legacy Discovery
                idx_url = f"https://www.sec.gov/Archives/edgar/data/{target['cik']}/{clean}/index.json"
                resp = requests.get(idx_url, headers=SEC_HEADERS, timeout=10)
                if resp.status_code == 200:
                    directory = resp.json().get("directory", {})
                    items = directory.get("item", [])
                    main_doc = None
                    for item in items:
                        if item["name"].lower().endswith((".htm", ".html", ".txt")):
                            main_doc = item["name"]
                            break
                    
                    if main_doc:
                        doc_url = f"https://www.sec.gov/Archives/edgar/data/{target['cik']}/{clean}/{main_doc}"
                        doc_resp = requests.get(doc_url, headers=SEC_HEADERS, timeout=15)
                        content = doc_resp.text
                        
                        if target["name_contains"].lower() in content.lower():
                            # Approximating holdings by regex'ing massive tables
                            approx_cusips = len(set(re.findall(r"\b[0-9A-Z]{9}\b", content)))
                            if approx_cusips > 50: # Legitimately contains a fund table
                                entry = {
                                    "accession": acc,
                                    "filing_date": filing_date,
                                    "reporting_date": filing_date, # Legacy doesn't always have repPdDate isolated
                                    "n_holdings": approx_cusips,
                                    "form_type": form_type,
                                    "doc_url": doc_url
                                }
                                found.append(entry)

            if found and progress_callback and found[-1]["accession"] == acc:
                progress_callback(
                    len(found), max_filings,
                    f"Found {target['short_name']} {form_type} filing: {found[-1]['reporting_date']} ({found[-1]['n_holdings']} est. holdings)"
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

    try:
        resp = requests.get(url, headers=SEC_HEADERS, timeout=15)
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
                holdings.append({
                    "issuer_name": name.text.strip(),
                    "cusip": cusip.text.strip() if cusip else None,
                    "value_usd": float(val.text) if val else None,
                    "shares": float(balance.text) if balance else None,
                    "pct_net_assets": float(pct.text) if pct else None,
                })
        
        if holdings:
            df = pd.DataFrame(holdings)
            if "cusip" in df.columns:
                df = df[df["cusip"].notna() & (df["cusip"] != "000000000") & (df["cusip"].str.len() >= 6)]
            df["reporting_date"] = reporting_date
            return df, reporting_date
            
    except requests.exceptions.RequestException:
        pass # Fallback to Legacy HTML Parsing

    # -----------------------------------------------------
    # Legacy HTML Regex Fallback (Pre-2019 N-Q & N-CSR)
    # -----------------------------------------------------
    idx_url = f"https://www.sec.gov/Archives/edgar/data/{target['cik']}/{clean}/index.json"
    idx_resp = requests.get(idx_url, headers=SEC_HEADERS, timeout=10)
    
    if idx_resp.status_code == 200:
        directory = idx_resp.json().get("directory", {})
        items = directory.get("item", [])
        
        main_doc = None
        for item in items:
            if item["name"].lower().endswith((".htm", ".html", ".txt")):
                main_doc = item["name"]
                break
                
        if main_doc:
            doc_url = f"https://www.sec.gov/Archives/edgar/data/{target['cik']}/{clean}/{main_doc}"
            doc_resp = requests.get(doc_url, headers=SEC_HEADERS, timeout=15)
            content = doc_resp.text
            
            # Use Regex to isolate explicit 9-character boundary CUSIPs organically from massive tables
            # Strict structural mapping: CUSIPs always begin with 3 numerics for US Equities.
            raw_cusips = set(re.findall(r"\b[0-9]{3}[0-9A-Z]{6}\b", content))
            
            if raw_cusips:
                df = pd.DataFrame([{"cusip": cusip, "issuer_name": "LEGACY_PROXY"} for cusip in raw_cusips])
                df["reporting_date"] = "LEGACY_" + clean
                return df, df["reporting_date"].iloc[0]

    # Return empty fallback
    return pd.DataFrame(), None
