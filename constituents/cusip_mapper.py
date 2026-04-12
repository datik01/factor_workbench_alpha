"""
cusip_mapper.py
Map CUSIPs to historical ticker symbols using the Massive API.

Uses the GET /v3/reference/tickers endpoint with:
  - cusip: The 9-char CUSIP from the SEC filing
  - date: The filing date (point-in-time lookup)

Critical note from the Massive API docs:
  "Although you can query by CUSIP, due to legal reasons we do not
   return the CUSIP in the response."
  → We must iterate one-by-one and track the mapping ourselves.
"""

import requests
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

MASSIVE_TICKERS_URL = "https://api.polygon.io/v3/reference/tickers"


def map_single_cusip(cusip: str, filing_date: str, api_key: str) -> dict:
    """
    Query the Massive API to find the ticker for a CUSIP on a given date.

    Parameters
    ----------
    cusip : str
        9-character CUSIP identifier
    filing_date : str
        Date string (YYYY-MM-DD) for point-in-time lookup
    api_key : str
        Massive API key

    Returns
    -------
    dict
        {"cusip": str, "ticker": str, "name": str} or None on failure
    """
    params = {
        "cusip": cusip,
        "date": filing_date,
        "market": "stocks",
        "active": "true",
        "limit": 1,
        "apiKey": api_key,
    }

    try:
        resp = requests.get(MASSIVE_TICKERS_URL, params=params, timeout=10)
        data = resp.json()

        results = data.get("results", [])
        if results:
            return {
                "cusip": cusip,
                "ticker": results[0].get("ticker", ""),
                "name": results[0].get("name", ""),
                "primary_exchange": results[0].get("primary_exchange", ""),
                "type": results[0].get("type", ""),
            }
    except Exception:
        pass

    return None


def map_cusips_to_tickers(
    holdings_df: pd.DataFrame,
    filing_date: str,
    api_key: str,
    max_workers: int = 10,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Map a DataFrame of SEC holdings (with CUSIP column) to historical tickers.

    Parameters
    ----------
    holdings_df : pd.DataFrame
        Must contain a 'cusip' column
    filing_date : str
        Date string for point-in-time CUSIP resolution
    api_key : str
        Massive API key
    max_workers : int
        Number of concurrent API threads
    progress_callback : callable(current, total, msg)
        Optional UI progress function

    Returns
    -------
    pd.DataFrame
        Original DataFrame joined with 'ticker', 'mapped_name', 'primary_exchange'
    """
    if "cusip" not in holdings_df.columns:
        raise ValueError("holdings_df must contain a 'cusip' column")

    # Deduplicate CUSIPs for efficient mapping
    unique_cusips = holdings_df["cusip"].dropna().unique().tolist()
    total = len(unique_cusips)

    if progress_callback:
        progress_callback(0, total, f"Mapping {total} CUSIPs for {filing_date}...")

    mapped = {}
    completed = 0
    failed = 0

    def _worker(cusip):
        return cusip, map_single_cusip(cusip, filing_date, api_key)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker, c): c for c in unique_cusips}

        for future in as_completed(futures):
            cusip_key = futures[future]
            try:
                cusip, result = future.result()
                if result:
                    mapped[cusip] = result
                else:
                    failed += 1
            except Exception:
                failed += 1

            completed += 1
            if progress_callback and completed % 50 == 0:
                progress_callback(completed, total, f"📡 {completed}/{total} CUSIPs mapped ({failed} unmapped)")

    if progress_callback:
        progress_callback(total, total, f"✅ {total - failed}/{total} CUSIPs resolved")

    # Build mapping DataFrame
    mapping_df = pd.DataFrame(mapped.values())

    if mapping_df.empty:
        holdings_df["ticker"] = None
        return holdings_df

    # Merge back onto holdings
    result_df = holdings_df.merge(
        mapping_df[["cusip", "ticker", "name"]].rename(columns={"name": "mapped_name"}),
        on="cusip",
        how="left",
    )

    return result_df
