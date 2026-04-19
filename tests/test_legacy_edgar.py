import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from constituents.edgar_scraper import _load_all_holdings_accessions, extract_etf_holdings
from constituents.universe_builder import build_historical_constituents

def test_legacy_discovery():
    # 1. Assert discovery pulls all form types
    accessions = _load_all_holdings_accessions("0001100663")
    
    forms_discovered = set([form for date, acc, form in accessions])
    
    assert "NPORT-P" in forms_discovered, "Failed to discover modern NPORT SEC filings."
    assert "N-Q" in forms_discovered or "N-CSR" in forms_discovered, "Failed to discover legacy pre-2019 SEC filings."
    
    # 2. Test fallback html extraction on a legacy accession natively via Regex
    # We will find the very first N-Q filing mapped and run the CUSIP regex on it!
    legacy_acc = None
    for date, acc, form in accessions:
        if form == "N-Q":
            legacy_acc = acc
            break
            
    print(f"Isolated Legacy N-Q Accession: {legacy_acc}")
    
    # Actually run the BeautifulSoup CUSIP ripper fallback!
    # Because N-Q filings for iShares are monolithic, we expect hundreds or thousands of regex CUSIP hits
    if legacy_acc:
        df, rep_date = extract_etf_holdings(legacy_acc, "R2K")
        assert not df.empty, "Regex Legcay fallback failed to match CUSIP boundaries!"
        print(f"Successfully scraped {len(df)} fuzzy CUSIP proxy matches from archaic N-Q HTML!")
        print(df.head())
        
if __name__ == "__main__":
    test_legacy_discovery()
