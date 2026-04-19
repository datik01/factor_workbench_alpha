import sys
import os

# Add root directory to python path implicitly
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from constituents.universe_builder import build_historical_constituents

def _progress(curr, total, msg):
    # Print securely to terminal during the long SEC crawl
    print(f"[{curr}/{total}] {msg}")

def main():
    print("==================================================")
    print(" INITIALIZING 20-YEAR SEC EDGAR WEB CRAWLER ")
    print(" Target: Russell 2000 (R2K) / iShares Trust ")
    print(" Bounds: 80 Quarters (Approx. 2005-2025) ")
    print("==================================================")
    
    print("==================================================")
    print(" CRAWLING S&P 500 (SP500) ")
    print("==================================================")
    df_sp500 = build_historical_constituents(
        etf_key="SP500",
        max_filings=80, 
        use_known=False,
        force_refresh=True,
        progress_callback=_progress
    )
    
    print("\n==================================================")
    print(" CRAWLING NASDAQ 100 (NDX) ")
    print("==================================================")
    df_ndx = build_historical_constituents(
        etf_key="NDX",
        max_filings=80, 
        use_known=False,
        force_refresh=True,
        progress_callback=_progress
    )
    
    print("\n==================================================")
    print(" MULTI-INDEX CRAWL COMPLETE AND PARQUET SHARDS CACHED ")
    print("==================================================")
    print(f"Total Unique Tickers Bound (SP500): {df_sp500['ticker'].nunique()}")
    print(f"Total Unique Tickers Bound (NDX): {df_ndx['ticker'].nunique()}")
    print("==================================================")

if __name__ == "__main__":
    main()
