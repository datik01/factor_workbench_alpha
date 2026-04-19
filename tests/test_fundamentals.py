
import pandas as pd
import numpy as np

# Directly test the foundational OHLCV + Fundamental fetcher
from tools import _fetch_single_ticker

def test_fetch_fundamental_injection():
    """
    Tests the standalone HTTP extraction of XBRL financials securely anchored 
    onto OHLCV dates via forward filling.
    """
    # Fetch AAPL over the last 3 years to guarantee fundamental filings exist
    df = _fetch_single_ticker("AAPL", "2021-01-01", "2024-01-01")
    
    # 1. Assert DataFrame returns successfully
    assert not df.empty, "DataFrame should not be empty."
    assert "date" in df.columns, "Date column must exist."
    assert "close" in df.columns, "Close column must exist."
    
    # 2. Assert Fundamental columns successfully injected
    fundamental_cols = ["pe_ratio", "pb_ratio", "ps_ratio", "eps", "equity", "revenues"]
    for col in fundamental_cols:
        assert col in df.columns, f"Fundamental column {col} is completely missing!"
        
    # 3. Assert Nans were correctly structurally filled
    for col in fundamental_cols:
        assert not df[col].isna().any(), f"Fundamental column {col} contains un-filled NaNs."
        
    # 4. Assert actual data populated securely (Values should not be strictly 0)
    # If the API limits successfully transferred, P/E ratio should be > 0 at the end of the history
    assert df["pe_ratio"].iloc[-1] > 0, "PE Ratio is zero! API likely failed to fetch or parse."
    assert df["revenues"].iloc[-1] > 0, "Revenues are zero! Fundamentals not injecting."
    assert df["eps"].iloc[-1] > 0, "EPS is zero! Formulas have no anchor."

    print("Fundamental Test Passed: ")
    print(df[['date', 'close', 'eps', 'pe_ratio', 'pb_ratio', 'revenues']].tail())

if __name__ == "__main__":
    test_fetch_fundamental_injection()
