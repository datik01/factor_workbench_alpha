import sys
import os

# Ensure we can import from the application directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from constituents.universe_builder import build_historical_constituents

def progress(c, t, msg):
    print(f'  [{c}/{t}] {msg}', flush=True)

TARGETS = ["R2K", "SP500", "NDX"]

print('Building full multi-universe point-in-time constituent pipelines...', flush=True)

for target in TARGETS:
    print(f'\n======================================================')
    print(f'🚀 Building SEC Historical Timeline: {target}')
    print(f'======================================================\n')
    try:
        df = build_historical_constituents(
            etf_key=target,
            max_filings=20, 
            use_known=True, # Will only apply to R2K natively inside builder
            progress_callback=progress,
            force_refresh=False
        )
        
        print(f'\nResult ({target}): {df.shape}', flush=True)
        print(f'Unique tickers overall: {df["ticker"].nunique()}', flush=True)
        print(f'Sample:')
        print(df[["reporting_date", "issuer_name", "cusip", "ticker"]].head(3).to_string(), flush=True)
        
    except Exception as e:
        print(f'Error during build for {target}: {e}')

print('\nComplete! All specified SEC Universes have been structurally cached locally.', flush=True)
