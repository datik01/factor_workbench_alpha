import datetime
from constituents.universe_builder import get_latest_constituents
import tools

print("Compiling active ticker list...")
# We will pull the master universe which contains elements over time
r2k = get_latest_constituents("R2K")
sp500 = get_latest_constituents("SP500")
ndx = get_latest_constituents("NDX")

tickers = list(set(r2k + sp500 + ndx))
print(f"Targeting {len(tickers)} unique tickers across all 3 universes for full cache extraction.")

def cb(curr, total, ticker, msg):
    print(f"[{curr}/{total}] {msg}")

# Force refresh to overwrite the parquet file with the new dictionary schema 
tools.fetch_universe_data(
    tickers=tickers,
    start_year=2021,
    end_year=datetime.datetime.now().year,
    progress_callback=cb,
    force_refresh=True
)
print("\nExtraction complete! Parquet cache safely overwritten.")
