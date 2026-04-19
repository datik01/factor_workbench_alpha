import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from constituents.universe_builder import get_latest_constituents
from tools import fetch_universe_data
from factor_miner import discover_alpha_factors

def run():
    print("Fetching universe...")
    # Get 10 tickers as proxy
    tickers = get_latest_constituents("R2K")[:10]
    df = fetch_universe_data(tickers, 2024, 2025, force_refresh=False)
    
    print("Running Miner...")
    discover_alpha_factors(
        df,
        generations=2,
        pop_size=50,
        horizon=63,
        fitness_metric="ic",
        syntax_set="all"
    )

if __name__ == "__main__":
    run()
