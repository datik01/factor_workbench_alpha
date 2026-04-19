import time
import os
if os.path.exists(".env"):
    with open(".env", "r") as f:
        for line in f:
            if "=" in line and not line.startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ[k] = v
from constituents.universe_builder import get_latest_constituents
from tools import portfolio_report

def main():
    print("Fetching subset tickers...")
    tickers = get_latest_constituents("R2K")
    
    t0 = time.time()
    last_t = t0
    
    def my_cb(pct, target, err, msg):
        nonlocal last_t
        curr_t = time.time()
        print(f"[{curr_t - t0:.2f}s overall | +{curr_t - last_t:.2f}s step] {pct}% - {msg}")
        last_t = curr_t
        
    print("Running portfolio report benchmark...")
    res = portfolio_report(
        tickers=tickers,
        themes=["Momentum"],
        start_year=2020,
        end_year=2024,
        portfolio_sizing_type="Absolute Count",
        portfolio_size=100.0,
        strategy_type="Long/Short",
        quantiles=10,
        rebalance_freq="W",
        invert_factor=False,
        progress_callback=my_cb,
        custom_formula="sma_20(rsi_14(Close))",
        benchmark_ticker="IWM",
        constituent_timeline=None
    )
    
    import json
    parsed = json.loads(res)
    if not parsed.get("success"):
        print("ERROR:", parsed.get("error"))
    else:
        print(f"Benchmark finished successfully in {time.time() - t0:.2f}s!")

if __name__ == "__main__":
    main()
