"""
constituents — Historical Russell 2000 Constituent Pipeline
Danny Atik - SYSEN 5381

Modules:
  edgar_scraper.py   — Fetch N-PORT filings from SEC EDGAR for iShares Trust
  cusip_mapper.py    — Map CUSIPs to historical tickers via Massive API
  universe_builder.py — Orchestrate the full pipeline and cache results
"""
