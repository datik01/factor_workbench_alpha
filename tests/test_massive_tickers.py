import os
from massive import RESTClient

api_key = os.getenv("MASSIVE_API_KEY")
if not api_key:
    from dotenv import load_dotenv
    load_dotenv(".env")
    api_key = os.getenv("MASSIVE_API_KEY")

client = RESTClient(api_key=api_key)
# let's try getting ticker details for AAPL
try:
    details = client.get_ticker_details("AAPL")
    print("AAPL Market Cap:", getattr(details, 'market_cap', None) or dict(details).get('market_cap'))
except Exception as e:
    print("Error:", e)
