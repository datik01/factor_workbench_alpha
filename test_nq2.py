import requests

SEC_HEADERS = {"User-Agent": "Danny Atik da494@cornell.edu", "Accept-Encoding": "gzip, deflate"}
url = "https://data.sec.gov/submissions/CIK0001100663.json"
resp = requests.get(url, headers=SEC_HEADERS).json()

filings = resp.get("filings", {})
for ef in reversed(filings.get("files", [])):
    url2 = f"https://data.sec.gov/submissions/{ef['name']}"
    d2 = requests.get(url2, headers=SEC_HEADERS).json()
    forms2 = d2.get("form", [])
    dates2 = d2.get("filingDate", [])
    for i in range(len(forms2)):
        if forms2[i] in ["N-Q", "N-CSR", "NPORT-P"]:
            print(f"Oldest Form Found: Date: {dates2[i]}, Form: {forms2[i]}")
    break # Just look at the oldest paginated block
