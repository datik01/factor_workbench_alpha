import requests

SEC_HEADERS = {"User-Agent": "Danny Atik da494@cornell.edu", "Accept-Encoding": "gzip, deflate"}
url = "https://data.sec.gov/submissions/CIK0001100663.json"
resp = requests.get(url, headers=SEC_HEADERS).json()

filings = resp.get("filings", {})
recent = filings.get("recent", {})
forms = recent.get("form", [])
dates = recent.get("filingDate", [])
acc = recent.get("accessionNumber", [])

nq_found = 0
for i in range(len(forms)):
    if forms[i] in ["N-Q", "N-CSR", "N-CSRS", "NPORT-P"]:
        print(f"Date: {dates[i]}, Form: {forms[i]}, Acc: {acc[i]}")
        nq_found += 1
        if nq_found > 10: break

print("\nChecking older paginated files...")
for ef in filings.get("files", []):
    url2 = f"https://data.sec.gov/submissions/{ef['name']}"
    d2 = requests.get(url2, headers=SEC_HEADERS).json()
    forms2 = d2.get("form", [])
    dates2 = d2.get("filingDate", [])
    for i in range(len(forms2)):
        if forms2[i] in ["N-Q", "N-CSR", "N-CSRS"]:
            print(f"OLDER Date: {dates2[i]}, Form: {forms2[i]}")
            nq_found += 1
            if nq_found > 20: break
    if nq_found > 20: break
