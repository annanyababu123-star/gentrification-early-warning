"""
Download human-readable names for the ACS variables used by the model.

Usage (cmd):
    set CENSUS_API_KEY=your_key_here
    python fetch_labels.py
Output: feature_labels.csv   (variable code, label, concept)
"""
import os
import re

import pandas as pd
import requests

KEY = os.environ.get("CENSUS_API_KEY")
BASE = "https://api.census.gov/data/2011/acs/acs5"
URLS = [f"{BASE}/subject/variables.json", f"{BASE}/profile/variables.json", f"{BASE}/variables.json"]

rows = []
for url in URLS:
    r = requests.get(url, params={"key": KEY} if KEY else None, timeout=120)
    r.raise_for_status()
    for code, d in r.json()["variables"].items():
        if re.match(r"(S2502|S2503|DP03|B25085)_", code):
            rows.append({"code": code, "label": d.get("label", ""), "concept": d.get("concept", "")})
    print("ok", url, len(rows))

pd.DataFrame(rows).to_csv("feature_labels.csv", index=False)
print("Wrote feature_labels.csv with", len(rows), "variables")
