"""
Download California census-tract ACS 5-year data (2010-2016) for the
gentrification early-warning mini-project.

Usage:
    pip install requests pandas
    export CENSUS_API_KEY=your_key      (Windows PowerShell: $env:CENSUS_API_KEY="your_key")
    python fetch_census.py

Output (in ./data):
    raw/{year}_{table}.csv     one file per table per year (estimates only)
    acs_{year}.csv             all tables merged per year, keyed by GEOID
    tl_2016_06_tract.zip       TIGER tract boundaries (for queen adjacency)
    fetch_log.txt              what succeeded / failed
"""
import os
import re
import sys
import time
import zipfile

import pandas as pd
import requests

KEY = os.environ.get("CENSUS_API_KEY")
if not KEY:
    sys.exit("Set CENSUS_API_KEY first (see top of file).")

STATE = "06"  # California
YEARS = range(2010, 2017)  # ACS 5-year releases 2010..2016
# (table id, endpoint suffix)
TABLES = [
    ("S2502", "subject"),
    ("S2503", "subject"),
    ("DP03", "profile"),
    ("B25085", ""),  # detailed table; may not exist in every year
]
OUT = "data"
RAW = os.path.join(OUT, "raw")
os.makedirs(RAW, exist_ok=True)
LOG = open(os.path.join(OUT, "fetch_log.txt"), "w")


def log(msg):
    print(msg)
    LOG.write(msg + "\n")
    LOG.flush()


def get_json(url, params, retries=3):
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            err = f"HTTP {r.status_code}: {r.text[:150]!r}"
        except requests.RequestException as e:
            err = str(e)
        time.sleep(2 * (i + 1))
    raise RuntimeError(err)


def to_df(rows):
    return pd.DataFrame(rows[1:], columns=rows[0])


def fetch_table(year, table, suffix):
    base = f"https://api.census.gov/data/{year}/acs/acs5" + (f"/{suffix}" if suffix else "")
    get = f"NAME,group({table})"
    try:
        # whole state in one call
        rows = get_json(base, {"get": get, "for": "tract:*", "in": f"state:{STATE}", "key": KEY})
        return to_df(rows)
    except RuntimeError:
        pass
    # fallback: loop over counties (older vintages require it)
    counties = get_json(base, {"get": "NAME", "for": "county:*", "in": f"state:{STATE}", "key": KEY})
    frames = []
    for row in counties[1:]:
        c = row[-1]
        rows = get_json(base, {"get": get, "for": "tract:*", "in": f"state:{STATE} county:{c}", "key": KEY})
        frames.append(to_df(rows))
    return pd.concat(frames, ignore_index=True)


def clean(df, table):
    df["GEOID"] = df["state"] + df["county"] + df["tract"]
    # keep estimate columns only (E / PE), drop margins (M) and annotations (EA, MA)
    keep = [c for c in df.columns if re.search(r"\d(P?)E$", c)]
    out = df[["GEOID"] + keep].copy()
    for c in keep:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out.loc[out[c] <= -555555555, c] = pd.NA  # Census sentinel codes
    return out


for year in YEARS:
    merged = None
    for table, suffix in TABLES:
        try:
            df = clean(fetch_table(year, table, suffix), table)
            df.to_csv(os.path.join(RAW, f"{year}_{table}.csv"), index=False)
            log(f"OK   {year} {table}: {df.shape[0]} tracts x {df.shape[1]-1} vars")
            merged = df if merged is None else merged.merge(df, on="GEOID", how="outer")
        except Exception as e:
            log(f"FAIL {year} {table}: {e}")
    if merged is not None:
        merged.to_csv(os.path.join(OUT, f"acs_{year}.csv"), index=False)

# TIGER tract boundaries for the adjacency graph
tiger = os.path.join(OUT, "tl_2016_06_tract.zip")
try:
    r = requests.get("https://www2.census.gov/geo/tiger/TIGER2016/TRACT/tl_2016_06_tract.zip", timeout=120)
    r.raise_for_status()
    open(tiger, "wb").write(r.content)
    log("OK   TIGER tract shapefile")
except Exception as e:
    log(f"FAIL TIGER: {e}")

log("Done. Zip the 'data' folder and send it back.")
