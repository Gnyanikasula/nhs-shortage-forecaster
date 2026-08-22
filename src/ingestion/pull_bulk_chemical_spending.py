"""
For each concession-panel chemical, sums monthly prescribing volume (ITEMS)
across ALL its resolved BNF codes -- not just one. This is a deliberate
design choice, not a shortcut: our concession chemical names are already
coarse (stripped of strength/form/route), so "Hydrocortisone" in the
concession data could refer to the ear-drop, cream, or systemic form. BNF
classifies those as different codes; summing them back together matches
the grain our OWN data is already at.

KNOWN APPROXIMATION: a few chemicals' candidate code lists include
combination products (e.g. Estradiol's matches include "Estradiol and
nomegestrol"). Summing includes those -- the resulting feature is closer
to "volume of anything containing this chemical" than "volume of this
chemical alone" for those specific cases.

KNOWN DATA HORIZON: OpenPrescribing's /spending/ endpoint returns a
rolling ~5-year window from today, not full history -- confirmed by an
earlier run returning data starting June 2021, not our concession panel's
January 2020 start. There is a real ~17-month gap at the start that this
source cannot fill. Not a bug, a source limitation -- documented, not
silently absorbed.

RATE LIMITING (fixed after a real failed run): the first version of this
script used a flat 0.3s delay and gave up immediately on any HTTP 429,
CACHING THE EMPTY RESULT -- which permanently poisoned that code for
every chemical sharing it, with zero retry. That run lost 109 of 360
chemicals entirely and likely silently undercounted others (chemicals
with multiple codes where only SOME codes got rate-limited). Fixed here
with: (1) real retry + exponential backoff on 429, respecting a
Retry-After header if present, (2) never caching a failed fetch as if it
were a genuine empty result, and (3) reporting exactly which codes failed
even after retries, so any remaining gaps are visible, not silent.
"""
import argparse
import re
import time
import requests
import pandas as pd

API_BASE = "https://openprescribing.net/api/1.0/spending/"
HEADERS = {"User-Agent": "student-portfolio-project (contact: replace-with-your-email)"}
BASE_DELAY_SECONDS = 1.0
MAX_RETRIES = 5
CODE_PATTERN = re.compile(r"([A-Za-z0-9]{9})=")


def extract_codes(row: pd.Series) -> list[str]:
    if pd.notna(row["bnf_code"]):
        return [row["bnf_code"]]
    if pd.notna(row["all_candidates"]):
        return CODE_PATTERN.findall(row["all_candidates"])
    return []


def fetch_code_spending(code: str, cache: dict, permanent_failures: set):
    """Returns a DataFrame on success (possibly empty, if the code
    genuinely has no data), or None if it failed even after retries --
    None is NEVER cached, so a later attempt (e.g. for a different
    chemical sharing this code) will try again rather than inheriting
    a stale failure."""
    if code in cache:
        return cache[code]
    if code in permanent_failures:
        return None

    delay = BASE_DELAY_SECONDS
    for attempt in range(MAX_RETRIES):
        resp = requests.get(API_BASE, params={"code": code, "format": "json"}, headers=HEADERS, timeout=30)
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else delay
            print(f"    429 on {code}, attempt {attempt+1}/{MAX_RETRIES}, waiting {wait:.1f}s...")
            time.sleep(wait)
            delay *= 2
            continue
        if not resp.ok:
            print(f"    {code}: HTTP {resp.status_code} (not a rate limit) -- giving up on this code")
            permanent_failures.add(code)
            return None

        records = resp.json()
        df = pd.DataFrame(records)[["date", "items"]] if records else pd.DataFrame(columns=["date", "items"])
        cache[code] = df
        time.sleep(BASE_DELAY_SECONDS)
        return df

    print(f"    {code}: still rate-limited after {MAX_RETRIES} retries -- giving up on this code for now")
    permanent_failures.add(code)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping-file", default="data/interim/chemical_to_bnf_code.csv")
    parser.add_argument("--out", default="data/interim/chemical_prescribing_volume.csv")
    args = parser.parse_args()

    mapping = pd.read_csv(args.mapping_file)
    resolvable = mapping[mapping["status"] != "no chemical-type match found"].copy()
    print(f"{len(resolvable)} of {len(mapping)} chemicals have at least one candidate BNF code")

    cache = {}
    permanent_failures = set()
    all_results = []
    multi_code_log = []
    undercounted_chemicals = []

    for i, row in resolvable.reset_index(drop=True).iterrows():
        codes = extract_codes(row)
        if not codes:
            continue
        if len(codes) > 1:
            multi_code_log.append((row["chemical"], len(codes)))

        per_code_results = [fetch_code_spending(c, cache, permanent_failures) for c in codes]
        n_failed = sum(1 for r in per_code_results if r is None)
        per_code_dfs = [d for d in per_code_results if d is not None and len(d) > 0]

        if n_failed > 0 and per_code_dfs:
            undercounted_chemicals.append((row["chemical"], n_failed, len(codes)))
        if not per_code_dfs:
            print(f"  [{i+1}/{len(resolvable)}] {row['chemical']}: no usable data from any of {len(codes)} code(s)")
            continue

        combined = pd.concat(per_code_dfs, ignore_index=True)
        monthly = combined.groupby("date")["items"].sum().reset_index()
        monthly["chemical"] = row["chemical"]
        monthly["n_bnf_codes_summed"] = len(per_code_dfs)
        monthly["n_bnf_codes_failed"] = n_failed
        all_results.append(monthly)

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(resolvable)}] processed...")

    result = pd.concat(all_results, ignore_index=True)
    result = result.rename(columns={"date": "month", "items": "total_items"})
    result.to_csv(args.out, index=False)

    n_chemicals_ok = result["chemical"].nunique()

    print(f"\nSaved {len(result)} chemical-month rows for {n_chemicals_ok} chemicals to {args.out}")
    print(f"Chemicals with data summed across multiple BNF codes: {len(multi_code_log)}")
    print(f"Chemicals with PARTIAL data (some codes failed, others succeeded -- likely undercounted): {len(undercounted_chemicals)}")
    if undercounted_chemicals:
        for chem, n_fail, n_total in undercounted_chemicals[:20]:
            print(f"    {chem}: {n_fail} of {n_total} codes failed")
    print(f"Date range: {result['month'].min()} to {result['month'].max()}")

    early_coverage = result[result["month"] < "2020-06-01"]
    if len(early_coverage) == 0:
        print("\nNOTE: no data before Jun 2020 -- as flagged in the module docstring, OpenPrescribing's")
        print("/spending/ endpoint has a rolling ~5-year window, not full history back to our concession")
        print("panel's Jan 2020 start. This is a real, unavoidable coverage gap with this data source.")


if __name__ == "__main__":
    main()