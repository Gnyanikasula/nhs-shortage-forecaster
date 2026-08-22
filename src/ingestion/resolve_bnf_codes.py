"""
Resolves our concession-panel chemical names (derived via text-stripping in
build_chemical_panel.py, e.g. "Atomoxetine") to BNF chemical codes, using
OpenPrescribing's name-search endpoint:

    /api/1.0/bnf_code/?q=<name>&format=json

This endpoint returns BNF sections, chemicals, AND presentations matching
the query -- we only want rows of type "chemical", and the query is a
substring/fuzzy match so a name can return multiple candidates (e.g.
different salt forms, or genuinely different chemicals sharing a word).

Usage:
    python resolve_bnf_codes.py --diagnose
    python resolve_bnf_codes.py --out chemical_to_bnf_code.csv
"""
import argparse
import time
import requests
import pandas as pd

API_BASE = "https://openprescribing.net/api/1.0/bnf_code/"
HEADERS = {"User-Agent": "student-portfolio-project (contact: replace-with-your-email)"}
REQUEST_DELAY_SECONDS = 0.5

KNOWN_ANSWERS = {
    "Dexamfetamine": "0404000L0",
    "Methylphenidate": "0404000M0",
    "Atomoxetine": "0404000S0",
}


def resolve_chemical(name: str) -> dict:
    """Returns a row with the resolved BNF code, or an explanation of why
    it couldn't resolve cleanly (never silently guesses on real ambiguity).

    Matching logic, in order:
      1. Exact case-insensitive PREFIX match among type=='chemical' results
         (our name is a prefix of theirs, e.g. "Atomoxetine" is a prefix of
         "Atomoxetine hydrochloride"). If there's exactly one, use it.
         Prefix, not substring: "Dexamfetamine" is a SUBSTRING of
         "Lisdexamfetamine" but NOT a prefix of it, so this correctly
         excludes that false match without extra special-casing.
      2. If zero prefix matches, fall back to substring matches, flagged
         as approximate rather than treated as equally trustworthy.
      3. If MULTIPLE prefix matches (e.g. "Aciclovir" matches 3 different
         BNF sections, AND "Aciclovir sodium" which is prefix-matched too
         since it also starts with "aciclovir"), this is genuine ambiguity
         that can't be resolved from the bare name alone -- flag ALL
         candidates for manual review rather than picking one arbitrarily.
    """
    resp = requests.get(API_BASE, params={"q": name, "format": "json"}, headers=HEADERS, timeout=30)
    if not resp.ok:
        return {"chemical": name, "bnf_code": None, "bnf_name": None,
                "status": f"HTTP {resp.status_code}", "all_candidates": None}

    data = resp.json()
    chemical_matches = [d for d in data if isinstance(d, dict) and d.get("type") == "chemical"]

    if not chemical_matches:
        return {"chemical": name, "bnf_code": None, "bnf_name": None,
                "status": "no chemical-type match found", "all_candidates": None}

    name_lower = name.strip().lower()
    prefix_matches = [
        m for m in chemical_matches
        if m.get("name", "").strip().lower().startswith(name_lower)
    ]

    candidates_str = "; ".join(f"{m['id']}={m['name']} ({m.get('section','?')})" for m in chemical_matches)

    if len(prefix_matches) == 1:
        m = prefix_matches[0]
        return {"chemical": name, "bnf_code": m.get("id"), "bnf_name": m.get("name"),
                "status": "ok (prefix match)", "all_candidates": candidates_str}

    if len(prefix_matches) > 1:
        return {"chemical": name, "bnf_code": None, "bnf_name": None,
                "status": f"AMBIGUOUS: {len(prefix_matches)} prefix matches across different sections -- manual review needed",
                "all_candidates": "; ".join(f"{m['id']}={m['name']} ({m.get('section','?')})" for m in prefix_matches)}

    return {"chemical": name, "bnf_code": chemical_matches[0].get("id"),
            "bnf_name": chemical_matches[0].get("name"),
            "status": f"APPROXIMATE: no exact/prefix match, {len(chemical_matches)} substring matches, took first -- verify",
            "all_candidates": candidates_str}


def diagnose(names: list[str]):
    for name in names:
        print(f"\n=== Query: '{name}' ===")
        resp = requests.get(API_BASE, params={"q": name, "format": "json"}, headers=HEADERS, timeout=30)
        print(f"HTTP status: {resp.status_code}")
        if not resp.ok:
            print(resp.text[:1000])
            continue
        data = resp.json()
        print(f"Response type: {type(data)}, length: {len(data) if hasattr(data, '__len__') else 'n/a'}")
        print("Raw response (first 5 entries):")
        for entry in (data[:5] if isinstance(data, list) else [data]):
            print(f"  {entry}")
        if name in KNOWN_ANSWERS:
            print(f"  >>> Expected BNF code for '{name}': {KNOWN_ANSWERS[name]} -- does any entry above match?")

        resolved = resolve_chemical(name)
        print(f"  RESOLVED: {resolved}")
        time.sleep(REQUEST_DELAY_SECONDS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--chemicals-file", default="data/interim/vmpp_to_chemical_map.csv")
    parser.add_argument("--out", default="chemical_to_bnf_code.csv")
    args = parser.parse_args()

    if args.diagnose:
        diagnose(["Atomoxetine", "Dexamfetamine", "Aciclovir", "Co-amilofruse"])
        return

    chem_map = pd.read_csv(args.chemicals_file)
    unique_chemicals = sorted(chem_map["canonical_chemical"].unique())
    print(f"Resolving {len(unique_chemicals)} distinct chemicals...")

    results = []
    for i, name in enumerate(unique_chemicals):
        r = resolve_chemical(name)
        results.append(r)
        if r["status"] != "ok (prefix match)":
            print(f"  [{i+1}/{len(unique_chemicals)}] {name}: {r['status']}")
        time.sleep(REQUEST_DELAY_SECONDS)

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.out, index=False)

    n_ok = (out_df["status"] == "ok (prefix match)").sum()
    n_ambiguous = out_df["status"].str.startswith("AMBIGUOUS").sum()
    n_approx = out_df["status"].str.startswith("APPROXIMATE").sum()
    n_failed = len(out_df) - n_ok - n_ambiguous - n_approx
    print(f"\nResolved: {n_ok} clean, {n_ambiguous} ambiguous (manual review), "
          f"{n_approx} approximate (verify), {n_failed} failed/not found")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()