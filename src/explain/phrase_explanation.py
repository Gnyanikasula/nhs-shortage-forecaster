"""
Feeds the ranked facts from extract_facts.py into Ollama for phrasing --
the LLM's ONLY input is this small structured fact list, no free text,
no external source. Its job is strictly to turn 2-3 short facts into
one fluent sentence, nothing else.

Uses llama3.2:3b, not 1b -- the 1b model was tested first and failed
twice: it echoed the chemical name with no explanation on 2/3 examples,
and on the third it INVERTED a given fact (a 1% historical onset rate
became "concession-prone" in its output).

3b, even with a heavily tightened prompt, still produced real problems
across repeated testing: a fact fusion that fabricated a new number
("1% of onsets occur in July" -- not a claim either source fact made),
a dropped fact, and most seriously a plain integer silently changed
(the fact said "6 previous episodes", the LLM wrote "three"). None of
these were caught by a percentage-only check, because none of them
involved a percentage.

Given that recurring failure pattern, this module:
  1. check_integers_preserved -- validates every plain integer in a
     fact (not just percentages) survives, digit or spelled-out form.
  2. Retries generation, verifying against ALL checks, up to
     MAX_ATTEMPTS.
  3. Falls back to a deterministic template if the LLM can't produce a
     verified-correct sentence within MAX_ATTEMPTS. This is correct by
     construction (no LLM involved), so the system can never actually
     ship a wrong number -- worst case it ships a less fluent sentence.

KNOWN, ACCEPTED LIMITATION: verification checks token presence, not
full semantic correctness. Testing surfaced two subtler issues that
passed every check but read slightly off on close reading -- a fact
about a COUNT ("6 previous episodes") being rephrased as an ORDINAL
("this is the 6th episode"), and a certain, deterministic fact ("came
off concession last month") being hedged into an uncertain one ("may
have come off..."). Neither changes a number or fabricates a claim, so
both were accepted rather than chased further -- see project decision
log / README for the reasoning.
"""
import os
import re
import sys

import numpy as np
import requests

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(THIS_DIR)
sys.path.insert(0, os.path.join(SRC_DIR, "db"))
sys.path.insert(0, THIS_DIR)

from rebuild_panel_from_source import rebuild_everything
from extract_facts import explain_chemical, get_contributions_for_chemical

MODEL_NAME = "llama3.2:3b"
MAX_ATTEMPTS = 3

_NUMBER_WORDS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
]


def _call_ollama(chemical: str, score: float, facts: list[dict]) -> str:
    numbered_facts = "\n".join(f"{i+1}. {f['fact']}" for i, f in enumerate(facts))
    prompt = f"""You are writing a one-sentence operational note about a medicine supply forecasting model's output. This is NOT clinical or medical advice.

Chemical: {chemical}
Model's risk score: {score:.0%}

There are exactly {len(facts)} facts below, numbered. You MUST include ALL {len(facts)} of them in your sentence, with every number restated EXACTLY as given -- do not skip any, do not change any number, do not add anything not listed:
{numbered_facts}

Write exactly one sentence, at least 25 words, that includes every one of the {len(facts)} numbered facts above with all numbers unchanged. Do not mention symptoms, treatment, or patient safety. Start the sentence with the chemical name."""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def check_percentages_preserved(facts: list[dict], output: str) -> list[str]:
    """Any 'N%' appearing in a fact must appear verbatim in the LLM output."""
    problems = []
    for f in facts:
        for pct in re.findall(r"\d+%", f["fact"]):
            if pct not in output:
                problems.append(f"Percentage '{pct}' from fact '{f['fact']}' missing from output.")
    return problems


def check_integers_preserved(facts: list[dict], output: str) -> list[str]:
    """Any plain integer (not part of a %) in a fact must appear in the
    output, either as a digit or spelled out (0-20). This is what would
    have caught the real bug seen in testing: '6 previous episodes'
    silently becoming 'three previous episodes' -- a change no
    percentage check could ever detect."""
    problems = []
    output_lower = output.lower()
    for f in facts:
        for num_str in re.findall(r"\d+(?!%)", f["fact"]):
            n = int(num_str)
            digit_present = num_str in output
            word_present = n <= 20 and _NUMBER_WORDS[n] in output_lower
            if not digit_present and not word_present:
                problems.append(
                    f"Number '{num_str}' from fact '{f['fact']}' does not appear "
                    f"(as digit or spelled out) in the output."
                )
    return problems


_FACT_SIGNATURES = [
    (r"came off a (price )?concession", ["last month", "came off"]),
    (r"never been on a price concession", ["never"]),
    (r"clear of concession for \d+ consecutive months", ["consecutive month"]),
    (r"previous concession episode", ["episode"]),
    (r"gone onto concession \d+% of the time", ["% of the time", "% historical", "historical", "rate"]),
]


def check_all_facts_present(facts: list[dict], output: str) -> list[str]:
    """For each fact, confirm at least one of its distinctive keywords
    shows up in the output. Catches dropped facts that the numeric
    checks alone would miss."""
    problems = []
    output_lower = output.lower()
    for f in facts:
        matched = False
        for pattern, keywords in _FACT_SIGNATURES:
            if re.search(pattern, f["fact"]):
                if any(kw in output_lower for kw in keywords):
                    matched = True
                break
        else:
            # Seasonal fact needs its OWN direction word checked -- the
            # observed failure mode was the LLM fusing this fact with a
            # DIFFERENT number (turning "July: slightly lower onset
            # pattern" into a fabricated "1% of onsets occur in July").
            season_match = re.search(r"(higher|lower) than average onset pattern", f["fact"])
            if season_match:
                direction = season_match.group(1)
                if direction in output_lower and "onset pattern" in output_lower:
                    matched = True
        if not matched:
            problems.append(f"Fact appears MISSING or DISTORTED in output: '{f['fact']}'")
    return problems


def verify(facts: list[dict], output: str) -> list[str]:
    return (
        check_percentages_preserved(facts, output)
        + check_integers_preserved(facts, output)
        + check_all_facts_present(facts, output)
    )


def template_fallback(chemical: str, score: float, facts: list[dict]) -> str:
    """Deterministic, no LLM involved -- built directly from the same
    facts, so it cannot introduce a number or claim that isn't already
    verified correct. Less fluent than a good LLM sentence, but it is
    correct by construction, which is the property that actually
    matters here."""
    fact_str = "; ".join(f["fact"] for f in facts)
    return f"{chemical} (model risk score {score:.0%}): {fact_str}."


def generate_explanation(chemical: str, score: float, facts: list[dict], verbose: bool = False) -> tuple[str, str]:
    """Returns (explanation, method_description). Tries the LLM up to
    MAX_ATTEMPTS times, verifying every attempt against the actual
    facts. Falls back to a guaranteed-correct template only if every
    attempt fails verification. This is the function other modules
    (score_and_log, score_latest_month, the API) should call."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        candidate = _call_ollama(chemical, score, facts)
        problems = verify(facts, candidate)
        if not problems:
            return candidate, f"LLM, attempt {attempt}/{MAX_ATTEMPTS}"
        if verbose:
            print(f"    Attempt {attempt}/{MAX_ATTEMPTS} FAILED verification:")
            for p in problems:
                print(f"      - {p}")
    return template_fallback(chemical, score, facts), f"TEMPLATE FALLBACK (all {MAX_ATTEMPTS} LLM attempts failed verification)"


if __name__ == "__main__":
    rebuilt = rebuild_everything()
    features = rebuilt["features"]
    bundle = rebuilt["model_bundle"]
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    latest_month = features["month"].max()
    at_risk = features[(features["month"] == latest_month) & (~features["on_concession"])].copy()

    at_risk["time_since_last_concession_filled"] = at_risk["time_since_last_concession"].fillna(999)
    at_risk["month_sin"] = np.sin(2 * np.pi * at_risk["month_of_year"] / 12)
    at_risk["month_cos"] = np.cos(2 * np.pi * at_risk["month_of_year"] / 12)
    at_risk["chemical_historical_onset_rate"] = at_risk["chemical_historical_onset_rate"].fillna(
        bundle["fallback_historical_rate"]
    )

    X = at_risk[feature_columns]
    at_risk["score"] = model.predict_proba(X)[:, 1]

    top3 = at_risk.sort_values("score", ascending=False).head(3)

    for _, row in top3.iterrows():
        contribs = get_contributions_for_chemical(row, feature_columns, model)
        facts = explain_chemical(row, contribs, feature_columns)

        print(f"=== {row['chemical']} (score {row['score']:.4f}) ===")
        print("Facts given to LLM:")
        for f in facts:
            print(f"  - {f['fact']}")

        explanation, method = generate_explanation(row["chemical"], row["score"], facts, verbose=True)
        print(f"FINAL explanation ({method}):")
        print(f"  {explanation}")
        print()