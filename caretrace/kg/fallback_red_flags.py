"""Dict-mode red-flag coverage.

When David's Neo4j KG is offline, we still need to catch the high-severity
signals the KG would normally fire: infant fever, very high fever, seizure,
breathing difficulty. Alex's pure dehydration rules don't cover these, so
we need this fallback for safe operation without a live graph.

The output shape matches KnowledgeRetrievalAgent.get_red_flags() so the
safety layer doesn't care which backend produced them.
"""

from typing import Any


# Each rule is a closure that takes the full state and returns a red flag
# dict or None. Dispositions use Alex's vocabulary.
def _infant_fever(state: dict) -> dict | None:
    age = state.get("age_months")
    fever = state.get("facts", {}).get("fever")
    if age is not None and age < 3 and fever == "yes":
        return {
            "rule_id": "FB_INFANT_FEVER",
            "description": "Any fever in an infant under 3 months is an emergency.",
            "disposition": "er_now",
            "triggered_by": "age < 3 months with fever",
        }
    return None


def _very_high_fever(state: dict) -> dict | None:
    temp = state.get("temperature_f")
    if temp is not None and temp >= 104.0:
        return {
            "rule_id": "FB_VERY_HIGH_FEVER",
            "description": "Temperature ≥104°F warrants urgent evaluation.",
            "disposition": "urgent_eval",
            "triggered_by": f"temperature_f = {temp}",
        }
    return None


def _seizure(state: dict) -> dict | None:
    if state.get("facts", {}).get("seizure") == "yes":
        return {
            "rule_id": "FB_SEIZURE",
            "description": "A febrile seizure requires immediate emergency evaluation.",
            "disposition": "er_now",
            "triggered_by": "seizure reported",
        }
    return None


def _breathing_difficulty(state: dict) -> dict | None:
    if state.get("facts", {}).get("breathing") == "difficulty":
        return {
            "rule_id": "FB_BREATHING",
            "description": "Difficulty breathing in a febrile child is an emergency.",
            "disposition": "er_now",
            "triggered_by": "breathing difficulty",
        }
    return None


def _petechial_rash(state: dict) -> dict | None:
    # Only fires if rash is present AND we also have a fever — proxy for
    # meningococcal concern. Pure fallback, not as nuanced as the KG would be.
    facts = state.get("facts", {})
    if facts.get("rash") == "yes" and facts.get("fever") == "yes":
        return {
            "rule_id": "FB_RASH_FEVER",
            "description": "New rash with fever — check for non-blanching spots urgently.",
            "disposition": "urgent_eval",
            "triggered_by": "rash with fever",
        }
    return None


_FALLBACK_RULES: list = [
    _infant_fever,
    _seizure,
    _breathing_difficulty,
    _very_high_fever,
    _petechial_rash,
]


def check_fallback_red_flags(state: dict) -> list[dict]:
    """Run all fallback red-flag checks against the state.

    Returns a list of red-flag dicts. Each dict has the same shape as a
    Neo4j-sourced red flag but with source='fallback' so the demo notebook
    can distinguish provenance.
    """
    hits = []
    for rule in _FALLBACK_RULES:
        result = rule(state)
        if result is not None:
            result["source"] = "fallback"
            hits.append(result)
    return hits


def highest_disposition(red_flags: list[dict]) -> str | None:
    """Return the most severe disposition across a list of red flags."""
    if not red_flags:
        return None
    priority = {"er_now": 0, "urgent_eval": 1, "home_monitor": 2}
    return min(red_flags, key=lambda r: priority.get(r.get("disposition", ""), 99))[
        "disposition"
    ]
