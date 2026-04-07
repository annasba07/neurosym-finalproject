"""
agent.py
CareTrace Knowledge Retrieval Agent.

Wraps the Neo4j graph queries into a clean interface for the LangGraph
orchestration layer. This is the only file other agents should import from.

Usage:
    from knowledge_graph_agent.agent import KnowledgeRetrievalAgent

    agent = KnowledgeRetrievalAgent()
    result = agent.ground_symptom("burning up")
    agent.close()

    # Or use as a context manager
    with KnowledgeRetrievalAgent() as agent:
        result = agent.ground_symptom("burning up")
"""

import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

from . import queries

load_dotenv()


class KnowledgeRetrievalAgent:
    """
    Retrieves structured clinical knowledge from the Neo4j graph.

    Responsibilities:
      - Ground free-text symptom mentions to SNOMED concepts
      - Expand concepts via IS_A hierarchy
      - Retrieve red flags for a set of grounded concepts
      - Retrieve authoritative medication dosing
      - Check medication contraindications against active concepts
    """

    def __init__(self):
        uri      = os.environ["NEO4J_URI"]
        user     = os.environ["NEO4J_USER"]
        password = os.environ["NEO4J_PASSWORD"]
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ── Core Interface ─────────────────────────────────────────────────────────

    def ground_symptom(self, text_mention: str) -> dict:
        """
        Map a free-text symptom mention to SNOMED concept(s) + ancestors.

        Args:
            text_mention: Raw text from the Interpretation Agent
                          e.g. "burning up", "won't stop throwing up", "dry lips"

        Returns:
            {
                "mention":  "burning up",
                "concepts": [
                    {
                        "sctid": "386661006",
                        "fsn":   "Fever (finding)",
                        "type":  "finding",
                        "depth": 0,
                        "ancestors": [...]
                    }
                ],
                "all_sctids": ["386661006"],   # flat list including ancestors — use this for downstream calls
                "grounded":  True
            }
        """
        with self._driver.session() as session:
            concepts = queries.ground_symptom(session, text_mention)

        # Flatten all sctids including ancestors for downstream queries
        all_sctids = set()
        for c in concepts:
            all_sctids.add(c["sctid"])
            for ancestor in c.get("ancestors", []):
                if ancestor.get("sctid"):
                    all_sctids.add(ancestor["sctid"])

        return {
            "mention":    text_mention,
            "concepts":   concepts,
            "all_sctids": list(all_sctids),
            "grounded":   len(concepts) > 0,
        }

    def ground_symptoms_batch(self, text_mentions: list[str]) -> dict:
        """
        Ground multiple symptom mentions at once and return a merged sctid list.
        This is the main entry point during a triage turn — pass all extracted
        symptoms from the Interpretation Agent in one call.

        Args:
            text_mentions: List of extracted symptom strings
                           e.g. ["fever", "not urinating", "very sleepy"]

        Returns:
            {
                "results":    { mention: ground_symptom_result, ... },
                "all_sctids": [deduplicated list of all sctids + ancestors],
                "ungrounded": [mentions that had no match]
            }
        """
        results    = {}
        all_sctids = set()
        ungrounded = []

        for mention in text_mentions:
            result = self.ground_symptom(mention)
            results[mention] = result
            if result["grounded"]:
                all_sctids.update(result["all_sctids"])
            else:
                ungrounded.append(mention)

        return {
            "results":    results,
            "all_sctids": list(all_sctids),
            "ungrounded": ungrounded,
        }

    def get_red_flags(self, sctids: list[str]) -> dict:
        """
        Return any red flags triggered by the given concept IDs.
        Always pass the full expanded sctid list (including ancestors)
        from ground_symptom / ground_symptoms_batch.

        Args:
            sctids: List of SNOMED concept IDs

        Returns:
            {
                "red_flags": [
                    {
                        "rule_id":      "RF_003",
                        "description":  "Non-blanching petechial rash with fever",
                        "disposition":  "ED_NOW",
                        "triggered_by": "Petechiae (finding)"
                    },
                    ...
                ],
                "highest_disposition": "ED_NOW" | "URGENT_CARE" | "HOME" | None,
                "has_red_flags": True | False
            }
        """
        with self._driver.session() as session:
            red_flags = queries.get_red_flags(session, sctids)

        highest = self._highest_disposition(red_flags)

        return {
            "red_flags":           red_flags,
            "highest_disposition": highest,
            "has_red_flags":       len(red_flags) > 0,
        }

    def get_dosing(self, medication_name: str, weight_kg: float) -> dict:
        """
        Return authoritative dosing for a medication given child weight.

        Args:
            medication_name: e.g. "ibuprofen", "acetaminophen"
            weight_kg:       Child's weight in kilograms

        Returns:
            {
                "medication":         "ibuprofen",
                "formulation":        "oral suspension 100mg/5ml",
                "dose_mg":            140.0,
                "dose_ml":            7.0,
                "max_dose_mg":        600,
                "min_interval_hours": 6,
                "max_daily_doses":    4,
                "min_age_months":     6,
                "source":             "AAP 2023",
                "safe_for_weight":    True,
                "found":              True
            }
        """
        with self._driver.session() as session:
            result = queries.get_dosing(session, medication_name, weight_kg)

        if result is None:
            return {"found": False, "medication": medication_name}

        return {**result, "found": True}

    def get_contraindications(self, medication_name: str, sctids: list[str]) -> dict:
        """
        Check whether a medication is contraindicated given active concepts.

        Args:
            medication_name: e.g. "ibuprofen"
            sctids:          Full expanded concept ID list (including ancestors)

        Returns:
            {
                "medication":      "ibuprofen",
                "contraindicated": True,
                "reasons": [
                    {
                        "contraindicated_in": "Dehydration (disorder)",
                        "sctid": "34095006"
                    }
                ]
            }
        """
        with self._driver.session() as session:
            hits = queries.get_contraindications(session, medication_name, sctids)

        return {
            "medication":      medication_name,
            "contraindicated": len(hits) > 0,
            "reasons":         [{"contraindicated_in": h["contraindicated_in"], "sctid": h["sctid"]} for h in hits],
        }

    def get_safe_medications(self, sctids: list[str], weight_kg: float, age_months: int) -> list[dict]:
        """
        Return all medications relevant to the active concepts, filtered for
        safety given the child's weight and age.
        Convenience method that combines dosing + contraindication checks.

        Args:
            sctids:      Full expanded concept ID list
            weight_kg:   Child weight in kg
            age_months:  Child age in months

        Returns:
            List of safe medication dosing dicts, each with a "safe" flag and
            "contraindication_reasons" if any.
        """
        # Medications to evaluate (hardcoded to scoped bundle)
        candidates = ["acetaminophen", "ibuprofen"]
        results = []

        for med in candidates:
            dosing = self.get_dosing(med, weight_kg)
            if not dosing["found"]:
                continue

            contra = self.get_contraindications(med, sctids)

            # Age check
            age_ok   = age_months >= dosing["min_age_months"]
            weight_ok = dosing["safe_for_weight"]
            contra_ok = not contra["contraindicated"]

            results.append({
                **dosing,
                "safe":                   age_ok and weight_ok and contra_ok,
                "age_eligible":           age_ok,
                "weight_eligible":        weight_ok,
                "contraindicated":        contra["contraindicated"],
                "contraindication_reasons": contra["reasons"],
            })

        return results

    # ── Audit / Traceability ───────────────────────────────────────────────────

    def explain_grounding(self, text_mention: str) -> str:
        """
        Return a human-readable explanation of how a mention was grounded.
        Used by the Explanation Agent for audit-grade rationale.
        """
        result = self.ground_symptom(text_mention)
        if not result["grounded"]:
            return f'"{text_mention}" could not be mapped to a known clinical concept.'

        lines = [f'"{text_mention}" grounded to:']
        for c in result["concepts"]:
            lines.append(f'  • {c["fsn"]} (SCTID: {c["sctid"]})')
            for ancestor in c.get("ancestors", []):
                if ancestor.get("sctid"):
                    lines.append(f'    └─ IS_A → {ancestor["fsn"]} (SCTID: {ancestor["sctid"]})')
        return "\n".join(lines)

    # ── Internal Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _highest_disposition(red_flags: list[dict]) -> str | None:
        """Return the most severe disposition across a list of red flags."""
        priority = {"ED_NOW": 0, "URGENT_CARE": 1, "HOME": 2}
        if not red_flags:
            return None
        return min(red_flags, key=lambda r: priority.get(r["disposition"], 99))["disposition"]